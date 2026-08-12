"""Measure resize throughput on a directory of raw videos.

Runs the real preprocessing path -- the same planner, ffmpeg command builder and
parallel executor the pipeline uses -- so the numbers describe the tool rather
than a synthetic proxy. Intended for comparing machines and parallelism settings.

    python3 -m lerobot_pipeline.scripts.bench_raw_videos \
        --input /scratch/data --pattern 'rgb.mp4' --sample 24 \
        --workers 8 --threads 1 --label c7gd-8w1t --json-out /scratch/out.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from ..bench import collect_videos, format_summary, sample, summarize
from ..steps.resize import (
    DEFAULT_MAX_AREA,
    DEFAULT_MULTIPLE,
    ResizePreserveAspectArea,
)
from ..transform import TranscodeJob, run_transcodes
from ..video_ops import EncodingParams, plan_parallelism, probe_video


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor() or "unknown"
    for label in ("model name", "Model name", "CPU implementer", "Processor"):
        for line in text.splitlines():
            if line.startswith(label):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def _physical_cores() -> int | None:
    try:
        out = subprocess.run(
            ["lscpu", "-p=Core,Socket"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pairs = {line for line in out.splitlines() if line and not line.startswith("#")}
    return len(pairs) or None


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=30
        ).stdout
        return out.splitlines()[0] if out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench_raw_videos")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--sample", type=int, default=0, help="0 = every file")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAX_AREA)
    parser.add_argument("--multiple", type=int, default=DEFAULT_MULTIPLE)
    parser.add_argument("--label", default="")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    videos = collect_videos(args.input, tuple(args.pattern or ("*.mp4",)))
    if not videos:
        parser.error(f"no videos matched under {args.input}")
    if args.sample:
        videos = sample(videos, args.sample)

    step = ResizePreserveAspectArea(max_area=args.max_area, multiple=args.multiple)
    encoding = EncodingParams(preset=args.preset, crf=args.crf, gop=args.gop)

    jobs: list[TranscodeJob] = []
    shapes: dict[str, int] = {}
    frames = 0
    in_pixels = 0
    unknown_frame_counts = 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(videos):
        info = probe_video(src)
        composed = step.plan(info.shape)
        if composed is None:
            continue
        key = f"{info.height}x{info.width}->{composed.out_shape[0]}x{composed.out_shape[1]}"
        shapes[key] = shapes.get(key, 0) + 1
        if info.frames is None:
            unknown_frame_counts += 1
        else:
            frames += info.frames
            in_pixels += info.frames * info.height * info.width
        jobs.append(
            TranscodeJob(
                src=src,
                dst=args.out_dir / f"{index:05d}_{src.stem}.mp4",
                filters=composed.filters,
                encoding=encoding,
            )
        )

    if not jobs:
        print("every video already matches the target shape; nothing to measure")
        return 1

    # longest-processing-time-first, matching the pipeline's own scheduling
    jobs.sort(key=lambda job: -job.src.stat().st_size)

    cores = args.cores or os.cpu_count() or 1
    parallelism = plan_parallelism(
        file_count=len(jobs),
        cores=cores,
        workers=args.workers,
        threads_per_ffmpeg=args.threads,
    )

    started = time.perf_counter()
    run_transcodes(jobs, parallelism)
    wall_clock = time.perf_counter() - started

    out_bytes = sum(job.dst.stat().st_size for job in jobs if job.dst.exists())
    in_bytes = sum(job.src.stat().st_size for job in jobs)

    summary = summarize(
        frames=frames,
        in_pixels=in_pixels,
        files=len(jobs),
        wall_clock_s=wall_clock,
        parallelism=parallelism,
        cores=cores,
    )

    payload = {
        "label": args.label or os.environ.get("BENCH_LABEL", ""),
        "instance_type": os.environ.get("BENCH_INSTANCE_TYPE", ""),
        "machine": {
            "arch": platform.machine(),
            "cpu_model": _cpu_model(),
            "vcpus": os.cpu_count(),
            "physical_cores": _physical_cores(),
            "ffmpeg": _ffmpeg_version(),
            "python": platform.python_version(),
        },
        "encoding": {
            "preset": args.preset,
            "crf": args.crf,
            "gop": args.gop,
            "max_area": args.max_area,
            "multiple": args.multiple,
        },
        "shapes": shapes,
        "unknown_frame_counts": unknown_frame_counts,
        "bytes": {"input": in_bytes, "output": out_bytes},
        "summary": summary.as_dict(),
    }

    print(format_summary(summary))
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
