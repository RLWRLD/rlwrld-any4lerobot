"""Benchmark helpers.

The best parallelism and encoder settings depend on source resolution, storage
bandwidth and core count, so they are measured on a sample rather than guessed.
Throughput is reported per core and in megapixels/s as well as raw fps, because
fps alone is not comparable across datasets -- decoding 1080p costs far more than
VGA at the same frame rate.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .meta import feature_shape, load_info, video_files, video_keys
from .registry import build_step
from .transform import TranscodeJob, plan_transform
from .video_ops import Parallelism, build_ffmpeg_command, plan_parallelism


@dataclass(frozen=True)
class BenchSummary:
    frames: int
    files: int
    wall_clock_s: float
    fps: float
    fps_per_core: float
    megapixels_per_s: float
    workers: int
    threads: int
    cores: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_videos(root: str | Path, patterns: Iterable[str]) -> list[Path]:
    """Find sample videos under ``root`` matching any of ``patterns``."""
    root = Path(root)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def summarize(
    frames: int,
    in_pixels: int,
    files: int,
    wall_clock_s: float,
    parallelism: Parallelism,
    cores: int,
) -> BenchSummary:
    fps = frames / wall_clock_s if wall_clock_s > 0 else 0.0
    megapixels_per_s = (in_pixels / wall_clock_s / 1e6) if wall_clock_s > 0 else 0.0

    return BenchSummary(
        frames=frames,
        files=files,
        wall_clock_s=wall_clock_s,
        fps=fps,
        fps_per_core=fps / cores if cores > 0 else 0.0,
        megapixels_per_s=megapixels_per_s,
        workers=parallelism.workers,
        threads=parallelism.threads,
        cores=cores,
    )


def project_seconds(summary: BenchSummary, total_frames: int) -> float:
    """Extrapolate a full-run wall clock from a sample measurement."""
    if summary.fps <= 0:
        return 0.0
    return total_frames / summary.fps


def sample(paths: Sequence[Path], count: int) -> list[Path]:
    """Take a size-spread sample.

    Episode lengths vary a lot; measuring only the smallest files would
    under-estimate the real run badly.
    """
    if count >= len(paths):
        return list(paths)
    ordered = sorted(paths, key=lambda path: path.stat().st_size)
    step = len(ordered) / count
    return [ordered[int(index * step)] for index in range(count)]


def format_summary(summary: BenchSummary, total_frames: int | None = None) -> str:
    lines = [
        f"{summary.files} file(s), {summary.frames:,} frames in {summary.wall_clock_s:.1f}s",
        f"  {summary.workers} workers x {summary.threads} threads on {summary.cores} cores",
        f"  {summary.fps:,.0f} fps  ({summary.fps_per_core:,.0f} fps/core, "
        f"{summary.megapixels_per_s:,.1f} MP/s)",
    ]
    if total_frames:
        projected = project_seconds(summary, total_frames)
        lines.append(
            f"  projected for {total_frames:,} frames: {projected / 60:,.1f} min"
        )
    return "\n".join(lines)


def _bench_transcode(job: "TranscodeJob", threads: int) -> int:
    """Run one job with progress reporting and return the frame count.

    ``-progress`` gives exact frame counts without a second decode pass, which a
    separate probe would cost.
    """
    command = build_ffmpeg_command(job.src, job.dst, job.filters, job.encoding, threads)
    command = command[:1] + ["-progress", "pipe:1", "-nostats"] + command[1:]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {job.src}: {result.stderr.strip()}")

    frames = 0
    for line in result.stdout.splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "frame":
            try:
                frames = int(value.strip())
            except ValueError:
                continue
    return frames


def measure(
    root: str | Path,
    sample_size: int = 10,
    workdir: str | Path | None = None,
    steps: Sequence[Any] | None = None,
    preset: str | None = None,
    crf: int | None = None,
    workers: int | None = None,
    threads_per_ffmpeg: int | None = None,
    cores: int | None = None,
) -> BenchSummary:
    """Transcode a sample of the dataset and report measured throughput.

    The sample is written into a scratch directory that is always removed, so a
    benchmark never leaves a half-converted dataset behind.
    """
    root = Path(root)
    cores = cores or os.cpu_count() or 1
    scratch = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix="lerobot-pipeline-bench-")
    )
    if steps is None:
        steps = [build_step({"type": "resize_preserve_aspect_area"})]

    source_info = load_info(root)
    plan = plan_transform(root, scratch / "out", steps, preset=preset, crf=crf)
    if not plan.transcodes:
        shutil.rmtree(scratch, ignore_errors=True)
        raise ValueError(
            "these steps would not re-encode anything in this dataset; "
            "there is nothing to benchmark"
        )

    pixels_by_source: dict[Path, int] = {}
    for key in video_keys(source_info):
        height, width = feature_shape(source_info, key)[:2]
        for path in video_files(root, source_info, key):
            pixels_by_source[path] = height * width

    chosen = set(sample([job.src for job in plan.transcodes], sample_size))
    jobs = [job for job in plan.transcodes if job.src in chosen]
    parallelism = plan_parallelism(len(jobs), cores, workers, threads_per_ffmpeg)

    for job in jobs:
        job.dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        if parallelism.workers == 1:
            frame_counts = [_bench_transcode(job, parallelism.threads) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=parallelism.workers) as executor:
                frame_counts = list(
                    executor.map(
                        _bench_transcode, jobs, [parallelism.threads] * len(jobs)
                    )
                )
        wall_clock_s = time.perf_counter() - started
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    frames = sum(frame_counts)
    in_pixels = sum(
        count * pixels_by_source.get(job.src, 0)
        for job, count in zip(jobs, frame_counts)
    )

    return summarize(
        frames=frames,
        in_pixels=in_pixels,
        files=len(jobs),
        wall_clock_s=wall_clock_s,
        parallelism=parallelism,
        cores=cores,
    )


def count_transcodes(root: str | Path, steps: Sequence[Any]) -> int:
    """How many video files these steps would actually re-encode.

    Keys already at the target size are hard-linked, so counting every video file
    would over-estimate a projected run.
    """
    plan = plan_transform(root, Path(root).parent / "__unused__", steps)
    return len(plan.transcodes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot_pipeline.bench",
        description="Measure preprocessing throughput on a sample before committing "
        "to a full run.",
    )
    parser.add_argument("--config", type=Path, required=True, help="pipeline config YAML")
    parser.add_argument(
        "--sample", type=int, default=10, help="how many video files to transcode"
    )
    parser.add_argument(
        "--presets",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=None,
        help="comma-separated x264 presets to compare, e.g. ultrafast,veryfast,fast",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-ffmpeg", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if not config.source.is_lerobot:
        raise SystemExit(
            "bench works on LeRobot sources; run the converter stage first for "
            f"source.type={config.source.type!r}"
        )

    total_files = count_transcodes(config.source.path, config.steps)

    for preset in args.presets or [config.runtime.preset]:
        summary = measure(
            config.source.path,
            sample_size=args.sample,
            steps=config.steps,
            preset=preset,
            crf=config.runtime.crf,
            workers=args.workers or config.runtime.workers,
            threads_per_ffmpeg=args.threads_per_ffmpeg
            or config.runtime.threads_per_ffmpeg,
        )
        estimated_total_frames = (
            int(summary.frames / summary.files * total_files) if summary.files else 0
        )
        print(f"preset={preset or 'source default'}")
        print(format_summary(summary, estimated_total_frames))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
