"""Compare a rebuilt dataset against the delivered copy it is meant to reproduce.

    uv run python -m dataset_registry.compare action_net --rebuilt /out --delivered /ref

The two are held to different standards, because they can be:

* **state and action must be identical.** Every slot is a float32 copied out of the
  source, so a rebuild that reads the same bytes and lines the clocks up the same
  way produces the same numbers exactly. A difference here is a real difference in
  what the dataset says, and the tool reports it as a failure, not a tolerance.
* **video must match in geometry, frame count and codec settings; its bytes will
  not match.** Two ffmpeg builds with the same flags do not emit the same file, so
  the size is reported as a ratio and judged loosely. If the geometry or the frame
  count differs, that is a real failure -- those are decided by our own code, not by
  the encoder.

Episodes carry no source id in the delivered copy, so they are aligned by position
and the alignment is *checked* rather than assumed: the task prompt of each pair
must agree. One dropped episode shifts everything after it, and comparing shifted
episodes would produce a wall of meaningless differences, so a prompt mismatch stops
the run.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import DatasetSpec, available, load

# ffmpeg is not deterministic across builds; this bounds "the same picture, encoded
# again" against "we encoded something else"
SIZE_TOLERANCE = 0.15


class CompareError(RuntimeError):
    pass


@dataclass
class EpisodeReport:
    index: int
    rows_rebuilt: int | None = None
    rows_delivered: int | None = None
    prompt_matches: bool | None = None
    columns: dict[str, str] = field(default_factory=dict)
    video: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def episode_prompts(root: Path) -> dict[int, str]:
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        tasks = entry.get("tasks") or []
        out[entry["episode_index"]] = tasks[0] if tasks else ""
    return out


def episode_parquet(root: Path, index: int) -> Path | None:
    matches = sorted(root.glob(f"data/**/episode_{index:06d}.parquet"))
    return matches[0] if matches else None


def episode_videos(root: Path, index: int) -> dict[str, Path]:
    out = {}
    for path in sorted(root.glob(f"videos/**/episode_{index:06d}.mp4")):
        out[path.parent.name] = path
    return out


def probe(path: Path) -> dict[str, Any]:
    """Geometry, frame count and codec settings of one mp4."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,has_b_frames,nb_read_frames",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as exc:
        raise CompareError("ffprobe not found; it is needed to compare video") from exc
    if result.returncode != 0:
        raise CompareError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or [{}]
    stream = streams[0]
    stream["bytes"] = path.stat().st_size
    return stream


def compare_vectors(rebuilt, delivered, keys: list[str]) -> tuple[dict, list[str]]:
    """State and action, slot by slot. Exact -- no tolerance."""
    import numpy as np

    summary: dict[str, str] = {}
    problems: list[str] = []
    for key in keys:
        if key not in rebuilt.columns or key not in delivered.columns:
            missing = "rebuilt" if key not in rebuilt.columns else "delivered"
            problems.append(f"{key} missing from the {missing} dataset")
            continue
        a = np.stack([np.atleast_1d(v) for v in rebuilt[key].to_numpy()])
        b = np.stack([np.atleast_1d(v) for v in delivered[key].to_numpy()])
        if a.shape != b.shape:
            problems.append(f"{key}: shape {a.shape} vs delivered {b.shape}")
            summary[key] = f"shape {a.shape} != {b.shape}"
            continue
        if np.array_equal(a, b):
            summary[key] = f"identical {a.shape}"
            continue
        differing = sorted(set(np.nonzero(~np.isclose(a, b, rtol=0, atol=0))[1].tolist()))
        worst = float(np.max(np.abs(a - b)))
        summary[key] = f"DIFFERS in slots {differing[:12]} (max |delta| {worst:.3e})"
        problems.append(f"{key}: {len(differing)} slot(s) differ, max |delta| {worst:.3e}")
    return summary, problems


def compare_episode(
    spec: DatasetSpec, rebuilt: Path, delivered: Path, index: int,
    rebuilt_prompts: dict, delivered_prompts: dict, check_video: bool,
) -> EpisodeReport:
    import pandas as pd

    report = EpisodeReport(index=index)

    a_path, b_path = episode_parquet(rebuilt, index), episode_parquet(delivered, index)
    if a_path is None or b_path is None:
        report.problems.append(
            f"episode {index} is missing from the "
            f"{'rebuilt' if a_path is None else 'delivered'} dataset"
        )
        return report

    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    report.rows_rebuilt, report.rows_delivered = len(a), len(b)

    if index in rebuilt_prompts and index in delivered_prompts:
        report.prompt_matches = rebuilt_prompts[index] == delivered_prompts[index]
        if not report.prompt_matches:
            report.problems.append(
                "episode alignment is wrong: this pair has different task prompts, "
                "so the two datasets do not have the same episode at this index"
            )
            return report

    if len(a) != len(b):
        report.problems.append(
            f"{len(a)} rows against {len(b)} delivered -- the clock alignment kept a "
            "different number of frames"
        )
        return report

    keys = [k for k in ("observation.state", "action") if spec.vector(
        "state" if k == "observation.state" else "action") is not None]
    report.columns, problems = compare_vectors(a, b, keys)
    report.problems += problems

    if check_video:
        report.video, problems = compare_video(rebuilt, delivered, index)
        report.problems += problems
    return report


def compare_video(rebuilt: Path, delivered: Path, index: int) -> tuple[dict, list[str]]:
    videos_a, videos_b = episode_videos(rebuilt, index), episode_videos(delivered, index)
    summary: dict[str, str] = {}
    problems: list[str] = []

    for key in sorted(set(videos_a) | set(videos_b)):
        if key not in videos_a or key not in videos_b:
            problems.append(f"video {key} only in the "
                            f"{'delivered' if key not in videos_a else 'rebuilt'} dataset")
            continue
        a, b = probe(videos_a[key]), probe(videos_b[key])

        # decided by our own code, so they must agree exactly
        for field_name in ("width", "height", "nb_read_frames", "codec_name",
                           "profile", "pix_fmt", "has_b_frames"):
            if a.get(field_name) != b.get(field_name):
                problems.append(
                    f"video {key}: {field_name} {a.get(field_name)!r} against "
                    f"delivered {b.get(field_name)!r}"
                )

        # decided by the encoder build, so only the magnitude is meaningful
        ratio = a["bytes"] / b["bytes"] if b["bytes"] else float("inf")
        verdict = "ok" if abs(ratio - 1) <= SIZE_TOLERANCE else "SIZE"
        if verdict == "SIZE":
            problems.append(
                f"video {key}: {a['bytes']} bytes against delivered {b['bytes']} "
                f"({ratio:.2f}x) -- beyond what a different ffmpeg build explains"
            )
        summary[key] = (
            f"{a.get('width')}x{a.get('height')} {a.get('nb_read_frames')}f "
            f"{a['bytes']}B vs {b['bytes']}B ({ratio:.2f}x) {verdict}"
        )
    return summary, problems


def run(
    spec: DatasetSpec, rebuilt: Path, delivered: Path,
    episodes: int, check_video: bool,
) -> list[EpisodeReport]:
    rebuilt_prompts = episode_prompts(rebuilt)
    delivered_prompts = episode_prompts(delivered)
    return [
        compare_episode(spec, rebuilt, delivered, index,
                        rebuilt_prompts, delivered_prompts, check_video)
        for index in range(episodes)
    ]


def report(reports: list[EpisodeReport]) -> str:
    lines = []
    for r in reports:
        mark = "ok  " if r.ok else "FAIL"
        rows = (f"{r.rows_rebuilt} rows" if r.rows_rebuilt == r.rows_delivered
                else f"{r.rows_rebuilt} vs {r.rows_delivered} rows")
        lines.append(f"[{mark}] episode {r.index:>5}  {rows}")
        for key, value in r.columns.items():
            lines.append(f"         {key:<20} {value}")
        for key, value in r.video.items():
            lines.append(f"         video {key:<14} {value}")
        for problem in r.problems:
            lines.append(f"         ! {problem}")

    failed = [r for r in reports if not r.ok]
    lines.append("")
    lines.append(f"{len(reports) - len(failed)}/{len(reports)} episodes reproduce "
                 "the delivered copy" + ("" if not failed else
                 f"; {len(failed)} differ"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help=f"one of: {', '.join(available())}")
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--delivered", type=Path, default=None,
                        help="defaults to the spec's delivered.path")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--no-video", action="store_true",
                        help="compare only the vectors; skips ffprobe")
    args = parser.parse_args(argv)

    spec = load(args.dataset)
    delivered = args.delivered or (
        Path(spec.delivered_path) if spec.delivered_path else None)
    if delivered is None:
        print("no --delivered given and the spec has no delivered.path", file=sys.stderr)
        return 2
    if not delivered.exists():
        print(f"delivered dataset not found: {delivered}", file=sys.stderr)
        return 2

    reports = run(spec, args.rebuilt, delivered, args.episodes, not args.no_video)
    print(report(reports))
    return 1 if any(not r.ok for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
