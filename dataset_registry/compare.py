"""Compare a rebuilt dataset against the delivered copy it is meant to reproduce.

    uv run python -m dataset_registry.compare action_net --rebuilt /out --delivered /ref

The two are held to different standards, because they can be:

* **state and action must be identical, row for row.** Every slot is a float32
  copied out of the source, so a rebuild that reads the same bytes and lines the
  clocks up the same way produces the same numbers exactly. A difference in the
  values is a real difference, reported as a failure rather than a tolerance.
* **the row count may differ slightly.** Which rows survive is decided by the clock
  strategy, and the delivered copy was produced by a script that no longer exists,
  so an episode landing a frame or two short is expected. Up to ``--row-tolerance``
  rows of difference is accepted, and the overlapping prefix is still compared in
  full -- which is the interesting part: if every shared row matches, the difference
  is a trimmed tail, and if they diverge partway the two are keeping *different*
  frames and the strategy is wrong.
* **video must match in geometry, frame count, codec and keyframe interval; its
  bytes will not match.** Two ffmpeg builds with the same flags do not emit the same
  file, so the size is reported as a ratio and judged loosely. Everything else there
  is decided by our own settings rather than by the encoder build, so a difference is
  a real failure. The keyframe interval is included deliberately: it is not in the
  stream header and has to be read off the frames, and it is the one encoder setting
  the training loader feels, since sampling a random frame from a 250-frame GOP means
  decoding back to the last keyframe.

Episodes carry no source id in the delivered copy, so they are aligned by position
and the alignment is *checked* rather than assumed: the task prompt of each pair
must agree. One dropped episode shifts everything after it, and comparing shifted
episodes would produce a wall of meaningless differences, so a prompt mismatch stops
the run.
"""

import argparse
import hashlib
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

# Rows the clock strategy may differ by before it counts as a wrong strategy rather
# than a boundary effect. The two upstream filters trim the tail of an episode, and
# exactly where they land depends on floating-point comparisons of timestamps.
ROW_TOLERANCE = 2


class CompareError(RuntimeError):
    pass


@dataclass
class EpisodeReport:
    index: int
    # which delivered episode this one was compared against; the same index unless
    # the run aligned by content
    delivered_index: int | None = None
    rows_rebuilt: int | None = None
    rows_delivered: int | None = None
    # rows compared when the two lengths differ, and where they first diverge
    compared_rows: int | None = None
    first_divergence: int | None = None
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


def episode_digests(root: Path) -> dict[str, list[int]]:
    """state+action bytes of every episode -> the indices that carry them.

    A list, not an index, because an episode can genuinely repeat: two of the 150
    ucsd_kitchen episodes are identical vectors.
    """
    import pandas as pd

    out: dict[str, list[int]] = {}
    for path in sorted(root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(path)
        for index, rows in frame.groupby("episode_index"):
            rows = rows.sort_values("frame_index")
            digest = hashlib.sha256()
            for column in ("observation.state", "action"):
                if column not in rows:
                    continue
                for value in rows[column]:
                    digest.update(bytes(memoryview(value.astype("float32"))))
            out.setdefault(digest.hexdigest(), []).append(int(index))
    return out


def pair_digests(
    a: dict[str, list[int]], b: dict[str, list[int]]
) -> dict[int, int]:
    """Rebuilt episode index -> the delivered index holding the same vectors.

    openx2lerobot writes episodes in tfds read order, which is not the order the
    delivered copies were written in, so the two agree on every episode and on none
    of the positions. Pairing on the vectors themselves says whether the *contents*
    reproduce; it deliberately says nothing about the order, which is what comparing
    by position is for.

    Episodes whose vectors appear a different number of times on the two sides are
    left unpaired rather than matched arbitrarily.
    """
    pairs: dict[int, int] = {}
    for digest, rebuilt_indices in a.items():
        delivered_indices = b.get(digest)
        if not delivered_indices or len(delivered_indices) != len(rebuilt_indices):
            continue
        for one, other in zip(sorted(rebuilt_indices), sorted(delivered_indices)):
            pairs[one] = other
    return pairs


def pair_by_content(rebuilt: Path, delivered: Path) -> dict[int, int]:
    """:func:`pair_digests` over two datasets on disk."""
    return pair_digests(episode_digests(rebuilt), episode_digests(delivered))


def episode_parquet(root: Path, index: int) -> Path | None:
    matches = sorted(root.glob(f"data/**/episode_{index:06d}.parquet"))
    return matches[0] if matches else None


def declared_cameras(root: Path) -> set[str] | None:
    """The cameras ``meta/modality.json`` exposes, or ``None`` if it declares none.

    A dataset can carry a camera it does not expose. bridge_orig keeps two spare
    views of four; humanoid_everyday keeps the unresized 640x480 original beside the
    256x192 the training stack actually reads. Those files are on disk and nothing
    opens them, so a rebuild that differs there differs in a way no one can see.

    ``None`` rather than every camera, because "declares nothing" and "declares all
    of them" are different: the first is a dataset that has not been given a modality
    file, and guessing on its behalf would quietly narrow a comparison.

    Both spellings of each camera come back, because the delivered copies use two
    conventions for the directory a camera's videos sit in: most name it after the
    ``original_key`` in full -- ``observation.images.rgb_static`` -- while
    humanoid_everyday names it ``egocentric_resized``, the last segment alone.
    Recognising one spelling would leave every dataset using the other with no
    cameras to compare at all, which reads as a pass.
    """
    path = root / "meta" / "modality.json"
    if not path.is_file():
        return None
    try:
        video = (json.loads(path.read_text()) or {}).get("video") or {}
    except json.JSONDecodeError:
        return None
    if not video:
        return None
    names: set[str] = set()
    for name, entry in video.items():
        key = str(entry.get("original_key") or f"observation.images.{name}")
        names.update({key, key.rsplit(".", 1)[-1]})
    return names


def episode_videos(
    root: Path, index: int, keep: set[str] | None = None
) -> dict[str, Path]:
    out = {}
    for path in sorted(root.glob(f"videos/**/episode_{index:06d}.mp4")):
        if keep is None or path.parent.name in keep:
            out[path.parent.name] = path
    return out


# How far into an episode to look for a second keyframe. Long enough to see a
# 250-frame GOP, short enough not to decode whole episodes.
GOP_PROBE_FRAMES = 320


def _ffprobe(command: list[str], path: Path) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as exc:
        raise CompareError("ffprobe not found; it is needed to compare video") from exc
    if result.returncode != 0:
        raise CompareError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    return result.stdout


def keyframe_interval(path: Path) -> int | None:
    """Frames between the first two keyframes, or ``None`` if there is only one.

    The keyframe interval is not in the stream header -- ffprobe reports the codec
    but not ``-g`` -- so it has to be read off the frames. It matters because it is
    the one encoder setting the training loader actually feels: sampling a random
    frame from a 250-frame GOP means decoding back to the last keyframe, which is
    why LeRobot's own writer uses 2.

    ``None`` also means "shorter than the interval", which is the common case for a
    250-frame setting and a 200-frame episode. Two files that both report ``None``
    agree as far as this can tell.
    """
    payload = _ffprobe([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=key_frame",
        "-read_intervals", f"%+#{GOP_PROBE_FRAMES}",
        "-of", "csv=p=0", str(path),
    ], path)
    # ffprobe appends a stray comma to the first frame's row, so a plain equality
    # test silently misses frame 0 -- which is always a keyframe, and skipping it
    # turns every interval into "only one keyframe found"
    flags = [value.strip().strip(",") for value in payload.split() if value.strip()]
    keyframes = [index for index, value in enumerate(flags) if value == "1"]
    return keyframes[1] - keyframes[0] if len(keyframes) > 1 else None


def probe(path: Path) -> dict[str, Any]:
    """Geometry, frame count, codec settings and keyframe interval of one mp4."""
    payload = _ffprobe([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,has_b_frames,nb_read_frames",
        "-of", "json", str(path),
    ], path)
    streams = json.loads(payload).get("streams") or [{}]
    stream = streams[0]
    stream["bytes"] = path.stat().st_size
    stream["gop"] = keyframe_interval(path)
    return stream


def compare_vectors(
    rebuilt, delivered, keys: list[str], rows: int
) -> tuple[dict, list[str], int | None]:
    """State and action over the first ``rows`` rows, slot by slot. Exact.

    Returns the first row index at which any vector diverges, which separates the
    two ways a rebuild can be wrong: a trimmed tail leaves the shared rows perfect,
    while a wrong clock strategy diverges partway through.
    """
    import numpy as np

    summary: dict[str, str] = {}
    problems: list[str] = []
    first_divergence: int | None = None

    for key in keys:
        if key not in rebuilt.columns or key not in delivered.columns:
            missing = "rebuilt" if key not in rebuilt.columns else "delivered"
            problems.append(f"{key} missing from the {missing} dataset")
            continue
        a = np.stack([np.atleast_1d(v) for v in rebuilt[key].to_numpy()])[:rows]
        b = np.stack([np.atleast_1d(v) for v in delivered[key].to_numpy()])[:rows]
        if a.shape[1] != b.shape[1]:
            problems.append(f"{key}: {a.shape[1]} wide against delivered {b.shape[1]}")
            summary[key] = f"width {a.shape[1]} != {b.shape[1]}"
            continue
        if np.array_equal(a, b):
            summary[key] = f"identical {a.shape}"
            continue

        unequal = a != b
        differing = sorted(set(np.nonzero(unequal)[1].tolist()))
        row = int(np.nonzero(unequal.any(axis=1))[0][0])
        first_divergence = row if first_divergence is None else min(first_divergence, row)
        worst = float(np.max(np.abs(a - b)))
        summary[key] = (f"DIFFERS from row {row} in slots {differing[:12]} "
                        f"(max |delta| {worst:.3e})")
        problems.append(
            f"{key}: {len(differing)} slot(s) differ from row {row}, "
            f"max |delta| {worst:.3e}"
        )
    return summary, problems, first_divergence


def compare_episode(
    spec: DatasetSpec, rebuilt: Path, delivered: Path, index: int,
    rebuilt_prompts: dict, delivered_prompts: dict, check_video: bool,
    row_tolerance: int = ROW_TOLERANCE, delivered_index: int | None = None,
) -> EpisodeReport:
    import pandas as pd

    other = index if delivered_index is None else delivered_index
    report = EpisodeReport(index=index, delivered_index=other)

    a_path, b_path = episode_parquet(rebuilt, index), episode_parquet(delivered, other)
    if a_path is None or b_path is None:
        report.problems.append(
            f"episode {index if a_path is None else other} is missing from the "
            f"{'rebuilt' if a_path is None else 'delivered'} dataset"
        )
        return report

    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    report.rows_rebuilt, report.rows_delivered = len(a), len(b)

    if index in rebuilt_prompts and other in delivered_prompts:
        report.prompt_matches = rebuilt_prompts[index] == delivered_prompts[other]
        if not report.prompt_matches:
            report.problems.append(
                "episode alignment is wrong: this pair has different task prompts, "
                "so the two datasets do not have the same episode at this index"
            )
            return report

    delta = len(a) - len(b)
    if abs(delta) > row_tolerance:
        report.problems.append(
            f"{len(a)} rows against {len(b)} delivered ({delta:+d}) -- beyond the "
            f"{row_tolerance}-row tolerance, so the clock strategy is keeping "
            "different frames rather than trimming differently"
        )
        return report

    rows = min(len(a), len(b))
    report.compared_rows = rows
    keys = [k for k in ("observation.state", "action") if spec.vector(
        "state" if k == "observation.state" else "action") is not None]
    report.columns, problems, report.first_divergence = compare_vectors(a, b, keys, rows)
    report.problems += problems

    if check_video:
        report.video, problems = compare_video(rebuilt, delivered, index, other)
        report.problems += problems
    return report


def compare_video(
    rebuilt: Path, delivered: Path, index: int, delivered_index: int | None = None
) -> tuple[dict, list[str]]:
    other = index if delivered_index is None else delivered_index
    # the delivered copy is the target, so it is the one that says which cameras the
    # comparison is about; reading the rebuild's would let a rebuild narrow its own
    # examination
    keep = declared_cameras(delivered)
    videos_a = episode_videos(rebuilt, index, keep)
    videos_b = episode_videos(delivered, other, keep)
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
                           "profile", "pix_fmt", "has_b_frames", "gop"):
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
            f"{a.get('codec_name')}/GOP{a.get('gop') or '>' + str(GOP_PROBE_FRAMES)} "
            f"{a['bytes']}B vs {b['bytes']}B ({ratio:.2f}x) {verdict}"
        )
    return summary, problems


ALIGNMENTS = ("position", "content")


def run(
    spec: DatasetSpec, rebuilt: Path, delivered: Path,
    episodes: int, check_video: bool, row_tolerance: int = ROW_TOLERANCE,
    align: str = "position",
) -> list[EpisodeReport]:
    """Compare ``episodes`` episodes, pairing them the way ``align`` says.

    ``position`` is the stronger question -- it asks whether the same episode is in
    the same place -- and is the default for that reason. ``content`` drops the order
    and asks only whether the episodes themselves reproduce, which is the right
    question for a converter that does not control the order it writes in.
    """
    if align not in ALIGNMENTS:
        raise CompareError(f"unknown alignment {align!r}; use one of {', '.join(ALIGNMENTS)}")

    rebuilt_prompts = episode_prompts(rebuilt)
    delivered_prompts = episode_prompts(delivered)

    pairs: dict[int, int] = {}
    unpaired = 0
    if align == "content":
        rebuilt_digests = episode_digests(rebuilt)
        pairs = pair_digests(rebuilt_digests, episode_digests(delivered))
        unpaired = sum(len(v) for v in rebuilt_digests.values()) - len(pairs)
        chosen = sorted(pairs)[:episodes]
    else:
        chosen = list(range(episodes))

    reports = [
        compare_episode(spec, rebuilt, delivered, index,
                        rebuilt_prompts, delivered_prompts, check_video, row_tolerance,
                        delivered_index=pairs.get(index))
        for index in chosen
    ]
    if unpaired:
        reports.append(unpaired_report(unpaired))
    return reports


def unpaired_report(count: int) -> EpisodeReport:
    """A finding about the datasets as a whole rather than about one episode.

    It rides along as a report so that it counts towards the exit status: episodes
    the delivered copy simply does not contain are a failure to reproduce, even when
    every episode that *was* compared came out identical.
    """
    report = EpisodeReport(index=-1)
    report.problems.append(
        f"{count} rebuilt episode(s) have vectors that appear nowhere in the "
        "delivered copy, so they were not compared"
    )
    return report


def report(reports: list[EpisodeReport]) -> str:
    lines = []
    for r in reports:
        mark = "ok  " if r.ok else "FAIL"
        if r.index < 0:
            lines += [f"[{mark}] {problem}" for problem in r.problems]
            continue
        if r.rows_rebuilt == r.rows_delivered:
            rows = f"{r.rows_rebuilt} rows"
        else:
            delta = (r.rows_rebuilt or 0) - (r.rows_delivered or 0)
            rows = f"{r.rows_rebuilt} vs {r.rows_delivered} rows ({delta:+d})"
        where = (
            f"{r.index:>5}"
            if r.delivered_index is None or r.delivered_index == r.index
            else f"{r.index:>5}->{r.delivered_index}"
        )
        lines.append(f"[{mark}] episode {where}  {rows}")
        for key, value in r.columns.items():
            lines.append(f"         {key:<20} {value}")
        for key, value in r.video.items():
            lines.append(f"         video {key:<14} {value}")
        for problem in r.problems:
            lines.append(f"         ! {problem}")

    episodes = [r for r in reports if r.index >= 0]
    failed = [r for r in episodes if not r.ok]
    short = [r for r in episodes if r.rows_rebuilt != r.rows_delivered and r.ok]
    lines.append("")
    lines.append(f"{len(episodes) - len(failed)}/{len(episodes)} episodes reproduce "
                 "the delivered copy" + ("" if not failed else
                 f"; {len(failed)} differ"))
    if short:
        # the distinction that matters: same frames, trimmed differently, versus
        # different frames kept
        lines.append(
            f"{len(short)} of those have a different row count but identical values "
            "in every shared row -- a trimmed tail, not a different frame selection"
        )
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
    parser.add_argument("--align", choices=ALIGNMENTS, default="position",
                        help="how to pair episodes: by index (default), or by their "
                             "own state/action bytes when the rebuild does not "
                             "control the order it writes in")
    parser.add_argument("--row-tolerance", type=int, default=ROW_TOLERANCE,
                        help="rows an episode may differ by before it counts as a "
                             "wrong clock strategy rather than a trimmed tail")
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

    reports = run(spec, args.rebuilt, delivered, args.episodes, not args.no_video,
                  args.row_tolerance, align=args.align)
    print(report(reports))
    return 1 if any(not r.ok for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
