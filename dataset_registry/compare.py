"""Ask how closely a rebuilt dataset matches the delivered copy it reproduces.

    uv run python -m dataset_registry.compare cmu_stretch --rebuilt /out --delivered /ref

Three questions in order, because one verdict was the wrong shape for this. The
rebuilds do not control the order they write episodes in, some episodes land a row or
two short, and the two copies do not even carry the same set of metadata files -- so
"does it reproduce" collapses several different answers into one and loses the useful
ones. Each question here explains the next:

1. **episodes** -- how much of the delivered copy is there. Episodes are paired on
   their own state and action bytes, because they carry no source id and the rebuild
   does not write them in the delivered order. Pairing happens twice: exactly, and
   then on the first few rows, so that an episode which is *present but a row short*
   is counted apart from one that is genuinely absent. Absence is reported, never
   failed -- which episodes a rebuild ends up with is decided outside this comparison.
2. **sample** -- whether what is there is the same, checked in full on a sample of
   paired episodes. state and action must be identical, row for row: every slot is a
   float32 copied out of the source, so a difference is a real difference and not a
   tolerance. Video must match in geometry, frame count, codec, keyframe interval and
   the pictures themselves; its *bytes* will not, because two ffmpeg builds given the
   same flags do not emit the same file, so size is judged as a loose ratio. The
   pictures are checked because the ratio cannot see them: a rebuild with red and blue
   exchanged came within 1% of the delivered size. The keyframe interval is checked
   because it is not in the stream header and it is the one encoder setting the
   training loader feels, since sampling a frame from a 250-frame GOP means decoding
   back to the last keyframe.
3. **distributions** -- whether the two describe the same data, per vector column,
   summarised by mean, std, min, max and three quantiles and scaled by the column's
   own range so that metres and radians answer the same number. Asked twice, over
   every episode and over the episodes the two copies share. That pair is the point of
   the arrangement: if they disagree, the difference is the missing episodes from the
   first question and nothing more; if the shared-episode number is also off, the
   values themselves are wrong. It is computed from the parquet rather than from
   ``meta``, because the delivered copies carry quantiles in ``episodes_stats.jsonl``
   and a rebuild does not -- the v3.0 to v2.1 downgrade keeps only the five legacy
   keys -- so comparing what each *says* about itself would compare the writers.

The exit status follows steps 2 and 3 over the shared episodes. It deliberately
ignores step 1: a run that lost episodes should still be able to say whether the ones
it has are right.
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


# Rows enough to tell two episodes apart while surviving a trimmed tail. Both upstream
# clock filters cut from the end, so a rebuild that trimmed differently still agrees at
# the front, while one that kept a *different* selection of frames does not.
PREFIX_ROWS = 8


@dataclass(frozen=True)
class Fingerprint:
    """What identifies an episode when its index cannot.

    ``whole`` answers "are these the same episode, to the byte". ``prefix`` answers
    the weaker question that has to be asked separately: "is this the same episode
    with its tail cut somewhere else". Keeping both is what lets a report say how many
    episodes are *missing* rather than lumping the trimmed ones in with them -- and
    that distinction decides whether a distribution difference further down needs
    explaining or not.

    An episode shorter than ``PREFIX_ROWS`` gets the same digest for both, so it can
    only ever match exactly. At eight rows against episodes of a hundred and more that
    is a corner rather than a case.
    """

    rows: int
    whole: str
    prefix: str


def _stack(column):
    """A parquet column of per-row arrays as one ``(rows, width)`` array."""
    import numpy as np

    values = column.to_numpy()
    return np.stack(values) if values.dtype == object else values.reshape(len(values), -1)


def episode_fingerprints(root: Path) -> dict[int, Fingerprint]:
    """Every episode in ``root``, keyed by its own index."""
    import pandas as pd

    out: dict[int, Fingerprint] = {}
    for path in sorted(root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(path)
        if "episode_index" not in frame:
            continue
        for index, rows in frame.groupby("episode_index"):
            if "frame_index" in rows:
                rows = rows.sort_values("frame_index")
            whole = hashlib.sha256()
            prefix = hashlib.sha256()
            # column by column, in the same order episode_digests uses, so the two
            # agree on what an episode's bytes are
            for column in ("observation.state", "action"):
                if column not in rows:
                    continue
                values = _stack(rows[column]).astype("float32")
                whole.update(values.tobytes())
                prefix.update(values[:PREFIX_ROWS].tobytes())
            out[int(index)] = Fingerprint(
                rows=len(rows), whole=whole.hexdigest(), prefix=prefix.hexdigest()
            )
    return out


def _group(fingerprints: dict[int, Fingerprint], key, only=None) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, print_ in sorted(fingerprints.items()):
        if only is not None and index not in only:
            continue
        grouped.setdefault(key(print_), []).append(index)
    return grouped


@dataclass
class Pairing:
    """Which episodes the two copies have in common, and on what basis.

    The counts are the first thing a verification run should report, because every
    later number depends on them: a distribution computed over episodes the other side
    does not have is expected to differ, and one computed over the episodes they share
    is not.
    """

    exact: dict[int, int] = field(default_factory=dict)
    trimmed: dict[int, int] = field(default_factory=dict)
    rebuilt_only: list[int] = field(default_factory=list)
    delivered_only: list[int] = field(default_factory=list)
    rebuilt_rows: int = 0
    delivered_rows: int = 0

    @property
    def pairs(self) -> dict[int, int]:
        return {**self.exact, **self.trimmed}

    @property
    def rebuilt_total(self) -> int:
        return len(self.exact) + len(self.trimmed) + len(self.rebuilt_only)

    @property
    def delivered_total(self) -> int:
        return len(self.exact) + len(self.trimmed) + len(self.delivered_only)

    @property
    def moved(self) -> int:
        """Paired episodes whose index is not the one they had. Not a defect on its
        own -- openx2lerobot writes in tfds read order -- but it is the reason a
        position-aligned comparison reports every episode as different."""
        return sum(1 for one, other in self.pairs.items() if one != other)


def pair_episodes(
    rebuilt: dict[int, Fingerprint], delivered: dict[int, Fingerprint]
) -> Pairing:
    """Match episodes on their own contents, exactly first and then by prefix.

    Two passes rather than one because the answers mean different things. An exact
    match says the episode reproduced; a prefix match says the same episode is there
    with a different number of rows, which is the clock strategy landing elsewhere and
    not a missing episode. Whatever survives both passes really is absent from one side.

    Multiplicity is respected as in :func:`pair_digests`: episodes whose contents
    appear a different number of times on the two sides are left unpaired rather than
    matched arbitrarily, because there is no way to say which went with which.
    """
    result = Pairing(
        rebuilt_rows=sum(f.rows for f in rebuilt.values()),
        delivered_rows=sum(f.rows for f in delivered.values()),
    )

    def take(key, only_rebuilt=None, only_delivered=None) -> dict[int, int]:
        return pair_digests(
            _group(rebuilt, key, only_rebuilt), _group(delivered, key, only_delivered)
        )

    result.exact = take(lambda f: f.whole)
    result.trimmed = take(
        lambda f: f.prefix,
        set(rebuilt) - set(result.exact),
        set(delivered) - set(result.exact.values()),
    )

    paired = result.pairs
    result.rebuilt_only = sorted(set(rebuilt) - set(paired))
    result.delivered_only = sorted(set(delivered) - set(paired.values()))
    return result


# What a distribution is summarised by. Quantiles rather than std alone because the
# question they answer is different: std moves when the values are rescaled, a quantile
# moves when the shape changes.
DISTRIBUTION_STATS = ("mean", "std", "min", "max", "q01", "q50", "q99")

VECTORS = ("observation.state", "action")


def distribution(
    root: Path, episodes: set[int] | None = None, columns: tuple[str, ...] = VECTORS
) -> dict[str, dict[str, Any]]:
    """Per-dimension summary of each vector column, over ``episodes`` or over all.

    Computed from the parquet rather than read out of ``meta``, on purpose. The two
    copies do not carry the same metadata -- the delivered ones have quantiles in
    ``episodes_stats.jsonl`` and a rebuild does not, because the v3.0 to v2.1
    downgrade keeps only the five legacy keys -- so a comparison of what each *says*
    about itself would be comparing the writers, not the data.
    """
    import numpy as np

    gathered: dict[str, list] = {column: [] for column in columns}
    for path in sorted(root.glob("data/**/*.parquet")):
        frame = pd_read(path)
        if episodes is not None and "episode_index" in frame:
            frame = frame[frame["episode_index"].isin(episodes)]
        if not len(frame):
            continue
        for column in columns:
            if column in frame:
                gathered[column].append(_stack(frame[column]).astype("float64"))

    out: dict[str, dict[str, Any]] = {}
    for column, parts in gathered.items():
        if not parts:
            continue
        values = np.concatenate(parts)
        out[column] = {
            "rows": len(values),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "q01": np.percentile(values, 1, axis=0),
            "q50": np.percentile(values, 50, axis=0),
            "q99": np.percentile(values, 99, axis=0),
        }
    return out


def pd_read(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def distribution_gap(rebuilt: dict, delivered: dict) -> dict[str, float]:
    """Column -> the worst difference across its dimensions, as a fraction of range.

    Scaled by the delivered column's own span so that one dimension in metres and
    another in radians are answerable by the same number. A column present on only one
    side is reported as ``inf`` rather than skipped.
    """
    import numpy as np

    gaps: dict[str, float] = {}
    for column in sorted(set(rebuilt) | set(delivered)):
        if column not in rebuilt or column not in delivered:
            gaps[column] = float("inf")
            continue
        mine, theirs = rebuilt[column], delivered[column]
        span = np.asarray(theirs["max"]) - np.asarray(theirs["min"])
        span = np.where(span > 0, span, 1.0)
        worst = 0.0
        for stat in DISTRIBUTION_STATS:
            a, b = np.asarray(mine[stat]), np.asarray(theirs[stat])
            if a.shape != b.shape:
                worst = float("inf")
                break
            worst = max(worst, float(np.max(np.abs(a - b) / span)))
        gaps[column] = worst
    return gaps


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


# Frames decoded per camera per episode. The point is to catch a picture that is
# not the one it should be -- reversed channels, a shifted clip, the wrong episode
# -- and any of those is wrong in every frame, so a handful spread across the
# episode says as much as all of them at a fraction of the cost.
PIXEL_FRAMES = 6

# Below this, the two are not the same picture. Measured 2026-08-19 on delivered
# episodes of ucsd_kitchen (av1), action_net (h264 crf 21) and humanoid_everyday
# (h264 fast crf 18), by decoding each and encoding it again with the settings its
# own spec names -- a second lossy generation, so a floor rather than an estimate:
#
#     re-encoded from the delivered frames      0.9993  0.9994  0.9997
#     red and blue exchanged                    0.8871
#     shifted four pixels and rescaled          0.8887
#     a different episode of the same task      0.55      (utaustin_mutex)
#
# Nothing lands between 0.89 and 0.999, so the threshold is not a judgement call
# about how close is close enough.
#
# The size ratio cannot stand in for this. The same channel reversal that costs
# action_net 24% of its bytes cost utaustin_mutex under 1%, which is how a reversed
# rebuild passed a size check at 64/64 -- how much a wrong picture changes the byte
# count depends on the picture.
PIXEL_AGREEMENT = 0.98


def sample_frames(path: Path, count: int = PIXEL_FRAMES):
    """``count`` frames spread through ``path``, as one uint8 array.

    Decoded through ffmpeg rather than a decoding library, for the same reason the
    rest of this module shells out to ffprobe: the comparison should see what ffmpeg
    sees, and a second decoder in the dependency list is a second answer to what a
    file contains.
    """
    import numpy as np

    total = int(probe(path).get("nb_read_frames") or 0)
    if total <= 0:
        raise CompareError(f"no frames to sample in {path}")
    picks = sorted({round(i * (total - 1) / max(count - 1, 1)) for i in range(count)})
    width, height = int(probe(path)["width"]), int(probe(path)["height"])
    # one select filter rather than one ffmpeg per frame: seeking is the expensive
    # part, and an episode is short enough to read straight through
    expression = "+".join(f"eq(n\\,{n})" for n in picks)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select={expression}",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=300,
    )
    if result.returncode != 0:
        raise CompareError(f"ffmpeg failed reading {path}: {result.stderr.decode()[:300]}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8)
    return frames.reshape(-1, height, width, 3)[: len(picks)]


def pixel_agreement(a: Path, b: Path, count: int = PIXEL_FRAMES) -> float:
    """How well two videos' frames agree, as the weakest of the sampled frames.

    The weakest rather than the average, because a difference that shows up in one
    frame is still a difference; averaging lets a good majority hide it.
    """
    import numpy as np

    left, right = sample_frames(a, count), sample_frames(b, count)
    if left.shape[1:] != right.shape[1:]:
        return 0.0
    n = min(len(left), len(right))
    scores = []
    for index in range(n):
        x = left[index].astype(np.float64).ravel()
        y = right[index].astype(np.float64).ravel()
        if x.std() == 0 and y.std() == 0:
            # a flat frame against a flat frame: correlation is undefined, so fall
            # back to whether the two flats are the same colour
            scores.append(1.0 if np.allclose(x, y, atol=2) else 0.0)
            continue
        if x.std() == 0 or y.std() == 0:
            scores.append(0.0)
            continue
        scores.append(float(np.corrcoef(x, y)[0, 1]))
    return min(scores) if scores else 0.0


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
            problem = (f"video {key} only in the "
                       f"{'delivered' if key not in videos_a else 'rebuilt'} dataset")
            if key not in videos_a:
                # A camera the rebuild wrote under a different name reads, through the
                # filter above, exactly like one it did not write at all -- and eight of
                # the delivered OXE datasets renamed their cameras to the modality
                # aliases while nineteen kept the source keys, so this is the common
                # case rather than a corner. Name what is actually there.
                elsewhere = sorted(set(episode_videos(rebuilt, index)) - set(videos_a))
                if elsewhere:
                    problem += f"; the rebuild has {', '.join(elsewhere)} instead"
            problems.append(problem)
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
        # the pictures themselves, which the size ratio cannot see: reversing the red
        # and blue channels leaves a file the same size and the frames unrecognisable
        agreement = pixel_agreement(videos_a[key], videos_b[key])
        if agreement < PIXEL_AGREEMENT:
            verdict = "PIXELS"
            problems.append(
                f"video {key}: frames agree {agreement:.3f} against delivered, below "
                f"{PIXEL_AGREEMENT} -- the same size encoded from a different picture"
            )

        summary[key] = (
            f"{a.get('width')}x{a.get('height')} {a.get('nb_read_frames')}f "
            f"{a.get('codec_name')}/GOP{a.get('gop') or '>' + str(GOP_PROBE_FRAMES)} "
            f"{a['bytes']}B vs {b['bytes']}B ({ratio:.2f}x) px{agreement:.3f} {verdict}"
        )
    return summary, problems


def run(
    spec: DatasetSpec, rebuilt: Path, delivered: Path,
    episodes: int, check_video: bool, row_tolerance: int = ROW_TOLERANCE,
) -> list[EpisodeReport]:
    """Compare the first ``episodes`` episodes of each, paired by index.

    The narrow question: is the same episode in the same place. :func:`measure` asks
    the broader one, pairing on contents instead, and uses :func:`compare_episode`
    the same way this does.
    """
    return compare_chosen(spec, rebuilt, delivered, list(range(episodes)), {},
                          check_video, row_tolerance)


def compare_chosen(
    spec: DatasetSpec, rebuilt: Path, delivered: Path,
    chosen: list[int], pairs: dict[int, int],
    check_video: bool, row_tolerance: int = ROW_TOLERANCE,
) -> list[EpisodeReport]:
    """``compare_episode`` over ``chosen``, reading the prompts once."""
    rebuilt_prompts = episode_prompts(rebuilt)
    delivered_prompts = episode_prompts(delivered)
    return [
        compare_episode(spec, rebuilt, delivered, index, rebuilt_prompts,
                        delivered_prompts, check_video, row_tolerance,
                        delivered_index=pairs.get(index))
        for index in chosen
    ]


def episode_lines(r: EpisodeReport) -> list[str]:
    """One episode's findings, without the run-level tally :func:`report` adds."""
    mark = "ok  " if r.ok else "FAIL"
    if r.index < 0:
        return [f"[{mark}] {problem}" for problem in r.problems]
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
    lines = [f"[{mark}] episode {where}  {rows}"]
    lines += [f"         {key:<20} {value}" for key, value in r.columns.items()]
    lines += [f"         video {key:<14} {value}" for key, value in r.video.items()]
    lines += [f"         ! {problem}" for problem in r.problems]
    return lines


def report(reports: list[EpisodeReport]) -> str:
    lines = []
    for r in reports:
        lines += episode_lines(r)

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


# What a rebuild's own float summation order can cost, with room to spare: a rebuild
# that reproduces the delivered episodes measures around 1e-15 here, so anything this
# far above it is a difference in the data rather than in the arithmetic.
DISTRIBUTION_TOLERANCE = 1e-6


@dataclass
class Funnel:
    """How closely a rebuild matches, asked in three questions rather than one.

    A single verdict was the wrong shape for this. The rebuilds do not control the
    order they write episodes in, some episodes land a row or two short, and the
    metadata each side carries is not even the same set of files -- so "does it
    reproduce" collapses several different answers into one, and the useful ones get
    lost. These are separate on purpose, and they are ordered so that each explains
    the next: how much is there, whether what is there is the same, and whether the
    two describe the same distribution.

    The third is asked twice, over everything and over the episodes the two share.
    That pair is the point of the whole arrangement: if they disagree, the difference
    is the missing episodes from the first question and nothing more, and if the
    shared-episode number is also off then something is wrong with the values.
    """

    dataset: str
    pairing: Pairing
    episodes: list[EpisodeReport] = field(default_factory=list)
    gap_overall: dict[str, float] = field(default_factory=dict)
    gap_shared: dict[str, float] = field(default_factory=dict)

    @property
    def values_agree(self) -> bool:
        """Whether every episode that was compared in full came out identical."""
        return all(r.ok for r in self.episodes if r.index >= 0)

    @property
    def distributions_agree(self) -> bool:
        return all(v <= DISTRIBUTION_TOLERANCE for v in self.gap_shared.values())


def measure(
    spec: DatasetSpec,
    rebuilt: Path,
    delivered: Path,
    episodes: int = 8,
    check_video: bool = True,
    row_tolerance: int = ROW_TOLERANCE,
) -> Funnel:
    """Walk the three questions in order, reusing the pairing for all of them."""
    pairing = pair_episodes(
        episode_fingerprints(rebuilt), episode_fingerprints(delivered)
    )

    pairs = pairing.pairs
    if pairs:
        chosen = sorted(pairs)[:episodes]
    else:
        # Nothing paired at all, which is what a rebuild whose *values* are wrong looks
        # like -- no episode on either side has a counterpart, so there is nothing to
        # sample. Fall back to position so that step 2 still says what differs; the
        # first question has already reported that none of them matched.
        chosen = list(range(min(episodes, pairing.rebuilt_total)))
    reports = compare_chosen(spec, rebuilt, delivered, chosen, pairs,
                             check_video, row_tolerance)

    shared = set(pairing.exact)
    return Funnel(
        dataset=spec.id,
        pairing=pairing,
        episodes=reports,
        gap_overall=distribution_gap(distribution(rebuilt), distribution(delivered)),
        gap_shared=distribution_gap(
            distribution(rebuilt, episodes=shared),
            distribution(delivered, episodes={pairing.exact[i] for i in shared}),
        ),
    )


def _gaps(gaps: dict[str, float]) -> str:
    return "   ".join(f"{column} {value:.1e}" for column, value in sorted(gaps.items()))


def funnel_report(f: Funnel, verbose: bool = False) -> str:
    p = f.pairing
    lines = [f"{f.dataset}", ""]

    lines.append("1  episodes")
    lines.append(f"     rebuilt      {p.rebuilt_total:>6} episodes  {p.rebuilt_rows:>9,} rows")
    lines.append(f"     delivered    {p.delivered_total:>6} episodes  {p.delivered_rows:>9,} rows")
    lines.append(f"     reproduced   {len(p.pairs):>6}  ({len(p.exact)} identical, "
                 f"{len(p.trimmed)} same episode with a different row count)")
    lines.append(f"     absent       {len(p.delivered_only):>6} of the delivered episodes; "
                 f"{len(p.rebuilt_only)} in the rebuild are not in the delivered copy")
    if p.moved:
        lines.append(f"     reordered    {p.moved:>6} of {len(p.pairs)} paired episodes "
                     "carry a different index")

    compared = [r for r in f.episodes if r.index >= 0]
    failed = [r for r in compared if not r.ok]
    lines += ["", "2  sample"]
    lines.append(f"     compared     {len(compared):>6} of {len(p.pairs)} paired episodes, in full")
    lines.append(f"     identical    {len(compared) - len(failed):>6}"
                 + (f"  ({len(failed)} differ)" if failed else ""))
    for r in compared if verbose else failed:
        lines += ["     " + line for line in episode_lines(r)]

    lines += ["", "3  distributions   (max difference per column, as a fraction of its range)"]
    lines.append(f"     every episode        {_gaps(f.gap_overall)}")
    lines.append(f"     the shared ones      {_gaps(f.gap_shared)}")
    if not f.distributions_agree:
        lines.append("     ! the shared episodes do not agree, so this is the values "
                     "differing and not the missing episodes")
    elif len(p.delivered_only) or len(p.rebuilt_only):
        lines.append(f"     the two rows differ because {len(p.delivered_only)} delivered "
                     "episode(s) are absent; over what both copies have, they agree")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help=f"one of: {', '.join(available())}")
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--delivered", type=Path, default=None,
                        help="defaults to the spec's delivered.path")
    parser.add_argument("--episodes", type=int, default=8,
                        help="how many paired episodes to compare in full (step 2)")
    parser.add_argument("--no-video", action="store_true",
                        help="compare only the vectors; skips ffprobe")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print every sampled episode, not only the ones that differ")
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

    if not args.rebuilt.exists():
        print(f"rebuilt dataset not found: {args.rebuilt}", file=sys.stderr)
        return 2

    measured = measure(spec, args.rebuilt, delivered, args.episodes,
                       not args.no_video, args.row_tolerance)
    print(funnel_report(measured, verbose=args.verbose))
    # Missing episodes are reported, not judged: which episodes a rebuild ends up with
    # is decided outside this comparison, and a run that lost some should still be able
    # to say whether the ones it has are right.
    return 0 if measured.values_agree and measured.distributions_agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
