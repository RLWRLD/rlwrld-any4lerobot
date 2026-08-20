"""Ask how closely a rebuilt dataset matches the delivered copy it reproduces.

    uv run python -m verification.compare cmu_stretch --rebuilt /out --delivered /ref

Four questions in order, cheapest and widest first, because one verdict was the wrong
shape for this. The rebuilds do not control the order they write episodes in, some
episodes land a row or two short, and the two copies do not even carry the same set of
metadata files -- so "does it reproduce" collapses several different answers into one
and loses the useful ones. Each question here explains the next:

1. **declaration** -- whether the two copies even claim to be the same dataset, read
   out of ``meta/info.json``: fps, robot type, feature names, shapes and dtypes, video
   codec and pixel format. Two small JSON files, so it costs nothing to ask, and
   anything wrong here is wrong in every episode.

   Only the fields **both** copies declare are judged. A field one side does not have
   is absence of evidence, not disagreement, and it is usually not the rebuild's
   doing: the delivered copies were written by an older LeRobot that recorded fewer
   encoder settings, which on cmu_stretch alone is eight one-sided fields against a
   rebuild whose 135 episodes are byte-identical. The count fields are set aside for
   a different reason -- they follow from the next question and would only repeat it.
2. **episodes** -- how much of the delivered copy is there. Episodes are paired on
   their own state and action bytes, because they carry no source id and the rebuild
   does not write them in the delivered order. Pairing happens twice: exactly, and
   then on the first few rows, so that an episode which is *present but a row short*
   is counted apart from one that is genuinely absent. Absence is reported, never
   failed -- which episodes a rebuild ends up with is decided outside this comparison.
3. **sample** -- whether what is there is the same, checked in full on a sample of
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

   Half the sample comes from each end of the paired index range rather than all of it
   from the front, because the converter writes in chunks of twenty-five episodes and
   the front of the range is one worker's output. Alongside the sample, every episode's
   video is counted and weighed -- files and mean size per camera, from ``stat`` alone
   -- since the frame-by-frame checks are affordable on a few dozen episodes and that
   leaves everything else unlooked-at, which is exactly where a rebuild that wrote a
   tenth of its videos would hide.
4. **distributions** -- whether the two describe the same data, per vector column,
   summarised by mean, std, min, max and three quantiles and scaled by the column's
   own range so that metres and radians answer the same number. Asked twice, over
   every episode and over the episodes the two copies share. That pair is the point of
   the arrangement: if they disagree, the difference is the missing episodes from the
   first question and nothing more; if the shared-episode number is also off, the
   values themselves are wrong. It is computed from the parquet rather than from
   ``meta``, because the delivered copies carry quantiles in ``episodes_stats.jsonl``
   and a rebuild does not -- the v3.0 to v2.1 downgrade keeps only the five legacy
   keys -- so comparing what each *says* about itself would compare the writers.

The exit status follows steps 1, 3 and 4, the last over the shared episodes. It
deliberately ignores step 2: a run that lost episodes should still be able to say
whether the ones it has are right.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset_registry import DatasetSpec, available, load

# ffmpeg is not deterministic across builds; this bounds "the same picture, encoded
# again" against "we encoded something else"
SIZE_TOLERANCE = 0.15

# Rows the clock strategy may differ by before it counts as a wrong strategy rather
# than a boundary effect. The two upstream filters trim the tail of an episode, and
# exactly where they land depends on floating-point comparisons of timestamps.
ROW_TOLERANCE = 2


class CompareError(RuntimeError):
    pass


# ------------------------------------------------------------- 1 · declaration

# ``info.json`` fields that follow from how many episodes a rebuild ended up with
# rather than from how it converted them. Set aside beside the episode counts and
# never failed: which episodes a rebuild has is the next question's subject, and
# failing here as well would only say the same thing twice.
COUNT_FIELDS = ("total_episodes", "total_frames", "total_videos", "total_chunks",
                "splits")

# Stands in for a field one side does not have, so that "declares 128 against 256"
# and "declares nothing at all" read as two different findings rather than one.
ABSENT = "(absent)"


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested mapping as dotted paths, so two of them can be diffed by key.

    Written against the file rather than against a schema of it. ``info.json`` has
    gained fields across LeRobot versions, and a comparison that listed the ones it
    knew about would fall silent on the ones it did not -- which is the direction
    that hides a difference instead of reporting it.
    """
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, item in value.items():
        out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return out or {prefix: {}}


def dataset_info(root: Path) -> dict[str, Any] | None:
    """``meta/info.json`` as a plain dict, or ``None`` if absent or unreadable."""
    path = root / "meta" / "info.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


@dataclass
class Declaration:
    """What the two copies say about themselves, field by field.

    The cheapest question here and the widest: it reads two small JSON files and
    settles whether the rebuild even claims to be the same dataset -- same fps, same
    robot, same feature names and shapes and dtypes, same codec and pixel format. A
    rebuild that has any of those wrong is wrong in every episode, so answering it
    first saves opening a single parquet.

    Only ``differences`` fails, and it holds the fields **both** copies declare. A
    field one copy does not have at all is absence of evidence rather than a
    disagreement, and it is not the rebuild's doing: the delivered copies were
    written by an older LeRobot, which recorded fewer encoder settings and kept
    ``is_depth_map`` one level further in. Measured on cmu_stretch, that alone was
    eight one-sided fields on a rebuild whose 135 episodes are byte-identical -- so
    failing on them would have marked a faithful rebuild wrong, and would have gone
    on doing it for every dataset in the collection.

    What the one-sided fields would have caught is caught better elsewhere. Their
    subject is the encoding, and the sample opens the actual files: codec, pixel
    format, geometry and keyframe interval come off the video rather than out of a
    writer's opinion of it. A camera the rebuild renamed is two one-sided feature
    keys here, and the sample reports that too, by name.
    """

    differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    counts: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    one_sided: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        return not self.differences and not self.missing


def compare_declarations(rebuilt: Path, delivered: Path) -> Declaration:
    """Diff the two ``meta/info.json`` files, holding the count fields aside.

    Neither copy having one is not a finding. The delivered OXE datasets carry it and
    so does every rebuild, but a partial tree may not, and inventing a failure for a
    file nobody wrote is noise. One copy having it and the other not *is* a finding.
    """
    result = Declaration()
    mine, theirs = dataset_info(rebuilt), dataset_info(delivered)
    if mine is None and theirs is None:
        return result
    if mine is None or theirs is None:
        result.missing.append("rebuilt" if mine is None else "delivered")
        return result

    a, b = flatten(mine), flatten(theirs)
    for path in sorted(set(a) | set(b)):
        one, other = a.get(path, ABSENT), b.get(path, ABSENT)
        if one == other:
            continue
        if path.split(".")[0] in COUNT_FIELDS:
            bucket = result.counts
        elif path not in a or path not in b:
            # membership, not equality against the sentinel: a field whose value
            # really is the string ABSENT would otherwise be filed as one-sided
            bucket = result.one_sided
        else:
            bucket = result.differences
        bucket[path] = (one, other)
    return result


# ---------------------------------------------------- the rest of ``meta``

# Every file under ``meta/`` a comparison covers. ``relative_stats.json`` is left out
# by decision rather than by oversight: it is two bytes -- ``{}`` -- in every delivered
# copy examined, so there is nothing in it to reproduce.
META_COMPARED = ("info.json", "tasks.jsonl", "modality.json", "stats.json",
                 "episodes.jsonl", "episodes_stats.jsonl")
META_EXCLUDED = ("relative_stats.json",)

# Stats features that are bookkeeping rather than data: which episode a row belongs to,
# where it sits in the dataset's global row order, and which line of ``tasks.jsonl``
# its prompt is on. All three move when a rebuild writes episodes in a different order
# or assigns task indices in a different order, and it does both -- which the episode
# question already reports, so failing here would report it again.
#
# That they are the *only* three was measured, not assumed. Over all 135 cmu_stretch
# pairs: index differs by 2.4e4, episode_index by 128 and task_index by 3, while
# observation.state, action, timestamp and frame_index are exact to the bit.
ORDER_FEATURES = ("index", "episode_index", "task_index")

# Statistics taken from decoded video cannot be exact -- a rebuild is a second lossy
# generation of the same picture. Image stats are stored normalised to 0-1, and on
# cmu_stretch the paired episodes differ by 0.0057 in mean, 0.016 in min and 0.035 in
# max: an extreme moves further than a mean does, because one clipped pixel decides it.
# Expressed as a fraction of the delivered range so one number covers uint8 frames and
# float depth maps.
IMAGE_STAT_TOLERANCE = 5e-2

# An extreme is decided by one pixel; a mean averages a hundred frames of them. So min
# and max move further under a second lossy generation than mean and std do, and on
# cmu_stretch 8 of 135 paired episodes put max between 0.05 and 0.0667 while every mean
# stayed under 0.0059. Holding an extreme to the mean's allowance fails a faithful
# rebuild for a single clipped pixel.
IMAGE_EXTREME_TOLERANCE = 1e-1
EXTREME_STATS = ("min", "max")

# What a statistic may differ by when neither video nor ordering explains it. The same
# reasoning as DISTRIBUTION_TOLERANCE: a rebuild that reproduces an episode agrees to
# around 1e-15, so anything this far above it is the data and not the arithmetic.
STAT_TOLERANCE = 1e-6


@dataclass
class Finding:
    """One metadata comparison: what disagrees, and what disagrees for a known reason.

    Two buckets rather than one, for the same reason the whole funnel has steps. Some
    of what differs here is the rebuild being wrong and some of it follows from
    something already reported -- the episode order, the task table's order, a lossy
    re-encode -- and a single list of differences cannot tell a reader which is which.
    Only ``differences`` fails; ``set_aside`` is printed with the ``reason`` beside it.
    """

    subject: str
    differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    set_aside: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    reason: str = ""
    absent: list[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        """A file the rebuild does not have counts against it; one neither has does not."""
        return not self.differences and "rebuilt" not in self.absent


def meta_json(root: Path, name: str) -> Any | None:
    path = root / "meta" / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def meta_lines(root: Path, name: str) -> list[dict] | None:
    path = root / "meta" / name
    if not path.is_file():
        return None
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _absence(rebuilt: Path, delivered: Path, name: str, result: Finding) -> bool:
    """Record which side lacks ``meta/<name>``. True when there is nothing to compare.

    A file the delivered copy has and the rebuild does not is a difference in the
    dataset even though no value differs -- a rebuild with no ``modality.json`` cannot
    be read by the training stack at all. Reporting nothing about an absent file, and
    printing "ok" beside it, is how that goes unnoticed.
    """
    mine = (rebuilt / "meta" / name).is_file()
    theirs = (delivered / "meta" / name).is_file()
    if mine and theirs:
        return False
    if not mine and not theirs:
        # neither wrote it; inventing a failure for a file nobody has is noise
        return True
    result.absent.append("rebuilt" if not mine else "delivered")
    result.differences[name] = ("present" if mine else ABSENT,
                                "present" if theirs else ABSENT)
    return True


def meta_inventory(rebuilt: Path, delivered: Path) -> Finding:
    """Which of the compared files each copy actually has.

    Its own finding because a file that is absent has no fields to disagree about, and
    reporting nothing about it reads as agreement. A rebuild that never wrote
    ``modality.json`` is not usable by the training stack -- that file is what says how
    to slice the flat vectors -- so its absence is a difference in the dataset even
    though no value differs.
    """
    result = Finding(subject="meta/", reason="excluded from the comparison by decision")
    for name in META_EXCLUDED:
        if (delivered / "meta" / name).is_file() or (rebuilt / "meta" / name).is_file():
            result.set_aside[name] = ("not compared", "not compared")
    for name in sorted({q.name for q in (delivered / "meta").glob("*")}
                       - set(META_COMPARED) - set(META_EXCLUDED)):
        # a file nobody thought to compare is worth one line: the collection has
        # grown metadata before and a silent new file is how a schema drifts
        result.set_aside[name] = ("not compared", "not compared")
    return result


def compare_tasks(rebuilt: Path, delivered: Path) -> Finding:
    """``meta/tasks.jsonl`` -- the task table.

    Compared as the *set* of prompts, because the index each is given is not a
    property of the data. The rebuild numbers them in the order it first meets them,
    which follows tfds read order; the delivered copy numbered them alphabetically. On
    cmu_stretch that is the same five prompts under five different indices, and it is
    also why the ``task_index`` column of the parquet differs while every episode's
    prompt matches.
    """
    result = Finding(
        subject="meta/tasks.jsonl",
        reason="the index a task is given follows the order it was first seen, "
               "not the data",
    )
    if _absence(rebuilt, delivered, "tasks.jsonl", result):
        return result
    mine = meta_lines(rebuilt, "tasks.jsonl")
    theirs = meta_lines(delivered, "tasks.jsonl")

    def table(lines):
        return {entry["task_index"]: entry["task"] for entry in lines}

    a, b = table(mine), table(theirs)
    for task in sorted(set(a.values()) ^ set(b.values())):
        side = "rebuilt" if task in a.values() else "delivered"
        result.differences[task] = ("present" if side == "rebuilt" else ABSENT,
                                   "present" if side == "delivered" else ABSENT)
    for index in sorted(set(a) | set(b)):
        if a.get(index) != b.get(index):
            result.set_aside[f"task_index {index}"] = (a.get(index, ABSENT),
                                                       b.get(index, ABSENT))
    return result


def compare_modality(rebuilt: Path, delivered: Path) -> Finding:
    """``meta/modality.json`` -- how the training stack slices the flat vectors.

    Not a LeRobot file: it is the GR00T-style view, and it is what decides which
    columns of ``observation.state`` mean which body part. Compared field for field,
    with nothing set aside, because every entry in it is a claim about the data.
    """
    result = Finding(subject="meta/modality.json")
    if _absence(rebuilt, delivered, "modality.json", result):
        return result
    mine = meta_json(rebuilt, "modality.json")
    theirs = meta_json(delivered, "modality.json")
    a, b = flatten(mine), flatten(theirs)
    for path in sorted(set(a) | set(b)):
        if a.get(path, ABSENT) != b.get(path, ABSENT):
            result.differences[path] = (a.get(path, ABSENT), b.get(path, ABSENT))
    return result


def _stat_gap(a: Any, b: Any) -> float:
    import numpy as np

    x, y = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if x.shape != y.shape:
        return float("inf")
    return float(np.max(np.abs(x - y)))


def _allowed(feature: str, stat: str, delivered_stats: dict) -> float:
    """How far one statistic of one feature may drift, in its own units."""
    import numpy as np

    if not feature.startswith("observation.images"):
        return STAT_TOLERANCE
    lo = np.asarray(delivered_stats.get("min", 0.0), dtype="float64")
    hi = np.asarray(delivered_stats.get("max", 1.0), dtype="float64")
    span = max(float(np.max(hi - lo)), 1.0)
    fraction = (IMAGE_EXTREME_TOLERANCE if stat in EXTREME_STATS
                else IMAGE_STAT_TOLERANCE)
    return fraction * span


def declared_features(root: Path) -> set[str]:
    """What a copy's own ``meta/info.json`` says it contains."""
    return set((meta_json(root, "info.json") or {}).get("features") or {})


def undeclared(feature: str, mine: dict, declared_a: set, declared_b: set) -> bool:
    """Is this a statistic for a column its own copy never declared as a feature?

    austin_buds' delivered parquet carries an eighth column, ``absolute_action``,
    that its own ``info.json`` does not list -- real derived data (``state + action``,
    the delta action resolved to an absolute pose) which nothing downstream reads,
    because a LeRobot loader builds its feature set from ``info.json``. The RLDS
    source has no such field, so nothing in the source says to build it either.

    Only the copy that *has* the statistic is consulted. A feature both copies
    declare and only one computes is a writer that dropped something, which stays a
    difference; this catches only the reverse -- a column that was never part of the
    schema on the side that carries it.
    """
    holder = declared_a if feature in mine else declared_b
    return feature not in holder


def compare_stat_blocks(
    mine: dict, theirs: dict, label: str, result: Finding,
    declared_a: set = frozenset(), declared_b: set = frozenset(),
) -> None:
    """One ``{feature: {stat: values}}`` pair, bucketed by what explains a difference."""
    for feature in sorted(set(mine) | set(theirs)):
        if feature not in mine or feature not in theirs:
            bucket = (result.set_aside
                      if undeclared(feature, mine, declared_a, declared_b)
                      else result.differences)
            bucket[f"{label}{feature}"] = (
                "present" if feature in mine else ABSENT,
                "present" if feature in theirs else ABSENT,
            )
            continue
        for stat in sorted(set(mine[feature]) | set(theirs[feature])):
            where = f"{label}{feature}.{stat}"
            allowed = _allowed(feature, stat, theirs[feature])
            if stat not in mine[feature] or stat not in theirs[feature]:
                # a quantile the rebuild does not carry: the v3.0 to v2.1 downgrade
                # keeps five keys and the delivered copies have ten
                result.set_aside[where] = (
                    "present" if stat in mine[feature] else ABSENT,
                    "present" if stat in theirs[feature] else ABSENT,
                )
                continue
            gap = _stat_gap(mine[feature][stat], theirs[feature][stat])
            if gap <= allowed:
                continue
            bucket = (result.set_aside if feature in ORDER_FEATURES
                      else result.differences)
            bucket[where] = (f"differs by {gap:.4g}", f"allowed {allowed:.2g}")


def compare_stats_json(rebuilt: Path, delivered: Path) -> Finding:
    """``meta/stats.json`` -- the statistics over the whole dataset."""
    result = Finding(
        subject="meta/stats.json",
        reason="ordering features follow the episode and task order; missing "
               "quantiles are the keys the v2.1 downgrade drops; a column its own "
               "info.json does not declare is read by nothing downstream",
    )
    if _absence(rebuilt, delivered, "stats.json", result):
        return result
    mine = meta_json(rebuilt, "stats.json")
    theirs = meta_json(delivered, "stats.json")
    compare_stat_blocks(mine, theirs, "", result,
                        declared_features(rebuilt), declared_features(delivered))
    return result


def compare_episode_meta(
    rebuilt: Path, delivered: Path, pairs: dict[int, int]
) -> Finding:
    """``meta/episodes.jsonl`` and ``meta/episodes_stats.jsonl``, through the pairing.

    These two cannot be compared line for line. Both are keyed by episode index and the
    rebuild does not write episodes in the delivered order -- 134 of cmu_stretch's 135
    sit somewhere else -- so a line-by-line diff would call every line different and
    say nothing. Pair by pair they are exact, which is the finding worth having.

    Every pair, not a sample: both files are read in full anyway, so checking some of
    them would save nothing and would miss a length or a statistic that went wrong
    outside the sample.

    Reported per statistic rather than per episode. A statistic that is wrong is
    usually wrong in every episode -- cmu_stretch's image ``std`` is wrong in all 135
    -- and 405 lines saying so is not 405 findings, it is one finding printed 405
    times. So each ``(feature, stat)`` gets one line: how many pairs differ, and the
    worst of them.
    """
    import numpy as np

    result = Finding(
        subject="meta/episodes.jsonl + meta/episodes_stats.jsonl",
        reason="ordering features follow the episode and task order, which the "
               "episode question already reports; absent quantiles are the five "
               "keys the v3.0 to v2.1 downgrade drops; a column its own info.json "
               "does not declare is read by nothing downstream",
    )
    episodes_a = {e["episode_index"]: e
                  for e in meta_lines(rebuilt, "episodes.jsonl") or []}
    episodes_b = {e["episode_index"]: e
                  for e in meta_lines(delivered, "episodes.jsonl") or []}
    stats_a = {e["episode_index"]: e["stats"]
               for e in meta_lines(rebuilt, "episodes_stats.jsonl") or []}
    stats_b = {e["episode_index"]: e["stats"]
               for e in meta_lines(delivered, "episodes_stats.jsonl") or []}
    declared_a, declared_b = declared_features(rebuilt), declared_features(delivered)

    # (feature, stat) -> [pairs compared, pairs differing, worst gap, allowed,
    #                     delivered was all zero]
    tally: dict[tuple[str, str], list] = {}
    for one, other in sorted(pairs.items()):
        if one in episodes_a and other in episodes_b:
            a, b = episodes_a[one], episodes_b[other]
            for key in ("tasks", "length"):
                if a.get(key) != b.get(key):
                    result.differences[f"episodes.jsonl[{one}].{key}"] = (
                        a.get(key, ABSENT), b.get(key, ABSENT))
        if one not in stats_a or other not in stats_b:
            continue
        mine, theirs = stats_a[one], stats_b[other]
        for feature in sorted(set(mine) | set(theirs)):
            if feature not in mine or feature not in theirs:
                bucket = (result.set_aside
                          if undeclared(feature, mine, declared_a, declared_b)
                          else result.differences)
                bucket[f"episodes_stats.jsonl.{feature}"] = (
                    "present" if feature in mine else ABSENT,
                    "present" if feature in theirs else ABSENT)
                continue
            for stat in sorted(set(mine[feature]) | set(theirs[feature])):
                key = (feature, stat)
                allowed = _allowed(feature, stat, theirs[feature])
                if stat not in mine[feature] or stat not in theirs[feature]:
                    result.set_aside[f"episodes_stats.jsonl.{feature}.{stat}"] = (
                        "present" if stat in mine[feature] else ABSENT,
                        "present" if stat in theirs[feature] else ABSENT)
                    continue
                row = tally.setdefault(key, [0, 0, 0.0, allowed, True])
                row[0] += 1
                gap = _stat_gap(mine[feature][stat], theirs[feature][stat])
                if gap > allowed:
                    row[1] += 1
                    row[2] = max(row[2], gap)
                if not np.all(np.asarray(theirs[feature][stat], dtype="float64") == 0):
                    row[4] = False

    for (feature, stat), (seen, bad, worst, allowed, zeroed) in sorted(tally.items()):
        if not bad:
            continue
        where = f"episodes_stats.jsonl.{feature}.{stat}"
        text = (f"{bad} of {seen} pairs differ, worst {worst:.4g}",
                f"allowed {allowed:.2g}")
        if feature in ORDER_FEATURES:
            result.set_aside[where] = text
        elif zeroed:
            # Not a rebuild defect. The delivered copy records this statistic as
            # exactly zero in every episode -- cmu_stretch's image std, where the
            # pixels have a spread of 0.24 -- so the two differ because the rebuild
            # computed it and the delivered copy did not. Reported as a difference,
            # since it is one, with the cause named rather than tolerated away.
            result.differences[where] = (
                text[0], "the delivered copy records zero in every episode")
        else:
            result.differences[where] = text
    return result


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
    # Paired on their first rows and not on all of them. Named for what the pass did
    # rather than for what it usually means, because it catches two different things
    # and only one of them is a trim: an episode a row or two short, and an episode
    # of the same length whose values go wrong past the prefix. The second is the
    # more serious of the two, and calling the pair "trimmed" would bury it.
    prefix_only: dict[int, int] = field(default_factory=dict)
    # how far each of those is off, as rebuilt rows minus delivered. Free from the
    # fingerprints, and what separates the two cases: zero here means the length was
    # right and the values were not.
    row_deltas: dict[int, int] = field(default_factory=dict)
    rebuilt_only: list[int] = field(default_factory=list)
    delivered_only: list[int] = field(default_factory=list)
    rebuilt_rows: int = 0
    delivered_rows: int = 0

    @property
    def pairs(self) -> dict[int, int]:
        return {**self.exact, **self.prefix_only}

    @property
    def worst_row_delta(self) -> int:
        """The largest row difference among the prefix-matched pairs, unsigned."""
        return max((abs(delta) for delta in self.row_deltas.values()), default=0)

    @property
    def same_length(self) -> list[int]:
        """Prefix-matched pairs whose lengths agree, so their values do not.

        The pairing cannot say more than that -- it only hashed the rows -- but the
        sample compares these in full and says where they diverge.
        """
        return sorted(one for one, delta in self.row_deltas.items() if delta == 0)

    @property
    def rebuilt_total(self) -> int:
        return len(self.exact) + len(self.prefix_only) + len(self.rebuilt_only)

    @property
    def delivered_total(self) -> int:
        return len(self.exact) + len(self.prefix_only) + len(self.delivered_only)

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
    result.prefix_only = take(
        lambda f: f.prefix,
        set(rebuilt) - set(result.exact),
        set(delivered) - set(result.exact.values()),
    )

    result.row_deltas = {
        one: rebuilt[one].rows - delivered[other].rows
        for one, other in result.prefix_only.items()
    }

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


@dataclass(frozen=True)
class CameraTotals:
    """One camera's files and bytes over a whole dataset."""

    files: int
    bytes: int

    @property
    def mean_bytes(self) -> float:
        return self.bytes / self.files if self.files else 0.0


def video_totals(root: Path, keep: set[str] | None = None) -> dict[str, CameraTotals]:
    """Every camera's file count and total size, across all episodes.

    ``stat`` and nothing else -- no ffprobe, no decoding -- which is what makes it
    affordable over the whole dataset rather than over a sample. That matters because
    the frame-by-frame checks are only affordable on a few dozen episodes, so
    everything past the sample goes unlooked-at, and a rebuild that wrote a tenth of
    its videos is exactly the failure that hides there.
    """
    tallies: dict[str, list[int]] = {}
    for path in root.glob("videos/**/*.mp4"):
        camera = path.parent.name
        if keep is not None and camera not in keep:
            continue
        tally = tallies.setdefault(camera, [0, 0])
        tally[0] += 1
        tally[1] += path.stat().st_size
    return {
        name: CameraTotals(files, size)
        for name, (files, size) in sorted(tallies.items())
    }


@dataclass
class VideoCoverage:
    """Is there a video for every episode, and are they about the right size.

    Two questions the sampled comparison cannot answer, both settled from file
    metadata. The counts are held as *videos per episode* rather than as totals,
    because a rebuild that is missing episodes is legitimately missing their videos
    too -- comparing totals would report the first question's finding a second time.
    The sizes are held as the mean per file for the same reason.
    """

    rebuilt: dict[str, CameraTotals] = field(default_factory=dict)
    delivered: dict[str, CameraTotals] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        return not self.problems


def compare_video_totals(
    rebuilt: Path, delivered: Path, rebuilt_episodes: int, delivered_episodes: int
) -> VideoCoverage:
    """Weigh every camera's videos on both sides, without opening any of them."""
    keep = declared_cameras(delivered)
    result = VideoCoverage(
        rebuilt=video_totals(rebuilt, keep), delivered=video_totals(delivered, keep)
    )
    for camera, theirs in result.delivered.items():
        mine = result.rebuilt.get(camera)
        if mine is None:
            result.problems.append(f"video {camera}: the rebuild has none at all")
            continue
        if rebuilt_episodes and delivered_episodes:
            per_episode = theirs.files / delivered_episodes
            ours = mine.files / rebuilt_episodes
            if abs(ours - per_episode) > 1e-9:
                result.problems.append(
                    f"video {camera}: {mine.files} files for {rebuilt_episodes} "
                    f"episodes against {theirs.files} for {delivered_episodes} "
                    f"delivered -- {ours:.3f} per episode against {per_episode:.3f}"
                )
        if theirs.mean_bytes:
            ratio = mine.mean_bytes / theirs.mean_bytes
            if abs(ratio - 1) > SIZE_TOLERANCE:
                result.problems.append(
                    f"video {camera}: {mine.mean_bytes:,.0f} bytes a file on average "
                    f"against delivered {theirs.mean_bytes:,.0f} ({ratio:.2f}x) -- "
                    "beyond what a different ffmpeg build explains"
                )
    return result


@dataclass
class PromptCheck:
    """Whether the paired episodes carry the same task prompt.

    Checked over every pair rather than over the sample, because ``episodes.jsonl``
    is read in full anyway -- so the only thing a sampled prompt check saves is the
    chance of catching a ``task_index`` mapping that shifted.

    The multiset difference is reported and not failed: a rebuild missing episodes is
    legitimately missing their prompts, which is the second question's finding.
    """

    pairs: int = 0
    mismatched: dict[int, tuple[str, str]] = field(default_factory=dict)
    rebuilt_only: dict[str, int] = field(default_factory=dict)
    delivered_only: dict[str, int] = field(default_factory=dict)

    @property
    def agree(self) -> bool:
        return not self.mismatched


def compare_prompts(
    rebuilt: Path, delivered: Path, pairs: dict[int, int]
) -> PromptCheck:
    from collections import Counter

    mine, theirs = episode_prompts(rebuilt), episode_prompts(delivered)
    result = PromptCheck()
    for one, other in sorted(pairs.items()):
        if one not in mine or other not in theirs:
            continue
        result.pairs += 1
        if mine[one] != theirs[other]:
            result.mismatched[one] = (mine[one], theirs[other])

    left, right = Counter(mine.values()), Counter(theirs.values())
    result.rebuilt_only = dict(sorted((left - right).items()))
    result.delivered_only = dict(sorted((right - left).items()))
    return result


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


def frame_agreement(left, right) -> float:
    """How well two stacks of frames agree, as the weakest single frame.

    The weakest rather than the average, because a difference that shows up in one
    frame is still a difference; averaging lets a good majority hide it. Pooling the
    frames into one correlation is worse still: a static background is common to both
    clips, so pooling scored 0.994 on a pair that scored 0.55 frame by frame.
    """
    import numpy as np

    if left.shape[1:] != right.shape[1:]:
        return 0.0
    scores = []
    for index in range(min(len(left), len(right))):
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


def pixel_agreement(a: Path, b: Path, count: int = PIXEL_FRAMES) -> float:
    """How well two videos' sampled frames agree."""
    return frame_agreement(sample_frames(a, count), sample_frames(b, count))


def pixel_verdict(
    a: Path, b: Path, count: int = PIXEL_FRAMES
) -> tuple[float, str | None]:
    """The agreement, and what explains it when there isn't enough of it.

    Channel order is deliberately *not* a check of its own. It was one difference
    among the many a wrong picture can be -- a time offset and a crop position are
    the next two -- and a dedicated test for each never ends, while the frame
    comparison catches all of them. So the reversal is only ever consulted after the
    frames have already disagreed, and only to name the cause: five of the upstream
    OXE transforms flip red and blue, so it is the first thing worth ruling out.
    """
    left, right = sample_frames(a, count), sample_frames(b, count)
    score = frame_agreement(left, right)
    if score >= PIXEL_AGREEMENT:
        return score, None
    reversed_score = frame_agreement(left[..., ::-1], right)
    if reversed_score >= PIXEL_AGREEMENT:
        return score, (
            f"the rebuild's red and blue channels are exchanged -- reversing them "
            f"agrees {reversed_score:.3f}"
        )
    return score, None


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
    # A supplied delivered index means the two were paired on their own state and
    # action bytes, so a differing prompt says the *prompt* is wrong. Without one the
    # pair is a guess from position, and a differing prompt says the guess was wrong.
    # The two readings are opposite, and only one of them is supportable at a time.
    paired = delivered_index is not None
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
                "this pair has different task prompts, so the rebuild's task mapping "
                "disagrees with the delivered copy's -- the vectors themselves paired"
                if paired else
                "episode alignment is wrong: this pair has different task prompts, "
                "so the two datasets do not have the same episode at this index"
            )
            if not paired:
                # nothing past here can be trusted: these are two different episodes,
                # and their vector and video differences would be a wall of noise
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
        agreement, diagnosis = pixel_verdict(videos_a[key], videos_b[key])
        if agreement < PIXEL_AGREEMENT:
            verdict = "PIXELS"
            problems.append(
                f"video {key}: frames agree {agreement:.3f} against delivered, below "
                f"{PIXEL_AGREEMENT} -- the same size encoded from a different picture"
                + (f"; {diagnosis}" if diagnosis else "")
            )

        summary[key] = (
            f"{a.get('width')}x{a.get('height')} {a.get('nb_read_frames')}f "
            f"{a.get('codec_name')}/GOP{a.get('gop') or '>' + str(GOP_PROBE_FRAMES)} "
            f"{a['bytes']}B vs {b['bytes']}B ({ratio:.2f}x) px {agreement:.3f} {verdict}"
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


# Paired episodes compared in full by default. Chosen to match the unit the
# verification record already reports against -- "64/64" in the per-dataset table --
# so that a new run's number means the same thing as the ones already written down.
SAMPLE_EPISODES = 64


def choose_sample(pairs: dict[int, int], episodes: int) -> list[int]:
    """Which paired episodes to compare in full: half from each end of the range.

    Not the first N, which is what this did at first and what made it weaker than its
    own numbers suggested. openx2lerobot converts in chunks of twenty-five episodes,
    one worker to a chunk, and the aggregate is written in chunk order -- so the front
    of the index range is a single worker's output, and a chunk that went wrong
    anywhere after the first was never in the sample. Both ends cost the same.
    """
    ordered = sorted(pairs)
    if len(ordered) <= episodes:
        return ordered
    back = episodes // 2
    return ordered[: episodes - back] + ordered[len(ordered) - back:]


@dataclass
class Funnel:
    """How closely a rebuild matches, asked in four questions rather than one.

    A single verdict was the wrong shape for this. The rebuilds do not control the
    order they write episodes in, some episodes land a row or two short, and the
    metadata each side carries is not even the same set of files -- so "does it
    reproduce" collapses several different answers into one, and the useful ones get
    lost. These are separate on purpose, and ordered cheapest and widest first so that
    each explains the next: what the two copies declare, how much is there, whether
    what is there is the same, and whether the two describe the same distribution.

    The last is asked twice, over everything and over the episodes the two share. That
    pair is the point of the whole arrangement: if they disagree, the difference is the
    missing episodes from the second question and nothing more, and if the
    shared-episode number is also off then something is wrong with the values.
    """

    dataset: str
    pairing: Pairing
    declaration: Declaration = field(default_factory=Declaration)
    # the rest of meta/: the inventory, tasks.jsonl, modality.json, stats.json
    meta: list[Finding] = field(default_factory=list)
    # episodes.jsonl and episodes_stats.jsonl, which need the pairing to compare
    episode_meta: Finding = field(
        default_factory=lambda: Finding(subject="meta/episodes*"))
    episodes: list[EpisodeReport] = field(default_factory=list)
    chosen: list[int] = field(default_factory=list)
    coverage: VideoCoverage = field(default_factory=VideoCoverage)
    prompts: PromptCheck = field(default_factory=PromptCheck)
    gap_overall: dict[str, float] = field(default_factory=dict)
    gap_shared: dict[str, float] = field(default_factory=dict)
    distributions: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @property
    def declarations_agree(self) -> bool:
        """Every file under meta/ that needs no pairing to compare."""
        return self.declaration.agree and all(f.agree for f in self.meta)

    @property
    def values_agree(self) -> bool:
        """The sampled episodes, plus the checks that span all of them."""
        return (
            all(r.ok for r in self.episodes if r.index >= 0)
            and self.coverage.agree
            and self.prompts.agree
            and self.episode_meta.agree
        )

    @property
    def distributions_agree(self) -> bool:
        return all(v <= DISTRIBUTION_TOLERANCE for v in self.gap_shared.values())

    @property
    def agree(self) -> bool:
        """Every judged question. The episode counts are deliberately not among them."""
        return (
            self.declarations_agree and self.values_agree and self.distributions_agree
        )

    @property
    def reasons(self) -> list[str]:
        """Why :attr:`agree` is false, in the order the questions are asked."""
        out = []
        if not self.declaration.agree:
            out.append("the two copies declare different datasets")
        for finding in self.meta:
            if not finding.agree:
                out.append(f"{finding.subject} differs")
        if not self.episode_meta.agree:
            out.append("per-episode metadata differs")
        if not all(r.ok for r in self.episodes if r.index >= 0):
            out.append("sampled episodes differ")
        if not self.coverage.agree:
            out.append("the videos do not add up over all episodes")
        if not self.prompts.agree:
            out.append("paired episodes carry different task prompts")
        if not self.distributions_agree:
            out.append("the shared episodes do not describe the same distribution")
        return out


def measure(
    spec: DatasetSpec,
    rebuilt: Path,
    delivered: Path,
    episodes: int = SAMPLE_EPISODES,
    check_video: bool = True,
    row_tolerance: int = ROW_TOLERANCE,
) -> Funnel:
    """Walk the four questions in order, reusing the pairing for all of them."""
    declaration = compare_declarations(rebuilt, delivered)
    meta = [
        meta_inventory(rebuilt, delivered),
        compare_tasks(rebuilt, delivered),
        compare_modality(rebuilt, delivered),
        compare_stats_json(rebuilt, delivered),
    ]
    pairing = pair_episodes(
        episode_fingerprints(rebuilt), episode_fingerprints(delivered)
    )

    pairs = pairing.pairs
    if pairs:
        chosen = choose_sample(pairs, episodes)
    else:
        # Nothing paired at all, which is what a rebuild whose *values* are wrong looks
        # like -- no episode on either side has a counterpart, so there is nothing to
        # sample. Fall back to position so that the third question still says what
        # differs; the second has already reported that none of them matched.
        chosen = list(range(min(episodes, pairing.rebuilt_total)))
    reports = compare_chosen(spec, rebuilt, delivered, chosen, pairs,
                             check_video, row_tolerance)

    shared = set(pairing.exact)
    overall = {"rebuilt": distribution(rebuilt), "delivered": distribution(delivered)}
    restricted = {
        "rebuilt": distribution(rebuilt, episodes=shared),
        "delivered": distribution(
            delivered, episodes={pairing.exact[i] for i in shared}
        ),
    }
    return Funnel(
        dataset=spec.id,
        pairing=pairing,
        declaration=declaration,
        meta=meta,
        episode_meta=compare_episode_meta(rebuilt, delivered, pairs),
        episodes=reports,
        chosen=chosen,
        coverage=(
            compare_video_totals(rebuilt, delivered, pairing.rebuilt_total,
                                 pairing.delivered_total)
            if check_video else VideoCoverage()
        ),
        prompts=compare_prompts(rebuilt, delivered, pairs),
        gap_overall=distribution_gap(overall["rebuilt"], overall["delivered"]),
        gap_shared=distribution_gap(restricted["rebuilt"], restricted["delivered"]),
        distributions={"overall": overall, "shared": restricted},
    )


def _gaps(gaps: dict[str, float]) -> str:
    return "   ".join(f"{column} {value:.1e}" for column, value in sorted(gaps.items()))


def _mark(ok: bool | None) -> str:
    """A step's verdict as a word, so the report says what its numbers mean.

    ``None`` is a step that is measured and not judged, which is the episode counts
    and only them -- printing "ok" there would claim a pass the step never asserts.
    """
    return "[ -- ]" if ok is None else ("[ ok ]" if ok else "[FAIL]")


def _pair_text(values: tuple[Any, Any]) -> str:
    return f"{values[0]!r} vs {values[1]!r}"


def declaration_lines(d: Declaration) -> list[str]:
    lines = []
    for side in d.missing:
        lines.append(f"     ! the {side} copy has no meta/info.json at all")
    for path, values in sorted(d.differences.items()):
        lines.append(f"     ! {path:<42} {_pair_text(values)}")
    if not d.differences and not d.missing:
        lines.append("     every field both copies declare is the same on both sides")
    for label, bucket, why in (
        ("counts", d.counts, "they follow from the episodes below"),
        ("one-sided", d.one_sided, "only one copy declares them, so there is "
                                  "nothing to compare"),
    ):
        if not bucket:
            continue
        lines.append(f"     {label:<12} {len(bucket)} field(s); {why}, and they are "
                     "not judged here")
        for path, values in sorted(bucket.items()):
            lines.append(f"                  {path:<42} {_pair_text(values)}")
    return lines


def finding_lines(f: Finding, limit: int = 12) -> list[str]:
    """One metadata finding, with whatever explains the set-aside half beside it."""
    lines = [f"     {_mark(f.agree)} {f.subject}"]
    for where, values in sorted(f.differences.items()):
        if ABSENT in values and f.absent:
            side = "rebuilt" if values[0] == ABSENT else "delivered"
            lines.append(f"       ! {where:<46} the {side} copy does not have it")
    plain = {k: v for k, v in f.differences.items() if not (ABSENT in v and f.absent)}
    for where, values in sorted(plain.items())[:limit]:
        lines.append(f"       ! {where:<46} {_pair_text(values)}")
    if len(plain) > limit:
        lines.append(f"       ! ... and {len(plain) - limit} more")
    if f.set_aside:
        lines.append(f"         {len(f.set_aside)} set aside -- {f.reason}")
        for where, values in sorted(f.set_aside.items())[:limit]:
            lines.append(f"           {where:<44} {_pair_text(values)}")
        if len(f.set_aside) > limit:
            lines.append(f"           ... and {len(f.set_aside) - limit} more")
    return lines


def coverage_lines(c: VideoCoverage) -> list[str]:
    lines = []
    for camera, theirs in c.delivered.items():
        mine = c.rebuilt.get(camera)
        if mine is None:
            lines.append(f"     video {camera:<14} none in the rebuild against "
                         f"{theirs.files} delivered")
            continue
        ratio = mine.mean_bytes / theirs.mean_bytes if theirs.mean_bytes else float("inf")
        lines.append(
            f"     video {camera:<14} {mine.files:>6} files, {mine.mean_bytes:>11,.0f}B "
            f"each against {theirs.files} / {theirs.mean_bytes:,.0f}B ({ratio:.2f}x)"
        )
    for problem in c.problems:
        lines.append(f"     ! {problem}")
    return lines


def funnel_report(f: Funnel, verbose: bool = False) -> str:
    p = f.pairing
    lines = [f"{f.dataset}", ""]

    lines.append(f"1  declaration    {_mark(f.declarations_agree)}  everything under "
                 f"meta/ that needs no pairing")
    lines.append(f"     {_mark(f.declaration.agree)} meta/info.json")
    lines += declaration_lines(f.declaration)
    for finding in f.meta:
        lines += finding_lines(finding)

    lines += ["", f"2  episodes       {_mark(None)}  counted, never failed -- which "
                  "episodes a rebuild has is decided elsewhere"]
    lines.append(f"     rebuilt      {p.rebuilt_total:>6} episodes  "
                 f"{p.rebuilt_rows:>9,} rows")
    lines.append(f"     delivered    {p.delivered_total:>6} episodes  "
                 f"{p.delivered_rows:>9,} rows")
    lines.append(f"     reproduced   {len(p.pairs):>6}  ({len(p.exact)} identical, "
                 f"{len(p.prefix_only)} agreeing on their first {PREFIX_ROWS} rows only)")
    if p.row_deltas:
        same = len(p.same_length)
        shorter = sorted(d for d in p.row_deltas.values() if d)
        # the range only when something is actually a different length; "(+0 to +0
        # rows)" beside a count of zero is noise dressed as a measurement
        span = f" ({shorter[0]:+d} to {shorter[-1]:+d} rows)" if shorter else ""
        lines.append(f"                  of those, {len(shorter)} differ in length{span} "
                     f"and {same} are the same length with values that go wrong "
                     "further in")
    lines.append(f"     unmatched    {len(p.delivered_only):>6} of the delivered "
                 f"episodes have no counterpart; {len(p.rebuilt_only)} of the "
                 "rebuild's do not either")
    if p.moved:
        lines.append(f"     reordered    {p.moved:>6} of {len(p.pairs)} paired episodes "
                     "carry a different index")

    compared = [r for r in f.episodes if r.index >= 0]
    failed = [r for r in compared if not r.ok]
    lines += ["", f"3  sample         {_mark(f.values_agree)}  the episodes that are "
                  "there, compared in full"]
    lines.append(f"     compared     {len(compared):>6} of {len(p.pairs)} paired episodes, "
                 "half from each end of the index range")
    lines.append(f"     identical    {len(compared) - len(failed):>6}"
                 + (f"  ({len(failed)} differ)" if failed else ""))
    for r in compared if verbose else failed:
        lines += ["     " + line for line in episode_lines(r)]
    lines += coverage_lines(f.coverage)
    prompts = f.prompts
    lines.append(f"     prompts      {prompts.pairs:>6} paired episodes checked; "
                 + ("all agree" if prompts.agree
                    else f"{len(prompts.mismatched)} carry different prompts"))
    for index, values in sorted(prompts.mismatched.items())[:8]:
        lines.append(f"     ! episode {index}: {_pair_text(values)}")
    lines += finding_lines(f.episode_meta)

    lines += ["", f"4  distributions  {_mark(f.distributions_agree)}  max difference per "
                  "column, as a fraction of its range"]
    lines.append(f"     every episode        {_gaps(f.gap_overall)}")
    lines.append(f"     the shared ones      {_gaps(f.gap_shared)}")
    if not f.distributions_agree:
        lines.append("     ! the shared episodes do not agree, so this is the values "
                     "differing and not the missing episodes")
    elif p.delivered_only or p.rebuilt_only or p.prefix_only:
        # Whatever is outside the shared set is what the two rows differ over, and it
        # is not always a missing episode: an episode that only matched on its prefix
        # is present on both sides and still excluded, because its rows are not the
        # delivered rows. Saying only "absent" there would leave the gap unexplained.
        outside = len(p.delivered_only) + len(p.rebuilt_only) + len(p.prefix_only)
        lines.append(f"     the two rows differ over the {outside} episode(s) outside the "
                     f"shared set -- {len(p.delivered_only)} absent from the rebuild, "
                     f"{len(p.rebuilt_only)} absent from the delivered copy, "
                     f"{len(p.prefix_only)} matching on their first rows only; over what "
                     "both copies hold identically, they agree")

    lines += ["", f"verdict          {_mark(f.agree)}  " + (
        "the rebuild reproduces the delivered copy" if f.agree
        else "; ".join(f.reasons))]
    if f.agree and (p.delivered_only or p.rebuilt_only):
        lines.append(f"                        over the {len(p.pairs)} episodes the two "
                     f"share; {len(p.delivered_only)} delivered episode(s) are not there "
                     "to compare")
    return "\n".join(lines)


# Significant digits kept for a recorded statistic. *Significant*, not decimal
# places: the interesting numbers here span 1e-15 to 1e+03, and rounding to twelve
# decimals would turn "agrees to 1.5e-15" into a flat zero -- which is the one
# distinction these records exist to preserve.
#
# Twelve rather than all of them because the records are committed. A float64's full
# repr changes in its last digits when a rebuild sums in a different order, so an
# unrounded record re-diffs entirely on a re-run that found nothing new, and a diff
# that always changes is a diff nobody reads. Twelve digits is six orders of
# magnitude finer than DISTRIBUTION_TOLERANCE.
RECORDED_DIGITS = 12


def _jsonable(value: Any) -> Any:
    """numpy scalars and arrays as plain JSON, recursively, floats rounded."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    as_list = getattr(value, "tolist", None)
    if callable(as_list):
        return _jsonable(as_list())
    if isinstance(value, float):
        # inf and nan have no digits to keep, and float() would not survive the format
        return value if value != value or value in (float("inf"), float("-inf")) else (
            float(f"{value:.{RECORDED_DIGITS}g}"))
    return value


def settings(episodes: int, row_tolerance: int, check_video: bool) -> dict[str, Any]:
    """The constants that decided the verdict, recorded alongside it.

    A verdict without its thresholds cannot be re-read a month later, and these are
    the ones a reader would otherwise have to go and look up in this file at whatever
    revision the run happened to use.
    """
    return {
        "sample_episodes": episodes,
        "row_tolerance": row_tolerance,
        "video_checked": check_video,
        "size_tolerance": SIZE_TOLERANCE,
        "pixel_agreement": PIXEL_AGREEMENT,
        "pixel_frames": PIXEL_FRAMES,
        "distribution_tolerance": DISTRIBUTION_TOLERANCE,
        "prefix_rows": PREFIX_ROWS,
        "gop_probe_frames": GOP_PROBE_FRAMES,
        "distribution_stats": list(DISTRIBUTION_STATS),
        "count_fields": list(COUNT_FIELDS),
    }


def as_dict(
    f: Funnel, rebuilt: Path, delivered: Path, run_settings: dict[str, Any]
) -> dict[str, Any]:
    """Everything the run measured, as JSON.

    Shaped so the record outlives the machine that made it. The runs happen on
    throwaway nodes, one dataset to a node, and the files are collected into one
    directory afterwards and committed -- so this carries the thresholds that decided
    the verdict rather than only the verdict, and the full per-dimension statistics
    rather than only the gap they were reduced to. A number nobody can recompute is a
    number nobody can argue with, and the datasets themselves are far too large to
    keep beside the record.
    """
    from dataclasses import asdict
    from datetime import datetime, timezone

    p = f.pairing
    # one pass over the whole payload at the end rather than at each numpy-bearing
    # field: a field added later should not have to remember to do this
    return _jsonable({
        "dataset": f.dataset,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rebuilt": str(rebuilt),
        "delivered": str(delivered),
        "settings": run_settings,
        "verdict": {
            "overall": f.agree,
            "declaration": f.declarations_agree,
            "sample": f.values_agree,
            "distributions": f.distributions_agree,
            "reasons": f.reasons,
        },
        "declaration": {
            "missing": f.declaration.missing,
            "differences": {
                k: list(v) for k, v in sorted(f.declaration.differences.items())
            },
            "counts": {k: list(v) for k, v in sorted(f.declaration.counts.items())},
            "one_sided": {
                k: list(v) for k, v in sorted(f.declaration.one_sided.items())
            },
        },
        "meta": [
            {
                "subject": finding.subject,
                "agree": finding.agree,
                "absent": finding.absent,
                "reason": finding.reason,
                "differences": {
                    k: list(v) for k, v in sorted(finding.differences.items())
                },
                "set_aside": {k: list(v) for k, v in sorted(finding.set_aside.items())},
            }
            for finding in [*f.meta, f.episode_meta]
        ],
        "episodes": {
            "rebuilt_total": p.rebuilt_total,
            "delivered_total": p.delivered_total,
            "rebuilt_rows": p.rebuilt_rows,
            "delivered_rows": p.delivered_rows,
            "exact": len(p.exact),
            "prefix_only": len(p.prefix_only),
            "same_length_but_differing": p.same_length,
            "reordered": p.moved,
            "worst_row_delta": p.worst_row_delta,
            # the index map itself, because it is the only record of which rebuilt
            # episode was compared against which delivered one -- and without it a
            # finding about "episode 12" cannot be looked up again
            "pairs": {str(k): v for k, v in sorted(p.pairs.items())},
            "row_deltas": {str(k): v for k, v in sorted(p.row_deltas.items())},
            "rebuilt_only": p.rebuilt_only,
            "delivered_only": p.delivered_only,
        },
        "sample": {
            "chosen": f.chosen,
            "reports": [asdict(r) for r in f.episodes],
        },
        "video": {
            "rebuilt": {k: asdict(v) for k, v in f.coverage.rebuilt.items()},
            "delivered": {k: asdict(v) for k, v in f.coverage.delivered.items()},
            "problems": f.coverage.problems,
        },
        "prompts": {
            "pairs_checked": f.prompts.pairs,
            "mismatched": {
                str(k): list(v) for k, v in sorted(f.prompts.mismatched.items())
            },
            "rebuilt_only": f.prompts.rebuilt_only,
            "delivered_only": f.prompts.delivered_only,
        },
        "distributions": {
            "gap_overall": f.gap_overall,
            "gap_shared": f.gap_shared,
            "detail": f.distributions,
        },
    })


def write_report(payload: dict[str, Any], text: str, into: Path) -> list[Path]:
    """One JSON and one text file per dataset, named after it.

    Two files rather than one because they answer to different readers: the JSON is
    what a later pass aggregates across datasets, and the text is what a person opens.
    Named after the dataset and nothing else, so that collecting several nodes' output
    into one directory cannot collide and re-running one dataset overwrites its own
    record instead of accumulating copies.
    """
    into.mkdir(parents=True, exist_ok=True)
    name = payload["dataset"]
    paths = [into / f"{name}.json", into / f"{name}.txt"]
    paths[0].write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    paths[1].write_text(text + "\n")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help=f"one of: {', '.join(available())}")
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--delivered", type=Path, default=None,
                        help="defaults to the spec's delivered.path")
    parser.add_argument("--episodes", type=int, default=SAMPLE_EPISODES,
                        help="how many paired episodes to compare in full, half taken "
                             "from each end of the index range (step 3)")
    parser.add_argument("--report", type=Path, default=None,
                        help="directory to write <dataset>.json and <dataset>.txt into, "
                             "for collecting several runs' records in one place")
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
    text = funnel_report(measured, verbose=args.verbose)
    print(text)

    if args.report:
        run_settings = settings(args.episodes, args.row_tolerance, not args.no_video)
        written = write_report(
            as_dict(measured, args.rebuilt, delivered, run_settings), text, args.report
        )
        print("\nwritten  " + "  ".join(str(path) for path in written))

    # Missing episodes are reported, not judged: which episodes a rebuild ends up with
    # is decided outside this comparison, and a run that lost some should still be able
    # to say whether the ones it has are right.
    return 0 if measured.agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
