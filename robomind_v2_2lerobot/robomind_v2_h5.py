"""RoboMIND 2.0 → LeRobot v3.0.

    python robomind_v2_h5.py --src-paths /data/robomind_2_0/... --output-path /out

어떤 embodiment 를 읽을지는 플래그로 받지 않는다. 원본이
``data/<embodiment>/<task>/success_episodes/<timestamp>/data/<name>.hdf5`` 로 시작하므로
``data/`` 밑 디렉토리 이름이 곧 config 이름이다. 이 릴리스는 로봇 하나를 repo 5 개로
쪼개 배포하는데 그 다섯이 모두 같은 ``data/<embodiment>`` 로 시작하므로, ``--src-paths``
를 복수로 받으면 한 번에 하나의 config 로 모인다.

에피소드가 하나도 안 만들어지면 **실패한다.** v1 은 이 경우 출력 디렉토리를 지우고 정상
종료했고, 그러면 "0 개 변환됨" 이 성공과 구별되지 않는다. 같은 이유로, 시작한 task 하나가
끝까지 못 가고 죽는 것도 **실패한다** -- 다른 task 들이 잘 끝났다고 해서 조용히 덮이지
않는다 (``TasksFailed``, ``TaskResult``).
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robomind_v2_utils.configs import ConfigError, EmbodimentConfig, available
from robomind_v2_utils.configs import load as load_config
from robomind_v2_utils.lerobot_utils import RoboMINDv2Dataset, build_features
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import (
    EpisodeRef,
    discover,
    read_episode,
    task_max_frames,
    task_profile,
)

# This module's own logger -- never `logging` (the root logger) directly.
# Importing this file must not reconfigure logging for everyone else who
# imports it (tests included): only the "run as a script" entry point at the
# bottom attaches a handler to *this* logger. That also sidesteps a real
# failure mode this had as a plain `logging.basicConfig(...)` call at import
# time: some dependency pulled in via `robomind_v2_utils.lerobot_utils`
# attaches its own handler to the root logger as an import side effect, and
# `basicConfig` is a no-op once a handler already exists there -- so the
# messages below silently never reached the console in a real run. A
# dedicated logger, configured only at the entry point, doesn't depend on
# root's state either way.
#
# It is also the only thing that reaches the console at all under Ray: worker
# processes' log records do not propagate back to the driver's terminal, so a
# per-episode `logger.warning(...)` inside `convert_task` is visible with
# `--debug` and invisible in a real run. `TaskResult.reasons` -- returned,
# not logged from the worker -- is what survives that boundary; see
# `_summarize` and `_write_summary`.
logger = logging.getLogger(__name__)

# An episode this short is a recording that went wrong rather than a short task.
# Carried over from v1, which used the same floor.
DEFAULT_MIN_FRAMES = 50

# An episode's own frame rate that disagrees with its task's median by more than
# this fraction is skipped rather than written onto the wrong time base. Real
# evidence this threshold catches: one release task's episodes measured
# 26.94 Hz and 31.33 Hz -- 16% apart -- and the slower one set the dataset's rate.
_FPS_DRIFT_THRESHOLD = 0.1


def _drift_tolerance(fps_span: int | None) -> float:
    """How much additional drift this one episode's own rate measurement
    cannot actually distinguish from a genuinely different rate, on top of
    the flat ``_FPS_DRIFT_THRESHOLD`` every episode is held to.

    A ``real`` episode's rate is ``frame_count / span`` from a span rounded
    to whole seconds (``reader.episode_fps``): the true span could be up to
    about a second shorter or longer than the recorded one, so the rate
    itself already carries roughly ``1/span`` of quantization noise before an
    episode has drifted from its task at all. Across this release's real
    sample episodes, spans of 8-34 seconds put that noise band at
    2.9%-12.5% -- comparable to, or larger than, the flat 10% threshold on
    its own -- so a flat threshold alone would skip perfectly good short
    episodes as "drift" that is really just how coarsely a short episode's
    own rate can ever be known (the I-A finding).

    ``fps_span`` is ``None`` for a simulated episode (see ``episode_fps``):
    its timestamps are milliseconds on a fine clock, which carries no
    comparable quantization, so it gets no extra tolerance here -- the flat
    threshold alone already fits it.
    """
    return 1.0 / fps_span if fps_span else 0.0


# Categories a skipped or failed episode is bucketed into, for the per-task,
# per-embodiment tally `main` logs and writes beside the output (`TaskResult`,
# `_summarize`, `_write_summary`). Deliberately coarse: counts per reason are
# enough to reconcile what landed against what was expected. The specific
# message for any one skip already reached the log the moment it happened
# (with `--debug`; see the module logger's docstring above for why that is not
# true under Ray) -- this tally exists for the count, not to repeat the message.
_MALFORMED = "malformed_or_incomplete_file"
_UNREADABLE = "unreadable"
_TOO_SHORT = "too_short"
_RATE_UNUSABLE = "rate_unusable"
_RATE_DRIFT = "rate_drift"
_RESOLUTION_MISMATCH = "resolution_mismatch"
_WRITE_FAILED = "write_failed"


class NothingConverted(RuntimeError):
    """No episode was written. Never a success."""


class TasksFailed(RuntimeError):
    """One or more tasks raised instead of completing.

    Deliberately distinct from ``NothingConverted``: a failed task can have
    written real episodes before it died (see ``convert_task``'s outer
    ``except``), so "nothing was converted" would be a false claim about a run
    that produced real, if incomplete, output. What actually failed is
    quarantined on disk (``_quarantine``) rather than left to look finished.
    """


def plan_tasks(src_paths: list[Path]) -> dict[tuple[str, str], list[EpisodeRef]]:
    """Episodes grouped into one conversion task per ``(embodiment, task)``.

    One LeRobot dataset per group, which is also what lets each group carry its
    own frame rate.
    """
    grouped: dict[tuple[str, str], list[EpisodeRef]] = defaultdict(list)
    for ref in discover(src_paths):
        grouped[(ref.embodiment, ref.task)].append(ref)
    return dict(grouped)


@dataclass(frozen=True)
class TaskResult:
    """One task's outcome: how many episodes it wrote, how many it skipped, and
    why. Returned rather than only logged so the driver can tally it even under
    Ray, where a worker's own log records never reach the console (see the
    module logger's docstring).

    A plain dataclass, deliberately: an earlier version used ``__slots__`` to
    save a little memory on an object created only a handful of times per run,
    and that made every task fail to round-trip through Ray's serializer --
    ``AttributeError: 'TaskResult' object has no attribute '__dict__'`` on the
    driver, for every single task, discovered only by actually running this
    through real Ray rather than the synthetic, ``--debug``-only test suite.
    Not worth the risk for the memory it saved.
    """

    written: int
    skipped: int
    reasons: dict[str, int]


def _shape_mismatch(dataset_shapes: dict, episode_shapes: dict) -> str | None:
    """``None`` if every key ``dataset_shapes`` names matches in ``episode_shapes``;
    otherwise a message naming each key that disagrees and both shapes.

    ``build_features`` fixes a task's video shapes once. That basis used to be
    whichever episode happened to create the dataset; it is now the task's own
    pre-pass majority shape (``reader.task_profile``, colour keys) merged with
    that first episode's own depth keys (see ``convert_task``) -- either way,
    every episode's own measured shape -- already in hand, ``read_episode``
    returns it as ``Episode.shapes`` -- is compared against the basis rather
    than trusted to match, including the episode that ends up creating the
    dataset itself. Both 720x1280 and 480x640 occur within a single release
    task, and the older converter had a workaround for exactly this case.
    """
    mismatches = [
        f"{key} is {episode_shapes.get(key)}, {dataset_shapes[key]} was opened"
        for key in dataset_shapes
        if episode_shapes.get(key) != dataset_shapes[key]
    ]
    return "; ".join(mismatches) if mismatches else None


def _quarantine(local_dir: Path) -> None:
    """Rename a task's output so a run that failed partway through it can never
    be mistaken for one that finished.

    Renamed, not removed: a task can fail after writing hundreds of real
    episodes (``convert_task``'s outer ``except`` runs this for any failure,
    including one deep into a long task), and this converter's source is
    114.28 TB an operator does not want to re-read solely to get back to
    diagnosing where a failed task already was -- keeping the partial bytes
    costs disk; deleting them costs a rerun from zero just to reproduce the
    same failure. Any previous quarantine at the same path is replaced rather
    than accumulated: by the time a task is retried, an earlier failure's
    evidence has presumably already been read.
    """
    if not local_dir.exists():
        return
    quarantined = local_dir.with_name(local_dir.name + ".failed")
    if quarantined.exists():
        shutil.rmtree(quarantined)
    local_dir.rename(quarantined)


_PARTIAL_RESULT_FILENAME = "partial.json"


def _write_partial_result(
    embodiment: str, task: str, local_dir: Path, written: int, skipped: int, reasons: dict
) -> None:
    """A task's own written/skipped tally, written into its own directory just
    before a fatal failure quarantines it.

    This is the one place that tally survives a task that dies after writing
    real episodes: threading it through the exception ``convert_task`` raises
    instead would depend on an exception object's own custom attributes
    surviving Ray's serialization of a failed remote task across the
    worker/driver boundary -- a real, previously-hit failure mode for a far
    simpler object (see ``TaskResult``'s own docstring) -- while this is a
    plain file, written synchronously from inside the worker before the
    task's process goes anywhere. ``_summarize`` reads it back
    (``_read_partial_result``) from the quarantined directory, so a task that
    died after writing real episodes is not reported as having written zero
    (the I-D finding).

    Any exception writing it is caught and logged, not raised: this runs from
    ``convert_task``'s own outer failure handler, which is already failing for
    its own reason -- a disk genuinely full enough to fail the original write
    (proven live: this really happens, not a hypothetical) is exactly the
    condition most likely to also fail this one, and letting that mask the
    original exception (skipping the ``_quarantine`` call right after it, in
    the same handler) would be strictly worse than simply not having this
    enrichment for this one task.
    """
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / _PARTIAL_RESULT_FILENAME).write_text(
            json.dumps({"written": written, "skipped": skipped, "reasons": dict(reasons)})
        )
    except OSError as exc:
        logger.error(
            "%s/%s: could not write %s -- its own written/skipped counts will read as "
            "zero in summary.json: %s",
            embodiment, task, _PARTIAL_RESULT_FILENAME, exc,
        )


def _read_partial_result(output_path: Path, embodiment: str, task: str) -> TaskResult:
    """A failed task's own written/skipped/reasons tally, recovered from the
    marker file it wrote into its own directory just before quarantine (see
    ``_write_partial_result``).

    Falls back to all-zero if that marker is itself missing or unreadable --
    e.g. a task that failed before ``_write_partial_result`` itself ran (a
    bug in this cleanup path, or a version of this file predating it) should
    not also crash the summary it is trying to make more honest.
    """
    path = output_path / embodiment / f"{task}.failed" / _PARTIAL_RESULT_FILENAME
    try:
        data = json.loads(path.read_text())
        return TaskResult(
            written=int(data.get("written", 0)),
            skipped=int(data.get("skipped", 0)),
            reasons=dict(data.get("reasons", {})),
        )
    except (OSError, ValueError, TypeError):
        return TaskResult(written=0, skipped=0, reasons={})


def convert_task(
    refs: list[EpisodeRef],
    embodiment: str,
    task: str,
    output_path: Path,
    save_depth: bool,
    min_frames: int,
) -> TaskResult:
    """Write one LeRobot dataset for one task.

    Everything from the frame-rate pre-pass through ``finalize()`` runs inside
    one try/except. A per-episode problem this converter can recover from
    safely -- an unreadable file, too few frames, a rate that doesn't belong
    on this task's time base, a resolution that doesn't match the dataset this
    task already created, a disk error while still only buffering frames --
    is caught narrowly at its own site below and only skips that episode.
    Anything else -- most importantly a failure raised by ``save_episode()``
    itself, once it may already have committed part of this episode to disk
    (see the comment at that call) -- is a condition this function cannot
    safely shrug off, so it quarantines whatever the task had written so far
    (``_quarantine``) and re-raises, rather than returning as if the task
    were merely empty. ``main`` tells a failed task apart from an empty one by
    exactly that: whether this function returned or raised (``TaskResult``,
    ``TasksFailed``).
    """
    config = load_config(embodiment)
    local_dir = (output_path / embodiment / task).resolve()
    if local_dir.exists():
        shutil.rmtree(local_dir)

    try:
        # One dataset per task, so its rate and its camera shape each have to
        # be one value -- but both used to come from whichever episode
        # happened to survive first, with every later disagreement either
        # only warned about and written onto that rate anyway (fps), or
        # skipped outright in favour of the first arrival (shape) regardless
        # of which one was actually the odd one out. This pre-pass (cheap:
        # `camera_observations/timestamp` is tiny and a shape costs one JPEG
        # decode per camera, nowhere near a full episode's worth of frames)
        # measures every episode's own rate and shape up front, so the task
        # is opened at their median rate and majority shape instead, and any
        # one episode's own measurements can be checked against those below
        # rather than trusted just for arriving first.
        #
        # The drift check just below compares against this unrounded median,
        # never against the integer `dataset_fps` a LeRobot dataset actually
        # opens at: `round()` alone can read as over 10% of "drift" against a
        # fractional true rate (10 frames / 3 s = 3.333 Hz rounds to 3, an
        # 11.1% gap from that integer with zero episodes actually disagreeing)
        # -- comparing episodes against each other's median, not against the
        # rounding that median happens to produce, is what this guards against.
        profile = task_profile(refs, config, min_frames=min_frames)
        median_fps = profile.fps
        dataset_fps = round(median_fps) if median_fps is not None else None
        # Colour keys only (see task_profile) -- empty when the pre-pass could
        # not measure a shape from any episode, which falls back to today's
        # behaviour of trusting whichever episode ends up creating the dataset.
        dataset_shapes: dict[str, tuple[int, ...]] = dict(profile.shapes or {})

        dataset = None
        written = 0
        skipped = 0
        reasons: Counter = Counter()

        for ref in refs:
            try:
                episode = read_episode(ref, config, save_depth=save_depth)
            except EpisodeSkipped as exc:
                logger.warning("skipped %s: %s", ref.path, exc)
                skipped += 1
                reasons[_MALFORMED] += 1
                continue
            except (OSError, ValueError) as exc:
                logger.warning("skipped %s: unreadable: %s", ref.path, exc)
                skipped += 1
                reasons[_UNREADABLE] += 1
                continue

            if len(episode.frames) < min_frames:
                logger.warning(
                    "skipped %s: %d frames < %d", ref.path, len(episode.frames), min_frames
                )
                skipped += 1
                reasons[_TOO_SHORT] += 1
                continue

            # A rate at or below 0.5 Hz rounds to 0, and a dataset can't be
            # opened at 0 fps -- nor can the drift check below divide by it.
            # `task_profile` already excludes a rate like this from the median
            # it computed (see its own docstring) for the same reason; this is
            # the same floor, just re-applied per episode rather than trusted
            # to have been excluded upstream.
            if round(episode.fps) < 1:
                logger.warning(
                    "skipped %s: measured rate %.3f Hz rounds to 0, not usable as a fps",
                    ref.path, episode.fps,
                )
                skipped += 1
                reasons[_RATE_UNUSABLE] += 1
                continue

            if median_fps is None:
                # Every episode task_profile looked at failed to open, or
                # failed to yield a usable rate -- and this ref's own file is
                # the same file `read_episode` above just read successfully
                # (including its own call to the same rate measurement), so
                # in practice this branch cannot fire once execution reaches
                # here. Kept anyway as a guard, not an assumption: dividing by
                # `median_fps` just below must never run against `None`, on
                # however slim a chance this file changed between the two reads.
                logger.warning(
                    "skipped %s: no episode in %s/%s produced a usable frame rate",
                    ref.path, embodiment, task,
                )
                skipped += 1
                reasons[_RATE_UNUSABLE] += 1
                continue

            drift = abs(episode.fps - median_fps) / median_fps
            threshold = max(_FPS_DRIFT_THRESHOLD, _drift_tolerance(episode.fps_span))
            if drift > threshold:
                logger.warning(
                    "skipped %s: measured %.1f Hz drifts from %s/%s's %d Hz base "
                    "by %.1f%% (span %s, tolerance %.1f%%) -- not written onto the "
                    "wrong time base",
                    ref.path, episode.fps, embodiment, task, dataset_fps, drift * 100,
                    f"{episode.fps_span}s" if episode.fps_span else "n/a",
                    threshold * 100,
                )
                skipped += 1
                reasons[_RATE_DRIFT] += 1
                continue

            # `dataset_shapes` is the task's own pre-pass majority (colour
            # keys only -- see task_profile), decided before this loop ever
            # started, so this same check already applies to the very first
            # episode reached, not just later ones: an odd-resolution episode
            # that happens to sort first in discovery order is what used to
            # get to define the dataset's shape and make every correctly-sized
            # episode the one that gets skipped instead (the I-B finding).
            # Trivially passes while `dataset_shapes` is still empty (the
            # majority itself could not be measured, or depth keys, which the
            # majority never covers -- see below).
            mismatch = _shape_mismatch(dataset_shapes, episode.shapes)
            if mismatch:
                logger.warning("skipped %s: shape mismatch: %s", ref.path, mismatch)
                skipped += 1
                reasons[_RESOLUTION_MISMATCH] += 1
                continue

            if dataset is None:
                # Fill in whatever the majority basis does not cover -- depth
                # keys (never part of it: majority-voting depth would mean
                # decoding a depth frame per camera per episode in the
                # pre-pass, for a feature most conversions do not even ask to
                # write), or every key if the majority itself could not be
                # measured -- from this specific episode, the one that ends
                # up actually creating the dataset. The majority basis, where
                # it exists, always wins for the keys it names: `episode.shapes`
                # is listed first only so its keys still contribute when
                # `dataset_shapes` does not already carry them.
                dataset_shapes = {**episode.shapes, **dataset_shapes}
                dataset = RoboMINDv2Dataset.create(
                    repo_id=f"{embodiment}/{task}",
                    root=local_dir,
                    fps=dataset_fps,
                    robot_type=config.robot_type,
                    features=build_features(config, dataset_shapes),
                )

            try:
                for frame in episode.frames:
                    dataset.add_frame(frame)
            except OSError as exc:
                # Nothing has been committed to the writer's own bookkeeping
                # yet at this point -- `save_episode` below is what does that
                # -- so discarding this episode's buffer and moving on is
                # safe. `clear_episode_buffer` both deletes this episode's
                # already-written per-frame image files and resets the buffer
                # a fresh `add_frame` call needs; without the reset, the next
                # episode's first `add_frame` would try to `.append` onto
                # arrays `save_episode` would have stacked, not lists.
                logger.warning("skipped %s: failed while adding frames: %s", ref.path, exc)
                dataset.writer.clear_episode_buffer(delete_images=True)
                skipped += 1
                reasons[_WRITE_FAILED] += 1
                continue

            try:
                dataset.save_episode()
            except Exception as exc:  # noqa: BLE001 - deliberately as broad as
                # the outer `except Exception` this feeds into: every failure
                # from save_episode leaves the same bookkeeping-ahead-of-
                # total_frames problem described below regardless of its
                # type, so catching only OSError here (as an earlier version
                # did) caught the common case (an ffmpeg or disk failure) but
                # let anything else -- a pyarrow error, a bug -- skip this
                # specific "failed to save" message while still being just as
                # fatal via the outer handler a few lines down. This is not a
                # narrower, more-recoverable catch than that one -- it exists
                # only to log a save-specific message before re-raising, so
                # its scope now actually matches what it is for instead of
                # only appearing to.
                #
                # save_episode writes this episode's non-video columns to the
                # data parquet (`_save_episode_data`) *before* it encodes
                # video -- so by the time encoding fails (any ffmpeg failure)
                # or a later write does (disk-full), the writer's own
                # bookkeeping (`_latest_episode`, `_recorded_frames`) can
                # already assume bytes that `self._meta.total_frames` was
                # never told about. There is no safe way to roll that back
                # short of reloading the dataset from what is already on
                # disk -- this project's not-yet-built resume path -- so this
                # is treated as fatal to the whole task, not just this
                # episode: propagate, let the outer `except` quarantine
                # whatever is on disk, and let `main` count the task as
                # failed rather than pretend the remaining episodes can still
                # land safely on a writer whose internal state may already
                # disagree with it.
                logger.error("%s failed to save: %s", ref.path, exc)
                raise

            written += 1
            logger.info("wrote %s (%d frames)", ref.path, len(episode.frames))

        if dataset is None:
            logger.warning("no episode survived for %s/%s", embodiment, task)
            return TaskResult(written=0, skipped=skipped, reasons=dict(reasons))

        dataset.finalize()
        return TaskResult(written=written, skipped=skipped, reasons=dict(reasons))
    except Exception:
        logger.error("%s/%s failed; quarantining %s", embodiment, task, local_dir)
        if dataset is not None:
            # A task-fatal failure can leave real, already-committed episodes
            # sitting only in this dataset's own in-memory metadata buffer --
            # LeRobotDatasetMetadata flushes it to disk only every
            # `metadata_buffer_size` episodes, or on finalize(). Left
            # unflushed, that buffer is lost the moment this object is
            # garbage-collected, since `_quarantine` below only renames
            # whatever is already *on disk*. Finalizing here -- while
            # `local_dir` is still at its live path -- flushes it there
            # before the rename, and, on a clean pass, marks the dataset, its
            # writer, and its writer's metadata finalized, so the garbage-
            # collection safety net below (`DatasetWriter.__del__`) becomes a
            # no-op wherever it eventually runs instead of recreating this
            # directory after it is gone (see the evidence doc for how this
            # was reproduced: a writer collected after `_quarantine` renamed
            # its directory away re-ran `_flush_metadata_buffer`, whose
            # `mkdir(parents=True, exist_ok=True)` against the writer's own,
            # still-stale `root` recreated the very directory that had just
            # been renamed).
            #
            # Any exception finalize() itself raises here is suppressed, not
            # re-raised: this function is already failing for its own
            # reason, and finalize() failing too -- plausibly from the exact
            # same disk error -- must not replace or mask that original
            # exception.
            try:
                dataset.finalize()
            except Exception as finalize_exc:  # noqa: BLE001 - see comment above
                logger.error(
                    "%s/%s: finalize during cleanup also failed: %s",
                    embodiment, task, finalize_exc,
                )
        _write_partial_result(embodiment, task, local_dir, written, skipped, reasons)
        _quarantine(local_dir)
        if dataset is not None:
            # finalize() sets its own `_finalized` flag -- on the dataset, its
            # writer, and the writer's metadata, three separate flags -- only
            # once *every* step inside it completes without raising. A
            # failure partway through (the one just above, suppressed, or an
            # earlier one during the episode loop that this exception
            # happened after) can leave one or more of them `False`. Left
            # `False`, `DatasetWriter.__del__`'s garbage-collection safety net
            # retries finalize() whenever this object is eventually collected
            # -- at a time this function does not control -- and by then
            # `_quarantine` above has already renamed this directory away, so
            # a retry reproduces exactly the recreate-the-directory bug the
            # finalize() call above exists to prevent. Forcing every flag
            # `True` here, regardless of whether finalize() actually
            # completed, is what makes `__del__` a guaranteed no-op from this
            # point on, whenever it eventually runs -- a stale flag is a far
            # smaller risk than a resurrected directory: nothing reads these
            # three private flags again once this task has failed and been
            # quarantined.
            dataset._is_finalized = True
            dataset.writer._finalized = True
            dataset.writer._meta._finalized = True
        raise


# Ray's `memory` option is admission control -- it changes how many tasks Ray
# will schedule onto a node at once -- not an enforced cap, but sizing it too
# low defeats the point, so this errs generous rather than tight.
#
# `read_episode` holds every camera's fully decoded frames in one dict before
# it ever builds the frame list, so the peak scales with frames x cameras x
# resolution -- all three, not just camera count. An earlier version of this
# reservation priced only a "per camera" constant fitted to one specific
# episode's own frame count (2,653), which silently implied every task runs
# about that many frames: fine for the episode it was measured on, wrong by
# roughly however far a task's real frame count differs from it -- for the
# six-camera, 720x1280 robot in this release, whose own sampled episode ran
# only 310 frames, that constant implied a ceiling far short of what its own
# longer episodes actually need (the I-C finding).
#
# Recomputed here on a per-frame, per-camera basis instead, from the same two
# measured episodes, so it scales with whatever frame count a task's own
# pre-pass (`reader.task_max_frames`) actually finds: 2,653 frames x 3 cameras
# x 480x640 colour is 6.83 GiB, 0.878 MiB/frame/camera; 310 frames x 6 cameras
# x 720x1280 colour is 4.79 GiB, 2.638 MiB/frame/camera -- the larger figure
# despite fewer frames, because each one is over 4x the pixels (720x1280 vs
# 480x640). The larger of the two per-frame figures is what every camera-
# frame is priced at below (rounded up a little further for margin: 2.7
# without depth). Depth adds 0.489 MiB/frame/camera on the first episode and
# 1.756 MiB/frame/camera on the second (measured the same way, from the same
# two episodes' own +depth totals) -- again the second, larger-resolution
# episode is worse, so combined colour+depth is priced at 4.5 MiB/frame/camera.
_WORST_MEASURED_MIB_PER_CAMERA_FRAME = 2.7             # colour only (720x1280 episode, 2.638 measured)
_WORST_MEASURED_MIB_PER_CAMERA_FRAME_WITH_DEPTH = 4.5  # same episode, colour + depth, 4.394 measured
_MEMORY_SAFETY_FACTOR = 1.5  # decode_color's transient extra copy, plus slop


def _task_memory_bytes(config: EmbodimentConfig, save_depth: bool, max_frames: int) -> int:
    """A per-task Ray memory reservation, in bytes. See the comment above.

    ``max_frames`` is the task's own largest episode frame count
    (``reader.task_max_frames``) -- every camera-frame this task could ever
    actually decode is priced, rather than a fixed, episode-specific frame
    count baked into the per-camera constant itself.
    """
    per_frame_mib = (
        _WORST_MEASURED_MIB_PER_CAMERA_FRAME_WITH_DEPTH if save_depth
        else _WORST_MEASURED_MIB_PER_CAMERA_FRAME
    )
    per_frame_bytes = per_frame_mib * (1024**2)
    return int(max_frames * len(config.cameras) * per_frame_bytes * _MEMORY_SAFETY_FACTOR)


def _max_node_memory(nodes: list[dict]) -> int:
    """The largest ``memory`` resource (bytes) any one live node reports, or 0
    if that can't be determined from ``nodes`` (typically ``ray.nodes()``).

    A task's ``memory`` reservation has to fit on *one* node -- Ray does not
    split a single task across several -- so a reservation is compared
    against the largest single node, not the cluster's summed total: a
    cluster of several smaller nodes can sum to more memory than one task
    needs while still having no single node that actually fits it. Taking
    plain dicts rather than calling ``ray.nodes()`` itself keeps this
    trivially testable without a real or mocked Ray cluster.
    """
    return max(
        (node.get("Resources", {}).get("memory", 0) for node in nodes if node.get("Alive")),
        default=0,
    )


def _summarize(
    output_path: Path,
    results: dict[tuple[str, str], TaskResult],
    failures: dict[tuple[str, str], str],
) -> dict:
    """Aggregate every task's outcome into a per-embodiment, per-task tally:
    written and skipped counts, skip reasons, and which tasks failed outright.

    With 4,502 known-broken files in the corpus and no persisted record beside
    the output before this, an operator had no way to reconcile what landed
    against what was expected. This is that record.

    A failed task's own written/skipped/reasons are read back from the
    marker file it left in its own (now quarantined) directory
    (``_read_partial_result``) rather than reported as zero: the bytes
    ``_quarantine`` preserves on disk are real, and a record that calls them
    zero because the task did not finish cleanly is not honest about what is
    actually sitting there (the I-D finding).

    Per-task counts are nested under each embodiment (``by_embodiment.
    <embodiment>.tasks.<task>``) rather than only rolled up: an embodiment-
    level total alone hides a single task that lost almost all of its
    episodes inside an otherwise-healthy embodiment's sum (the I-B finding).
    A task whose skipped count exceeds what it wrote is also logged as a
    warning here -- not failed outright: a skip is already a deliberately
    recoverable, reported outcome (that is the whole point of ``TaskResult``
    over letting one bad episode fail a task), and majority-shape-basing
    (see ``convert_task``) already fixes the single worst known cause of a
    task losing most of its episodes. A residual case after that is a real
    signal worth an operator's attention, not by itself evidence of a bug
    that should quarantine otherwise-good, already-written episodes.
    """
    by_embodiment: dict[str, dict] = {}

    def bucket(embodiment: str) -> dict:
        return by_embodiment.setdefault(
            embodiment, {"written": 0, "skipped": 0, "reasons": {}, "failed_tasks": [], "tasks": {}}
        )

    def record_task(embodiment: str, task: str, result: TaskResult) -> None:
        entry = bucket(embodiment)
        entry["written"] += result.written
        entry["skipped"] += result.skipped
        for reason, count in result.reasons.items():
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + count
        entry["tasks"][task] = {
            "written": result.written,
            "skipped": result.skipped,
            "reasons": dict(result.reasons),
        }
        if result.skipped > result.written:
            logger.warning(
                "%s/%s: skipped %d episode(s) but only wrote %d -- most of this "
                "task's episodes did not survive; see its reasons in summary.json",
                embodiment, task, result.skipped, result.written,
            )

    for (embodiment, task), result in results.items():
        record_task(embodiment, task, result)

    for (embodiment, task), error in failures.items():
        record_task(embodiment, task, _read_partial_result(output_path, embodiment, task))
        bucket(embodiment)["failed_tasks"].append({"task": task, "error": error})

    return {
        "written": sum(entry["written"] for entry in by_embodiment.values()),
        "skipped": sum(entry["skipped"] for entry in by_embodiment.values()),
        "tasks_found": len(results) + len(failures),
        "tasks_failed": len(failures),
        "by_embodiment": by_embodiment,
    }


def _write_summary(path: Path, summary: dict) -> None:
    """Persist ``_summarize``'s tally at ``path``, so a run's outcome survives
    past its own log lines -- the log itself is not kept anywhere once a
    run's terminal is gone, and Ray never brings a worker's per-episode log
    records back to the driver's console in the first place (see the module
    logger's docstring).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def _checkpoint_summary(
    output_path: Path,
    run_summary_path: Path,
    results: dict[tuple[str, str], TaskResult],
    failures: dict[tuple[str, str], str],
) -> dict:
    """Recompute and persist the run's tally so far -- to the stable
    ``summary.json`` (always the most recent run's tally, for whatever
    already expects that fixed name) and to this run's own timestamped copy
    at ``run_summary_path`` (so a later run into the same ``--output-path``
    does not destroy this one's -- see ``main``) -- and return it, so the
    caller can reuse the final call's return value for its own per-embodiment
    log lines once every task has resolved.

    Called after *every* task resolves, not only once after the last one:
    writing only at the very end means a run killed hours into a 114 TB
    conversion leaves no record at all of the tasks that had already finished
    (the I-D finding). Each call recomputes the whole tally from ``results``/
    ``failures`` rather than patching the previous write, which is simple to
    reason about and cheap enough at this scale: both dicts hold at most one
    entry per task -- a few hundred, for the full release -- not per episode.

    Always returns the computed tally even if writing either copy fails (a
    genuinely full output disk, proven live, is exactly the condition under
    which this is most likely to happen): a checkpoint that cannot currently
    be persisted must not crash the whole run out from under whatever tasks
    are still in flight, or replace the real reason a task failed with an
    unrelated "could not write summary.json" one.
    """
    summary = _summarize(output_path, results, failures)
    for path in (output_path / "summary.json", run_summary_path):
        try:
            _write_summary(path, summary)
        except OSError as exc:
            logger.error("could not write %s: %s", path, exc)
    return summary


def _describe_failure(exc: BaseException) -> str:
    """A short, readable description of a task's fatal exception.

    Under Ray, ``str(exc)`` on a remote task's exception is the *entire*
    worker traceback -- ANSI colour escapes included -- with the actual
    exception type and message only on its own last line (every one of them
    leads with something like ``ray::convert_task() (pid=..., ip=...)``
    first). Keeping only that last line, alongside the raised (possibly
    Ray-wrapped) exception's own type name, is what actually says what went
    wrong and is short enough to live in ``failures``/``summary.json``; the
    full traceback already reached the worker's own stderr once. A
    ``--debug`` run's exception is not Ray-wrapped and is usually one line
    already, so this is close to a no-op for it.

    Using this for both the log line and ``failures`` (rather than the
    exception object itself, or its full ``str()``) also means nothing here
    embeds the exception *object* in a log record's ``args`` -- which, kept
    alive by a log-capturing handler, is what used to pin a failed task's
    entire traceback (and everything it references, including the dataset
    object that failed) alive for longer than this function call, regardless
    of whether the object holding it was actually still needed. See
    ``test_a_fatal_write_failure_fails_the_task_and_quarantines_its_output``
    in ``tests/test_cli.py`` for exactly where that mattered.
    """
    text = str(exc).strip()
    last_line = text.splitlines()[-1] if text else ""
    return f"{type(exc).__name__}: {last_line}" if last_line else type(exc).__name__


def main(
    src_paths: list[Path],
    output_path: Path,
    save_depth: bool = False,
    min_frames: int = DEFAULT_MIN_FRAMES,
    cpus_per_task: int = 2,
    debug: bool = False,
) -> Path:
    if not debug:
        # Imported before plan_tasks walks the source tree -- which, at the
        # full release's size, means enumerating 269,569 objects -- rather
        # than after: a node bootstrapped without ray installed then fails
        # immediately and legibly instead of after minutes of directory
        # discovery it was always going to throw away.
        import ray
        from ray.runtime_env import RuntimeEnv

    grouped = plan_tasks(src_paths)
    if not grouped:
        raise NothingConverted(
            f"no episodes under {', '.join(str(path) for path in src_paths)}: "
            "expected data/<embodiment>/<task>/success_episodes/<stamp>/data/*.hdf5"
        )

    known = set(available())
    discovered = sorted({embodiment for embodiment, _ in grouped})
    unknown = sorted(set(discovered) - known)
    if unknown:
        raise NothingConverted(
            f"no config for {', '.join(unknown)}. available: {', '.join(sorted(known))}"
        )

    # Every discovered embodiment's config is loaded here, in the driver,
    # before any work is handed off -- a malformed YAML then fails the whole
    # run immediately with a message naming the embodiment and the reason,
    # instead of raising inside whichever worker happens to draw that
    # embodiment's first task, which under Ray would cost only that one task
    # (see TasksFailed) while every other embodiment quietly kept going.
    configs: dict[str, EmbodimentConfig] = {}
    bad_configs = []
    for embodiment in discovered:
        try:
            configs[embodiment] = load_config(embodiment)
        except ConfigError as exc:
            bad_configs.append(f"{embodiment}: {exc}")
    if bad_configs:
        raise NothingConverted("invalid config for " + "; ".join(bad_configs))

    results: dict[tuple[str, str], TaskResult] = {}
    failures: dict[tuple[str, str], str] = {}

    # Timestamped so a later run into the same --output-path accumulates
    # rather than clobbers: summary.json alone used to be the *only* record,
    # so consecutive per-robot runs each destroyed the previous run's tally
    # (the I-D finding). summary.json itself is still written too, on every
    # checkpoint below, and always reflects whichever run wrote it most
    # recently -- for anything that wants "the latest tally" under a fixed
    # name -- while this run's own copy is never overwritten by a later one.
    # Microsecond resolution, not just seconds: two runs (or, in a test, two
    # calls) can start well under a second apart, and a same-second collision
    # would silently make the second run overwrite the first's timestamped
    # copy -- exactly the clobbering this exists to prevent.
    run_summary_path = output_path / f"summary-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}.json"

    if debug:
        for (embodiment, task), refs in sorted(grouped.items()):
            try:
                results[(embodiment, task)] = convert_task(
                    refs, embodiment, task, output_path, save_depth, min_frames
                )
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                summary_text = _describe_failure(exc)
                logger.error("failed %s/%s: %s", embodiment, task, summary_text)
                failures[(embodiment, task)] = summary_text
            summary = _checkpoint_summary(output_path, run_summary_path, results, failures)
    else:
        ray.init(
            runtime_env=RuntimeEnv(
                env_vars={
                    "HDF5_USE_FILE_LOCKING": "FALSE",
                    "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                }
            )
        )
        # Advisory only: a task whose reservation exceeds every node's own
        # memory can never be scheduled anywhere, and Ray's response to that
        # is not to fail the task but to leave it pending forever (the I-C
        # finding) -- silent unless someone happens to go looking at the
        # dashboard. Warning here at least puts it in the log up front,
        # before a run spends hours waiting on a task that was never going to
        # start.
        max_node_memory = _max_node_memory(ray.nodes())

        remote = ray.remote(convert_task)
        futures = {}
        for (embodiment, task), refs in sorted(grouped.items()):
            reservation = _task_memory_bytes(
                configs[embodiment], save_depth, task_max_frames(refs)
            )
            if max_node_memory and reservation > max_node_memory:
                logger.warning(
                    "%s/%s: %.1f GiB memory reservation exceeds every node's own "
                    "memory (largest: %.1f GiB) -- Ray will never schedule this "
                    "task; it will sit pending rather than fail",
                    embodiment, task, reservation / 1024**3, max_node_memory / 1024**3,
                )
            futures[(embodiment, task)] = remote.options(
                num_cpus=cpus_per_task, memory=reservation,
            ).remote(refs, embodiment, task, output_path, save_depth, min_frames)

        for (embodiment, task), future in futures.items():
            try:
                results[(embodiment, task)] = ray.get(future)
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                summary_text = _describe_failure(exc)
                logger.error("failed %s/%s: %s", embodiment, task, summary_text)
                failures[(embodiment, task)] = summary_text
            summary = _checkpoint_summary(output_path, run_summary_path, results, failures)

    total_written = sum(result.written for result in results.values())
    total_skipped = sum(result.skipped for result in results.values())

    for embodiment, tally in sorted(summary["by_embodiment"].items()):
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(tally["reasons"].items())
        )
        failed_note = (
            f", {len(tally['failed_tasks'])} task(s) failed" if tally["failed_tasks"] else ""
        )
        logger.info(
            "%s: wrote %d, skipped %d%s%s",
            embodiment, tally["written"], tally["skipped"],
            f" ({reasons})" if reasons else "", failed_note,
        )

    if failures:
        detail = "; ".join(
            f"{embodiment}/{task}: {error}"
            for (embodiment, task), error in sorted(failures.items())
        )
        raise TasksFailed(f"{len(failures)} of {len(grouped)} task(s) failed outright: {detail}")

    if total_written == 0:
        raise NothingConverted(
            f"{len(grouped)} task(s) found but no episode was written. Every episode "
            "was skipped -- check the log for the reasons rather than treating this "
            "as an empty dataset."
        )

    logger.info(
        "wrote %d episode(s), skipped %d, across %d task(s)",
        total_written, total_skipped, len(grouped),
    )
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-paths",
        type=Path,
        nargs="+",
        required=True,
        help="one or more repo roots; each holds a data/<embodiment>/ tree",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--debug", action="store_true", help="run serially, no Ray")
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Configured here, not at import time: this is the one place this file
    # acts as a script rather than a library, so this is its one chance to
    # decide how its own records reach the console. A handler on this
    # module's own logger, not on `logging.root`, so nothing belonging to
    # another package -- e.g. whatever attaches a handler to the root logger
    # as an import side effect (see the comment above `logger`'s definition)
    # -- is touched, replaced, or silenced.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    args = parse_args()
    try:
        print(main(**vars(args)))
    except (NothingConverted, ConfigError, TasksFailed) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error
