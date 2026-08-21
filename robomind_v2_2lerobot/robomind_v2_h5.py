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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robomind_v2_utils.configs import ConfigError, EmbodimentConfig, available
from robomind_v2_utils.configs import load as load_config
from robomind_v2_utils.lerobot_utils import RoboMINDv2Dataset, build_features
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import EpisodeRef, discover, read_episode, task_fps

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

    ``build_features`` fixes a task's video shapes once, from whichever episode
    happens to create the dataset; every later episode's own measured shape --
    already in hand, ``read_episode`` returns it as ``Episode.shapes`` -- is
    compared against that rather than trusted to match. Both 720x1280 and
    480x640 occur within a single release task, and the older converter had a
    workaround for exactly this case.
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
        # One dataset per task, so its rate has to be one number -- but that
        # number used to come from whichever episode happened to survive
        # first, with every later disagreement only warned about and written
        # onto that base anyway. This pre-pass (cheap: `camera_observations/
        # timestamp` is tiny, and nothing here decodes a camera frame) measures
        # every episode's own rate up front so the task is opened at their
        # median instead, and any one episode's own rate can be checked
        # against it below rather than trusted just for arriving first.
        #
        # The drift check just below compares against this unrounded median,
        # never against the integer `dataset_fps` a LeRobot dataset actually
        # opens at: `round()` alone can read as over 10% of "drift" against a
        # fractional true rate (10 frames / 3 s = 3.333 Hz rounds to 3, an
        # 11.1% gap from that integer with zero episodes actually disagreeing)
        # -- comparing episodes against each other's median, not against the
        # rounding that median happens to produce, is what this guards against.
        median_fps = task_fps(refs, config, min_frames=min_frames)
        dataset_fps = round(median_fps) if median_fps is not None else None

        dataset = None
        dataset_shapes: dict[str, tuple[int, ...]] = {}
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
            # `task_fps` already excludes a rate like this from the median it
            # computed (see its own docstring) for the same reason; this is
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
                # Every episode task_fps looked at failed to open, or failed
                # to yield a usable rate -- and this ref's own file is the
                # same file `read_episode` above just read successfully
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

            if abs(episode.fps - median_fps) / median_fps > _FPS_DRIFT_THRESHOLD:
                logger.warning(
                    "skipped %s: measured %.1f Hz drifts from %s/%s's %d Hz base "
                    "by more than %d%% -- not written onto the wrong time base",
                    ref.path, episode.fps, embodiment, task, dataset_fps,
                    int(_FPS_DRIFT_THRESHOLD * 100),
                )
                skipped += 1
                reasons[_RATE_DRIFT] += 1
                continue

            if dataset is None:
                dataset = RoboMINDv2Dataset.create(
                    repo_id=f"{embodiment}/{task}",
                    root=local_dir,
                    fps=dataset_fps,
                    robot_type=config.robot_type,
                    features=build_features(config, episode.shapes),
                )
                dataset_shapes = episode.shapes
            else:
                mismatch = _shape_mismatch(dataset_shapes, episode.shapes)
                if mismatch:
                    logger.warning("skipped %s: shape mismatch: %s", ref.path, mismatch)
                    skipped += 1
                    reasons[_RESOLUTION_MISMATCH] += 1
                    continue

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
            except OSError as exc:
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
        _quarantine(local_dir)
        raise


# Ray's `memory` option is admission control -- it changes how many tasks Ray
# will schedule onto a node at once -- not an enforced cap, but sizing it too
# low defeats the point, so this errs generous rather than tight.
#
# `read_episode` holds every camera's fully decoded frames in one dict before
# it ever builds the frame list, so the peak scales with frames x cameras x
# resolution. Measured directly (not assumed) on two real episodes: 2,653
# frames x 3 cameras x 480x640 colour is 6.83 GiB (2.28 GiB/camera), +3.80 GiB
# more with depth at 400x640 (3.54 GiB/camera combined). A 310-frame x
# 6-camera x 720x1280 episode is 4.79 GiB colour (0.80 GiB/camera), +3.19 GiB
# depth (1.33 GiB/camera combined) -- despite the larger frames, because it has
# far fewer of them. The first episode's per-camera figures are the worse of
# the two and are what every camera is priced at below, rounded up a little
# further for margin (2.3 and 3.6): resolution is measured per episode, never
# configured (see `build_features`), so a task's own peak can't be known
# without decoding a frame from it, which is what this avoids paying for every
# task before Ray even schedules it.
_WORST_MEASURED_GIB_PER_CAMERA = 2.3          # colour only (the 480x640 episode, 2.28 measured)
_WORST_MEASURED_GIB_PER_CAMERA_WITH_DEPTH = 3.6   # same episode, colour + depth, 3.54 measured
_MEMORY_SAFETY_FACTOR = 1.5  # decode_color's transient extra copy, plus slop


def _task_memory_bytes(config: EmbodimentConfig, save_depth: bool) -> int:
    """A per-task Ray memory reservation, in bytes. See the comment above."""
    per_camera_gib = (
        _WORST_MEASURED_GIB_PER_CAMERA_WITH_DEPTH if save_depth else _WORST_MEASURED_GIB_PER_CAMERA
    )
    per_camera_bytes = per_camera_gib * (1024**3)
    return int(len(config.cameras) * per_camera_bytes * _MEMORY_SAFETY_FACTOR)


def _summarize(
    results: dict[tuple[str, str], TaskResult],
    failures: dict[tuple[str, str], str],
) -> dict:
    """Aggregate every task's outcome into a per-embodiment tally: written and
    skipped counts, skip reasons, and which tasks failed outright.

    With 4,502 known-broken files in the corpus and no persisted record beside
    the output before this, an operator had no way to reconcile what landed
    against what was expected. This is that record.
    """
    by_embodiment: dict[str, dict] = {}

    def bucket(embodiment: str) -> dict:
        return by_embodiment.setdefault(
            embodiment, {"written": 0, "skipped": 0, "reasons": {}, "failed_tasks": []}
        )

    for (embodiment, _task), result in results.items():
        entry = bucket(embodiment)
        entry["written"] += result.written
        entry["skipped"] += result.skipped
        for reason, count in result.reasons.items():
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + count

    for (embodiment, task), error in failures.items():
        bucket(embodiment)["failed_tasks"].append({"task": task, "error": error})

    return {
        "written": sum(entry["written"] for entry in by_embodiment.values()),
        "skipped": sum(entry["skipped"] for entry in by_embodiment.values()),
        "tasks_found": len(results) + len(failures),
        "tasks_failed": len(failures),
        "by_embodiment": by_embodiment,
    }


def _write_summary(output_path: Path, summary: dict) -> None:
    """Persist ``_summarize``'s tally beside the output, so a run's outcome
    survives past its own log lines -- the log itself is not kept anywhere
    once a run's terminal is gone, and Ray never brings a worker's per-episode
    log records back to the driver's console in the first place (see the
    module logger's docstring).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


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

    if debug:
        for (embodiment, task), refs in sorted(grouped.items()):
            try:
                results[(embodiment, task)] = convert_task(
                    refs, embodiment, task, output_path, save_depth, min_frames
                )
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                logger.error("failed %s/%s: %s", embodiment, task, exc)
                failures[(embodiment, task)] = str(exc)
    else:
        ray.init(
            runtime_env=RuntimeEnv(
                env_vars={
                    "HDF5_USE_FILE_LOCKING": "FALSE",
                    "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                }
            )
        )
        remote = ray.remote(convert_task)
        futures = {
            (embodiment, task): remote.options(
                num_cpus=cpus_per_task,
                memory=_task_memory_bytes(configs[embodiment], save_depth),
            ).remote(refs, embodiment, task, output_path, save_depth, min_frames)
            for (embodiment, task), refs in sorted(grouped.items())
        }
        for (embodiment, task), future in futures.items():
            try:
                results[(embodiment, task)] = ray.get(future)
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                logger.error("failed %s/%s: %s", embodiment, task, exc)
                failures[(embodiment, task)] = str(exc)

    total_written = sum(result.written for result in results.values())
    total_skipped = sum(result.skipped for result in results.values())

    summary = _summarize(results, failures)
    _write_summary(output_path, summary)
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
