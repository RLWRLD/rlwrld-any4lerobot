"""RoboMIND 2.0 의 원본 트리와 HDF5 를 읽는다.

이 파일에는 embodiment 이름이 없다. 무엇을 읽을지는 전부 ``EmbodimentConfig`` 가
말하고, 어떤 config 를 쓸지는 경로가 말한다:

    data/<embodiment>/<task>/success_episodes/<timestamp>/data/<name>.hdf5

``<name>`` 은 real 이면 ``trajectory.hdf5``, sim 이면 ``<timestamp>.hdf5`` 다.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .configs import EmbodimentConfig
from .errors import EpisodeSkipped
from .images import decode_color, decode_depth, frame_shape


@dataclass(frozen=True)
class EpisodeRef:
    embodiment: str
    task: str
    path: Path


def discover(src_paths: list[Path]) -> Iterator[EpisodeRef]:
    """Every episode under the given source roots, in a stable order.

    Several roots can share one embodiment: in this release, one robot ships
    as five separate repos that all start at the same ``data/<embodiment>``
    path, so they carry one config between them.
    """
    for src_path in sorted(Path(root).expanduser().resolve() for root in src_paths):
        data = src_path / "data"
        if not data.is_dir():
            continue
        for embodiment_dir in sorted(path for path in data.iterdir() if path.is_dir()):
            for task_dir in sorted(
                path for path in embodiment_dir.iterdir() if path.is_dir()
            ):
                episodes = task_dir / "success_episodes"
                if not episodes.is_dir():
                    continue
                for hdf5 in sorted(episodes.glob("*/data/*.hdf5")):
                    yield EpisodeRef(
                        embodiment=embodiment_dir.name,
                        task=task_dir.name,
                        path=hdf5,
                    )


def _has_any_dataset(handle) -> bool:
    """Whether the file holds a dataset anywhere in it, at any depth.

    A broken file can carry the full group skeleton -- ``camera_model``,
    ``master/<stream>_align``, and the rest -- and still hold zero datasets if
    the write that would have populated them never happened. ``handle.keys()``
    only sees the top-level group names and stays non-empty either way, so
    telling a truly empty file from a skeleton-only one needs a walk of the
    whole tree, not a look at the top level.
    """
    import h5py

    return (
        handle.visititems(
            lambda _, obj: True if isinstance(obj, h5py.Dataset) else None
        )
        is not None
    )


def check_usable(handle, config: EmbodimentConfig, *, save_depth: bool = False) -> None:
    """Raise ``EpisodeSkipped`` unless every dataset the config names is present.

    Three kinds of broken file exist upstream. 4,500 files in one repo of the
    release are recognisable by size alone, at exactly 6,144 bytes: a valid
    hdf5 that carries the group skeleton -- six top-level groups such as
    ``camera_model`` and ``master`` -- but not one dataset anywhere inside it,
    at any depth. That is checked first and needs no config, since there is
    nothing yet to compare against. The other two both need the config's key
    list checked against the file, and are told apart by how many keys survive:
    two more files were truncated mid-write and hold only their first stream,
    so a few of the config's keys are present and the rest are missing; a
    config pointed at the wrong embodiment names keys the file never had, so
    none of them are present at all.

    A camera's ``depth: true`` only makes its depth dataset *required* when
    ``save_depth`` is also true. All twelve configs in this release set
    ``depth: true`` on every camera, but ``--save-depth`` defaults off and no
    real conversion has passed it yet -- gating on the config field alone,
    regardless of ``save_depth``, would fail every episode of every robot the
    moment a run actually left depth off, for a feature nothing was asked to
    write.
    """
    if not _has_any_dataset(handle):
        raise EpisodeSkipped("no datasets in the file")

    own_keys: list[str] = []
    for camera in config.cameras:
        own_keys.append(f"camera_observations/color_images/{camera.name}")
        if camera.depth and save_depth:
            own_keys.append(f"camera_observations/depth_images/{camera.name}")
    for stream in config.streams:
        own_keys.append(f"puppet/{stream.name}_align/data")
        own_keys.append(f"master/{stream.name}_align/data")
    for extra in config.extras:
        own_keys.append(f"{extra.group}/{extra.name}_align/data")
    required = ["camera_observations/timestamp", *own_keys]

    missing = [key for key in required if key not in handle]
    if not missing:
        return

    matched = len(required) - len(missing)
    sample = ", ".join(missing[:4])
    if len(missing) > 4:
        sample += f", and {len(missing) - 4} more"

    # camera_observations/timestamp is the one key every layout carries regardless
    # of embodiment, so it says nothing about whether the config matches this
    # file. Ignore it here and ask whether any of the config's *own* keys --
    # the ones its cameras/streams/extras actually name -- showed up at all.
    if any(key in handle for key in own_keys):
        raise EpisodeSkipped(
            f"missing {len(missing)} of {len(required)} dataset(s), {matched} "
            f"present -- looks like a damaged or partial file: {sample}"
        )
    raise EpisodeSkipped(
        f"missing {len(missing)} of {len(required)} dataset(s), {matched} "
        f"present -- looks like the config does not match this file's embodiment: {sample}"
    )


def _stream_data(handle, path: str, width: int) -> "np.ndarray":
    values = np.asarray(handle[f"{path}/data"][()], dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise EpisodeSkipped(f"{path} is {values.shape}, expected 1-D or 2-D")
    if values.shape[1] != width:
        raise EpisodeSkipped(
            f"{path.rsplit('/', 1)[-1].removesuffix('_align')} is "
            f"{values.shape[1]} wide, expected {width}"
        )
    return values


def read_streams(handle, config, *, save_depth: bool = False) -> dict[str, "np.ndarray"]:
    """The per-frame vectors this config names, as ``(T, width)`` arrays.

    Only ``_align`` is read. Simulated episodes carry a ``_raw`` copy of every
    stream at roughly twice the rate, sampled on its own clock; ``_align`` is the
    one lined up with the cameras.

    Widths are checked rather than trusted. Two embodiments in this release name
    their streams identically and differ only in that ``end_effector_*_position``
    is one column wide on one and twelve on the other, so reading by name alone
    would silently reinterpret a dexterous hand as a gripper.

    ``save_depth`` only affects which datasets ``check_usable`` requires -- see
    its own docstring. It never gates anything read here: none of the vectors
    this function returns are camera data.
    """
    check_usable(handle, config, save_depth=save_depth)

    columns: dict[str, np.ndarray] = {}
    for stream in config.streams:
        columns[f"observation.states.{stream.name}"] = _stream_data(
            handle, f"puppet/{stream.name}_align", stream.width
        )
        columns[f"actions.{stream.name}"] = _stream_data(
            handle, f"master/{stream.name}_align", stream.width
        )

    for extra in config.extras:
        values = np.asarray(
            handle[f"{extra.group}/{extra.name}_align/data"][()], dtype=np.float32
        )
        if values.shape[1:] != extra.shape:
            raise EpisodeSkipped(
                f"{extra.name} is {values.shape[1:]}, expected {extra.shape}"
            )
        columns[f"observation.{extra.name}"] = values

    frames = len(handle["camera_observations/timestamp"])
    wrong = {key: value.shape[0] for key, value in columns.items() if value.shape[0] != frames}
    if wrong:
        raise EpisodeSkipped(
            f"length mismatch against {frames} camera frames: "
            + ", ".join(f"{key}={count}" for key, count in list(wrong.items())[:3])
        )
    return columns


MIN_FRAMES = 2


def episode_fps(handle, config) -> tuple[float, int | None]:
    """This episode's frame rate, and (for ``real`` episodes only) the span in
    whole seconds the rate was divided out of.

    The span is what a caller uses to size its own tolerance for how far this
    one episode's rate may drift from its task's median before that counts as
    a real disagreement rather than measurement noise (see
    ``robomind_v2_h5``'s ``_drift_tolerance``): a ``real`` rate is
    ``frame_count / span`` from a span rounded to whole seconds, so the span
    itself bounds how precisely that rate can be known -- a short span means
    a coarse rate, not a wrong one. ``sim``'s timestamps carry no comparable
    quantization (see below), so its span is reported as ``None`` -- there is
    nothing for a caller to size a tolerance from, and none is needed.

    Measured, not configured: the rate runs from about 7 Hz to about 101 Hz
    across the release's robots, and moves between episodes of one embodiment.
    v1 wrote 30 for everything. There is no config fallback for either layout --
    see ``configs.py`` for why the field was removed.

    ``camera_observations/timestamp`` means a different unit depending on
    ``config.layout``, confirmed by measuring real files rather than assumed:

    * ``real`` -- whole *seconds* (epoch-like values, stepping by 0 or 1 between
      consecutive frames). A short episode carries a boundary error of up to one
      second at each end, so the rate is computed from the episode's total span
      (``frames / (last - first)``): good enough to put a dataset on the right
      time base, not a precise measurement.
    * ``sim`` -- *milliseconds*, on a much finer clock (consecutive frames step
      by roughly 33-34, i.e. a ~30 Hz tick) -- not seconds, and not frozen. An
      earlier version of this function assumed a simulated clock never advances
      at all, and fell back to a config-stated rate whenever the span came out
      to 0; measuring real simulated-layout files showed that premise was
      simply wrong. Both layouts advance, every episode checked, just in
      different units -- there was never a case where the rate genuinely
      could not be measured, only a wrong assumption about which branch would
      fire.

    The two branches estimate different quantities, not just different units,
    so do not expect them to agree even on identical clean data: ``sim``
    estimates a frame *period* (the typical gap between one frame and the
    next) and inverts it into a rate, while ``real`` divides a frame *count*
    by a span. For T uniformly-spaced frames the two differ by a factor of
    T/(T-1) -- count-over-span implicitly treats T frames as T intervals,
    while a period estimate correctly divides by the T-1 gaps that actually
    exist between them.

    One simulated episode was found with a single mid-episode backward jump (one
    frame-to-frame step of roughly -2600, amid hundreds of otherwise constant
    +33/+34 steps) -- the clock resets partway through, rather than merely
    failing to advance. Using the endpoint span (``stamps[-1] - stamps[0]``) for
    ``sim`` would silently bake that one glitch into the total and understate
    the elapsed time by however much the clock fell (measured: it turns a ~30 Hz
    episode into an apparent ~35 Hz one). Because the per-frame step is
    otherwise extremely uniform, ``sim`` instead averages the *positive*
    consecutive steps -- discarding the one backward step rather than letting
    it drag the total down -- and the rate is the reciprocal of that mean.

    Mean, not median: the timestamps are whole milliseconds, so a true
    33.333 ms period (an exact 30.0000 Hz tick) still has to be stored as a 33
    or a 34 every single frame, never the fractional true value. Whichever of
    the two is the majority dominates the median outright -- 33, in every
    simulated episode sampled -- reading a systematically fast 30.3030 Hz, a
    +1% bias with no episode-to-episode variation to average away. The mean of
    those same whole-millisecond steps is under no such constraint and
    converges on the true 33.333 instead, recovering 30.0000 Hz. Measured on
    the release's four simulated sample episodes, the mean reads within 0.005%
    of 30 Hz on all four, including the one with the backward jump; the naive
    endpoint span is off by about 18% on that one.

    ``real`` keeps the endpoint span rather than a mean or median of steps:
    its timestamps are whole seconds, so most consecutive steps are 0 (several
    frames land in the same second) and only a few are 1 -- an average of
    mostly zeroes would read most real episodes as frozen. The coarse
    total-span count is the only usable signal there, and no backward jump was
    ever observed on a real episode, so there is nothing for ``real`` to be
    robust against.

    A ``sim`` episode whose clock is frozen or runs only backward has no
    positive step to average -- raised as unknown rather than computed from an
    empty mean, which would otherwise be a silent ``nan`` (or a
    ``ZeroDivisionError`` on its reciprocal).
    """
    if "camera_observations/timestamp" not in handle:
        raise EpisodeSkipped("camera_observations/timestamp is missing")
    stamps = np.asarray(handle["camera_observations/timestamp"][()], dtype=np.int64)
    if stamps.size < MIN_FRAMES:
        raise EpisodeSkipped(f"too few frames: {stamps.size}")

    if config.layout == "sim":
        steps = np.diff(stamps)
        positive_steps = steps[steps > 0]
        if positive_steps.size > 0:
            step_ms = float(np.mean(positive_steps))
            return 1000.0 / step_ms, None
        raise EpisodeSkipped(
            "timestamps do not advance, so the frame rate is unknown"
        )

    span = int(stamps[-1] - stamps[0])
    if span > 0:
        return float(stamps.size) / span, span
    raise EpisodeSkipped(
        "timestamps do not advance, so the frame rate is unknown"
    )


# A rate that rounds below this cannot be a dataset's fps -- see task_profile
# below, and convert_task's own drift check, which both hold single episodes to
# the same floor.
_MIN_TASK_FPS = 1


@dataclass(frozen=True)
class TaskProfile:
    """A task's own basics, measured once up front from every episode's file
    rather than assumed from whichever one happens to be looked at first (see
    ``convert_task``, and the I1/I-B findings this closes together).

    ``fps`` is the median of every episode's own measured rate. ``shapes`` is
    the per-camera colour shape (``observation.images.<camera> -> (H, W, 3)``)
    that the largest number of episodes actually agree on -- a task's dataset
    is created at this shape, not at whichever episode happens to be opened
    first, so one odd-resolution episode arriving first in discovery order
    can no longer make every other episode the "wrong" one. Either field is
    ``None`` when not one episode produced a usable measurement of it.
    """

    fps: float | None
    shapes: dict[str, tuple[int, ...]] | None


def task_profile(refs: list[EpisodeRef], config, min_frames: int = 0) -> TaskProfile:
    """The whole task's frame rate and camera shape, from one pass over every
    episode's file.

    **fps**: one LeRobot dataset is opened per ``(embodiment, task)`` at a
    single fps, but the rate moves between episodes of the same task -- real
    evidence from this release: one task's episodes measured 26.94 Hz and
    31.33 Hz, 16% apart. Taking it from whichever episode happens to survive
    first and write everything else onto that base, silently or with a mere
    warning, is what the median here replaces: the caller fixes the task at
    this median instead, and skips any episode whose own rate drifts too far
    from it (see ``convert_task``).

    **shapes**: built the same way, for the same reason -- both 720x1280 and
    480x640 occur within a single release task, ``build_features`` fixes a
    task's video shape once, and an odd-resolution episode arriving first in
    discovery order used to make *it* the dataset's shape and every
    correctly-sized episode the one that gets skipped (see the I-B finding).
    The most common exact per-camera shape combination across this task's
    episodes is returned, not a per-key majority computed independently --
    picking per-key would synthesize a shape combination no real episode
    actually has, if different episodes disagree on different cameras.

    Measuring a shape costs one JPEG decode per camera per episode
    (``images.frame_shape``, on the first frame only) -- not free like the fps
    half of this pre-pass, which reads only the tiny ``camera_observations/
    timestamp`` array, but still nowhere near ``read_episode``'s full
    per-frame decode of every frame of every camera, paid once per episode
    that actually gets written rather than once per episode in the task.

    ``min_frames`` should be the same floor the caller applies to a full
    episode (``convert_task``'s own ``min_frames``): an episode this short
    will be skipped there regardless of its rate or shape, and a
    broken-recording-length episode's own measurements are exactly the kind
    least likely to be trustworthy (``episode_fps``'s own docstring notes the
    boundary error a short ``real`` episode carries) -- so it is excluded
    from both the median and the shape vote here too, rather than letting a
    handful of them quietly pull either away from what the task's real
    episodes actually look like.

    An episode this function cannot open, or cannot measure a rate or a shape
    for, is simply left out of that measurement's own tally -- ``read_episode``
    reaches the same file again and logs the specific reason when the caller
    processes it for real. A measured rate that rounds below ``_MIN_TASK_FPS``
    is left out of the median too, the same floor a lone surviving episode was
    always held to: unusable as a dataset's fps on its own, so it must not get
    to drag the median toward zero either.
    """
    import h5py

    rates = []
    shape_votes: Counter = Counter()
    shape_for_signature: dict[tuple, dict[str, tuple[int, ...]]] = {}

    for ref in refs:
        try:
            with h5py.File(ref.path, "r") as handle:
                key = "camera_observations/timestamp"
                if key in handle and len(handle[key]) < min_frames:
                    continue

                try:
                    rate, _span = episode_fps(handle, config)
                except (OSError, ValueError, EpisodeSkipped):
                    pass
                else:
                    if round(rate) >= _MIN_TASK_FPS:
                        rates.append(rate)

                try:
                    shapes = {
                        f"observation.images.{camera.name}": frame_shape(handle, camera.name)
                        for camera in config.cameras
                    }
                except (OSError, ValueError, KeyError, EpisodeSkipped):
                    pass
                else:
                    signature = tuple(sorted(shapes.items()))
                    shape_votes[signature] += 1
                    shape_for_signature[signature] = shapes
        except (OSError, ValueError):
            continue

    fps = float(np.median(rates)) if rates else None
    shapes = shape_for_signature[shape_votes.most_common(1)[0][0]] if shape_votes else None
    return TaskProfile(fps=fps, shapes=shapes)


def task_max_frames(refs: list[EpisodeRef]) -> int:
    """The largest frame count among this task's episodes, or 0 if none of
    them could even be opened.

    Sizing a Ray memory reservation before a task is scheduled (see
    ``robomind_v2_h5``'s ``_task_memory_bytes``) needs to know, before any
    episode is actually read, how large the biggest one might be: peak memory
    during ``read_episode`` scales with an episode's own frame count, not
    just its camera count (see the I-C finding). This reads only
    ``camera_observations/timestamp``'s length for every episode -- the same
    tiny dataset ``task_profile`` already opens each file for, no frame
    decode -- so on its own it costs about what discovering the task's
    episodes already did, nowhere near what actually converting one does.

    Deliberately independent of ``task_profile``: that pre-pass runs inside
    each Ray worker, in parallel across however many tasks Ray schedules at
    once; this one has to run on the driver, before any task is scheduled at
    all, so it stays as cheap as possible rather than also paying for
    ``task_profile``'s per-camera shape decode serially for the whole corpus
    before a single worker starts.
    """
    import h5py

    max_frames = 0
    for ref in refs:
        try:
            with h5py.File(ref.path, "r") as handle:
                key = "camera_observations/timestamp"
                if key in handle:
                    max_frames = max(max_frames, len(handle[key]))
        except (OSError, ValueError):
            continue
    return max_frames


_SIM_TASK_PREFIX = re.compile(r"^\d+-")


def _from_dirname(task: str) -> str:
    """The task directory name as a sentence.

    The directory names are English even where the Chinese description file is the
    only stated instruction, which makes them a usable last resort. Simulated tasks
    carry a numeric prefix (``101-pick_cube_into_plate``) that is an id, not words.

    The prefix is stripped unconditionally, with no layout check, because that
    cannot misfire: across all 738 task directory names in this release, a numeric
    prefix appears only in the two simulated subsets and in none of the 679 real
    ones, so gating the strip on layout would add a parameter for a case that does
    not exist.

    Stripping the prefix, swapping underscores for spaces, and trimming whitespace
    can still leave nothing behind -- ``"42-"`` and ``"_"`` are both legal directory
    names that reduce to an empty string this way. This function backs the
    guarantee that ``instruction()`` never returns an empty prompt, so it must
    never return one either: when the transformation empties the result, the
    untouched directory name is returned instead.
    """
    sentence = _SIM_TASK_PREFIX.sub("", task).replace("_", " ").strip()
    return sentence or task


def instruction(ref: EpisodeRef, config, handle) -> str:
    """This episode's task instruction, from wherever the config says it lives.

    Three sources exist upstream and one embodiment has none of the first two:

    ==============  ============================================================
    ``zh_file``     ``<task>/zh_description.txt``, Chinese -- the ten real sets
    ``h5_metadata`` ``metadata/language_instruction``, an English sentence -- sim
    ``dirname``     nothing on disk; the directory name is the instruction
    ==============  ============================================================

    A named source that turns out to be missing falls back to the directory name
    rather than skipping the episode: one real task (521 episodes) has no
    description file, and dropping it would cost more than a coarser prompt.

    A named source that is present but unreadable gets the same fallback instead
    of raising: reading ``zh_file`` and decoding ``h5_metadata`` are each wrapped
    in a ``try`` that catches only ``UnicodeDecodeError`` and ``OSError`` -- never
    a bare ``except`` or ``Exception`` -- and falls through to the directory name
    below. A corrupt or mis-encoded source should cost this episode a coarser
    prompt, the same as a missing one, not the episode itself.

    ``zh_file`` is read as UTF-8, which is measured rather than assumed: one
    ``zh_description.txt`` was sampled from each of six repos of the release,
    six different robots, and all six decoded cleanly as UTF-8, while GBK and
    GB18030 -- the other Chinese encoding upstream could plausibly have used --
    failed to decode five of the six.
    """
    if config.instruction_source == "h5_metadata":
        key = "metadata/language_instruction"
        if key in handle:
            try:
                value = handle[key][()]
                text = (value.decode() if isinstance(value, bytes) else str(value)).strip()
            except (UnicodeDecodeError, OSError):
                text = ""
            if text:
                return text
    elif config.instruction_source == "zh_file":
        # <...>/<task>/success_episodes/<stamp>/data/<file>.hdf5 -> <task>
        path = ref.path.parents[3] / "zh_description.txt"
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, OSError):
                text = ""
            if text:
                return text

    return _from_dirname(ref.task)


@dataclass(frozen=True)
class Episode:
    ref: EpisodeRef
    frames: list[dict]
    fps: float
    # The span (whole seconds) this episode's own fps was divided out of --
    # `None` for a `sim` episode, which has no comparable span to report (see
    # `episode_fps`). `convert_task`'s drift check uses this to size how much
    # a `real` episode's own rate may disagree with its task's median before
    # that counts as real drift rather than this episode's own measurement
    # noise (the I-A finding).
    fps_span: int | None
    task: str
    # LeRobot needs each video key's shape up front, and it is measured per
    # episode rather than configured, so it travels with the episode.
    shapes: dict[str, tuple[int, ...]]


def read_episode(ref: EpisodeRef, config, *, save_depth: bool = False) -> Episode:
    """One episode as the per-frame dicts LeRobot's ``add_frame`` takes.

    Raises ``EpisodeSkipped`` for anything wrong with this one file -- a path
    that cannot be opened, a broken write, a width that contradicts the config,
    a clock that never advances -- so that a run drops the episode and keeps
    going.
    """
    import h5py

    try:
        handle = h5py.File(ref.path, "r")
    except (OSError, ValueError) as error:
        raise EpisodeSkipped(f"cannot open {ref.path}: {error}") from error

    with handle:
        columns = read_streams(handle, config, save_depth=save_depth)
        # Always present regardless of what a config's cameras/streams/extras
        # name, unlike picking an arbitrary entry out of ``columns`` -- which
        # would be a bare ``StopIteration`` instead of ``EpisodeSkipped`` for a
        # config with no streams. Also the same value read_streams already
        # checked every column's length against, so nothing else changes.
        count = len(handle["camera_observations/timestamp"])
        fps, fps_span = episode_fps(handle, config)
        task = instruction(ref, config, handle)

        images: dict[str, np.ndarray] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for camera in config.cameras:
            key = f"observation.images.{camera.name}"
            images[key] = decode_color(
                handle[f"camera_observations/color_images/{camera.name}"]
            )
            shapes[key] = frame_shape(handle, camera.name)
            if camera.depth and save_depth:
                depth_key = f"{key}_depth"
                depth = decode_depth(
                    handle[f"camera_observations/depth_images/{camera.name}"]
                )
                images[depth_key] = depth
                # Measured from the decoded array itself, not derived from
                # colour's shape: a stated resolution can silently disagree
                # with the pixels, and assuming depth's resolution equals
                # colour's is the same class of unverified assumption.
                shapes[depth_key] = depth.shape[1:]

    for key, value in images.items():
        if len(value) != count:
            raise EpisodeSkipped(f"{key} has {len(value)} frames, streams have {count}")

    frames = [
        {
            **{key: value[index] for key, value in images.items()},
            **{key: value[index] for key, value in columns.items()},
            "task": task,
        }
        for index in range(count)
    ]
    return Episode(ref=ref, frames=frames, fps=fps, fps_span=fps_span, task=task, shapes=shapes)
