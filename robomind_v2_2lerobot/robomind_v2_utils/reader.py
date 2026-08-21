"""RoboMIND 2.0 의 원본 트리와 HDF5 를 읽는다.

이 파일에는 embodiment 이름이 없다. 무엇을 읽을지는 전부 ``EmbodimentConfig`` 가
말하고, 어떤 config 를 쓸지는 경로가 말한다:

    data/<embodiment>/<task>/success_episodes/<timestamp>/data/<name>.hdf5

``<name>`` 은 real 이면 ``trajectory.hdf5``, sim 이면 ``<timestamp>.hdf5`` 다.
"""

import re
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


def check_usable(handle, config: EmbodimentConfig) -> None:
    """Raise ``EpisodeSkipped`` unless every dataset the config names is present.

    Three kinds of broken file exist upstream. 4,500 files in one repo of the
    release are a valid hdf5 with nothing in it -- recognisable by size alone,
    and caught before the config's key list is even built. The other two both
    need that key list checked against the file, and are told apart by how many
    keys survive: two more files were truncated mid-write and hold only their
    first stream, so a few of the config's keys are present and the rest are
    missing; a config pointed at the wrong embodiment names keys the file never
    had, so none of them are present at all.
    """
    if not handle.keys():
        raise EpisodeSkipped("no objects in the file")

    own_keys: list[str] = []
    for camera in config.cameras:
        own_keys.append(f"camera_observations/color_images/{camera.name}")
        if camera.depth:
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


def read_streams(handle, config) -> dict[str, "np.ndarray"]:
    """The per-frame vectors this config names, as ``(T, width)`` arrays.

    Only ``_align`` is read. Simulated episodes carry a ``_raw`` copy of every
    stream at roughly twice the rate, sampled on its own clock; ``_align`` is the
    one lined up with the cameras.

    Widths are checked rather than trusted. Two embodiments in this release name
    their streams identically and differ only in that ``end_effector_*_position``
    is one column wide on one and twelve on the other, so reading by name alone
    would silently reinterpret a dexterous hand as a gripper.
    """
    check_usable(handle, config)

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


def episode_fps(handle, config) -> float:
    """This episode's frame rate.

    Measured, not configured: the rate runs from about 7 Hz to about 101 Hz
    across the release's robots, and moves between episodes of one embodiment.
    v1 wrote 30 for everything.

    ``camera_observations/timestamp`` is whole seconds, so a short episode carries
    a boundary error of up to one second at each end. The value is good enough to
    put a dataset on the right time base and is not a precise measurement.

    A simulated episode's timestamps never advance, so its rate comes from the
    config instead -- the one number a config is allowed to state.
    """
    if "camera_observations/timestamp" not in handle:
        raise EpisodeSkipped("camera_observations/timestamp is missing")
    stamps = np.asarray(handle["camera_observations/timestamp"][()], dtype=np.int64)
    if stamps.size < MIN_FRAMES:
        raise EpisodeSkipped(f"too few frames: {stamps.size}")

    span = int(stamps[-1] - stamps[0])
    if span > 0:
        return float(stamps.size) / span
    if config.fps is not None:
        return float(config.fps)
    raise EpisodeSkipped(
        "timestamps do not advance and the config states no fps, so the frame "
        "rate is unknown"
    )


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
    task: str
    # LeRobot needs each video key's shape up front, and it is measured per
    # episode rather than configured, so it travels with the episode.
    shapes: dict[str, tuple[int, ...]]


def read_episode(ref: EpisodeRef, config, *, save_depth: bool = False) -> Episode:
    """One episode as the per-frame dicts LeRobot's ``add_frame`` takes.

    Raises ``EpisodeSkipped`` for anything wrong with this one file -- a broken
    write, a width that contradicts the config, a clock that never advances -- so
    that a run drops the episode and keeps going.
    """
    import h5py

    with h5py.File(ref.path, "r") as handle:
        columns = read_streams(handle, config)
        fps = episode_fps(handle, config)
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
                images[depth_key] = decode_depth(
                    handle[f"camera_observations/depth_images/{camera.name}"]
                )
                height, width, _ = shapes[key]
                shapes[depth_key] = (height, width, 1)

    count = len(next(iter(columns.values())))
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
    return Episode(ref=ref, frames=frames, fps=fps, task=task, shapes=shapes)
