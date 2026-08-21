"""RoboMIND 2.0 의 원본 트리와 HDF5 를 읽는다.

이 파일에는 embodiment 이름이 없다. 무엇을 읽을지는 전부 ``EmbodimentConfig`` 가
말하고, 어떤 config 를 쓸지는 경로가 말한다:

    data/<embodiment>/<task>/success_episodes/<timestamp>/data/<name>.hdf5

``<name>`` 은 real 이면 ``trajectory.hdf5``, sim 이면 ``<timestamp>.hdf5`` 다.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .configs import EmbodimentConfig
from .errors import EpisodeSkipped


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

    Three kinds of broken file exist upstream. 4,500 UR5 files are a valid hdf5
    with nothing in it -- recognisable by size alone, and caught before the
    config's key list is even built. The other two both need that key list
    checked against the file, and are told apart by how many keys survive: two
    more files were truncated mid-write and hold only their first stream, so a
    few of the config's keys are present and the rest are missing; a config
    pointed at the wrong embodiment names keys the file never had, so none of
    them are present at all.
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
