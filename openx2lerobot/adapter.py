"""openx2lerobot as a datatrove adapter, so one dataset can use a whole machine.

The upstream converter reads the RLDS stream and writes episodes one after another
in a single process. Its only parallelism is LeRobot's image writer, so a
conversion occupies about five cores no matter how many the machine has -- measured
at 245 frames/s on a 48-core instance, where the 19 million frames of the smaller
OXE sources would take a day.

RLDS splits cleanly: ``train[a:b]`` is a deterministic slice of the same stream, so
a chunk of episodes is a task, tasks run in parallel, and the temporary datasets are
concatenated in task order. That last part matters -- ``aggregate_tasks`` preserves
list order, so a sharded run writes the episodes in the same order a single-process
run does.

Nothing here opens tensorflow at import time. The builder is constructed inside the
worker that uses it, which keeps the adapter picklable across processes and lets the
task arithmetic be tested without the conversion environment installed.
"""

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from generic_converter import BaseAdapter, ConversionTask  # noqa: E402

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def read_raw_dir(raw_dir: Path) -> tuple[str, str, Path]:
    """``--raw-dir`` as tfds wants it: ``(dataset_name, version, data_dir)``.

    Both forms are accepted because both occur. A mirror is synced to
    ``<raw_root>/<name>/`` with the version directory still inside it, and a hand-
    staged copy is often pointed at directly.
    """
    last = raw_dir.name
    if _VERSION.match(last):
        return raw_dir.parent.name, last, raw_dir.parent.parent
    return last, "", raw_dir.parent


def file_format(version_dir: Path) -> str | None:
    """Which format a prepared tfds directory is written in, or ``None`` if unknown.

    tfds writes two, and offers a different API for each: ``tfrecord`` is read with
    ``as_dataset`` and ``array_record`` only with ``as_data_source``. Asking for the
    wrong one fails with an assertion telling you to call ``download_and_prepare``,
    which says nothing about the format and sends you looking for a missing file.

    Worth knowing before a run rather than during one: bc_z's mirror is array_record
    across 1024 shards, so finding out from the converter means finding out after
    56 GB has been fetched.

    A directory whose ``dataset_info.json`` predates the field reads as tfrecord,
    which is what it will be -- the field arrived after the format. A directory with
    no metadata at all reads as ``None`` rather than as a guess.
    """
    path = version_dir / "dataset_info.json"
    if not path.is_file():
        # a mirror is synced as <raw_root>/<name>/<version>/, and --raw-dir is often
        # the name rather than the version
        inner = [d for d in sorted(version_dir.glob("*/dataset_info.json"))]
        if not inner:
            return None
        path = inner[0]
    try:
        info = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return info.get("fileFormat") or "tfrecord"


def episode_chunks(total: int, per_task: int) -> list[tuple[int, int]]:
    """``(start, stop)`` for each task, covering ``total`` episodes in order."""
    if per_task < 1:
        raise ValueError(f"episodes_per_task must be at least 1, got {per_task}")
    return [(start, min(start + per_task, total)) for start in range(0, total, per_task)]


class OpenXAdapter(BaseAdapter):
    dataset_type = "openx"

    def __init__(
        self,
        raw_dir: Path,
        output_path: Path,
        *,
        episodes_per_task: int = 100,
        resize: Any = None,
        encoding: Any = None,
        channels: Any = None,
        fps: int | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        image_writer_process: int = 5,
        image_writer_threads: int = 10,
        max_episodes: int | None = None,
    ):
        super().__init__(output_path)
        self.raw_dir = Path(raw_dir)
        self.dataset_name, self.version, self.data_dir = read_raw_dir(self.raw_dir)
        self.episodes_per_task = episodes_per_task
        self.max_episodes = max_episodes
        self.resize = resize
        self.encoding = encoding
        self.channels = channels
        self.use_videos = use_videos
        self.image_writer_process = image_writer_process
        self.image_writer_threads = image_writer_threads
        self.tags = (self.dataset_name, "rlds")
        self._fps = fps
        self._robot_type = robot_type
        # filled in on first use, in whichever process asks; never pickled
        self._features: dict[str, Any] | None = None

    # --- what the pipeline reads off the adapter -----------------------------

    @property
    def features(self) -> dict[str, Any]:
        if self._features is None:
            from openx_rlds import generate_features_from_raw
            from video_rules import parse_rule

            self._features = generate_features_from_raw(
                self._builder(), self.use_videos, resize=parse_rule(self.resize)
            )
        return self._features

    @features.setter
    def features(self, value):  # BaseAdapter declares it as a plain attribute
        self._features = value

    @property
    def fps(self) -> int:
        if self._fps is None:
            self._fps = self._config().get("control_frequency", 10)
        return int(self._fps)

    @property
    def robot_type(self) -> str:
        if self._robot_type is None:
            raw = self._config().get("robot_type", "unknown")
            self._robot_type = raw.lower().replace(" ", "_").replace("-", "_")
        return self._robot_type

    # --- the work ------------------------------------------------------------

    def load_tasks(self) -> list[ConversionTask]:
        # before anything imports tensorflow: an array_record mirror cannot be read by
        # the path below, and finding that out from tfds means finding it out after the
        # source has been fetched
        written_as = file_format(self.raw_dir)
        if written_as == "array_record":
            raise NotImplementedError(
                f"{self.raw_dir} is written in array_record, which this converter does "
                "not read yet: it opens a builder with as_dataset, and tfds serves "
                "array_record only through as_data_source. Nothing else about the "
                "mirror is wrong."
            )

        total = self._count_episodes()
        if self.max_episodes is not None:
            total = min(total, self.max_episodes)
        if total < 1:
            raise ValueError(
                f"{self.raw_dir} has no episodes in its train split; nothing to convert"
            )

        tasks = []
        for index, (start, stop) in enumerate(
            episode_chunks(total, self.episodes_per_task)
        ):
            name = f"{self.dataset_name}_chunk_{index:06d}_{start}_{stop - 1}"
            tasks.append(
                ConversionTask(
                    input_path=self.raw_dir,
                    output_path=(self.temp_output_path / name).resolve(),
                    local_repo_id=name,
                    metadata={"split": f"train[{start}:{stop}]"},
                )
            )
        return tasks

    def load_subset(self, task: ConversionTask) -> Iterable[Any]:
        from functools import partial

        from openx_rlds import transform_raw_dataset

        # kuka is the one source that ships failures alongside successes
        keep = (
            (lambda episode: episode["success"])
            if self.dataset_name == "kuka"
            else (lambda episode: True)
        )
        dataset = (
            self._builder()
            .as_dataset(split=task.metadata["split"])
            .filter(keep)
            .map(partial(transform_raw_dataset, dataset_name=self.dataset_name))
        )
        yield from dataset.as_numpy_iterator()

    def create_dataset(self, task: ConversionTask):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from video_rules import rgb_encoder

        return LeRobotDataset.create(
            repo_id=task.local_repo_id,
            root=task.output_path,
            fps=self.fps,
            robot_type=self.robot_type,
            use_videos=self.use_videos,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_process,
            rgb_encoder=rgb_encoder(self.encoding),
        )

    def save_episode(self, dataset, episode_data, task: ConversionTask) -> bool:
        from openx_rlds import camera_shapes, frame_images

        shapes = camera_shapes(self.features)
        traj = episode_data["steps"]
        for index in range(traj["action"].shape[0]):
            dataset.add_frame(
                {
                    **frame_images(
                        traj["observation"], index, shapes,
                        self.dataset_name, self.channels,
                    ),
                    "observation.state": traj["proprio"][index],
                    "action": traj["action"][index],
                    "task": traj["task"][0].decode(),
                }
            )
        dataset.save_episode()
        return True

    def get_episode_length(self, episode_data: Any) -> int:
        return int(episode_data["steps"]["action"].shape[0])

    # --- tfds, opened where it is used --------------------------------------

    def _builder(self):
        import tensorflow_datasets as tfds

        return tfds.builder(
            self.dataset_name, data_dir=self.data_dir, version=self.version
        )

    def _count_episodes(self) -> int:
        return int(self._builder().info.splits["train"].num_examples)

    def _config(self) -> dict[str, Any]:
        from oxe_utils.configs import OXE_DATASET_CONFIGS

        return OXE_DATASET_CONFIGS.get(self.dataset_name, {})
