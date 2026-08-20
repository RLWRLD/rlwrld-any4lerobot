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
run does. The slice means the same thing under both readers below, so the task
arithmetic does not care which one a mirror needs.

Nothing here opens tensorflow at import time. The builder is constructed inside the
worker that uses it, which keeps the adapter picklable across processes and lets the
task arithmetic be tested without the conversion environment installed.
"""

import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

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

    This is what picks the reader. tfds writes two formats and offers a different API
    for each: ``tfrecord`` is read with ``as_dataset`` and ``array_record`` only with
    ``as_data_source``. Asking for the wrong one fails with an assertion telling you to
    call ``download_and_prepare``, which says nothing about the format and sends you
    looking for a missing file -- so the format is read from the metadata rather than
    inferred from that failure. Of the collection, only bc_z is array_record, across
    1024 shards.

    A directory whose ``dataset_info.json`` predates the field reads as tfrecord,
    which is what it will be -- the field arrived after the format. A directory with
    no metadata at all reads as ``None`` rather than as a guess, and ``None`` reads the
    stream, which is what every mirror but one is.
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


def stack_steps(steps) -> dict[str, Any]:
    """One episode's steps as arrays over time, which is the shape a transform reads.

    ``as_data_source`` hands the steps back as a sequence of per-step dicts, decoded
    one at a time as they are asked for, where the tf.data reader gets the whole
    trajectory from ``steps.batch(steps.cardinality())``. This is that batch, done in
    Python. Nesting is kept rather than flattened: bc_z's action is itself a dict, and
    its standardization transform indexes into that dict by name.
    """
    frames = list(steps)
    if not frames:
        raise ValueError("episode has no steps")
    return _stacked(frames)


def _stacked(values: list[Any]) -> Any:
    if isinstance(values[0], Mapping):
        return {key: _stacked([value[key] for value in values]) for key in values[0]}
    return np.stack(values)


def to_numpy(trajectory: Any) -> Any:
    """Eager tensors back to arrays, leaving whatever is already an array alone.

    Standardizing a trajectory runs TensorFlow ops, so parts of the result come back as
    eager tensors no matter which reader produced the input. The tf.data reader converts
    a whole stream at once with ``as_numpy_iterator``; this is the same conversion for
    one episode. Written as a duck-typed walk rather than through ``tf.nest`` so that
    the module keeps its promise not to import TensorFlow.
    """
    if isinstance(trajectory, Mapping):
        return {key: to_numpy(value) for key, value in trajectory.items()}
    as_array = getattr(trajectory, "numpy", None)
    return as_array() if callable(as_array) else trajectory


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
        split = task.metadata["split"]
        if file_format(self.raw_dir) == "array_record":
            return self._episodes_by_index(split)
        return self._episodes_by_stream(split)

    def _keep(self):
        """Whether an episode is converted at all.

        kuka is the one source that ships failures alongside successes. Written to
        return the field rather than a bool so the same predicate serves both readers:
        tf.data traces it into a graph, where ``bool`` on a tensor raises.
        """
        if self.dataset_name != "kuka":
            return lambda episode: True
        return lambda episode: episode["success"]

    def _episodes_by_stream(self, split: str) -> Iterable[Any]:
        """RLDS from a tfrecord mirror: filter, standardize and convert as a stream."""
        from functools import partial

        from openx_rlds import transform_raw_dataset

        dataset = (
            self._builder()
            .as_dataset(split=split)
            .filter(self._keep())
            .map(partial(transform_raw_dataset, dataset_name=self.dataset_name))
        )
        yield from dataset.as_numpy_iterator()

    def _episodes_by_index(self, split: str) -> Iterable[Any]:
        """RLDS from an array_record mirror, which tfds serves only by random access.

        ``as_data_source`` returns a sequence of episodes rather than a stream, and the
        steps of each episode are a sequence of their own that decodes an element when
        it is indexed. So the three things tf.data was doing as stream operations --
        filtering, batching the steps, converting to numpy -- happen here as plain
        Python, around the very same standardization transform. An episode costs the
        same memory either way: both readers hold one whole trajectory at a time.

        The images come out RGB under both readers -- tfds decodes them with OpenCV
        here and reorders BGR back itself -- so this is not a second place a channel
        order can differ. video_rules.flips_channels stays the only thing that decides
        colour order, and it sees the same bytes whichever reader ran.
        """
        from openx_rlds import standardize_trajectory

        keep = self._keep()
        source = self._builder().as_data_source(split=split)
        for index in range(len(source)):
            episode = dict(source[index])
            if not keep(episode):
                continue
            episode["steps"] = to_numpy(
                standardize_trajectory(stack_steps(episode["steps"]), self.dataset_name)
            )
            yield episode

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
        from video_rules import parse_rule, resize_filter

        shapes = camera_shapes(self.features)
        # The same mapping the shapes came from, so geometry and resampler are read
        # off one declaration rather than two.
        filter = resize_filter(parse_rule(self.resize))
        traj = episode_data["steps"]
        for index in range(traj["action"].shape[0]):
            dataset.add_frame(
                {
                    **frame_images(
                        traj["observation"], index, shapes,
                        self.dataset_name, self.channels, filter,
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
