"""One adapter for every dataset in the registry.

The dataset-specific parts of a conversion -- where the files are, what the hdf5
keys are called, how the clocks line up, what the robot is called, how fast it runs
-- all come from ``dataset_registry``. No dataset is named here.

What this emits is deliberately incomplete: the source's own feature vectors and
the video, and nothing else. ``observation.state`` and ``action`` are assembled
afterwards by ``lerobot_pipeline``'s ``state_layout`` step, from the same layout
spec, so that a dataset arriving as raw hdf5 and a dataset arriving as LeRobot go
through one implementation of the convention rather than two.
"""

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generic_converter import BaseAdapter, ConversionTask  # noqa: E402
from generic_converter.prerendered_video import PrerenderedDataset  # noqa: E402

from .formats import EpisodeSkipped, build_reader  # noqa: E402


def motor_names(count: int) -> list[str]:
    """The delivered datasets name motors positionally rather than by joint."""
    return [f"m{index}" for index in range(count)]


def vector_feature(count: int) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (count,),
        "names": {"motors": motor_names(count)},
    }


def source_features(spec) -> dict[str, dict[str, Any]]:
    """The LeRobot features a reader emits: the video keys and the source vectors.

    Widths come from ``source.feature_widths`` where the spec states them, and
    otherwise from how far into each feature the layout reads.
    """
    features: dict[str, dict[str, Any]] = {}
    for key in spec.cameras:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": tuple(spec.camera_shape(key) or ()),
            "names": ["height", "width", "rgb"],
        }

    declared = spec.source.feature_widths if spec.source else {}
    for side in ("state", "action"):
        vector = spec.vector(side)
        if vector is None:
            continue
        for name, sides in vector.source_features.items():
            width = declared.get(name) or max(
                (b.src_end for b in vector.blocks if b.feature == name), default=0
            )
            if width:
                features[sides[side]] = vector_feature(width)
    return features


class SpecAdapter(BaseAdapter):
    dataset_type = "spec"

    def __init__(
        self,
        spec,
        src_path: Path,
        output_path: Path,
        episodes_per_task: int,
        max_episodes: int | None = None,
    ):
        super().__init__(output_path)
        if episodes_per_task < 1:
            raise ValueError("--episodes-per-task must be >= 1")
        problems = spec.buildable()
        if problems:
            raise ValueError(
                f"{spec.id} cannot be rebuilt from its source:\n  "
                + "\n  ".join(problems)
            )

        self.spec = spec
        self.src_path = src_path.expanduser().resolve()
        self.episodes_per_task = episodes_per_task
        self.max_episodes = max_episodes
        self.fps = spec.fps
        self.robot_type = spec.robot_type
        self.tags = (spec.id,)
        self.features = source_features(spec)
        self.reader = build_reader(spec, self.src_path)

    def load_tasks(self) -> list[ConversionTask]:
        episode_ids = self.reader.episode_ids()
        if self.max_episodes is not None:
            episode_ids = episode_ids[: self.max_episodes]
        if not episode_ids:
            raise ValueError(
                f"no episodes matching {self.spec.source.discover!r} under "
                f"{self.src_path}"
            )

        tasks = []
        for chunk_index, chunk in enumerate(self._chunks(episode_ids)):
            name = f"{self.spec.id}_chunk_{chunk_index:06d}_{chunk[0]}_{chunk[-1]}"
            tasks.append(
                ConversionTask(
                    input_path=self.src_path,
                    output_path=(self.temp_output_path / self.spec.id / name).resolve(),
                    local_repo_id=name,
                    metadata={"episode_ids": tuple(chunk)},
                )
            )
        return tasks

    def load_subset(self, task: ConversionTask) -> Iterable[dict[str, Any]]:
        prompts = self.reader.prompts()
        for episode_id in task.metadata["episode_ids"]:
            try:
                episode = self.reader.read_episode(
                    episode_id, prompt=prompts.get(episode_id, "")
                )
            except EpisodeSkipped as exc:
                print(f"{episode_id}: skipping ({exc})")
                continue
            except (OSError, ValueError) as exc:
                print(f"{episode_id}: skipping (unreadable: {exc})")
                continue
            if not episode.frames:
                print(f"{episode_id}: skipping (no frames)")
                continue
            yield {
                "episode_id": episode_id,
                "frames": episode.frames,
                "videos": episode.videos,
            }

    def create_dataset(self, task: ConversionTask) -> PrerenderedDataset:
        return PrerenderedDataset.create(
            repo_id=task.local_repo_id,
            root=task.output_path,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
        )

    def save_episode(
        self,
        dataset: PrerenderedDataset,
        episode_data: dict[str, Any],
        task: ConversionTask,
    ) -> bool:
        for frame in episode_data["frames"]:
            dataset.add_frame(frame)
        try:
            dataset.save_episode(videos=episode_data["videos"])
        except Exception as exc:  # noqa: BLE001 - one bad mp4 must not kill the chunk
            print(f"{episode_data['episode_id']}: save failed ({exc})")
            dataset.clear_episode_buffer(delete_images=False)
            return False
        return True

    def get_episode_length(self, episode_data: dict[str, Any]) -> int:
        return len(episode_data["frames"])

    def _chunks(self, episode_ids: list[str]):
        for start in range(0, len(episode_ids), self.episodes_per_task):
            yield episode_ids[start : start + self.episodes_per_task]
