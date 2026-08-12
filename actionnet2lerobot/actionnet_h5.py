"""Convert the Fourier ActionNet dataset to LeRobot.

ActionNet ships one hdf5 and one already-encoded mp4 per episode, so conversion is
mostly bookkeeping: line the ~60 Hz robot stream up with the ~30 Hz camera stream,
write the per-frame arrays, and reuse the mp4 rather than decoding and re-encoding
it. Episodes are independent, so the work is split into chunks and handed to the
generic datatrove executor.

    python actionnet_h5.py --src-path /data/action_net --output-path /out
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ACTIONNET_DIR = Path(__file__).resolve().parent
REPO_ROOT = ACTIONNET_DIR.parent
for import_path in (REPO_ROOT, ACTIONNET_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from actionnet_utils.actionnet_utils import (  # noqa: E402
    EpisodePaths,
    EpisodeSkipped,
    discover_episode_ids,
    load_episode,
    load_prompts,
)
from actionnet_utils.config import (  # noqa: E402
    FPS,
    ROBOT_TYPE,
    generate_features,
    generate_modality,
)

from generic_converter import BaseAdapter, ConversionTask, run_converter  # noqa: E402
from generic_converter.prerendered_video import PrerenderedDataset  # noqa: E402


class ActionNetAdapter(BaseAdapter):
    dataset_type = "actionnet"
    fps = FPS
    robot_type = ROBOT_TYPE
    tags = ("fourier", "action-net", "gr1t1")

    def __init__(
        self,
        src_path: Path,
        output_path: Path,
        episodes_per_task: int,
        max_episodes: int | None = None,
    ):
        super().__init__(output_path)
        if episodes_per_task < 1:
            raise ValueError("--episodes-per-task must be >= 1")
        self.src_path = src_path.expanduser().resolve()
        self.episodes_per_task = episodes_per_task
        self.max_episodes = max_episodes
        self.features = generate_features()

    def load_tasks(self) -> list[ConversionTask]:
        episode_ids = discover_episode_ids(self.src_path)
        if self.max_episodes is not None:
            episode_ids = episode_ids[: self.max_episodes]
        if not episode_ids:
            raise ValueError(f"no *.hdf5 episodes found under {self.src_path}")

        tasks = []
        for chunk_index, chunk in enumerate(self._chunks(episode_ids)):
            name = f"actionnet_chunk_{chunk_index:06d}_{chunk[0]}_{chunk[-1]}"
            tasks.append(
                ConversionTask(
                    input_path=self.src_path,
                    output_path=(self.temp_output_path / "actionnet" / name).resolve(),
                    local_repo_id=name,
                    metadata={"episode_ids": tuple(chunk)},
                )
            )
        return tasks

    def load_subset(self, task: ConversionTask) -> Iterable[dict[str, Any]]:
        prompts = load_prompts(self.src_path)
        for episode_id in task.metadata["episode_ids"]:
            paths = EpisodePaths.build(self.src_path, episode_id)
            try:
                frames, videos = load_episode(
                    paths, prompt=prompts.get(episode_id, "")
                )
            except EpisodeSkipped as exc:
                print(f"{episode_id}: skipping ({exc})")
                continue
            except (OSError, ValueError) as exc:
                print(f"{episode_id}: skipping (unreadable: {exc})")
                continue
            if not frames:
                print(f"{episode_id}: skipping (no frames)")
                continue
            yield {"episode_id": episode_id, "frames": frames, "videos": videos}

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


def write_modality(output_path: Path) -> Path:
    """Emit ``meta/modality.json``, which LeRobot itself does not write.

    Every dataset in the RLDX-1 pre-training collection has one, and the training
    stack reads it to find the state/action blocks inside the flat vectors.
    """
    import json

    path = output_path / "meta" / "modality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(generate_modality(), indent=4) + "\n")
    return path


def main(
    src_path: Path,
    output_path: Path,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    episodes_per_task: int,
    max_episodes: int | None = None,
    resume_dir: Path | None = None,
    debug: bool = False,
    repo_id: str | None = None,
    push_to_hub: bool = False,
):
    adapter = ActionNetAdapter(
        src_path=src_path,
        output_path=output_path,
        episodes_per_task=episodes_per_task,
        max_episodes=max_episodes,
    )
    result = run_converter(
        adapter=adapter,
        executor=executor,
        cpus_per_task=cpus_per_task,
        tasks_per_job=tasks_per_job,
        workers=workers,
        resume_dir=resume_dir,
        debug=debug,
        local_repo_id=repo_id,
        hub_repo_id=repo_id,
        push_to_hub=push_to_hub,
    )
    print(f"wrote {write_modality(result)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--executor", type=str, choices=["local", "ray"], default="local"
    )
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--tasks-per-job", type=int, default=1)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=100,
        help="episodes per temporary dataset; one task per chunk",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="convert only the first N episodes; for smoke tests",
    )
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        src_path=args.src_path,
        output_path=args.output_path,
        executor=args.executor,
        cpus_per_task=args.cpus_per_task,
        tasks_per_job=args.tasks_per_job,
        workers=args.workers,
        episodes_per_task=args.episodes_per_task,
        max_episodes=args.max_episodes,
        resume_dir=args.resume_dir,
        debug=args.debug,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
    )
