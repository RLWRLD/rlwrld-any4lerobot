"""Convert any registry dataset to LeRobot.

    python -m spec2lerobot --dataset action_net --src-path /data/action_net \
        --output-path /out

Which dataset is a flag, not a module. Everything the conversion needs -- file
layout, hdf5 keys, clock strategy, robot type, frame rate -- comes from
``dataset_registry/datasets/<name>.yaml``. Adding a dataset that uses a format
already implemented means writing that YAML and nothing else.

The output carries the source's own feature vectors and its video, but no
``observation.state`` or ``action``: those are assembled by the ``state_layout``
pipeline step from the same spec. Run this through ``lerobot_pipeline`` rather than
alone unless you specifically want the unassembled form.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import available, load  # noqa: E402
from generic_converter import run_converter  # noqa: E402

from .adapter import SpecAdapter  # noqa: E402


def main(
    dataset: str,
    src_path: Path,
    output_path: Path,
    executor: str = "local",
    cpus_per_task: int = 1,
    tasks_per_job: int = 1,
    workers: int = -1,
    episodes_per_task: int = 100,
    max_episodes: int | None = None,
    resume_dir: Path | None = None,
    debug: bool = False,
    repo_id: str | None = None,
    push_to_hub: bool = False,
) -> Path:
    adapter = SpecAdapter(
        spec=load(dataset),
        src_path=src_path,
        output_path=output_path,
        episodes_per_task=episodes_per_task,
        max_episodes=max_episodes,
    )
    return run_converter(
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help=f"one of: {', '.join(available())}"
    )
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--executor", choices=["local", "ray"], default="local")
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
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(main(**vars(args)))
