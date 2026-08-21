"""RoboMIND 2.0 → LeRobot v3.0.

    python robomind_v2_h5.py --src-paths /data/robomind_2_0/... --output-path /out

어떤 embodiment 를 읽을지는 플래그로 받지 않는다. 원본이
``data/<embodiment>/<task>/success_episodes/<timestamp>/data/<name>.hdf5`` 로 시작하므로
``data/`` 밑 디렉토리 이름이 곧 config 이름이다. 이 릴리스는 로봇 하나를 repo 5 개로
쪼개 배포하는데 그 다섯이 모두 같은 ``data/<embodiment>`` 로 시작하므로, ``--src-paths``
를 복수로 받으면 한 번에 하나의 config 로 모인다.

에피소드가 하나도 안 만들어지면 **실패한다.** v1 은 이 경우 출력 디렉토리를 지우고 정상
종료했고, 그러면 "0 개 변환됨" 이 성공과 구별되지 않는다.
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robomind_v2_utils.configs import ConfigError, available
from robomind_v2_utils.configs import load as load_config
from robomind_v2_utils.lerobot_utils import RoboMINDv2Dataset, build_features
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import EpisodeRef, discover, read_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# An episode this short is a recording that went wrong rather than a short task.
# Carried over from v1, which used the same floor.
DEFAULT_MIN_FRAMES = 50


class NothingConverted(RuntimeError):
    """No episode was written. Never a success."""


def plan_tasks(src_paths: list[Path]) -> dict[tuple[str, str], list[EpisodeRef]]:
    """Episodes grouped into one conversion task per ``(embodiment, task)``.

    One LeRobot dataset per group, which is also what lets each group carry its
    own frame rate.
    """
    grouped: dict[tuple[str, str], list[EpisodeRef]] = defaultdict(list)
    for ref in discover(src_paths):
        grouped[(ref.embodiment, ref.task)].append(ref)
    return dict(grouped)


def convert_task(
    refs: list[EpisodeRef],
    embodiment: str,
    task: str,
    output_path: Path,
    save_depth: bool,
    min_frames: int,
) -> int:
    """Write one LeRobot dataset for one task. Returns the episode count."""
    config = load_config(embodiment)
    local_dir = (output_path / embodiment / task).resolve()
    if local_dir.exists():
        shutil.rmtree(local_dir)

    dataset = None
    written = 0
    for ref in refs:
        try:
            episode = read_episode(ref, config, save_depth=save_depth)
        except EpisodeSkipped as exc:
            logging.warning("skipped %s: %s", ref.path, exc)
            continue
        except (OSError, ValueError) as exc:
            logging.warning("skipped %s: unreadable: %s", ref.path, exc)
            continue

        if len(episode.frames) < min_frames:
            logging.warning(
                "skipped %s: %d frames < %d", ref.path, len(episode.frames), min_frames
            )
            continue

        if dataset is None:
            # One dataset per task, so the rate is per task rather than per
            # episode. The first surviving episode sets it; anything that
            # disagrees by more than a tenth is logged rather than silently
            # written onto the wrong time base.
            fps = round(episode.fps)
            if fps < 1:
                # A rate at or below 0.5 Hz rounds to 0, and a dataset can't be opened at
                # 0 fps -- nor can the drift check below divide by it. Treat
                # this measurement the same as any other unusable episode:
                # skip it and let the next surviving episode set the rate.
                logging.warning(
                    "skipped %s: measured rate %.3f Hz rounds to 0, not usable as a fps",
                    ref.path, episode.fps,
                )
                continue
            dataset = RoboMINDv2Dataset.create(
                repo_id=f"{embodiment}/{task}",
                root=local_dir,
                fps=fps,
                robot_type=config.robot_type,
                features=build_features(config, episode.shapes),
            )
        elif abs(episode.fps - fps) / fps > 0.1:
            logging.warning(
                "%s runs at %.1f Hz but %s/%s was opened at %d Hz",
                ref.path, episode.fps, embodiment, task, fps,
            )

        for frame in episode.frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        written += 1
        logging.info("wrote %s (%d frames)", ref.path, len(episode.frames))

    if dataset is None:
        logging.warning("no episode survived for %s/%s", embodiment, task)
        return 0

    dataset.finalize()
    return written


def main(
    src_paths: list[Path],
    output_path: Path,
    save_depth: bool = False,
    min_frames: int = DEFAULT_MIN_FRAMES,
    cpus_per_task: int = 2,
    debug: bool = False,
) -> Path:
    grouped = plan_tasks(src_paths)
    if not grouped:
        raise NothingConverted(
            f"no episodes under {', '.join(str(path) for path in src_paths)}: "
            "expected data/<embodiment>/<task>/success_episodes/<stamp>/data/*.hdf5"
        )

    known = set(available())
    unknown = sorted({embodiment for embodiment, _ in grouped} - known)
    if unknown:
        raise NothingConverted(
            f"no config for {', '.join(unknown)}. available: {', '.join(sorted(known))}"
        )

    if debug:
        results = [
            convert_task(refs, embodiment, task, output_path, save_depth, min_frames)
            for (embodiment, task), refs in sorted(grouped.items())
        ]
    else:
        import ray
        from ray.runtime_env import RuntimeEnv

        ray.init(
            runtime_env=RuntimeEnv(
                env_vars={
                    "HDF5_USE_FILE_LOCKING": "FALSE",
                    "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                }
            )
        )
        remote = ray.remote(convert_task).options(num_cpus=cpus_per_task)
        futures = {
            (embodiment, task): remote.remote(
                refs, embodiment, task, output_path, save_depth, min_frames
            )
            for (embodiment, task), refs in sorted(grouped.items())
        }
        results = []
        for (embodiment, task), future in futures.items():
            try:
                results.append(ray.get(future))
            except Exception as exc:  # noqa: BLE001 - one task must not kill the run
                logging.error("failed %s/%s: %s", embodiment, task, exc)
                results.append(0)

    total = sum(results)
    if total == 0:
        raise NothingConverted(
            f"{len(grouped)} task(s) found but no episode was written. Every episode "
            "was skipped -- check the log for the reasons rather than treating this "
            "as an empty dataset."
        )
    logging.info("wrote %d episodes across %d tasks", total, len(grouped))
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
    args = parse_args()
    try:
        print(main(**vars(args)))
    except (NothingConverted, ConfigError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error
