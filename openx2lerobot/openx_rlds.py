#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
For all datasets in the RLDS format.
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

NOTE: You need to install tensorflow and tensorflow_datsets before running this script.

Example:
    python openx_rlds.py \
        --raw-dir /path/to/bridge_orig/1.0.0 \
        --local-dir /path/to/local_dir \
        --repo-id your_id \
        --use-videos \
        --push-to-hub
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from lerobot.utils.constants import HF_LEROBOT_HOME

from adapter import OpenXAdapter, read_raw_dir
from generic_converter import run_converter
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS
from video_rules import resize_frame, target_shape

np.set_printoptions(precision=2)


def transform_raw_dataset(episode, dataset_name):
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    proprio = tf.concat(
        [
            (
                tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)  # padding
                if key is None
                else tf.cast(traj["observation"][key], tf.float32)
            )
            for key in state_obs_keys
        ],
        axis=1,
    )

    traj.update(
        {
            "proprio": proprio,
            "task": traj.pop("language_instruction"),
            "action": tf.cast(traj["action"], tf.float32),
        }
    )

    episode["steps"] = traj
    return episode


def generate_features_from_raw(
    builder: tfds.core.DatasetBuilder, use_videos: bool = True, resize=None
):
    dataset_name = Path(builder.data_dir).parent.name

    state_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
        if state_encoding == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
            if "libero" in dataset_name:
                state_names = [
                    "x",
                    "y",
                    "z",
                    "axis_angle1",
                    "axis_angle2",
                    "axis_angle3",
                    "gripper",
                    "gripper",
                ]  # 2D gripper state
        elif state_encoding == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif state_encoding == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
            state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
            pad_count = state_obs_keys[:-1].count(None)
            state_names[-pad_count - 1 : -1] = ["pad"] * pad_count
            state_names[-1] = "pad" if state_obs_keys[-1] is None else state_names[-1]

    action_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
        if action_encoding == ActionEncoding.EEF_POS:
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            if "libero" in dataset_name:
                action_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
        elif action_encoding == ActionEncoding.JOINT_POS:
            action_names = [f"motor_{i}" for i in range(7)] + ["gripper"]

    DEFAULT_FEATURES = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"motors": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"motors": action_names},
        },
    }

    obs = builder.info.features["steps"]["observation"]
    features = {
        f"observation.images.{key}": {
            "dtype": "video" if use_videos else "image",
            # the size after the resize rule, because that is what gets written
            "shape": (*target_shape(resize, key, tuple(value.shape[:2])), value.shape[2]),
            "names": ["height", "width", "rgb"],
        }
        for key, value in obs.items()
        if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }
    return {**features, **DEFAULT_FEATURES}


def camera_shapes(features) -> dict[str, tuple[int, int]]:
    """Video key -> the ``(height, width)`` its frames are written at."""
    return {
        key: tuple(value["shape"][:2])
        for key, value in features.items()
        if key.startswith("observation.images.")
    }


def frame_images(observation, index: int, shapes: dict[str, tuple[int, int]]):
    """One frame's camera images, keyed and sized the way they will be written."""
    out = {}
    for key, value in observation.items():
        if "depth" in key or not any(x in key for x in ["image", "rgb"]):
            continue
        name = f"observation.images.{key}"
        shape = shapes.get(name)
        out[name] = value[index] if shape is None else resize_frame(value[index], shape)
    return out


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    image_writer_process: int = 5,
    image_writer_threads: int = 10,
    resize=None,
    encoding=None,
    executor: str = "local",
    workers: int = -1,
    cpus_per_task: int = 1,
    tasks_per_job: int = 1,
    episodes_per_task: int = 100,
    max_episodes: int | None = None,
    resume_dir: Path | None = None,
    debug: bool = False,
) -> Path:
    """Convert one RLDS source, a chunk of episodes per task.

    A single-process run is ``--workers 1``: the same code path, one task at a time.
    There is no separate serial implementation to keep in step with this one.
    """
    dataset_name, version, _ = read_raw_dir(raw_dir)
    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    output_path = local_dir / f"{dataset_name}_{version}_lerobot"
    if output_path.exists():
        shutil.rmtree(output_path)

    adapter = OpenXAdapter(
        raw_dir=raw_dir,
        output_path=output_path,
        episodes_per_task=episodes_per_task,
        resize=resize,
        encoding=encoding,
        fps=fps,
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_process=image_writer_process,
        image_writer_threads=image_writer_threads,
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
        extra_tags=("openx",) if dataset_name in OXE_DATASET_CONFIGS else (),
        # spawn, not the forkserver datatrove defaults to: this process has already
        # imported TensorFlow, and workers forked from that state can inherit a lock
        # held by a thread they do not have. It does not fail, it waits -- one run
        # held a 48-core instance for fifteen hours without starting a task.
        start_method="spawn",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type of this dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate used to collect videos. Default fps equals to the control frequency of the robot.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help=(
            "How to size frames before they are written: a step name "
            "(`resize_preserve_aspect_area`) or the JSON of one with its parameters. "
            "Resizing happens here rather than in a later stage because this "
            "converter encodes the video itself -- see video_rules.py."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help=(
            "How to encode the video: the name of a profile in "
            "lerobot_pipeline/configs/encoding, or the JSON of one. Default: "
            "LeRobot's own writer settings."
        ),
    )
    parser.add_argument(
        "--executor",
        choices=["local", "ray"],
        default="local",
        help="local spreads tasks across this machine's cores; ray across a cluster",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="concurrent tasks; -1 fills the machine. 1 is a single-process run.",
    )
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--tasks-per-job", type=int, default=1)
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
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=5,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of threads per process of image writer for saving images.",
    )

    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()
