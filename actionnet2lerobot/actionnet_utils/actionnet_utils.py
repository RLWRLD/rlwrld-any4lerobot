"""Reading one ActionNet episode and lining its streams up on the video clock.

Layout on disk, after untarring the dataset::

    <episode_id>.hdf5          robot side: state/, action/, timestamp
    <episode_id>/top/rgb.mp4   camera side: h264
    <episode_id>/top/depth.mkv camera side: z16 depth (unused here)
    <episode_id>/top/timestamps.json  wall-clock time of every video frame
    metadata.json              id -> prompt for the whole dataset

The robot side is sampled at ~60 Hz and the camera at ~30 Hz, on separate clocks, so
the two are matched by timestamp rather than by index. The matching follows the
reference converter published with the dataset (``FFTAI/fourier-lerobot``,
``scripts/convert_hdf5_to_lerobot.py``) so that the result lines up with what
Fourier's own training pipeline expects; the mp4 is then reused as-is rather than
being decoded and re-encoded, which is also what the reference converter does.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import (
    HAND_JOINTS,
    PERMUTATION,
    ROBOT_JOINTS,
    SOURCE_CAMERA_DIR,
    VIDEO_KEY,
)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S_%f"


class EpisodeSkipped(Exception):
    """Raised when an episode cannot be converted; carries the reason for logs."""


@dataclass(frozen=True)
class EpisodePaths:
    episode_id: str
    hdf5: Path
    rgb: Path
    timestamps: Path

    @classmethod
    def build(cls, root: Path, episode_id: str) -> "EpisodePaths":
        camera_dir = root / episode_id / SOURCE_CAMERA_DIR
        return cls(
            episode_id=episode_id,
            hdf5=root / f"{episode_id}.hdf5",
            rgb=camera_dir / "rgb.mp4",
            timestamps=camera_dir / "timestamps.json",
        )

    def missing(self) -> list[str]:
        return [
            name
            for name, path in (
                ("hdf5", self.hdf5),
                ("rgb.mp4", self.rgb),
                ("timestamps.json", self.timestamps),
            )
            if not path.exists()
        ]


def iso_to_epoch(value: str) -> float:
    """``2025-01-02T02-14-12_142154`` -> seconds since the epoch (UTC)."""
    base = value.split(".")[0]
    return (
        datetime.strptime(base, TIMESTAMP_FORMAT)
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def match_timestamps(candidate: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """For every time in ``ref``, the index of the nearest time in ``candidate``.

    Ported from the reference converter (``FFTAI/fourier-lerobot``,
    ``scripts/convert_hdf5_to_lerobot.py``) so the emitted dataset matches the one
    Fourier's own training pipeline was built against. ``candidate`` must be sorted
    ascending.

    A robot sample is claimed by at most one video frame: when the nearest one is
    already taken the next is used instead, and if that is taken too the frame is
    dropped. So the result can be shorter than ``ref``.
    """
    if candidate.size == 0:
        raise EpisodeSkipped("no robot samples")

    closest_indices: list[int] = []
    already_matched: set[int] = set()
    for time in ref:
        idx = int(np.searchsorted(candidate, time, side="left"))
        if idx > 0 and (
            idx == candidate.size
            or np.fabs(time - candidate[idx - 1]) < np.fabs(time - candidate[idx])
        ):
            idx -= 1
        if idx not in already_matched:
            closest_indices.append(idx)
            already_matched.add(idx)
        elif idx + 1 not in already_matched and idx + 1 < candidate.size:
            closest_indices.append(idx + 1)
            already_matched.add(idx + 1)
    return np.array(closest_indices, dtype=np.int64)


def assemble(robot: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """Reorder ``[robot | hand]`` into the GR1 body-part vector.

    Source order is the robot's own (legs, waist, neck, left arm, right arm) plus
    the two hands; the emitted order groups by body part, left side first. See
    ``config.GR1_BLOCKS``.
    """
    return np.concatenate([robot, hand], axis=1, dtype=np.float32)[:, PERMUTATION]


def load_episode(
    paths: EpisodePaths,
    prompt: str,
) -> tuple[list[dict], dict[str, Path]]:
    """Return per-frame features for one episode plus its video files."""
    missing = paths.missing()
    if missing:
        raise EpisodeSkipped(f"missing {', '.join(missing)}")

    try:
        image_times = np.array(
            [iso_to_epoch(value) for value in json.loads(paths.timestamps.read_text())],
            dtype=np.float64,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise EpisodeSkipped(f"unreadable timestamps.json: {exc}") from exc

    if image_times.size == 0:
        raise EpisodeSkipped("timestamps.json is empty")

    with h5py.File(paths.hdf5, "r") as handle:
        try:
            data_times = np.asarray(handle["timestamp"], dtype=np.float64)
            state_robot = np.asarray(handle["state/robot"], dtype=np.float32)
            state_hand = np.asarray(handle["state/hand"], dtype=np.float32)
            action_robot = np.asarray(handle["action/robot"], dtype=np.float32)
            action_hand = np.asarray(handle["action/hand"], dtype=np.float32)
        except KeyError as exc:
            raise EpisodeSkipped(f"hdf5 missing dataset {exc}") from exc

    for name, array, expected in (
        ("state/robot", state_robot, ROBOT_JOINTS),
        ("state/hand", state_hand, HAND_JOINTS),
        ("action/robot", action_robot, ROBOT_JOINTS),
        ("action/hand", action_hand, HAND_JOINTS),
    ):
        if array.ndim != 2 or array.shape[1] != expected:
            # GR2 reports 29 joints and the 12-DoF hand reports 24 values; both
            # would need their own feature schema and cannot share this dataset.
            raise EpisodeSkipped(
                f"{name} is {array.shape}, expected (n, {expected})"
            )

    state = assemble(state_robot, state_hand)
    action = assemble(action_robot, action_hand)

    # Both filters are the reference converter's: keep only robot samples whose
    # timestamp advances, and only video frames recorded before the last robot
    # sample. The second one typically drops the final frame, so the emitted
    # episode can be one row shorter than the mp4 -- the reference converter copies
    # the mp4 unchanged and leaves that trailing frame unreferenced, and so do we.
    usable = np.where(np.diff(data_times) > 0)[0]
    if usable.size == 0:
        raise EpisodeSkipped("no usable robot timestamps")
    image_times = image_times[image_times < data_times[-1]]
    if image_times.size == 0:
        raise EpisodeSkipped("no video frames inside the robot time range")
    # match_timestamps indexes the filtered array; map back to the full one. The
    # two coincide whenever the timestamps are strictly increasing, which is what
    # every episode inspected so far looks like.
    matched = usable[match_timestamps(data_times[usable], image_times)]

    frames = [
        {
            "observation.state": state[index],
            "action": action[index],
            # actions are absolute joint targets, so this is the same vector; the
            # training config reads it to tell absolute from delta
            "absolute_action": action[index],
            "observation.robot_joints": state_robot[index],
            "observation.hand_joints": state_hand[index],
            "action.robot_joints": action_robot[index],
            "action.hand_joints": action_hand[index],
            "task": prompt,
        }
        for index in matched
    ]
    return frames, {VIDEO_KEY: paths.rgb}


def load_prompts(root: Path) -> dict[str, str]:
    """``metadata.json`` maps every episode id to its natural-language prompt."""
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found; it holds the prompt for every episode"
        )
    entries = json.loads(metadata_path.read_text())
    return {entry["id"]: entry.get("prompt", "") for entry in entries}


def discover_episode_ids(root: Path) -> list[str]:
    """Episode ids present on disk, in sorted (ULID, so chronological) order."""
    return sorted(path.stem for path in root.glob("*.hdf5"))
