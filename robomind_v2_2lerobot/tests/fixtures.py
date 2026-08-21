"""실측한 RoboMIND 2.0 구조를 그대로 만드는 합성 에피소드.

실물은 에피소드 하나가 24 MB ~ 738 MB 라서 단위 테스트에 쓸 수 없다. 여기서 만드는
것은 몇 KB 지만 키 구조·dtype·인코딩이 실물과 같다: color 는 JPEG, depth 는 PNG,
스트림은 `<name>_align/{data,is_intervene,timestamp}` 3 개짜리 그룹, timestamp 는
초 단위 정수.
"""

from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_STREAMS = {
    "arm_left_position": 7,
    "arm_right_position": 7,
    "end_effector_left_pose": 7,
    "end_effector_left_position": 1,
    "end_effector_right_pose": 7,
    "end_effector_right_position": 1,
}


def _jpeg(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.reshape(-1)


def _png_depth(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000)
    depth = rng.integers(0, 4000, (height, width), dtype=np.uint16)
    ok, buffer = cv2.imencode(".png", depth)
    assert ok
    return buffer.reshape(-1)


def write_episode(
    root: Path,
    embodiment: str,
    task: str,
    stamp: str,
    *,
    frames: int = 6,
    cameras: tuple[str, ...] = ("camera_top",),
    streams: dict[str, int] | None = None,
    layout: str = "real",
    instruction: str | None = None,
    broken: str | None = None,
    seconds: int = 2,
    resolution: tuple[int, int] = (48, 64),
    depth_resolution: tuple[int, int] | None = None,
    resolution_field: tuple[int, int] | None = None,
    extras: dict[str, tuple[int, ...]] | None = None,
) -> Path:
    """One episode on disk. Returns the hdf5 path.

    ``broken="empty"`` writes a valid hdf5 with no objects -- the 6,144-byte shape
    that 4,500 UR5 files have. ``broken="truncated"`` writes only the first stream,
    the shape the two 9-13 KB files have.

    ``resolution_field`` overrides what ``camera_color_resolution`` claims, so a
    test can prove the reader ignores it. ``sim`` stores (W, H) where real stores
    (H, W), and decoding shows both are really (H, W).

    ``depth_resolution`` defaults to ``resolution``, so every existing caller
    keeps writing depth at the same size as colour. Passing a different size
    lets a test prove depth's shape is measured from its own decoded pixels
    rather than borrowed from colour's.
    """
    streams = dict(streams or DEFAULT_STREAMS)
    height, width = resolution
    depth_height, depth_width = depth_resolution or resolution
    name = "trajectory.hdf5" if layout == "real" else f"{stamp}.hdf5"
    directory = root / "data" / embodiment / task / "success_episodes" / stamp / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name

    if broken == "empty":
        with h5py.File(path, "w"):
            pass
        return path

    stamps = np.linspace(0, seconds, frames).astype(np.int64) + 1_747_982_392

    with h5py.File(path, "w") as handle:
        if broken == "truncated":
            first = next(iter(streams))
            group = handle.create_group(f"master/{first}_align")
            group.create_dataset(
                "data", data=np.zeros((frames, streams[first]), dtype=np.float32)
            )
            return path

        blob = h5py.special_dtype(vlen=np.uint8)
        claimed = resolution_field or (
            (width, height) if layout == "sim" else (height, width)
        )
        claimed_depth = (
            (depth_width, depth_height) if layout == "sim" else (depth_height, depth_width)
        )
        for index, camera in enumerate(cameras):
            handle.create_dataset(
                f"camera_color_resolution/{camera}", data=np.array(claimed, dtype=np.int64)
            )
            handle.create_dataset(
                f"camera_depth_resolution/{camera}", data=np.array(claimed_depth, dtype=np.int64)
            )
            handle.create_dataset(f"camera_color_channel/{camera}", data=b"rgb")
            handle.create_dataset(f"camera_model/{camera}", data=b"RealSense_D435if")

            colors = handle.create_dataset(
                f"camera_observations/color_images/{camera}", (frames,), dtype=blob
            )
            depths = handle.create_dataset(
                f"camera_observations/depth_images/{camera}", (frames,), dtype=blob
            )
            for frame in range(frames):
                colors[frame] = _jpeg(height, width, seed=index * 100 + frame)
                depths[frame] = _png_depth(depth_height, depth_width, seed=index * 100 + frame)

        handle.create_dataset("camera_observations/timestamp", data=stamps)
        handle.create_dataset(
            "camera_observations/is_intervene", data=np.zeros(frames, dtype=bool)
        )

        for side in ("puppet", "master"):
            # Add a side-dependent offset so puppet and master data are distinguishable,
            # making it possible to catch bugs that swap their paths.
            side_offset = 0.0 if side == "puppet" else 1000.0
            for stream, stream_width in streams.items():
                suffixes = ["_align", "_raw"] if layout == "sim" else ["_align"]
                for suffix in suffixes:
                    count = frames * 2 if suffix == "_raw" else frames
                    group = handle.create_group(f"{side}/{stream}{suffix}")
                    shape = (count,) if stream_width == 1 else (count, stream_width)
                    group.create_dataset(
                        "data",
                        data=np.arange(count * stream_width, dtype=np.float32).reshape(shape) + side_offset,
                    )
                    group.create_dataset(
                        "is_intervene", data=np.zeros(count, dtype=bool)
                    )
                    group.create_dataset(
                        "timestamp",
                        data=stamps if count == frames else np.repeat(stamps, 2),
                    )

        for group_name, members in (extras or {}).items():
            for member, shape in members.items():
                group = handle.create_group(f"{group_name}/{member}_align")
                group.create_dataset(
                    "data", data=np.zeros((frames, *shape), dtype=np.float32)
                )
                group.create_dataset("is_intervene", data=np.zeros(frames, dtype=bool))
                group.create_dataset("timestamp", data=stamps)

        if layout == "sim":
            handle.create_dataset(
                "metadata/language_instruction", data=(instruction or "do the thing").encode()
            )
            handle.create_dataset("metadata/trajectory_length", data=frames)
            handle.create_dataset("metadata/data_type", data=b"sim")

    if layout == "real" and instruction is not None:
        (root / "data" / embodiment / task / "zh_description.txt").write_text(instruction)

    return path
