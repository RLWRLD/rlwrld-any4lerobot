"""HDF5 안에 인코딩된 채로 들어 있는 프레임을 픽셀로 되돌린다.

color 는 JPEG(`ffd8ff`), depth 는 PNG(`89504e47`) 다. v1 의 ``decode_images`` 와 같은
방식이고, 압축이 아닌 raw 바이트로 저장된 프레임에 대한 대비도 그대로 가져왔다.

해상도는 ``camera_color_resolution`` 을 읽지 않고 디코드한 결과에서 얻는다. 그 필드는
real 이 (H, W), sim 이 (W, H) 로 축 순서가 반대다.
"""

import cv2
import numpy as np

from .errors import EpisodeSkipped

# Frames that were stored raw rather than encoded, keyed by byte count.
_RAW_COLOR = {2_764_800: (720, 1280, 3), 921_600: (480, 640, 3)}
_RAW_DEPTH = {921_600: (720, 1280), 307_200: (480, 640)}


def _blob(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.frombuffer(array.tobytes(), dtype=np.uint8)
    return array


def decode_color(blobs) -> np.ndarray:
    frames = []
    for value in blobs:
        buffer = _blob(value)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            shape = _RAW_COLOR.get(buffer.size)
            if shape is None:
                raise EpisodeSkipped(f"undecodable colour frame of {buffer.size} bytes")
            frame = buffer.reshape(shape)
        frames.append(frame)
    if not frames:
        raise EpisodeSkipped("no colour frames")
    return np.asarray(frames, dtype=np.uint8)


def decode_depth(blobs) -> np.ndarray:
    frames = []
    for value in blobs:
        buffer = _blob(value)
        frame = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if frame is None:
            shape = _RAW_DEPTH.get(buffer.size)
            if shape is None:
                raise EpisodeSkipped(f"undecodable depth frame of {buffer.size} bytes")
            frame = buffer.reshape(shape)
        frames.append(frame)
    if not frames:
        raise EpisodeSkipped("no depth frames")
    return np.asarray(frames)[..., None]


def frame_shape(handle, camera: str) -> tuple[int, int, int]:
    """``(height, width, 3)`` from the first decoded frame.

    Measured rather than read off ``camera_color_resolution``: that field stores
    (H, W) for a real episode and (W, H) for a simulated one, while the pixels are
    (H, W) in both.
    """
    blobs = handle[f"camera_observations/color_images/{camera}"]
    if len(blobs) == 0:
        raise EpisodeSkipped(f"{camera} has no frames")
    frame = decode_color(blobs[:1])[0]
    return (int(frame.shape[0]), int(frame.shape[1]), 3)
