"""프레임은 HDF5 안에 인코딩된 바이트로 들어 있다 — color 는 JPEG, depth 는 PNG.

해상도는 camera_color_resolution 필드가 아니라 첫 프레임을 디코드해서 얻는다.
그 필드는 real 이 (H, W) 인데 sim 이 (W, H) 로 축 순서가 반대이고, 실제 픽셀은
양쪽 다 (H, W) 다.
"""

import cv2
import h5py
import numpy as np
from fixtures import write_episode

from robomind_v2_utils.images import decode_color, decode_depth, frame_shape


def test_color_decodes_to_hwc_uint8(tmp_path):
    path = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=4, resolution=(48, 64))

    with h5py.File(path, "r") as handle:
        frames = decode_color(handle["camera_observations/color_images/camera_top"])

    assert frames.shape == (4, 48, 64, 3)
    assert frames.dtype.name == "uint8"


def test_color_channel_order_is_not_flipped():
    """실측(task 11): 이 컨버터가 지금 다루는 config 는 채널을 뒤집지 않아야
    한다 -- 실물 프레임을 ``PIL.Image.fromarray`` 로 있는 그대로(재해석 없이)
    렌더링해서 눈으로 확인했다(빨간 사과 task 의 실제 프레임이, 뒤집지 않았을
    때 자연스러운 빨강으로 나오고 뒤집으면 파랗게 나온다). 근거와, 이 판정에서
    ``cv2.imwrite`` 기반 비교가 결론을 반대로 뒤집을 뻔했던 함정은
    ``decode_color`` 의 docstring 을 본다.

    여기서는 그 "뒤집지 않는다" 는 계약 자체를 고정한다: ``decode_color`` 의
    출력이 ``cv2.imdecode`` 의 원시 출력과 채널 순서까지 완전히 같아야 한다.
    """
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 1] = 50
    image[..., 2] = 10
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    blob = buffer.reshape(-1)

    raw = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    decoded = decode_color([blob])

    assert decoded[0].tolist() == raw.tolist()


def test_depth_decodes_with_a_trailing_channel(tmp_path):
    path = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=4, resolution=(48, 64))

    with h5py.File(path, "r") as handle:
        frames = decode_depth(handle["camera_observations/depth_images/camera_top"])

    assert frames.shape == (4, 48, 64, 1)


def test_shape_comes_from_the_pixels_not_the_field(tmp_path):
    """sim 의 resolution 필드는 (W, H) 다. 그걸 믿으면 shape 이 뒤집힌다."""
    path = write_episode(
        tmp_path, "franka_sim", "task", "0003_000000", resolution=(48, 64), layout="sim"
    )

    with h5py.File(path, "r") as handle:
        claimed = tuple(int(value) for value in handle["camera_color_resolution/camera_top"][()])
        measured = frame_shape(handle, "camera_top")

    assert claimed == (64, 48)          # the file says (W, H)
    assert measured == (48, 64, 3)      # the pixels say (H, W, 3)


def test_a_lying_resolution_field_is_ignored(tmp_path):
    path = write_episode(
        tmp_path,
        "tienyi",
        "task",
        "0004_000000",
        resolution=(48, 64),
        resolution_field=(999, 111),
    )

    with h5py.File(path, "r") as handle:
        assert frame_shape(handle, "camera_top") == (48, 64, 3)
