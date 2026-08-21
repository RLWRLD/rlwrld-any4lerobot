"""깨진 파일은 조용히 통과해서도, 런을 죽여서도 안 된다 — 건너뛰고 이유를 남긴다.

실측으로 두 종류가 있다:
  * 6,144 B 짜리 유효한 HDF5, 객체 0 개 — ur 에 4,500 개
  * 9,485 / 13,105 B 짜리 잘린 파일, 첫 스트림만 있고 EOF — ur·franka 에 각 1 개
두 번째는 크기로 못 걸러진다.
"""

import h5py
import pytest
from fixtures import write_episode

from robomind_v2_utils.configs import load
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import check_usable


@pytest.fixture
def config():
    return load("tienyi")


def test_empty_hdf5_is_skipped(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0001_000000", broken="empty")

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="no objects"):
            check_usable(handle, config)


def test_truncated_hdf5_is_skipped(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0002_000000", broken="truncated")

    with h5py.File(path, "r") as handle:
        try:
            check_usable(handle, config)
            assert False, "Expected EpisodeSkipped to be raised"
        except EpisodeSkipped as e:
            # Verify the message matches and elision tail is present
            assert "damaged or partial" in str(e)
            assert ", and " in str(e), f"Expected elision tail in message: {e}"


def test_a_wrong_embodiment_is_skipped(tmp_path, config):
    """config 의 카메라·스트림과 하나도 안 겹치는, 완전한 파일은 손상이 아니라
    config 불일치로 알려준다."""
    path = write_episode(
        tmp_path,
        "other",
        "task",
        "0006_000000",
        cameras=("camera_wrist",),
        streams={"gripper_position": 6},
    )

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="config does not match"):
            check_usable(handle, config)


def test_a_whole_episode_passes(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0003_000000")

    with h5py.File(path, "r") as handle:
        check_usable(handle, config)  # does not raise


def test_a_missing_camera_is_skipped(tmp_path, config):
    """config 가 camera_top 을 요구하는데 파일에 없으면 건너뛴다."""
    path = write_episode(
        tmp_path, "tienyi", "task", "0004_000000", cameras=("camera_front",)
    )

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="camera_top"):
            check_usable(handle, config)


def test_a_missing_stream_is_skipped(tmp_path, config):
    streams = {"arm_left_position": 7}

    path = write_episode(tmp_path, "tienyi", "task", "0005_000000", streams=streams)

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="arm_right_position"):
            check_usable(handle, config)
