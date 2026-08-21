"""폭은 검증한다 — 이름이 같고 폭만 다른 embodiment 가 실제로 있다.

`ur` 의 end_effector_*_position 은 (T, 1) gripper 이고 `ur_dex` 의 같은 이름은
(T, 12) dexterous hand 다. 폭을 확인하지 않으면 손을 gripper 로 읽고 에러도 안 난다.
"""

import h5py
import pytest
from fixtures import DEFAULT_STREAMS, write_episode

from robomind_v2_utils.configs import load
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import read_streams


@pytest.fixture
def config():
    return load("tienyi")


def read(path, config):
    with h5py.File(path, "r") as handle:
        return read_streams(handle, config)


def test_puppet_becomes_state_and_master_becomes_action(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6)

    columns = read(path, config)

    assert "observation.states.arm_left_position" in columns
    assert "actions.arm_left_position" in columns
    assert columns["observation.states.arm_left_position"].shape == (6, 7)


def test_a_one_wide_stream_is_still_two_dimensional(tmp_path, config):
    """agilex 의 end_effector_left_position 은 파일에서 (T,) 1 차원으로 저장돼 있다.
    폭 1 짜리 열로 만들어야 뒤에서 다른 스트림과 같이 다룰 수 있다."""
    path = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=6)

    columns = read(path, config)

    assert columns["observation.states.end_effector_left_position"].shape == (6, 1)


def test_wrong_width_skips_the_episode(tmp_path, config):
    """ur config 로 ur_dex 데이터를 읽는 상황의 회귀 테스트."""
    wide = {**DEFAULT_STREAMS, "end_effector_left_position": 12}
    path = write_episode(tmp_path, "tienyi", "task", "0003_000000", streams=wide)

    with pytest.raises(EpisodeSkipped, match="end_effector_left_position is 12 wide, expected 1"):
        read(path, config)


def test_raw_streams_are_ignored(tmp_path, config):
    """sim 은 모든 스트림이 _align 과 _raw 두 벌이고 _raw 는 샘플 수가 두 배다.
    카메라에 맞춰진 것은 _align 뿐이다."""
    path = write_episode(tmp_path, "tienyi", "task", "0004_000000", frames=6, layout="sim")

    columns = read(path, config)

    assert all(value.shape[0] == 6 for value in columns.values())
    assert not any("_raw" in key for key in columns)


def test_mismatched_lengths_skip_the_episode(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0005_000000", frames=6)
    with h5py.File(path, "a") as handle:
        del handle["master/arm_right_position_align/data"]
        handle.create_dataset(
            "master/arm_right_position_align/data", data=[[0.0] * 7] * 5
        )

    with pytest.raises(EpisodeSkipped, match="length"):
        read(path, config)


def test_extras_keep_their_shape(tmp_path):
    """agilex_mobile 의 tactile 은 (T, 2, 6) 이다 — 평탄한 벡터가 아니다."""
    import yaml

    from robomind_v2_utils import configs

    body = {
        "robot_type": "probe",
        "cameras": {"camera_top": {"depth": True}},
        "streams": {"arm_left_position": {"width": 7}},
        "extra": {"tactile_observations": {"tactile_left": {"shape": [2, 6]}}},
        "instruction": {"source": "dirname"},
        "layout": "real",
    }
    (tmp_path / "probe.yaml").write_text(yaml.safe_dump(body))
    original, configs.CONFIG_DIR = configs.CONFIG_DIR, tmp_path
    try:
        probe = configs.load("probe")
    finally:
        configs.CONFIG_DIR = original

    path = write_episode(
        tmp_path,
        "probe",
        "task",
        "0006_000000",
        frames=6,
        streams={"arm_left_position": 7},
        extras={"tactile_observations": {"tactile_left": (2, 6)}},
    )

    columns = read(path, probe)

    assert columns["observation.tactile_left"].shape == (6, 2, 6)
