"""embodiment config 는 데이터다. 로더는 그것을 엄격하게 읽는다.

오타 하나가 조용히 무시되면 잘못된 폭으로 변환이 돌아간다 — v1 이 gripper 를
12-DoF 손으로 오해할 수 있었던 것과 같은 종류의 실패다. 그래서 모르는 키는 에러다.
"""

import pytest
import yaml

from robomind_v2_utils.configs import ConfigError, available, load, load_all


def test_tienyi_loads():
    config = load("tienyi")
    assert config.embodiment == "tienyi"
    assert config.robot_type == "tienyi"
    assert config.layout == "real"
    assert config.fps is None
    assert [camera.name for camera in config.cameras] == ["camera_top"]
    assert config.cameras[0].depth is True
    assert config.stream("arm_left_position").width == 7
    assert config.stream("end_effector_left_position").width == 1
    assert config.instruction_source == "zh_file"
    assert config.extras == ()


def test_unknown_stream_returns_none():
    assert load("tienyi").stream("chassis_pose") is None


def test_available_lists_config_stems():
    assert "tienyi" in available()


def test_load_all_loads_every_config():
    assert {config.embodiment for config in load_all()} == set(available())


def test_unknown_name_is_an_error():
    with pytest.raises(ConfigError, match="unknown embodiment 'nope'"):
        load("nope")


def test_path_traversal_is_an_error():
    with pytest.raises(ConfigError, match="unknown embodiment"):
        load("../tienyi")


def write(tmp_path, monkeypatch, body: dict, name: str = "probe"):
    from robomind_v2_utils import configs

    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body))
    monkeypatch.setattr(configs, "CONFIG_DIR", tmp_path)
    return name


VALID = {
    "robot_type": "probe",
    "cameras": {"camera_top": {"depth": True}},
    "streams": {"arm_left_position": {"width": 7}},
    "instruction": {"source": "zh_file"},
    "layout": "real",
}


def test_valid_minimal_config(tmp_path, monkeypatch):
    name = write(tmp_path, monkeypatch, VALID)
    assert load(name).robot_type == "probe"


def test_unknown_top_level_key_is_an_error(tmp_path, monkeypatch):
    name = write(tmp_path, monkeypatch, {**VALID, "camerass": {}})
    with pytest.raises(ConfigError, match="unknown key 'camerass'"):
        load(name)


def test_missing_cameras_is_an_error(tmp_path, monkeypatch):
    body = {key: value for key, value in VALID.items() if key != "cameras"}
    name = write(tmp_path, monkeypatch, body)
    with pytest.raises(ConfigError, match="cameras must not be empty"):
        load(name)


def test_bad_instruction_source_is_an_error(tmp_path, monkeypatch):
    name = write(tmp_path, monkeypatch, {**VALID, "instruction": {"source": "guess"}})
    with pytest.raises(ConfigError, match="instruction.source must be one of"):
        load(name)


def test_bad_layout_is_an_error(tmp_path, monkeypatch):
    name = write(tmp_path, monkeypatch, {**VALID, "layout": "half"})
    with pytest.raises(ConfigError, match="layout must be one of"):
        load(name)


def test_zero_width_stream_is_an_error(tmp_path, monkeypatch):
    body = {**VALID, "streams": {"arm_left_position": {"width": 0}}}
    name = write(tmp_path, monkeypatch, body)
    with pytest.raises(ConfigError, match="width must be >= 1"):
        load(name)


def test_sim_layout_requires_fps(tmp_path, monkeypatch):
    """sim 에는 시계가 없다 — timestamp span 이 0 이라 fps 를 계산할 수 없다.
    그래서 sim 만 fps 를 config 에 적고, 빠뜨리면 에러다."""
    name = write(tmp_path, monkeypatch, {**VALID, "layout": "sim"})
    with pytest.raises(ConfigError, match="layout: sim requires fps"):
        load(name)


def test_real_layout_rejects_fps(tmp_path, monkeypatch):
    """real 의 fps 는 에피소드마다 다르다(7~101 Hz). 하나로 못 적는다."""
    name = write(tmp_path, monkeypatch, {**VALID, "fps": 30})
    with pytest.raises(ConfigError, match="layout: real must not set fps"):
        load(name)


def test_extras_shape_becomes_a_tuple(tmp_path, monkeypatch):
    body = {
        **VALID,
        "extra": {"tactile_observations": {"tactile_left": {"shape": [2, 6]}}},
    }
    name = write(tmp_path, monkeypatch, body)
    (extra,) = load(name).extras
    assert (extra.group, extra.name, extra.shape) == (
        "tactile_observations",
        "tactile_left",
        (2, 6),
    )


def test_ur_has_six_cameras_and_a_one_wide_gripper():
    """ur 과 ur_dex 는 스트림 이름이 같고 폭만 다르다. 그 차이를 config 가 들고 있다."""
    config = load("ur")

    assert len(config.cameras) == 6
    assert [camera.name for camera in config.cameras] == [
        "camera_front", "camera_left", "camera_right",
        "camera_top", "camera_wrist_left", "camera_wrist_right",
    ]
    assert config.stream("arm_left_position").width == 6
    assert config.stream("end_effector_left_position").width == 1
    assert config.stream("end_effector_left_pose").width == 7
    assert config.layout == "real"
    assert config.fps is None


def test_ur_dex_differs_from_ur_only_in_end_effector_width():
    """이름은 같고 폭만 다르다 — 이 데이터셋의 가장 조용한 함정이다."""
    ur = load("ur")
    dex = load("ur_dex")

    assert {stream.name for stream in ur.streams} == {stream.name for stream in dex.streams}
    assert {camera.name for camera in ur.cameras} == {camera.name for camera in dex.cameras}
    assert dex.stream("end_effector_left_position").width == 12
    assert dex.stream("end_effector_right_position").width == 12
    assert dex.stream("arm_left_position").width == 6


def test_ur_dex_has_no_description_file():
    """이 repo 에는 zh_description.txt 가 하나도 없다 — 디렉토리 이름뿐이다."""
    assert load("ur_dex").instruction_source == "dirname"


CAMERA_COUNTS = {
    "franka": 6, "agilex": 3, "ark": 3, "tienkung": 1,
    "ur": 6, "ur_dex": 6, "tienyi": 1,
}
ARM_WIDTHS = {
    "franka": 8, "agilex": 6, "ark": 6, "tienkung": 7,
    "ur": 6, "ur_dex": 6, "tienyi": 7,
}


@pytest.mark.parametrize("name", sorted(CAMERA_COUNTS))
def test_measured_camera_count(name):
    assert len(load(name).cameras) == CAMERA_COUNTS[name]


@pytest.mark.parametrize("name", sorted(ARM_WIDTHS))
def test_measured_arm_width(name):
    assert load(name).stream("arm_left_position").width == ARM_WIDTHS[name]
    assert load(name).stream("arm_right_position").width == ARM_WIDTHS[name]


def test_ark_cameras_are_left_right_top_not_front():
    """ark 는 카메라 3 대인데 이름이 agilex 와 다르다."""
    assert [camera.name for camera in load("ark").cameras] == [
        "camera_left", "camera_right", "camera_top"
    ]


def test_agilex_cameras_are_front_left_right():
    assert [camera.name for camera in load("agilex").cameras] == [
        "camera_front", "camera_left", "camera_right"
    ]
