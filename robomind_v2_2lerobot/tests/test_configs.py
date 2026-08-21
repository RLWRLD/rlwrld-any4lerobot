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
    assert not hasattr(config, "fps")
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


@pytest.mark.parametrize("layout", ["real", "sim"])
def test_fps_is_rejected_as_an_unknown_key_under_either_layout(tmp_path, monkeypatch, layout):
    """이전 규칙: sim 은 fps 를 필수로 요구하고 real 은 금지했다 — sim 시계가
    멈춰 있어 계산이 불가능하다고 믿었기 때문이다. 실측해보니 sim 도 흐르고
    있었다(단위만 밀리초로 다르다, reader.episode_fps 참고). 그래서 fps 는
    이제 계산되는 값이지 config 가 적는 값이 아니다 — 어느 layout 이든 이 키
    자체가 알 수 없는 키다."""
    name = write(tmp_path, monkeypatch, {**VALID, "layout": layout, "fps": 30})
    with pytest.raises(ConfigError, match="unknown key 'fps'"):
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
    assert not hasattr(config, "fps")


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
    "franka": 6, "agilex": 3, "agilex_mobile": 3, "ark": 3, "ark_mobile": 3, "tienkung": 1,
    "ur": 6, "ur_dex": 6, "tienyi": 1, "tienyi_mobile": 1,
}
ARM_WIDTHS = {
    "franka": 8, "agilex": 6, "agilex_mobile": 6, "ark": 6, "ark_mobile": 6, "tienkung": 7,
    "ur": 6, "ur_dex": 6, "tienyi": 7, "tienyi_mobile": 7,
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


def test_agilex_mobile_adds_chassis_and_tactile():
    config = load("agilex_mobile")

    assert config.stream("chassis_pose").width == 7
    assert config.stream("chassis_twist").width == 6
    assert {extra.name for extra in config.extras} == {"tactile_left", "tactile_right"}
    assert all(extra.group == "tactile_observations" for extra in config.extras)
    assert all(extra.shape == (2, 6) for extra in config.extras)


def test_ark_mobile_adds_chassis_but_no_tactile():
    config = load("ark_mobile")

    assert config.stream("chassis_pose").width == 7
    assert config.stream("chassis_twist").width == 6
    assert config.extras == ()


def test_tienyi_mobile_adds_head_position_and_only_twist():
    """tienyi_mobile 만 head_position 을 갖고, chassis 는 twist 만 있다."""
    config = load("tienyi_mobile")

    assert config.stream("head_position") is not None
    assert config.stream("head_position").width == 3
    assert config.stream("chassis_twist").width == 6
    assert config.stream("chassis_pose") is None


def test_tactile_becomes_a_two_dimensional_feature():
    """평탄한 벡터가 아니다 — build_features 가 shape 을 그대로 넘겨야 한다."""
    from robomind_v2_utils.lerobot_utils import build_features

    config = load("agilex_mobile")
    features = build_features(config, {})

    assert features["observation.tactile_left"]["shape"] == (2, 6)
    assert features["observation.tactile_left"]["dtype"] == "float32"
    assert features["observation.tactile_right"]["shape"] == (2, 6)
    assert features["observation.tactile_right"]["dtype"] == "float32"


NO_POSE_EMBODIMENTS = {"tienkung", "tienyi", "tienyi_mobile"}


@pytest.mark.parametrize("name", sorted(NO_POSE_EMBODIMENTS))
def test_tien_kung_family_declares_no_end_effector_pose(name):
    """실측: 코퍼스 12 종 중 이 3 종만 end_effector_*_pose 데이터셋이 실물에서
    (0,) 로 비어 있어서, config 가 그 스트림을 아예 선언하지 않는다. 선언하면
    (Task 11 실행 기록대로) 전 에피소드가 skip 된다. 이 비대칭은 실측이지 결함이
    아니다 -- 이 테스트가 없으면 나중에 '빠진 것 같으니 채운다'로 조용히
    되돌려질 수 있다."""
    config = load(name)

    assert config.stream("end_effector_left_pose") is None
    assert config.stream("end_effector_right_pose") is None


def test_robot_outside_that_group_still_declares_end_effector_pose():
    """대조군: Tien Kung 계열이 아닌 embodiment 는 그대로 선언한다."""
    config = load("agilex")

    assert config.stream("end_effector_left_pose") is not None
    assert config.stream("end_effector_right_pose") is not None
