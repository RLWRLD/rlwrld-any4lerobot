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
