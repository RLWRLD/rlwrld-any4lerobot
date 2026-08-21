"""LeRobot feature 사전은 config 와 실측 shape 에서 나온다.

이름은 v1 의 관례를 지킨다 — RoboMIND 의 원본 필드 이름을 그대로 쓴다:
observation.states.<name> / actions.<name>. state 를 조립하지 않는다.
"""

import pytest

from robomind_v2_utils.configs import load
from robomind_v2_utils.lerobot_utils import build_features


@pytest.fixture
def config():
    return load("tienyi")


SHAPES = {
    "observation.images.camera_top": (480, 640, 3),
    "observation.images.camera_top_depth": (480, 640, 1),
}


def test_a_camera_becomes_a_video_feature(config):
    features = build_features(config, SHAPES)

    entry = features["observation.images.camera_top"]
    assert entry["dtype"] == "video"
    assert entry["shape"] == (480, 640, 3)
    assert entry["names"] == ["height", "width", "rgb"]


def test_depth_is_an_image_feature_not_video(config):
    """depth 는 인코딩하지 않는다 — 손실 압축이 거리값을 망친다."""
    features = build_features(config, SHAPES)

    entry = features["observation.images.camera_top_depth"]
    assert entry["dtype"] == "image"
    assert entry["shape"] == (480, 640, 1)
    assert entry["names"] == ["height", "width", "channel"]


def test_depth_is_absent_when_no_shape_was_measured(config):
    features = build_features(config, {"observation.images.camera_top": (480, 640, 3)})

    assert "observation.images.camera_top_depth" not in features


def test_streams_become_state_and_action_vectors(config):
    features = build_features(config, SHAPES)

    state = features["observation.states.arm_left_position"]
    assert state["dtype"] == "float32"
    assert state["shape"] == (7,)
    assert features["actions.arm_left_position"]["shape"] == (7,)


def test_a_one_wide_stream_keeps_width_one(config):
    features = build_features(config, SHAPES)

    assert features["observation.states.end_effector_left_position"]["shape"] == (1,)


def test_no_assembled_state_or_action_is_emitted(config):
    """RoboMIND 는 원본 이름을 보존한다. 평탄한 observation.state 를 만들지 않는다."""
    features = build_features(config, SHAPES)

    assert "observation.state" not in features
    assert "action" not in features
