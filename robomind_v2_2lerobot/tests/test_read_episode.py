"""에피소드 하나가 LeRobot 이 받을 프레임 목록이 된다."""

import numpy as np
import pytest
from fixtures import write_episode

from robomind_v2_utils.configs import load
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import EpisodeRef, read_episode


@pytest.fixture
def config():
    return load("tienyi")


def ref_for(path, task="flip_white_cup_upright"):
    return EpisodeRef(embodiment="tienyi", task=task, path=path)


def test_one_frame_per_camera_sample(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)

    episode = read_episode(ref_for(path), config)

    assert len(episode.frames) == 6


def test_a_frame_carries_images_states_and_actions(tmp_path, config):
    path = write_episode(
        tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2, resolution=(48, 64)
    )

    frame = read_episode(ref_for(path), config).frames[0]

    assert frame["observation.images.camera_top"].shape == (48, 64, 3)
    assert frame["observation.states.arm_left_position"].shape == (7,)
    assert frame["actions.arm_left_position"].shape == (7,)
    assert frame["task"] == "flip white cup upright"


def test_depth_is_left_out_unless_asked_for(tmp_path, config):
    """depth 는 embodiment 전부에 있지만 기본으로는 안 쓴다 — 용량이 크다."""
    path = write_episode(tmp_path, "tienyi", "task", "0003_000000", frames=4, seconds=2)

    without = read_episode(ref_for(path), config).frames[0]
    with_depth = read_episode(ref_for(path), config, save_depth=True).frames[0]

    assert "observation.images.camera_top_depth" not in without
    assert with_depth["observation.images.camera_top_depth"].shape == (48, 64, 1)


def test_fps_and_shapes_travel_with_the_episode(tmp_path, config):
    path = write_episode(
        tmp_path, "tienyi", "task", "0004_000000", frames=101, seconds=10, resolution=(48, 64)
    )

    episode = read_episode(ref_for(path), config)

    assert episode.fps == pytest.approx(10.1, abs=0.05)
    assert episode.shapes["observation.images.camera_top"] == (48, 64, 3)


def test_a_broken_episode_raises_skipped(tmp_path, config):
    path = write_episode(tmp_path, "tienyi", "task", "0005_000000", broken="empty")

    with pytest.raises(EpisodeSkipped):
        read_episode(ref_for(path), config)


def test_state_and_action_differ(tmp_path, config):
    """puppet 과 master 를 같은 배열로 채우는 실수를 잡는다."""
    path = write_episode(tmp_path, "tienyi", "task", "0006_000000", frames=4, seconds=2)

    frame = read_episode(ref_for(path), config).frames[0]

    assert frame["observation.states.arm_left_position"].dtype == np.float32
    assert frame["actions.arm_left_position"].dtype == np.float32
