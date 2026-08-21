"""fps 는 에피소드마다 계산한다 — 전역 상수로 둘 수 없다.

실측: ur ~7 · franka 10-14 · tienkung 20-29 · tienyi ~48 · ark 56-61 ·
agilex_mobile 63-66 · agilex ~101 Hz. v1 은 30 으로 박아뒀다.

timestamp 가 초 단위 정수라 짧은 에피소드는 경계 오차가 크다. 그래서 계산은 하되
정밀도를 주장하지 않는다.
"""

import h5py
import pytest
import yaml
from fixtures import write_episode

from robomind_v2_utils import configs
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import episode_fps


def sim_config(tmp_path):
    body = {
        "robot_type": "probe_sim",
        "cameras": {"camera_top": {"depth": True}},
        "streams": {"arm_left_position": {"width": 7}},
        "instruction": {"source": "h5_metadata"},
        "layout": "sim",
        "fps": 30,
    }
    (tmp_path / "probe_sim.yaml").write_text(yaml.safe_dump(body))
    original, configs.CONFIG_DIR = configs.CONFIG_DIR, tmp_path
    try:
        return configs.load("probe_sim")
    finally:
        configs.CONFIG_DIR = original


def test_fps_is_frames_over_span(tmp_path):
    """101 프레임을 10 초에 걸쳐 찍었으면 ~10 Hz 다."""
    config = configs.load("tienyi")
    path = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=101, seconds=10)

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) == pytest.approx(10.1, abs=0.05)


def test_a_fast_episode_reads_high(tmp_path):
    """agilex 는 2,225 프레임 / 22 초 = ~101 Hz 다. 30 이 아니다."""
    config = configs.load("tienyi")
    path = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=203, seconds=2)

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) > 90


def test_sim_uses_the_config_because_its_clock_does_not_advance(tmp_path):
    config = sim_config(tmp_path)
    path = write_episode(
        tmp_path, "probe_sim", "task", "0003_000000", frames=8, seconds=0,
        streams={"arm_left_position": 7}, layout="sim",
    )

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) == 30.0


def test_a_real_episode_with_a_dead_clock_is_skipped(tmp_path):
    """real 인데 span 이 0 이면 계산할 근거가 없다 — 30 으로 추측하지 않는다."""
    config = configs.load("tienyi")
    path = write_episode(tmp_path, "tienyi", "task", "0004_000000", frames=8, seconds=0)

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="timestamps do not advance"):
            episode_fps(handle, config)


def test_too_few_frames_is_skipped(tmp_path):
    config = configs.load("tienyi")
    path = write_episode(tmp_path, "tienyi", "task", "0005_000000", frames=1, seconds=0)

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="too few frames"):
            episode_fps(handle, config)
