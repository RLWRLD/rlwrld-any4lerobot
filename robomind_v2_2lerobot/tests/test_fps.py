"""fps 는 에피소드마다 계산한다 — 전역 상수로 둘 수 없다.

실측: ur ~7 · franka 10-14 · tienkung 20-29 · tienyi ~48 · ark 56-61 ·
agilex_mobile 63-66 · agilex ~101 Hz. v1 은 30 으로 박아뒀다.

timestamp 가 초 단위 정수라 짧은 에피소드는 경계 오차가 크다. 그래서 계산은 하되
정밀도를 주장하지 않는다.

sim 은 단위가 다르다 — timestamp 가 초가 아니라 밀리초로 흐른다 (프레임 간 대략
33-34 씩, ~30 Hz). 예전엔 sim 시계가 아예 멈춰 있다고 보고 config 의 fps 로
대체했지만, 실측해보니 sim 도 흐르고 있었다 — 단위만 달랐다. 그래서 config 에는
이제 fps 필드 자체가 없다 (양쪽 layout 다). 실측 중 한 에피소드는 중간에 시계가
한 번 거꾸로(-2600 근처) 튀는 것도 발견됐다 — 양끝 차이(span)를 그대로 쓰면 이
튐 하나가 전체 경과 시간을 깎아먹으므로, sim 은 프레임 간 간격 중 양수만 골라
평균을 쓴다. 중앙값이 아니라 평균인 이유: timestamp 가 정수 밀리초라 33.333 처럼
참값이 정수가 아니면 33/34 를 오가는데, 중앙값은 그중 더 흔한 쪽(예: 33)을 그대로
답으로 내놓아 30.3030 Hz 처럼 +1% 오차를 체계적으로 만든다. 평균은 정수에 묶이지
않으므로 33.333 에 수렴해 30.0000 Hz 를 돌려준다.
"""

import h5py
import pytest
import yaml
from fixtures import write_episode

from robomind_v2_utils import configs
from robomind_v2_utils.errors import EpisodeSkipped
from robomind_v2_utils.reader import EpisodeRef, episode_fps, task_fps


def sim_config(tmp_path):
    body = {
        "robot_type": "probe_sim",
        "cameras": {"camera_top": {"depth": True}},
        "streams": {"arm_left_position": {"width": 7}},
        "instruction": {"source": "h5_metadata"},
        "layout": "sim",
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


def test_sim_fps_comes_from_millisecond_timestamps(tmp_path):
    """sim's clock is real, just finer-grained than real's -- milliseconds,
    stepping by about 33 rather than real's seconds stepping by 0 or 1
    (measured; see episode_fps's docstring). 30 frames at a clean 33 ms step
    is a ~30 Hz episode, not the ~0.03 Hz the old code computed by treating
    that span as seconds, and not a number invented by a config field either
    -- there is no fps field left to invent one from."""
    config = sim_config(tmp_path)
    path = write_episode(
        tmp_path, "probe_sim", "task", "0002_000000", frames=30,
        streams={"arm_left_position": 7}, layout="sim", milliseconds=33 * 29,
    )

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) == pytest.approx(1000 / 33, rel=1e-6)


def test_a_whole_millisecond_tick_yields_the_unbiased_rate(tmp_path):
    """This is not test_sim_fps_comes_from_millisecond_timestamps's scenario:
    that fixture's ground truth is a literal, exact 33 ms period (30.3030 Hz
    is the *correct* answer there). Here the ground truth period is 33.333 ms
    -- an exact 30.0000 Hz tick -- which cannot be stored as a constant
    whole-millisecond step at all: it has to land on 33 six times out of nine
    and 34 the other three to average out (see episode_fps's docstring). The
    old median-based estimate picked whichever of the two was more common --
    33 -- and returned a systematically fast 30.3030 Hz, indistinguishable
    from the other test's genuinely-33ms case even though the true rate here
    is different. Averaging the same nine steps instead recovers the true
    33.333 ms and the true 30.0000 Hz: the +1% bias this change removes.
    """
    config = sim_config(tmp_path)
    path = write_episode(
        tmp_path, "probe_sim", "task", "0002_000001", frames=10,
        streams={"arm_left_position": 7}, layout="sim", milliseconds=300,
    )

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) == pytest.approx(30.0)


def test_a_mid_episode_backward_jump_does_not_skew_the_sim_rate(tmp_path):
    """Synthetic mirror of the one release episode whose clock jumps backward
    once, mid-episode, amid an otherwise constant step (see episode_fps's
    docstring for why the mean of the *positive* steps is used for sim instead
    of the endpoint span). 10 frames at a clean 100 ms step (10 Hz) with a
    single -500 ms jump injected at frame 5: the endpoint span reads this as
    25 Hz (400 ms of apparent span for 10 frames); averaging only the positive
    steps reads through the one negative outlier to the true 10 Hz.
    """
    config = sim_config(tmp_path)
    path = write_episode(
        tmp_path, "probe_sim", "task", "0003_000000", frames=10,
        streams={"arm_left_position": 7}, layout="sim", milliseconds=900,
    )
    with h5py.File(path, "a") as handle:
        stamps = handle["camera_observations/timestamp"][()]
        stamps[5:] -= 500
        del handle["camera_observations/timestamp"]
        handle.create_dataset("camera_observations/timestamp", data=stamps)

    with h5py.File(path, "r") as handle:
        assert episode_fps(handle, config) == pytest.approx(10.0)


def test_a_sim_episode_with_a_dead_clock_is_skipped(tmp_path):
    """sim's clock can freeze too, same as real's -- and there is no config
    fallback left to invent a rate from (configs.py has no fps field at all
    now; see test_sim_fps_comes_from_millisecond_timestamps for the case
    where it does advance)."""
    config = sim_config(tmp_path)
    path = write_episode(
        tmp_path, "probe_sim", "task", "0004_000000", frames=8, milliseconds=0,
        streams={"arm_left_position": 7}, layout="sim",
    )

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="timestamps do not advance"):
            episode_fps(handle, config)


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


def test_missing_timestamp_is_skipped(tmp_path):
    """An empty file with no camera_observations/timestamp raises EpisodeSkipped."""
    config = configs.load("tienyi")
    path = write_episode(tmp_path, "tienyi", "task", "0006_000000", frames=0, broken="empty")

    with h5py.File(path, "r") as handle:
        with pytest.raises(EpisodeSkipped, match="camera_observations/timestamp is missing"):
            episode_fps(handle, config)


def _refs(paths, embodiment="tienyi", task="task"):
    return [EpisodeRef(embodiment=embodiment, task=task, path=path) for path in paths]


def test_task_fps_is_the_median_of_its_episodes(tmp_path):
    """One dataset is opened per task at a single fps, so a task with more than
    one episode needs one rate for all of them -- the median of what each
    episode actually measured, not whichever happens to be looked at first
    (see convert_task and the I1 finding this closes). Two ~10 Hz episodes and
    one much faster one: the median is the middle value, not pulled toward the
    outlier the way a mean would be.
    """
    config = configs.load("tienyi")
    slow_a = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=101, seconds=10)
    slow_b = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=101, seconds=10)
    fast = write_episode(tmp_path, "tienyi", "task", "0003_000000", frames=203, seconds=2)

    rate = task_fps(_refs([slow_a, slow_b, fast]), config)

    assert rate == pytest.approx(10.1, abs=0.05)


def test_task_fps_excludes_a_rate_that_rounds_to_zero(tmp_path):
    """A rate this slow can't be a dataset's fps on its own -- convert_task holds
    a single surviving episode to the same floor (round(fps) >= 1) -- so it must
    not get to drag the task's median toward zero either.
    """
    config = configs.load("tienyi")
    slow = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=2, seconds=10)
    normal = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2)

    rate = task_fps(_refs([slow, normal]), config)

    assert rate == pytest.approx(3.0, abs=0.1)


def test_task_fps_skips_a_file_it_cannot_open(tmp_path):
    """A ref this function can't even open contributes nothing to the median --
    read_episode hits the same file again for real and logs the specific
    reason when the caller actually processes it.
    """
    config = configs.load("tienyi")
    good = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)
    missing = tmp_path / "does_not_exist.hdf5"

    rate = task_fps(_refs([missing, good]), config)

    assert rate == pytest.approx(3.0, abs=0.1)


def test_task_fps_is_none_when_no_episode_is_measurable(tmp_path):
    config = configs.load("tienyi")
    dead_clock = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=8, seconds=0)

    assert task_fps(_refs([dead_clock]), config) is None


def test_task_fps_excludes_episodes_shorter_than_min_frames(tmp_path):
    """An episode this short will be skipped by convert_task's own min_frames
    check regardless of its rate, and a broken-recording-length episode's own
    timestamp span is exactly the least trustworthy measurement (see
    episode_fps's docstring on the boundary error a short `real` episode
    carries) -- so it must not get a vote in the task's median either. Without
    threading min_frames through, a 2-frame episode measuring 1.0 Hz here would
    have pulled the two-episode median down to 2.0 Hz instead of the 6-frame
    episode's own 3.0 Hz -- a real gap this test pins shut.
    """
    config = configs.load("tienyi")
    too_short = write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=2, seconds=2)
    normal = write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2)

    rate = task_fps(_refs([too_short, normal]), config, min_frames=3)

    assert rate == pytest.approx(3.0, abs=0.1)
