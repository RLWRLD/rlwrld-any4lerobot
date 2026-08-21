"""합성 에피소드 하나가 실제로 LeRobot 데이터셋이 되는지 끝까지 확인한다.

``configs``/``reader``/``images``/``build_features`` 는 각자 단위 테스트가 있지만,
여기서 처음으로 ``add_frame``/``save_episode`` 를 실제 ``RoboMINDv2Dataset`` 에 대고
불러본다 -- ffmpeg 인코딩까지 포함해서. 카메라가 있는 데이터셋은 전부 이 경로를
지나가므로, ``sample_images`` 가 ``add_frame`` 이 디스크에 남기는 경로 목록을
못 읽으면 여기서 바로 드러난다.
"""

import json

import pytest
from fixtures import write_episode

from robomind_v2_utils.configs import load
from robomind_v2_utils.lerobot_utils import RoboMINDv2Dataset, build_features
from robomind_v2_utils.reader import EpisodeRef, read_episode


@pytest.fixture
def config():
    return load("tienyi")


def test_a_synthetic_episode_becomes_a_real_lerobot_dataset(tmp_path, config):
    path = write_episode(
        tmp_path, "tienyi", "task", "0001_000000", frames=4, seconds=1, resolution=(32, 32)
    )
    episode = read_episode(EpisodeRef(embodiment="tienyi", task="task", path=path), config)
    features = build_features(config, episode.shapes)

    root = tmp_path / "dataset"
    dataset = RoboMINDv2Dataset.create(
        repo_id="test/write",
        fps=round(episode.fps),
        features=features,
        root=root,
        robot_type=config.robot_type,
        use_videos=True,
    )
    for frame in episode.frames:
        dataset.add_frame(frame)
    # This is where the bug bites: `add_frame` left `observation.images.camera_top`
    # as a list of on-disk paths, and `save_episode` -> `compute_episode_stats` ->
    # `sample_images` has to read them back. Without the path branch this raises
    # `AttributeError: 'list' object has no attribute 'ndim'`.
    dataset.save_episode()
    dataset.finalize()

    assert dataset.meta.info.total_episodes == 1
    assert dataset.meta.info.total_frames == len(episode.frames)

    feature_keys = set(dataset.meta.features)
    assert "observation.images.camera_top" in feature_keys
    assert "observation.states.arm_left_position" in feature_keys
    assert "actions.arm_left_position" in feature_keys
    assert "observation.state" not in feature_keys
    assert "action" not in feature_keys

    # Re-read straight off disk, not just the in-memory object, so this proves the
    # dataset actually landed rather than that the Python object merely looks right.
    on_disk = json.loads((root / "meta" / "info.json").read_text())
    assert on_disk["total_episodes"] == 1
    assert on_disk["total_frames"] == len(episode.frames)
    assert "observation.images.camera_top" in on_disk["features"]
    assert "observation.state" not in on_disk["features"]
    assert "action" not in on_disk["features"]

    assert list(root.rglob("*.mp4")), "expected a real encoded video file on disk"


def test_a_depth_feature_writes_successfully_end_to_end(tmp_path, config):
    """``tienyi``'s one camera carries depth (see its config). Before the writer's
    stacking loop also skipped ``image``-dtype keys (not just ``video``), a depth key's
    list of on-disk paths got stacked into an ndarray of path *strings*, which
    `sample_images` then fed to its ndarray branch -- `input[:, None, :, :]` on a 1-D
    array of strings raises `IndexError: too many indices for array`. This test asks
    for depth explicitly (`save_depth=True`) so that path is actually exercised.
    """
    path = write_episode(
        tmp_path, "tienyi", "task", "0001_000000", frames=4, seconds=1, resolution=(32, 32)
    )
    episode = read_episode(
        EpisodeRef(embodiment="tienyi", task="task", path=path), config, save_depth=True
    )
    features = build_features(config, episode.shapes)
    assert features["observation.images.camera_top_depth"]["dtype"] == "image"

    root = tmp_path / "dataset"
    dataset = RoboMINDv2Dataset.create(
        repo_id="test/depth",
        fps=round(episode.fps),
        features=features,
        root=root,
        robot_type=config.robot_type,
        use_videos=True,
    )
    for frame in episode.frames:
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()

    feature_keys = set(dataset.meta.features)
    assert "observation.images.camera_top" in feature_keys
    assert "observation.images.camera_top_depth" in feature_keys

    on_disk = json.loads((root / "meta" / "info.json").read_text())
    assert on_disk["total_episodes"] == 1
    assert "observation.images.camera_top_depth" in on_disk["features"]
