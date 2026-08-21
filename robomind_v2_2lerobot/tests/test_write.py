"""합성 에피소드 하나가 실제로 LeRobot 데이터셋이 되는지 끝까지 확인한다.

``configs``/``reader``/``images``/``build_features`` 는 각자 단위 테스트가 있지만,
여기서 처음으로 ``add_frame``/``save_episode`` 를 실제 ``RoboMINDv2Dataset`` 에 대고
불러본다 -- ffmpeg 인코딩까지 포함해서. 카메라가 있는 데이터셋은 전부 이 경로를
지나가므로, ``sample_images`` 가 ``add_frame`` 이 디스크에 남기는 경로 목록을
못 읽으면 여기서 바로 드러난다.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from fixtures import write_episode

from robomind_v2_utils import configs
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

    # The port deliberately keeps a single `train` split and drops v1's `validation`
    # bookkeeping (see the module docstring). This is the one line that removal left
    # behind, and it now comes from `LeRobotDatasetMetadata.save_episode` -- inherited,
    # not overridden (see `RoboMINDv2DatasetMetadata`'s docstring) -- so this also pins
    # that the inherited method produces exactly what the port intends.
    assert on_disk["splits"] == {"train": "0:1"}

    # `compute_episode_stats` is the one place normalisation (divide by 255) and the
    # channel-first reduction both happen; a mistake there is quiet, not loud. Checking
    # the camera key's mean shape catches a wrong reduction axis, and the [0, 1] bound
    # catches a skipped or doubled normalisation.
    stats = json.loads((root / "meta" / "stats.json").read_text())
    camera_mean = np.array(stats["observation.images.camera_top"]["mean"])
    assert camera_mean.shape == (3, 1, 1)
    assert np.all(camera_mean >= 0.0) and np.all(camera_mean <= 1.0)

    # `action_config` was the second thing this port authorises dropping (see the
    # module docstring) but, unlike `split`, nothing pinned that it actually stayed
    # gone from what lands on disk. Check every episode-metadata column, not just the
    # in-memory dict, since a stray key could in principle survive only the parquet
    # write path.
    episode_files = list((root / "meta" / "episodes").rglob("*.parquet"))
    assert episode_files, "expected an episode-metadata parquet file on disk"
    columns = pq.read_table(episode_files[0]).column_names
    assert not any(name == "action_config" or name.startswith("action_config.") for name in columns)


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


def two_camera_config(tmp_path):
    """A throwaway two-camera embodiment, built the way the other tests in this
    suite build one: write a YAML body, point `configs.CONFIG_DIR` at `tmp_path`
    for the load, then restore it. Both cameras carry depth so the parallel
    encoding test below can check for both depth keys landing alongside both
    video keys.
    """
    body = {
        "robot_type": "probe_multi",
        "cameras": {
            "camera_a": {"depth": True},
            "camera_b": {"depth": True},
        },
        "streams": {"arm_left_position": {"width": 7}},
        "instruction": {"source": "dirname"},
        "layout": "real",
    }
    (tmp_path / "probe_multi.yaml").write_text(yaml.safe_dump(body))
    original, configs.CONFIG_DIR = configs.CONFIG_DIR, tmp_path
    try:
        return configs.load("probe_multi")
    finally:
        configs.CONFIG_DIR = original


def test_two_cameras_with_depth_exercise_the_parallel_encoding_branch(tmp_path):
    """``RoboMINDv2DatasetWriter.save_episode``'s ``ProcessPoolExecutor`` branch only
    runs when ``parallel_encoding`` (default ``True``) and ``num_cameras > 1``. Every
    other test in this file uses one camera, so only the sequential branch has ever
    run. Real robots typically carry more than one camera, so the parallel branch is
    the common production path, not an edge case, and this is the only test that
    actually exercises it.
    """
    config = two_camera_config(tmp_path)
    path = write_episode(
        tmp_path,
        "probe_multi",
        "task",
        "0001_000000",
        frames=4,
        seconds=1,
        resolution=(32, 32),
        cameras=("camera_a", "camera_b"),
    )
    episode = read_episode(
        EpisodeRef(embodiment="probe_multi", task="task", path=path), config, save_depth=True
    )
    features = build_features(config, episode.shapes)

    root = tmp_path / "dataset"
    dataset = RoboMINDv2Dataset.create(
        repo_id="test/multi_camera",
        fps=round(episode.fps),
        features=features,
        root=root,
        robot_type=config.robot_type,
        use_videos=True,
    )
    for frame in episode.frames:
        dataset.add_frame(frame)
    dataset.save_episode()  # parallel_encoding=True by default; num_cameras == 2 here
    dataset.finalize()

    feature_keys = set(dataset.meta.features)
    assert "observation.images.camera_a" in feature_keys
    assert "observation.images.camera_b" in feature_keys
    assert "observation.images.camera_a_depth" in feature_keys
    assert "observation.images.camera_b_depth" in feature_keys

    on_disk = json.loads((root / "meta" / "info.json").read_text())
    assert "observation.images.camera_a" in on_disk["features"]
    assert "observation.images.camera_b" in on_disk["features"]
    assert "observation.images.camera_a_depth" in on_disk["features"]
    assert "observation.images.camera_b_depth" in on_disk["features"]

    assert len(list(root.rglob("*.mp4"))) == 2, "expected two real encoded video files on disk"
