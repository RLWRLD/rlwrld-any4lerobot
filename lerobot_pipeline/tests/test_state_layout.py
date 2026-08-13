"""The layout step against a small synthetic dataset.

Three episodes is enough: the step is per-row arithmetic driven by the slot map, so
what needs checking is that the right column lands in the right slot, that video is
carried through untouched, and that the metadata beside the vector agrees with it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import load  # noqa: E402
from lerobot_pipeline.registry import build_step  # noqa: E402
from lerobot_pipeline.steps.state_layout import LayoutError  # noqa: E402

pd = pytest.importorskip("pandas")

ROWS_PER_EPISODE = 5
EPISODES = 3


@pytest.fixture
def dataset(tmp_path) -> Path:
    """An ActionNet-shaped dataset carrying source features but no state/action."""
    root = tmp_path / "converted"
    (root / "meta").mkdir(parents=True)
    (root / "videos" / "chunk-000").mkdir(parents=True)

    features = {
        "observation.images.primary": {"dtype": "video", "shape": [800, 1280, 3]},
        "observation.robot_joints": {"dtype": "float32", "shape": [32]},
        "observation.hand_joints": {"dtype": "float32", "shape": [12]},
        "action.robot_joints": {"dtype": "float32", "shape": [32]},
        "action.hand_joints": {"dtype": "float32", "shape": [12]},
    }
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30, "features": features})
    )
    (root / "videos" / "chunk-000" / "file-000.mp4").write_bytes(b"video bytes")

    rng = np.random.default_rng(7)
    for episode in range(EPISODES):
        rows = {
            name: [
                rng.random(spec["shape"][0]).astype(np.float32)
                for _ in range(ROWS_PER_EPISODE)
            ]
            for name, spec in features.items()
            if spec["dtype"] != "video"
        }
        directory = root / "data" / "chunk-000"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(directory / f"file-{episode:03d}.parquet")
    return root


@pytest.fixture
def step():
    return build_step({"type": "state_layout", "dataset": "action_net"})


class TestAssembly:
    def test_every_slot_takes_its_declared_source_column(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)

        slots = load("action_net").state.slot_map("state")
        before = pd.read_parquet(dataset / "data" / "chunk-000" / "file-000.parquet")
        after = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")

        for row in range(ROWS_PER_EPISODE):
            state = after["observation.state"].iloc[row]
            assert len(state) == 44
            for slot, source in enumerate(slots):
                name, column = source
                assert state[slot] == pytest.approx(before[name].iloc[row][column])

    def test_action_uses_the_action_side_sources(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        before = pd.read_parquet(dataset / "data" / "chunk-000" / "file-000.parquet")
        after = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
        # slot 0 is left_arm[0] <- robot_joints[18], on the action side
        assert after["action"].iloc[0][0] == pytest.approx(
            before["action.robot_joints"].iloc[0][18]
        )

    def test_source_features_are_kept_alongside(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        after = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
        assert "observation.robot_joints" in after.columns

    def test_every_episode_is_rewritten(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        files = sorted((out / "data" / "chunk-000").glob("*.parquet"))
        assert len(files) == EPISODES
        for path in files:
            assert "observation.state" in pd.read_parquet(path).columns


class TestVideoIsUntouched:
    def test_video_is_hard_linked_not_copied(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        source = dataset / "videos" / "chunk-000" / "file-000.mp4"
        target = out / "videos" / "chunk-000" / "file-000.mp4"
        assert target.read_bytes() == b"video bytes"
        assert target.stat().st_ino == source.stat().st_ino

    def test_the_source_dataset_is_not_modified(self, step, dataset, tmp_path):
        before = pd.read_parquet(dataset / "data" / "chunk-000" / "file-000.parquet")
        step.apply(dataset, tmp_path / "out")
        after = pd.read_parquet(dataset / "data" / "chunk-000" / "file-000.parquet")
        assert list(before.columns) == list(after.columns)


class TestMetadata:
    def test_info_gains_the_assembled_features(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        features = json.loads((out / "meta" / "info.json").read_text())["features"]
        assert features["observation.state"]["shape"] == [44]
        assert features["action"]["names"]["motors"][:2] == ["m0", "m1"]

    def test_modality_matches_the_vector_width(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        step.apply(dataset, out)
        modality = json.loads((out / "meta" / "modality.json").read_text())
        assert modality["state"] == {"state": {"start": 0, "end": 44}}
        assert modality["action"]["action"]["absolute"] is True
        assert modality["video"] == {"primary": {"original_key": "observation.images.primary"}}


class TestRefusal:
    def test_a_dataset_with_unrecovered_columns_is_refused(self):
        """Galaxea's 26-wide action was never recovered. Emitting zeros for it would
        train quietly on a blank quarter of the vector, so the step will not build."""
        with pytest.raises(LayoutError, match="unrecovered"):
            build_step({"type": "state_layout", "dataset": "galaxea"})

    def test_a_dataset_with_no_sources_at_all_is_refused(self):
        with pytest.raises(LayoutError, match="no source column"):
            build_step({"type": "state_layout", "dataset": "neural_robocurate"})

    def test_a_missing_source_feature_is_reported(self, step, dataset, tmp_path):
        info_path = dataset / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        del info["features"]["observation.hand_joints"]
        info_path.write_text(json.dumps(info))
        with pytest.raises(LayoutError, match="observation.hand_joints"):
            step.apply(dataset, tmp_path / "out")

    def test_it_will_not_overwrite_an_existing_output(self, step, dataset, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(LayoutError, match="already exists"):
            step.apply(dataset, out)
