"""The spec-driven reader against the hand-written one it replaces.

``actionnet2lerobot`` read ActionNet with the dataset's paths, keys and joint
counts written into Python. This reads the same bytes with all of that coming from
``datasets/action_net.yaml``. The two must agree on every row, or the move from
code to data silently changed the dataset.

The fixture is synthetic rather than a real episode: 60 Hz robot against 30 Hz
camera, with the two conditions the clock strategy exists to handle -- a stretch of
non-advancing timestamps, and video frames recorded past the last robot sample.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import load  # noqa: E402
from spec2lerobot import EpisodeSkipped, build_reader  # noqa: E402
from spec2lerobot.clocks import align, parse_times  # noqa: E402

h5py = pytest.importorskip("h5py")

EPISODE_ID = "01JQ0000000000000000000000"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S_%f"
ROBOT_JOINTS = 32
HAND_JOINTS = 12


def iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(TIMESTAMP_FORMAT)


@pytest.fixture
def source(tmp_path) -> Path:
    """One ActionNet-shaped episode on disk."""
    rng = np.random.default_rng(0)
    base = 1_735_000_000.0

    robot_times = base + np.arange(120) / 60.0
    # a stall: four samples that do not advance, which the strategy must drop
    robot_times[40:44] = robot_times[40]
    # video runs a little past the robot, so the tail frames fall outside the range
    image_times = base + np.arange(64) / 30.0

    with h5py.File(tmp_path / f"{EPISODE_ID}.hdf5", "w") as handle:
        handle["timestamp"] = robot_times
        handle["state/robot"] = rng.random((120, ROBOT_JOINTS), dtype=np.float32)
        handle["state/hand"] = rng.random((120, HAND_JOINTS), dtype=np.float32)
        handle["action/robot"] = rng.random((120, ROBOT_JOINTS), dtype=np.float32)
        handle["action/hand"] = rng.random((120, HAND_JOINTS), dtype=np.float32)

    camera = tmp_path / EPISODE_ID / "top"
    camera.mkdir(parents=True)
    (camera / "rgb.mp4").write_bytes(b"not a real mp4; the reader only carries it")
    (camera / "timestamps.json").write_text(
        json.dumps([iso(value) for value in image_times])
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps([{"id": EPISODE_ID, "prompt": "pick up the cup"}])
    )
    return tmp_path


@pytest.fixture
def reader(source):
    return build_reader(load("action_net"), source)


class TestDiscovery:
    def test_finds_the_episode(self, reader):
        assert reader.episode_ids() == [EPISODE_ID]

    def test_reads_the_prompt(self, reader):
        assert reader.prompts() == {EPISODE_ID: "pick up the cup"}


class TestReadEpisode:
    def test_emits_the_source_features_under_their_lerobot_names(self, reader):
        episode = reader.read_episode(EPISODE_ID, prompt="p")
        assert set(episode.frames[0]) == {
            "observation.robot_joints",
            "observation.hand_joints",
            "action.robot_joints",
            "action.hand_joints",
            "task",
        }

    def test_does_not_assemble_state_or_action(self, reader):
        """Assembly is the layout step's job; a reader that did it too would be a
        second implementation of the convention."""
        episode = reader.read_episode(EPISODE_ID, prompt="p")
        assert "observation.state" not in episode.frames[0]
        assert "action" not in episode.frames[0]

    def test_carries_the_video_without_decoding_it(self, reader, source):
        episode = reader.read_episode(EPISODE_ID, prompt="p")
        assert episode.videos == {
            "observation.images.primary": source / EPISODE_ID / "top" / "rgb.mp4"
        }

    def test_rows_match_the_clock_strategy(self, reader, source):
        with h5py.File(source / f"{EPISODE_ID}.hdf5", "r") as handle:
            robot_times = np.asarray(handle["timestamp"])
            robot = np.asarray(handle["state/robot"])
        image_times = parse_times(
            json.loads((source / EPISODE_ID / "top" / "timestamps.json").read_text()),
            TIMESTAMP_FORMAT,
        )
        expected = align("nearest_timestamp_dedup", robot_times, image_times)

        episode = reader.read_episode(EPISODE_ID, prompt="p")
        assert len(episode.frames) == len(expected)
        for frame, index in zip(episode.frames, expected):
            assert np.array_equal(frame["observation.robot_joints"], robot[index])

    def test_drops_frames_past_the_last_robot_sample(self, reader, source):
        """The two upstream filters together make the episode shorter than the mp4."""
        image_count = len(
            json.loads((source / EPISODE_ID / "top" / "timestamps.json").read_text())
        )
        assert len(reader.read_episode(EPISODE_ID, prompt="p").frames) < image_count


class TestRejection:
    def test_a_different_robot_is_skipped_not_reinterpreted(self, source, reader):
        """A GR2 reports 29 joints where a GR1 reports 32; its columns mean other
        things, so the episode is skipped rather than silently truncated."""
        with h5py.File(source / f"{EPISODE_ID}.hdf5", "a") as handle:
            del handle["state/robot"]
            handle["state/robot"] = np.zeros((120, 29), dtype=np.float32)
        with pytest.raises(EpisodeSkipped, match="32 wide"):
            reader.read_episode(EPISODE_ID, prompt="p")

    def test_a_missing_hdf5_dataset_is_skipped(self, source, reader):
        with h5py.File(source / f"{EPISODE_ID}.hdf5", "a") as handle:
            del handle["action/hand"]
        with pytest.raises(EpisodeSkipped, match="missing dataset"):
            reader.read_episode(EPISODE_ID, prompt="p")

    def test_a_missing_video_is_skipped(self, source, reader):
        (source / EPISODE_ID / "top" / "rgb.mp4").unlink()
        with pytest.raises(EpisodeSkipped, match="rgb.mp4"):
            reader.read_episode(EPISODE_ID, prompt="p")


class TestReproducesTheHandWrittenConverter:
    """The layout the deleted ActionNet converter hardcoded, stated independently.

    ``actionnet2lerobot`` concatenated ``[robot | hand]`` and applied a fixed
    44-element permutation. That permutation is pinned below, so the claim that
    moving the layout into YAML changed nothing survives the code it replaced --
    a live import would have skipped forever once the package was deleted.
    """

    # column i of the emitted vector took column PERMUTATION[i] of [robot | hand]
    PERMUTATION = (
        *range(18, 25),  # left_arm
        *range(32, 38),  # left_hand  (hand columns start at 32 in the concatenation)
        *range(0, 6),    # left_leg
        *range(15, 18),  # neck
        *range(25, 32),  # right_arm
        *range(38, 44),  # right_hand
        *range(6, 12),   # right_leg
        *range(12, 15),  # waist
    )

    def test_the_slot_map_is_that_permutation(self):
        slots = load("action_net").state.slot_map("state")
        flattened = tuple(
            column if path == "observation.robot_joints" else ROBOT_JOINTS + column
            for path, column in slots
        )
        assert flattened == self.PERMUTATION

    def test_assembling_the_readers_output_matches_it(self, reader):
        """End to end: what the reader emits, laid out by the spec, is what the
        hand-written converter would have written."""
        episode = reader.read_episode(EPISODE_ID, prompt="p")
        slots = load("action_net").state.slot_map("state")

        for frame in episode.frames:
            concatenated = np.concatenate(
                [frame["observation.robot_joints"], frame["observation.hand_joints"]]
            )
            assembled = np.array([frame[path][column] for path, column in slots])
            assert np.array_equal(assembled, concatenated[list(self.PERMUTATION)])
