"""The 44-dim layout was recovered, not documented, so it is pinned by tests.

The expected values here come from two independent sources that agree:
  * column-by-column comparison of `observation.state` against
    `observation.robot_joints` / `observation.hand_joints` in a delivered parquet
    (/data/taeyoung/data/vla_pretrain_dataset/action_net, episode 0);
  * `neural_gr1`'s `modality.json` in RLDX-1, which names the same eight blocks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionnet_utils.actionnet_utils import assemble, match_timestamps  # noqa: E402
from actionnet_utils.config import (  # noqa: E402
    HAND_JOINTS,
    PERMUTATION,
    ROBOT_JOINTS,
    STATE_WIDTH,
    block_ranges,
    generate_features,
    generate_modality,
)

# as declared by neural_gr1 / march_robocurate / oneuniverse_simul
REFERENCE_BLOCKS = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}


class TestBlockLayout:
    def test_blocks_match_the_reference_modality_json(self):
        assert block_ranges() == REFERENCE_BLOCKS

    def test_width_is_44(self):
        assert STATE_WIDTH == 44 == ROBOT_JOINTS + HAND_JOINTS

    def test_permutation_uses_every_source_column_exactly_once(self):
        assert sorted(PERMUTATION) == list(range(44))


class TestPermutation:
    """Source columns, as recovered from the delivered parquet."""

    @pytest.mark.parametrize(
        "block, source_columns",
        [
            ("left_arm", list(range(18, 25))),
            ("left_hand", list(range(32, 38))),
            ("left_leg", list(range(0, 6))),
            ("neck", list(range(15, 18))),
            ("right_arm", list(range(25, 32))),
            ("right_hand", list(range(38, 44))),
            ("right_leg", list(range(6, 12))),
            ("waist", list(range(12, 15))),
        ],
    )
    def test_block_takes_the_expected_source_columns(self, block, source_columns):
        start, end = block_ranges()[block]
        assert list(PERMUTATION[start:end]) == source_columns

    def test_assemble_moves_values_where_the_layout_says(self):
        # each source column carries its own index, so the output reads back as
        # the permutation itself
        robot = np.arange(ROBOT_JOINTS, dtype=np.float32)[None, :]
        hand = np.arange(ROBOT_JOINTS, 44, dtype=np.float32)[None, :]
        assert list(assemble(robot, hand)[0]) == list(PERMUTATION)

    def test_assemble_keeps_the_row_count(self):
        robot = np.zeros((7, ROBOT_JOINTS), dtype=np.float32)
        hand = np.zeros((7, HAND_JOINTS), dtype=np.float32)
        assert assemble(robot, hand).shape == (7, 44)


class TestFeatures:
    def test_feature_widths_match_the_reference_dataset(self):
        features = generate_features()
        widths = {key: value["shape"][0] for key, value in features.items()
                  if value["dtype"] == "float32"}
        assert widths == {
            "observation.state": 44,
            "action": 44,
            "absolute_action": 44,
            "observation.robot_joints": 32,
            "observation.hand_joints": 12,
            "action.robot_joints": 32,
            "action.hand_joints": 12,
        }

    def test_camera_key_is_primary(self):
        # the collection names the main camera `primary`, not after its folder
        assert "observation.images.primary" in generate_features()

    def test_motors_are_named_positionally(self):
        names = generate_features()["observation.state"]["names"]["motors"]
        assert names[:2] == ["m0", "m1"] and names[-1] == "m43"


class TestModality:
    def test_state_and_action_are_one_flat_block(self):
        # RLDX-1 asks ActionNet for modality_keys=["state"], so splitting the
        # vector into the eight named blocks would not match the training config
        modality = generate_modality()
        assert modality["state"] == {"state": {"start": 0, "end": 44}}
        assert modality["action"]["action"]["absolute"] is True

    def test_video_points_at_the_primary_camera(self):
        modality = generate_modality()
        assert modality["video"] == {
            "primary": {"original_key": "observation.images.primary"}
        }


class TestTimestampMatching:
    def test_each_robot_sample_is_claimed_at_most_once(self):
        candidate = np.array([0.0, 1.0, 2.0, 3.0])
        # two frames land nearest the same sample; the second takes the next one
        matched = match_timestamps(candidate, np.array([1.0, 1.05, 2.9]))
        assert len(set(matched.tolist())) == len(matched)

    def test_result_is_ordered_by_video_frame(self):
        candidate = np.arange(10, dtype=float)
        matched = match_timestamps(candidate, np.array([0.1, 4.2, 8.9]))
        assert list(matched) == [0, 4, 9]
