"""The slot layouts, pinned as literals.

Slots used to be written into each dataset spec by hand. They are now derived from
a block's width and the order its layout declares, which removed the copies but also
removed the thing that was being reviewed: nobody reads a spec and checks a prefix
sum. So the previously hand-written values are pinned here, once, as the record of
what the delivered datasets actually look like.

These are the numbers recovered from delivered data (see
``docs/vla-pretrain-state-action-layout.md``). A diff here means either a real
mistake or a deliberate convention change -- and a convention change has to break a
test, because an existing checkpoint's projector was trained on this order.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_registry import SpecError, load  # noqa: E402

# dataset -> block name -> (start, end)
GOLDEN: dict[str, dict[str, tuple[int, int]]] = {
    "action_net": {
        "left_arm": (0, 7), "left_hand": (7, 13), "left_leg": (13, 19),
        "neck": (19, 22), "right_arm": (22, 29), "right_hand": (29, 35),
        "right_leg": (35, 41), "waist": (41, 44),
    },
    "agibot_dexhand": {
        "left_arm": (0, 7), "left_hand": (7, 13), "left_leg": (13, 19),
        "neck": (19, 22), "right_arm": (22, 29), "right_hand": (29, 35),
        "right_leg": (35, 41), "waist": (41, 44),
    },
    "agibot_gripper": {
        "left_arm": (0, 7), "left_hand": (7, 8), "left_leg": (8, 14),
        "neck": (14, 17), "right_arm": (17, 24), "right_hand": (24, 25),
        "right_leg": (25, 31), "waist": (31, 34),
    },
    "neural_robocurate": {
        "left_arm": (0, 7), "left_hand": (7, 13), "left_leg": (13, 19),
        "neck": (19, 22), "right_arm": (22, 29), "right_hand": (29, 35),
        "right_leg": (35, 41), "waist": (41, 44),
    },
    "humanoid_everyday_g1": {
        "left_arm": (0, 7), "right_arm": (7, 14),
        "left_hand": (14, 21), "right_hand": (21, 28),
    },
    "galaxea": {
        "arm_left": (0, 6), "gripper_left": (6, 7), "arm_right": (7, 13),
        "gripper_right": (13, 14), "torso": (14, 18),
    },
}

# dataset -> the (source path, column) each state slot takes, for the datasets whose
# columns were recovered rather than merely named
GOLDEN_STATE_SLOTS: dict[str, list[tuple[str, int] | None]] = {
    "action_net": (
        [("observation.robot_joints", c) for c in range(18, 25)]        # left_arm
        + [("observation.hand_joints", c) for c in range(0, 6)]         # left_hand
        + [("observation.robot_joints", c) for c in range(0, 6)]        # left_leg
        + [("observation.robot_joints", c) for c in range(15, 18)]      # neck
        + [("observation.robot_joints", c) for c in range(25, 32)]      # right_arm
        + [("observation.hand_joints", c) for c in range(6, 12)]        # right_hand
        + [("observation.robot_joints", c) for c in range(6, 12)]       # right_leg
        + [("observation.robot_joints", c) for c in range(12, 15)]      # waist
    ),
}


@pytest.mark.parametrize("dataset", sorted(GOLDEN))
def test_derived_slots_match_the_recovered_layout(dataset):
    state = load(dataset).state
    assert {b.name: (b.start, b.end) for b in state.blocks} == GOLDEN[dataset]


@pytest.mark.parametrize("dataset", sorted(GOLDEN))
def test_block_order_follows_the_layout(dataset):
    state = load(dataset).state
    starts = [b.start for b in state.blocks]
    assert starts == sorted(starts), "blocks must be emitted in slot order"


@pytest.mark.parametrize("dataset", sorted(GOLDEN_STATE_SLOTS))
def test_every_state_slot_takes_the_recovered_column(dataset):
    assert load(dataset).state.slot_map("state") == GOLDEN_STATE_SLOTS[dataset]


def test_action_side_mirrors_the_state_side_for_action_net(dataset="action_net"):
    """Same columns on both sides, read from the action-side feature."""
    state = load(dataset).state
    for (state_path, state_col), (action_path, action_col) in zip(
        state.slot_map("state"), state.slot_map("action")
    ):
        assert state_col == action_col
        assert state_path.removeprefix("observation.") == action_path.removeprefix(
            "action."
        )


class TestSwitchingTheConvention:
    """A profile changes the layout; no dataset spec is touched.

    This is what the layout layer exists for. The delivered order is load-bearing
    while the current checkpoint is in use -- its per-embodiment projector was
    trained on it -- so switching has to be possible without editing eight specs by
    hand, and has to be all-or-nothing rather than per dataset.
    """

    @pytest.fixture
    def layouts(self, tmp_path, monkeypatch):
        from dataset_registry import schema

        (tmp_path / "gr1_body_parts.yaml").write_text(
            "order: [left_arm, left_hand, left_leg, neck, "
            "right_arm, right_hand, right_leg, waist]\n"
        )
        # the same blocks, right side first
        (tmp_path / "gr1_right_first.yaml").write_text(
            "order: [right_arm, right_hand, right_leg, waist, "
            "left_arm, left_hand, left_leg, neck]\n"
        )
        monkeypatch.setattr(schema, "LAYOUTS_DIR", tmp_path)
        return tmp_path

    def test_a_substitution_reorders_every_slot(self, layouts):
        switched = load("action_net", layouts={"gr1_body_parts": "gr1_right_first"})
        assert [b.name for b in switched.state.blocks][0] == "right_arm"
        # right_arm was at 22 under the delivered order and is at 0 under this one
        assert switched.state.blocks[0].start == 0
        assert switched.state.width == 44

    def test_the_same_columns_are_still_read(self, layouts):
        """Reordering moves slots, it does not change what fills them."""
        original = load("action_net")
        switched = load("action_net", layouts={"gr1_body_parts": "gr1_right_first"})
        assert sorted(filter(None, original.state.slot_map("state"))) == sorted(
            filter(None, switched.state.slot_map("state"))
        )

    def test_a_layout_with_different_blocks_is_refused(self, layouts):
        """Substitution is a reordering, not a redefinition: a layout naming other
        body parts would silently drop or invent slots."""
        (layouts / "two_blocks.yaml").write_text("order: [left_arm, right_arm]\n")
        with pytest.raises(SpecError, match="unexpected"):
            load("action_net", layouts={"gr1_body_parts": "two_blocks"})
