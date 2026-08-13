import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_registry import SpecError, available, load, load_all, parse  # noqa: E402

# Uses a real layout rather than a synthetic one: block order comes from
# layouts/*.yaml and there is no inline form, which is the point -- a spec cannot
# state an order of its own.
MINIMAL = {
    "id": "demo",
    "name": "Demo",
    "lerobot": {
        "state": {
            "width": 6,
            "layout": "arms_then_hands",
            "source_features": {"arm": {"state": "obs.arm", "action": "act.arm"}},
            "blocks": {
                "left_arm": {
                    "width": 3,
                    "source": {"feature": "arm", "columns": [2, 5]},
                    "evidence": "measured",
                },
                "right_arm": {"width": 1, "evidence": "constant"},
                "left_hand": {"width": 1, "evidence": "constant"},
                "right_hand": {"width": 1, "evidence": "constant"},
            },
        }
    },
}


def spec(**overrides):
    import copy

    raw = copy.deepcopy(MINIMAL)
    raw.update(overrides)
    return raw


class TestShippedSpecs:
    def test_every_spec_parses(self):
        assert load_all()

    def test_action_net_is_registered(self):
        assert "action_net" in available()

    @pytest.mark.parametrize("name", available())
    def test_spec_declares_an_embodiment_tag(self, name):
        # the tag is what selects the model's per-embodiment weights; a spec without
        # one cannot be pointed at a training run
        assert load(name).embodiment_tag

    @pytest.mark.parametrize("name", available())
    def test_blocks_tile_the_vector(self, name):
        state = load(name).state
        if state is None:
            pytest.skip("no state layout")
        assert sum(b.width for b in state.blocks) == state.width

    def test_unknown_dataset_lists_the_available_ones(self):
        with pytest.raises(SpecError, match="action_net"):
            load("no_such_dataset")


class TestActionNetLayout:
    """The layout the ActionNet converter implements, stated independently."""

    def test_the_eight_gr1_blocks_are_present_in_order(self):
        state = load("action_net").state
        assert [b.name for b in state.blocks] == [
            "left_arm", "left_hand", "left_leg", "neck",
            "right_arm", "right_hand", "right_leg", "waist",
        ]

    def test_leg_blocks_are_marked_constant(self):
        # they are all-zero in the data and therefore unverifiable; saying so is the
        # point of the evidence field
        state = load("action_net").state
        legs = [b for b in state.blocks if b.name.endswith("_leg")]
        assert legs and all(b.evidence == "constant" for b in legs)

    def test_most_slots_are_measured(self):
        counts = load("action_net").state.evidence_counts()
        assert counts["measured"] == 32
        assert counts["constant"] == 12


class TestSourceMapping:
    def test_foundry_uri_is_recorded_when_mirrored(self):
        assert load("action_net").foundry_uri.startswith("s3://rlwrld-foundry-data/")

    def test_a_source_not_mirrored_yet_has_no_foundry_entry(self):
        # only three sources are in foundry so far; the rest must still parse
        assert load("agibot_dexhand").foundry_uri is None

    def test_camera_keys_map_to_source_cameras(self):
        spec = load("action_net")
        assert list(spec.cameras) == ["primary"]
        assert spec.camera_source("primary") == "top"
        # the source geometry, before the resize step -- not what was delivered
        assert spec.camera_shape("primary") == (800, 1280, 3)

    def test_delivered_geometry_is_recorded_separately(self):
        # a rebuild is checked by comparing the two, so they must not be the same field
        assert load("action_net").delivered_video["primary"]["shape"] == [192, 288, 3]

    @pytest.mark.parametrize("name", available())
    def test_every_spec_records_what_was_delivered(self, name):
        spec = load(name)
        assert spec.delivered_video, "no delivered.video to check a rebuild against"


class TestValidation:
    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(SpecError, match="colour"):
            parse(spec(colour="red"))

    def test_a_missing_block_is_rejected(self):
        raw = spec()
        del raw["lerobot"]["state"]["blocks"]["right_hand"]
        with pytest.raises(SpecError, match="missing right_hand"):
            parse(raw)

    def test_a_block_the_layout_does_not_name_is_rejected(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["tail"] = {"width": 1, "evidence": "constant"}
        with pytest.raises(SpecError, match="unexpected tail"):
            parse(raw)

    def test_blocks_as_a_list_are_rejected(self):
        # a list would let the spec imply an order that disagrees with the layout,
        # which is exactly what moving order into layouts/ was meant to prevent
        raw = spec()
        raw["lerobot"]["state"]["blocks"] = [{"width": 6, "evidence": "constant"}]
        with pytest.raises(SpecError, match="must be a mapping"):
            parse(raw)

    def test_widths_must_sum_to_the_declared_width(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["right_hand"]["width"] = 2
        with pytest.raises(SpecError, match="sum to 7"):
            parse(raw)

    def test_slot_and_source_widths_must_agree(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["left_arm"]["source"]["columns"] = [2, 4]
        with pytest.raises(SpecError, match="source columns"):
            parse(raw)

    def test_an_unknown_layout_is_rejected(self):
        raw = spec()
        raw["lerobot"]["state"]["layout"] = "no_such_layout"
        with pytest.raises(SpecError, match="unknown layout"):
            parse(raw)

    def test_unknown_source_feature_is_rejected(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["left_arm"]["source"]["feature"] = "leg"
        with pytest.raises(SpecError, match="not in source_features"):
            parse(raw)

    def test_a_sourceless_block_cannot_claim_to_be_measured(self):
        # "measured" means it was matched against a source column; with no source
        # named there is nothing it could have been matched against
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["right_arm"]["evidence"] = "measured"
        with pytest.raises(SpecError, match="block with no source"):
            parse(raw)

    def test_a_sourceless_block_may_be_declared(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["right_arm"]["evidence"] = "declared"
        assert parse(raw).state.blocks[1].evidence == "declared"

    def test_a_sourceless_block_cannot_be_padded(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["left_arm"] = {
            "width": 3, "pad": 1, "evidence": "constant"
        }
        with pytest.raises(SpecError, match="pad is only meaningful"):
            parse(raw)

    def test_unknown_evidence_value_is_rejected(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["left_arm"]["evidence"] = "probably"
        with pytest.raises(SpecError, match="evidence must be one of"):
            parse(raw)

    def test_unknown_mirror_layout_is_rejected(self):
        with pytest.raises(SpecError, match="layout must be one of"):
            parse(spec(mirrors=[{"kind": "foundry", "uri": "s3://x/", "layout": "zipfile"}]))

    def test_unknown_mirror_kind_is_rejected(self):
        with pytest.raises(SpecError, match="kind must be one of"):
            parse(spec(mirrors=[{"kind": "dropbox", "uri": "s3://x/"}]))

    def test_foundry_mirror_must_carry_the_upstream_revision(self):
        # foundry lays external sources out under external/<name>/<revision>/, so a
        # mirror path that omits it is pointing at a different release
        raw = spec(
            upstream={"huggingface": "org/repo", "revision": "abc123"},
            mirrors=[{"kind": "foundry", "uri": "s3://b/external/demo/deadbeef/"}],
        )
        with pytest.raises(SpecError, match="does not contain the upstream revision"):
            parse(raw)

    def test_a_matching_foundry_mirror_is_accepted(self):
        raw = spec(
            upstream={"huggingface": "org/repo", "revision": "abc123"},
            mirrors=[{"kind": "foundry", "uri": "s3://b/external/demo/abc123/"}],
        )
        assert parse(raw).foundry_uri.endswith("abc123/")


class TestSlotMap:
    def test_maps_each_slot_to_its_source_column(self):
        state = parse(spec()).state
        assert state.slot_map("state") == [
            ("obs.arm", 2), ("obs.arm", 3), ("obs.arm", 4), None, None, None
        ]

    def test_action_side_uses_the_action_source_paths(self):
        state = parse(spec()).state
        assert state.slot_map("action")[0] == ("act.arm", 2)

    def test_pad_slots_take_no_source(self):
        raw = spec()
        raw["lerobot"]["state"]["blocks"]["left_arm"]["pad"] = 1
        raw["lerobot"]["state"]["blocks"]["left_arm"]["source"]["columns"] = [2, 4]
        state = parse(raw).state
        assert state.slot_map("state")[:3] == [("obs.arm", 2), ("obs.arm", 3), None]
        # the pad is not counted as measured; nothing was measured about it
        assert state.evidence_counts() == {"measured": 2, "pad": 1, "constant": 3}

    def test_side_must_be_state_or_action(self):
        with pytest.raises(SpecError, match="must be 'state' or 'action'"):
            parse(spec()).state.slot_map("video")
