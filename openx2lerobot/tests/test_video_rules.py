import json

import pytest

from video_rules import (
    BGR_CAMERAS,
    VideoRuleError,
    flips_channels,
    parse_rule,
    resize_frame,
    rgb_encoder,
    target_shape,
)

RLDX1_RESIZE = {"type": "resize_preserve_aspect_area", "max_area": 65536, "multiple": 32}


class TestParseRule:
    def test_a_bare_name_is_a_rule_with_its_own_defaults(self):
        assert parse_rule("resize_preserve_aspect_area") == {
            "type": "resize_preserve_aspect_area"
        }

    def test_json_carries_the_parameters_the_pipeline_resolved(self):
        assert parse_rule(json.dumps(RLDX1_RESIZE)) == RLDX1_RESIZE

    def test_no_rule_means_no_rule(self):
        assert parse_rule(None) is None
        assert parse_rule("") is None

    def test_malformed_json_says_so_rather_than_being_read_as_a_name(self):
        with pytest.raises(VideoRuleError):
            parse_rule('{"type": ')


class TestTargetShape:
    def test_the_geometry_is_the_transform_stage_s_own(self):
        """640x480 is what the delivered ucsd_kitchen source is, 256x192 what it became."""
        assert target_shape(RLDX1_RESIZE, "image", (480, 640)) == (192, 256)

    def test_a_frame_already_on_the_grid_is_left_alone(self):
        assert target_shape(RLDX1_RESIZE, "image", (128, 128)) == (128, 128)

    def test_without_a_rule_the_source_size_is_kept(self):
        assert target_shape(None, "image", (480, 640)) == (480, 640)


class TestResizeFrame:
    def test_it_produces_exactly_the_requested_shape(self):
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert resize_frame(frame, (192, 256)).shape == (192, 256, 3)

    def test_a_frame_that_is_already_right_is_not_touched(self):
        import numpy as np

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        assert resize_frame(frame, (128, 128)) is frame


class TestRgbEncoder:
    def test_lerobot_av1_default_is_the_writer_s_own_settings(self):
        config = rgb_encoder("lerobot_av1_default")
        assert (config.vcodec, config.g, config.pix_fmt) == ("libsvtav1", 2, "yuv420p")

    def test_no_rule_leaves_the_writer_alone(self):
        assert rgb_encoder(None) is None

    def test_an_encoding_the_writer_cannot_produce_is_refused(self):
        """rldx1_reference is H.264 with three B-frames -- an ffmpeg command line,
        not something LeRobot's writer can be asked for. Half-applying it would
        produce a file that matches the name and nothing else."""
        with pytest.raises(VideoRuleError) as exc:
            rgb_encoder("rldx1_reference")
        assert "bframes=3" in str(exc.value)

    def test_an_encoding_may_arrive_as_the_settings_themselves(self):
        """--encoding advertises "a name ... or the JSON of one", which is the shape
        settings take crossing a command line: the pipeline sends JSON whenever the
        run asked for an encoding the collection has no file for."""
        assert rgb_encoder('{"gop": 50}').g == 50


class TestFlipsChannels:
    """The rule decides the channel order; the table decides which cameras it reaches."""

    def test_as_source_never_flips(self):
        assert not flips_channels("as_source", "utaustin_mutex", "image")

    def test_bgr_to_rgb_flips_the_cameras_oxe_reads_as_bgr(self):
        assert flips_channels("bgr_to_rgb", "utaustin_mutex", "image")
        assert flips_channels("bgr_to_rgb", "berkeley_autolab_ur5", "hand_image")

    def test_it_leaves_the_other_cameras_of_a_listed_dataset_alone(self):
        """berkeley_autolab_ur5 has several cameras and OXE reads only hand_image
        as BGR, so naming the dataset is not enough to flip the rest."""
        assert not flips_channels("bgr_to_rgb", "berkeley_autolab_ur5", "image")

    def test_a_dataset_with_no_bgr_camera_is_the_same_under_both_rules(self):
        for rule in ("as_source", "bgr_to_rgb"):
            assert not flips_channels(rule, "taco_play", "rgb_static")

    def test_no_rule_at_all_keeps_the_source_order(self):
        assert not flips_channels(None, "utaustin_mutex", "image")

    def test_an_unknown_rule_is_refused_rather_than_ignored(self):
        with pytest.raises(VideoRuleError, match="unknown channel rule"):
            flips_channels("rgb_to_bgr", "utaustin_mutex", "image")

    def test_the_table_names_the_datasets_whose_transforms_used_to_flip(self):
        """The flip moved out of oxe_utils/transforms.py and into this table. If a
        transform grows one back, the two would disagree and this catches it."""
        from pathlib import Path

        transforms = Path(__file__).resolve().parents[1] / "oxe_utils" / "transforms.py"
        assert "[..., ::-1]" not in transforms.read_text()
        assert set(BGR_CAMERAS) == {
            "berkeley_autolab_ur5",
            "stanford_hydra_dataset_converted_externally_to_rlds",
            "utaustin_mutex",
            "berkeley_fanuc_manipulation",
            "fmb_dataset",
        }
