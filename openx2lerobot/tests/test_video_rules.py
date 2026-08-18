import json

import pytest

from video_rules import (
    VideoRuleError,
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
