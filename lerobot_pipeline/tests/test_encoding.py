import pytest

from lerobot_pipeline.encoding import (
    EncodingProfileError,
    apply_profile,
    available_profiles,
    load_profile,
)
from lerobot_pipeline.video_ops import EncodingParams, build_ffmpeg_command


def flag_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


class TestProfileFiles:
    def test_shipped_profiles_are_discoverable(self):
        assert "rldx1_reference" in available_profiles()
        assert "random_access" in available_profiles()

    def test_reference_profile_matches_what_was_measured(self):
        # recovered by probing the delivered RLDX-1 datasets; see the profile file
        assert load_profile("rldx1_reference") == {
            "codec": "libx264",
            "profile": "high",
            "pix_fmt": "yuv420p",
            "preset": "medium",
            "crf": 23,
            "gop": 250,
            "bframes": 3,
            "sc_threshold": None,
        }

    def test_random_access_profile_keyframes_every_other_frame(self):
        profile = load_profile("random_access")
        assert profile["gop"] == 2
        # -g only means something if scene-cut detection cannot add keyframes
        assert profile["sc_threshold"] == 0

    def test_unknown_profile_lists_the_available_ones(self):
        with pytest.raises(EncodingProfileError, match="rldx1_reference"):
            load_profile("no_such_profile")

    def test_profile_name_may_not_be_a_path(self):
        with pytest.raises(EncodingProfileError, match="must not contain a path"):
            load_profile("../../etc/passwd")


class TestInlineProfiles:
    def test_inline_mapping_is_accepted(self):
        assert load_profile({"crf": 30, "gop": 12}) == {"crf": 30, "gop": 12}

    def test_unknown_key_is_an_error(self):
        with pytest.raises(EncodingProfileError, match="bitrate"):
            load_profile({"bitrate": "2M"})

    def test_wrong_type_is_an_error(self):
        with pytest.raises(EncodingProfileError, match="crf must be int"):
            load_profile({"crf": "high"})

    def test_explicit_null_is_kept_as_a_setting(self):
        # null means "do not pass the flag", which is different from absent
        assert load_profile({"sc_threshold": None}) == {"sc_threshold": None}


class TestApplyProfile:
    def test_absent_profile_leaves_source_settings_alone(self):
        source = EncodingParams(codec="libx265", crf=23)
        assert apply_profile(source, None) is source

    def test_profile_only_overrides_what_it_sets(self):
        source = EncodingParams(codec="libx265", preset="fast", crf=23)
        result = apply_profile(source, {"crf": 30})
        assert result.crf == 30
        assert result.codec == "libx265"
        assert result.preset == "fast"


class TestCommandConstruction:
    def test_reference_profile_produces_the_measured_settings(self):
        encoding = apply_profile(EncodingParams(), load_profile("rldx1_reference"))
        command = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=288:192",), encoding, 1)
        assert flag_value(command, "-c:v") == "libx264"
        assert flag_value(command, "-profile:v") == "high"
        assert flag_value(command, "-g") == "250"
        assert flag_value(command, "-bf") == "3"
        assert flag_value(command, "-crf") == "23"

    def test_null_setting_omits_the_flag_entirely(self):
        encoding = apply_profile(EncodingParams(), load_profile("rldx1_reference"))
        # the original encode left x264's scene-cut detection on
        assert "-sc_threshold" not in command_of(encoding)

    def test_random_access_profile_pins_scene_cut_detection(self):
        encoding = apply_profile(EncodingParams(), load_profile("random_access"))
        command = command_of(encoding)
        assert flag_value(command, "-sc_threshold") == "0"
        assert flag_value(command, "-g") == "2"

    def test_unset_optional_flags_are_not_emitted(self):
        encoding = EncodingParams(preset=None, crf=None, gop=None, sc_threshold=None)
        command = command_of(encoding)
        for flag in ("-preset", "-crf", "-g", "-sc_threshold", "-bf", "-profile:v"):
            assert flag not in command


def command_of(encoding):
    return build_ffmpeg_command("in.mp4", "out.mp4", ("scale=2:2",), encoding, 1)
