"""Profiles, and what a run config resolves to under one."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot_pipeline.config import ConfigError, parse_config  # noqa: E402
from lerobot_pipeline.profiles import (  # noqa: E402
    ProfileError,
    available_profiles,
    load_profile,
)
from lerobot_pipeline.stages import plan_stages  # noqa: E402

RUN = {
    "name": "demo",
    "dataset": "action_net",
    "profile": "rldx1",
    "source": {"path": "/in"},
    "dest": {"path": "/out"},
}


class TestLoading:
    def test_rldx1_ships(self):
        assert "rldx1" in available_profiles()

    def test_unknown_profile_lists_the_available_ones(self):
        with pytest.raises(ProfileError, match="rldx1"):
            load_profile("no_such_profile")

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ProfileError, match="colour"):
            load_profile({"colour": "red"})

    def test_a_resize_without_a_type_is_rejected(self):
        with pytest.raises(ProfileError, match="missing 'type'"):
            load_profile({"video": {"resize": {"max_area": 1}}})

    def test_layouts_must_map_names_to_names(self):
        with pytest.raises(ProfileError, match="layout name"):
            load_profile({"state": {"layouts": {"gr1_body_parts": 3}}})


class TestResolution:
    def test_the_profile_supplies_the_steps(self):
        config = parse_config(RUN)
        assert [s.config_name for s in config.steps] == [
            "state_layout",
            "resize_preserve_aspect_area",
        ]

    def test_the_profile_supplies_the_output_version_and_encoding(self):
        config = parse_config(RUN)
        assert config.dest.type == "lerobot_v21"
        assert config.runtime.encoding["gop"] == 250

    def test_the_dataset_supplies_the_source_type(self):
        config = parse_config(RUN)
        assert config.source.type == "spec"
        assert config.source.args["dataset"] == "action_net"

    def test_the_run_config_wins_over_the_profile(self):
        # a profile is a default, so a one-off run can differ without a second file
        config = parse_config({**RUN, "dest": {"path": "/out", "type": "lerobot_v30"}})
        assert config.dest.type == "lerobot_v30"

    def test_stages_run_the_layout_before_the_video(self):
        # a table step is cheap and hard-links the video, so failing there costs
        # seconds rather than a re-encode
        stages = plan_stages(parse_config(RUN), "/work")
        assert [s.kind for s in stages] == [
            "convert",
            "state_layout",
            "transform",
            "version_convert",
        ]

    def test_a_config_without_a_profile_still_works(self):
        config = parse_config(
            {
                "name": "plain",
                "source": {"type": "lerobot_v21", "path": "/in"},
                "dest": {"type": "lerobot_v21", "path": "/out"},
                "steps": [{"type": "resize_preserve_aspect_area"}],
            }
        )
        assert config.dataset is None and config.profile is None


class TestRefusal:
    def test_an_unbuildable_dataset_fails_at_config_time(self):
        """Galaxea's action columns were never recovered. The run stops here rather
        than after the conversion has already written most of a dataset."""
        with pytest.raises(ConfigError, match="unrecovered"):
            parse_config({**RUN, "dataset": "galaxea"})

    def test_an_unknown_dataset_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown dataset"):
            parse_config({**RUN, "dataset": "no_such_dataset"})
