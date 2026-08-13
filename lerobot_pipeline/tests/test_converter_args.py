"""Rejecting args a converter would reject, at config time.

``source.args`` goes straight onto a converter's command line, so a wrong key is a
run that dies at argument parsing -- after the machine is up and the source staged,
which is the expensive moment to discover a typo.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot_pipeline.config import ConfigError, load_config, parse_config  # noqa: E402
from lerobot_pipeline.converter_args import check, converter_flags  # noqa: E402

CONFIGS = REPO_ROOT / "lerobot_pipeline" / "configs" / "datasets"


class TestReadingFlags:
    @pytest.mark.parametrize("script", [
        "spec2lerobot",
        "openx2lerobot/openx_rlds.py",
        "agibot2lerobot/agibot_h5.py",
    ])
    def test_flags_are_read_from_the_converter_itself(self, script):
        """Read rather than listed here, so the list cannot drift from the code."""
        flags = converter_flags(script)
        assert flags and "push_to_hub" in flags

    def test_reading_does_not_import_the_converter(self):
        """Several converters import tensorflow or h5py at module scope; validating
        a config must not need the whole conversion environment."""
        assert "tensorflow" not in sys.modules
        converter_flags("openx2lerobot/openx_rlds.py")
        assert "tensorflow" not in sys.modules


class TestChecking:
    def test_a_typo_is_caught_and_a_correction_offered(self):
        problems = check("spec2lerobot", {"excutor": "local"}, "spec")
        assert problems and "did you mean executor?" in problems[0]

    def test_a_flag_the_pipeline_supplies_is_rejected(self):
        problems = check("spec2lerobot", {"output_path": "/tmp"}, "spec")
        assert "supplied by the pipeline itself" in problems[0]

    def test_valid_args_pass(self):
        assert check("spec2lerobot", {"executor": "local", "debug": True}, "spec") == []


class TestThroughTheConfig:
    def test_a_bad_arg_fails_the_config_not_the_run(self):
        with pytest.raises(ConfigError, match="not a flag of the spec converter"):
            parse_config({
                "name": "demo", "dataset": "action_net", "profile": "rldx1",
                "source": {"path": "/in", "args": {"nonsense": 1}},
                "dest": {"path": "/out"},
            })

    @pytest.mark.parametrize(
        "path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem
    )
    def test_every_shipped_config_passes_its_converter(self, path):
        try:
            load_config(path)
        except ConfigError as exc:
            # an unbuildable dataset fails for a different, already-reported reason
            if "cannot be laid out" in str(exc):
                pytest.skip("dataset is not buildable; covered elsewhere")
            raise


class TestBuilderArgs:
    def test_the_spec_supplies_the_flag_only_it_knows(self):
        """agibot2lerobot converts one end-effector type per run. Without this the
        dexhand and gripper configs would produce the same dataset."""
        dexhand = load_config(CONFIGS / "agibot_dexhand.yaml")
        gripper = load_config(CONFIGS / "agibot_gripper.yaml")
        assert dexhand.source.args["eef_type"] == "dexhand"
        assert gripper.source.args["eef_type"] == "gripper"

    def test_the_run_config_can_override_a_builder_arg(self):
        config = parse_config({
            "name": "demo", "dataset": "agibot_gripper", "profile": "rldx1",
            "source": {"path": "/in", "args": {"eef_type": "dexhand"}},
            "dest": {"path": "/out"},
        })
        assert config.source.args["eef_type"] == "dexhand"
