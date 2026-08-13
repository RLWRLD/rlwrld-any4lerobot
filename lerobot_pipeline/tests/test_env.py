"""One file per machine, and the dataset as a flag.

The thing these guard is that no dataset-specific information leaks back into the
environment file. It says where this machine keeps data and how hard it may work;
everything about a dataset stays in its spec, which is what makes one env file cover
all 36 and keeps a new dataset from being a two-file change.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import available, load  # noqa: E402
from lerobot_pipeline.env import (  # noqa: E402
    EnvError,
    available_envs,
    build_config,
    load_env,
)


@pytest.fixture
def env_file(tmp_path):
    def write(body: str) -> Path:
        path = tmp_path / "demo.yaml"
        path.write_text(body)
        return path

    return write


class TestLoading:
    def test_the_shipped_environments_load(self):
        assert set(available_envs()) >= {"local", "ec2"}
        assert load_env("ec2").profile == "rldx1"

    def test_an_unknown_name_lists_the_available_ones(self):
        with pytest.raises(EnvError, match="ec2"):
            load_env("no_such_env")

    def test_an_unknown_key_is_rejected(self, env_file):
        with pytest.raises(EnvError, match="colour"):
            load_env(env_file("colour: red\n"))

    def test_an_unknown_runtime_key_is_rejected(self, env_file):
        with pytest.raises(EnvError, match="nonsense"):
            load_env(env_file("runtime: {nonsense: 1}\n"))


class TestPaths:
    def test_a_dataset_lands_under_the_roots(self):
        env = load_env("ec2")
        config = build_config(env, "action_net")
        assert config.source.path == Path("/scratch/raw/action_net")
        assert config.dest.path == Path("/scratch/out/action_net")

    def test_two_datasets_can_share_one_source_tree(self):
        """AgiBot's dexhand and gripper subsets are one tree read twice, differing
        only by --eef-type, so the directory is a dataset fact and not raw_root/<id>."""
        env = load_env("ec2")
        assert (build_config(env, "agibot_dexhand").source.path
                == Path("/scratch/raw/AgiBotWorld-Beta"))

    def test_a_machine_can_place_one_dataset_anywhere(self, env_file):
        env = load_env(env_file(
            "profile: rldx1\nraw_root: /raw\nout_root: /out\n"
            "paths: {action_net: /mnt/big/an}\n"))
        assert build_config(env, "action_net").source.path == Path("/mnt/big/an")

    def test_a_missing_root_is_reported_rather_than_guessed(self, env_file):
        env = load_env(env_file("profile: rldx1\nout_root: /out\n"))
        with pytest.raises(EnvError, match="no raw_root"):
            build_config(env, "action_net")

    def test_an_environment_without_a_profile_is_refused(self, env_file):
        """The output version and the video handling are conventions, not machine
        settings, so there is no sensible default to fall back on."""
        env = load_env(env_file("raw_root: /raw\nout_root: /out\n"))
        with pytest.raises(EnvError, match="no profile"):
            build_config(env, "action_net")


class TestBuilderFlags:
    def test_the_three_layers_compose(self, env_file):
        """dataset < profile < environment, most specific last."""
        env = load_env(env_file(
            "profile: rldx1\nraw_root: /raw\nout_root: /out\n"
            "builders: {agibot: {executor: ray}}\n"))
        args = build_config(env, "agibot_gripper").source.args
        assert args["eef_type"] == "gripper"   # from the dataset spec
        assert args["executor"] == "ray"       # from this machine

    def test_the_profile_supplies_a_convention_wide_flag(self):
        """--use-videos is what every delivered copy is, whichever machine runs it."""
        assert build_config(load_env("ec2"), "viola").source.args["use_videos"] is True


class TestCoversTheCollection:
    @pytest.mark.parametrize("dataset", available())
    def test_every_buildable_dataset_resolves_from_one_env_file(self, dataset):
        if load(dataset).buildable():
            pytest.skip("blocked for reasons of its own; reported by plan --all")
        config = build_config(load_env("ec2"), dataset)
        assert config.dataset.id == dataset

    def test_the_env_file_names_no_dataset(self):
        """The whole point: adding a dataset must not be an edit here."""
        for name in available_envs():
            body = (Path(__file__).resolve().parents[1] / "configs" / "env"
                    / f"{name}.yaml").read_text()
            named = [d for d in available() if d in body]
            assert not named, f"{name}.yaml mentions {named}"
