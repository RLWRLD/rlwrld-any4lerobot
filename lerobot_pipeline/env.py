"""Where this machine keeps its data, and how hard it should work.

A run used to need a config file per dataset. Those files turned out to hold no
dataset-specific information at all -- across all 36 the only differences were the
dataset's own name substituted into two paths, plus a default that belonged to the
builder. What actually varies between one run and the next is the *machine*: where
the raw sources are staged, where output goes, how many cores there are.

So there is one file per machine instead, and the dataset is a flag:

    python -m lerobot_pipeline.run --env configs/env/ec2.yaml --dataset action_net
    python -m lerobot_pipeline.run --env configs/env/ec2.yaml --all

Which leaves exactly one file per dataset, in ``dataset_registry/datasets``, and
nothing about a dataset written anywhere else.

Builder flags are layered, most specific last::

    spec.source.args   what only the dataset knows      (--eef-type gripper)
    profile.builders   what the convention wants        (--use-videos)
    env.builders       what this machine can do         (--executor ray)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_DIR = Path(__file__).resolve().parent / "configs" / "env"

_TOP_LEVEL = {
    "name",
    "profile",
    "raw_root",
    "out_root",
    "state_root",
    "paths",
    "builders",
    "batch",
    "runtime",
}
# How hard this machine may work, and nothing about what comes out of it. The
# encoder settings used to be settable here too; they belong to the profile, whose
# name a build records -- see _PROFILE_KEYS.
_RUNTIME = {"workers", "threads_per_ffmpeg", "nic_rate"}
# Rules that decide what is built rather than how fast. Named so that finding one in
# an environment file can say where it belongs instead of just "unknown key".
_PROFILE_KEYS = {"preset", "crf", "encoding"}
_BATCH = {"max_datasets", "target_episodes"}

# Deliberately modest: enough to keep a small dataset from leaving a machine idle,
# not so much that a mistake in one dataset takes two others down with it. Both are
# properties of the machine, so both are overridable per environment.
_DEFAULT_MAX_DATASETS = 3
_DEFAULT_TARGET_EPISODES = 768


class EnvError(ValueError):
    """Raised for an unknown environment name or a malformed environment file."""


@dataclass(frozen=True)
class Environment:
    name: str
    profile: str | None = None
    raw_root: Path | None = None
    out_root: Path | None = None
    # where the orchestrator records how far each dataset has got
    state_root: Path | None = None
    # dataset id -> an explicit raw path, for sources this machine keeps somewhere
    # that does not follow raw_root/<dir>
    paths: Mapping[str, str] = field(default_factory=dict)
    # builder name -> flags this machine wants for it
    builders: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    batch: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)

    @property
    def nic_rate(self) -> str | None:
        """What this machine's link can carry, e.g. ``30Gb/s``.

        The aws CLI's ``target_bandwidth``, which decides how hard the CRT client
        parallelises. It is worth naming because leaving it unset costs half the
        download: measured on m8i.16xlarge, 130.7 GB of toto onto a four-way stripe,
        1,330.8 MB/s unset against 2,687.7 MB/s at ``30Gb/s``.

        An earlier note in this repo had it at 3%, from a node whose single gp3 volume
        was the limit at 677 MB/s -- where nothing about the client could have mattered.
        It matters as soon as the disk is not the constraint.
        """
        rate = self.runtime.get("nic_rate")
        return str(rate) if rate else None

    @property
    def max_datasets(self) -> int:
        return int(self.batch.get("max_datasets") or _DEFAULT_MAX_DATASETS)

    @property
    def target_episodes(self) -> int:
        return int(self.batch.get("target_episodes") or _DEFAULT_TARGET_EPISODES)

    def is_staged(self, spec) -> bool:
        """Whether this machine keeps ``spec``'s source somewhere it was put by hand.

        Such a source was not downloaded by the pipeline, so the pipeline must not
        delete it -- and, before that, must not try to fetch over it.
        """
        return spec.id in self.paths

    def raw_path(self, spec) -> Path:
        """Where this machine keeps ``spec``'s raw source.

        ``source.raw_dir`` is the dataset's own answer to "which directory of a raw
        collection am I": the AgiBot dexhand and gripper subsets both name
        AgiBotWorld-Beta, because they are one tree read twice with different flags.
        """
        explicit = self.paths.get(spec.id)
        if explicit:
            return Path(explicit).expanduser()
        if self.raw_root is None:
            raise EnvError(
                f"{self.name}: no raw_root, and no paths entry for {spec.id!r}"
            )
        directory = (spec.source.raw_dir if spec.source else None) or spec.id
        return self.raw_root / directory

    def out_path(self, spec) -> Path:
        if self.out_root is None:
            raise EnvError(f"{self.name}: no out_root to write {spec.id!r} to")
        return self.out_root / spec.id


def available_envs() -> list[str]:
    if not ENV_DIR.is_dir():
        return []
    return sorted(path.stem for path in ENV_DIR.glob("*.yaml"))


def load_env(source: str | Path) -> Environment:
    """Load by name from ``configs/env`` or by path."""
    import yaml

    path = Path(source)
    if path.suffix != ".yaml":
        path = ENV_DIR / f"{source}.yaml"
        if Path(str(source)).name != str(source) or not path.is_file():
            raise EnvError(
                f"unknown environment {source!r}. "
                f"available: {', '.join(available_envs()) or '(none)'}"
            )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise EnvError(f"environment file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise EnvError(f"could not parse {path} as YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise EnvError(f"{path} must contain a YAML mapping at the top level")
    unknown = sorted(set(raw) - _TOP_LEVEL)
    if unknown:
        raise EnvError(
            f"{path}: unknown key(s) {', '.join(unknown)}. "
            f"allowed: {', '.join(sorted(_TOP_LEVEL))}"
        )

    runtime = raw.get("runtime") or {}
    misplaced = sorted(set(runtime) & _PROFILE_KEYS)
    if misplaced:
        raise EnvError(
            f"{path}.runtime: {', '.join(misplaced)} decides what is built rather "
            "than how fast, so it belongs to a profile. Building the collection a "
            "different way is a second file in configs/profiles, which a build "
            "records the name of; setting it here would leave a build that differs "
            "from the collection looking identical to one that does not."
        )
    unknown = sorted(set(runtime) - _RUNTIME)
    if unknown:
        raise EnvError(f"{path}.runtime: unknown key(s) {', '.join(unknown)}")

    builders = raw.get("builders") or {}
    if not isinstance(builders, Mapping) or not all(
        isinstance(value, Mapping) for value in builders.values()
    ):
        raise EnvError(f"{path}.builders must map builder name -> flags")

    batch = raw.get("batch") or {}
    if not isinstance(batch, Mapping):
        raise EnvError(f"{path}.batch must be a mapping")
    unknown = sorted(set(batch) - _BATCH)
    if unknown:
        raise EnvError(f"{path}.batch: unknown key(s) {', '.join(unknown)}")

    def directory(key: str) -> Path | None:
        return Path(raw[key]).expanduser() if raw.get(key) else None

    return Environment(
        name=raw.get("name") or path.stem,
        profile=raw.get("profile"),
        raw_root=directory("raw_root"),
        out_root=directory("out_root"),
        state_root=directory("state_root"),
        paths=dict(raw.get("paths") or {}),
        builders={name: dict(flags) for name, flags in builders.items()},
        batch=dict(batch),
        runtime=dict(runtime),
    )


def build_config(env: Environment, dataset: str, profile: str | None = None):
    """The pipeline config for one dataset on this machine."""
    from dataset_registry import SpecError, load

    from .config import ConfigError, parse_config
    from .profiles import ProfileError, load_profile

    name = profile or env.profile
    if not name:
        raise EnvError(
            f"{env.name}: no profile. The output version and the video handling are "
            "conventions rather than machine settings, so a run needs one -- set "
            "`profile:` in the environment or pass --profile"
        )
    try:
        spec = load(dataset)
    except SpecError as exc:
        raise EnvError(str(exc)) from exc

    builder = spec.source.builder if spec.source else "none"
    args = dict(env.builders.get(builder) or {})
    try:
        args = {**(load_profile(name).get("builders") or {}).get(builder, {}), **args}
    except ProfileError as exc:
        raise EnvError(f"profile: {exc}") from exc

    raw: dict[str, Any] = {
        "name": dataset,
        "dataset": dataset,
        "profile": name,
        "source": {"path": str(env.raw_path(spec))},
        "dest": {"path": str(env.out_path(spec))},
    }
    if args:
        raw["source"]["args"] = args
    if env.runtime:
        raw["runtime"] = dict(env.runtime)

    try:
        return parse_config(raw)
    except ConfigError:
        raise
