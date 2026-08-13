"""Pipeline config: schema, loading and validation.

One config describes a whole run -- where data comes from, which preprocessing
steps to apply in order, and what the final output should be.

Validation is deliberately strict: unknown keys are errors rather than being
ignored, because a silently dropped key means a run that looks successful but
did the wrong thing.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import UnknownStepError, build_step

LEROBOT_SOURCES = ("lerobot_v21", "lerobot_v30")
CONVERTER_SOURCES = (
    # spec-driven: which dataset comes from the registry, not from the source type
    "spec",
    "agibot",
    "libero",
    "openx",
    "robocasa",
    "robomind",
)
SOURCE_TYPES = LEROBOT_SOURCES + CONVERTER_SOURCES
DEST_TYPES = LEROBOT_SOURCES

_TOP_LEVEL_KEYS = {"name", "dataset", "profile", "source", "steps", "dest", "runtime"}
_SOURCE_KEYS = {"type", "path", "args"}
_DEST_KEYS = {"type", "path"}
_RUNTIME_KEYS = {"workers", "threads_per_ffmpeg", "preset", "crf", "encoding"}


class ConfigError(ValueError):
    """Raised for any malformed pipeline config."""


@dataclass(frozen=True)
class SourceConfig:
    type: str
    path: Path
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_lerobot(self) -> bool:
        return self.type in LEROBOT_SOURCES


@dataclass(frozen=True)
class DestConfig:
    type: str
    path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """``None`` everywhere means "decide automatically at run time"."""

    workers: int | None = None
    threads_per_ffmpeg: int | None = None
    preset: str | None = None
    crf: int | None = None
    # overrides layered on the source-derived encoder settings; comes from a named
    # profile in configs/encoding or from an inline mapping
    encoding: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    source: SourceConfig
    dest: DestConfig
    steps: tuple[Any, ...]
    runtime: RuntimeConfig
    # the registry entry this run builds, when the config names one. Carries the
    # source layout, the state layout and the upstream identity.
    dataset: Any = None
    # the processing convention the run was resolved under, for reporting
    profile: str | None = None


def load_config(path: str | Path) -> PipelineConfig:
    import yaml

    path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path} as YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    return parse_config(raw, base_dir=path.parent)


def parse_config(raw: Mapping[str, Any], base_dir: Path | None = None) -> PipelineConfig:
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "config")

    name = _require(raw, "name", "config")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("'name' must be a non-empty string")

    profile_name = raw.get("profile")
    profile = _load_profile(profile_name)
    spec = _load_spec(raw.get("dataset"), profile)

    raw = _apply_profile(raw, profile, spec)

    source = _parse_source(_require(raw, "source", "config"), base_dir)
    dest = _parse_dest(_require(raw, "dest", "config"), base_dir)
    steps = _parse_steps(raw.get("steps") or [])
    runtime = _parse_runtime(raw.get("runtime") or {})

    _check_converter_args(source)

    if source.path == dest.path:
        raise ConfigError(
            f"source.path and dest.path are the same ({source.path}); "
            "the pipeline never modifies its input in place"
        )

    return PipelineConfig(
        name=name,
        source=source,
        dest=dest,
        steps=steps,
        runtime=runtime,
        dataset=spec,
        profile=profile_name if isinstance(profile_name, str) else None,
    )


def _check_converter_args(source: SourceConfig) -> None:
    """Reject args the converter would reject, here rather than hours into a run."""
    if source.is_lerobot or not source.args:
        return
    from .converter_args import check
    from .stages import _CONVERTERS

    entry = _CONVERTERS.get(source.type)
    if entry is None:
        return
    problems = check(entry[0], source.args, source.type)
    if problems:
        raise ConfigError("; ".join(problems))


def _load_profile(name: Any) -> dict[str, Any]:
    if name is None:
        return {}
    from .profiles import ProfileError, load_profile

    try:
        return load_profile(name)
    except ProfileError as exc:
        raise ConfigError(f"profile: {exc}") from exc


def _load_spec(name: Any, profile: Mapping[str, Any]) -> Any:
    """The registry entry, built under whatever layouts the profile asks for."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise ConfigError(f"'dataset' must be a dataset name, got {name!r}")

    from dataset_registry import SpecError, load

    layouts = (profile.get("state") or {}).get("layouts") or {}
    try:
        return load(name, layouts=layouts)
    except SpecError as exc:
        raise ConfigError(f"dataset: {exc}") from exc


def _apply_profile(
    raw: Mapping[str, Any], profile: Mapping[str, Any], spec: Any
) -> dict[str, Any]:
    """Fill in whatever the run config left to the convention.

    Anything written in the run config wins: a profile is a default, so a one-off
    run can differ from the collection without a second profile file. What a profile
    supplies is the video step, the encoder settings, the output version, and -- when
    a dataset is named -- the source type and the state_layout step.
    """
    filled = dict(raw)

    if spec is not None:
        source = dict(filled.get("source") or {})
        builder = spec.source.builder if spec.source else "none"
        # `none` means the source is already LeRobot, so there is nothing to convert
        source.setdefault("type", "lerobot_v21" if builder == "none" else builder)
        args = dict(spec.source.args) if spec.source else {}
        if source["type"] == "spec":
            # the spec-driven converter is dataset-agnostic; which dataset is a flag
            args["dataset"] = spec.id
        # the run config still wins, so a one-off run can override
        source["args"] = {**args, **(source.get("args") or {})}
        filled["source"] = source

    if "steps" not in filled:
        steps: list[Any] = []
        if spec is not None and not (
            spec.source and spec.source.builds_its_own_vectors
        ):
            # a pre-existing converter writes observation.state itself; running the
            # layout step over its output would restate a convention it already applied
            # the already-resolved spec, so the step sees the profile's layouts
            steps.append({"type": "state_layout", "spec": spec})
        # the spec says whether this dataset is resized at all; the profile says how
        resize = (profile.get("video") or {}).get("resize")
        if resize and (spec is None or spec.is_resized):
            steps.append(dict(resize))
        if steps:
            filled["steps"] = steps

    version = (profile.get("dest") or {}).get("version")
    if version and "type" not in (filled.get("dest") or {}):
        filled["dest"] = {**(filled.get("dest") or {}), "type": version}

    encoding = (profile.get("video") or {}).get("encoding")
    if encoding and "encoding" not in (filled.get("runtime") or {}):
        filled["runtime"] = {**(filled.get("runtime") or {}), "encoding": encoding}

    return filled


def _parse_source(raw: Any, base_dir: Path | None) -> SourceConfig:
    raw = _require_mapping(raw, "source")
    _reject_unknown_keys(raw, _SOURCE_KEYS, "source")

    type_name = _require(raw, "type", "source")
    if type_name not in SOURCE_TYPES:
        raise ConfigError(
            f"unknown source.type {type_name!r}. "
            f"supported: {', '.join(SOURCE_TYPES)}"
        )

    args = raw.get("args") or {}
    if args and type_name in LEROBOT_SOURCES:
        raise ConfigError(
            f"source.args is only meaningful for converter sources "
            f"({', '.join(CONVERTER_SOURCES)}); source.type={type_name!r} takes none"
        )
    if not isinstance(args, Mapping):
        raise ConfigError("source.args must be a mapping")

    return SourceConfig(
        type=type_name,
        path=_resolve(_require(raw, "path", "source"), base_dir),
        args=dict(args),
    )


def _parse_dest(raw: Any, base_dir: Path | None) -> DestConfig:
    raw = _require_mapping(raw, "dest")
    _reject_unknown_keys(raw, _DEST_KEYS, "dest")

    type_name = _require(raw, "type", "dest")
    if type_name not in DEST_TYPES:
        raise ConfigError(
            f"unknown dest.type {type_name!r}. supported: {', '.join(DEST_TYPES)}"
        )

    return DestConfig(type=type_name, path=_resolve(_require(raw, "path", "dest"), base_dir))


def _parse_steps(raw: Any) -> tuple[Any, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ConfigError("'steps' must be a list")

    steps = []
    for index, spec in enumerate(raw):
        if not isinstance(spec, Mapping):
            raise ConfigError(f"steps[{index}] must be a mapping, got {spec!r}")
        try:
            steps.append(build_step(spec))
        except (UnknownStepError, ValueError) as exc:
            raise ConfigError(f"steps[{index}]: {exc}") from exc
    return tuple(steps)


def _parse_runtime(raw: Any) -> RuntimeConfig:
    raw = _require_mapping(raw, "runtime")
    _reject_unknown_keys(raw, _RUNTIME_KEYS, "runtime")

    def positive_int(key: str) -> int | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"runtime.{key} must be a positive integer, got {value!r}")
        return value

    crf = raw.get("crf")
    if crf is not None and (not isinstance(crf, int) or isinstance(crf, bool)):
        raise ConfigError(f"runtime.crf must be an integer, got {crf!r}")

    preset = raw.get("preset")
    if preset is not None and not isinstance(preset, str):
        raise ConfigError(f"runtime.preset must be a string, got {preset!r}")

    encoding = raw.get("encoding")
    if encoding is not None:
        if not isinstance(encoding, (str, Mapping)):
            raise ConfigError(
                "runtime.encoding must be a profile name or a mapping of encoder "
                f"settings, got {encoding!r}"
            )
        from .encoding import EncodingProfileError, load_profile

        try:
            encoding = load_profile(encoding)
        except EncodingProfileError as exc:
            raise ConfigError(f"runtime.encoding: {exc}") from exc

    return RuntimeConfig(
        workers=positive_int("workers"),
        threads_per_ffmpeg=positive_int("threads_per_ffmpeg"),
        preset=preset,
        crf=crf,
        encoding=encoding,
    )


def _require(raw: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in raw:
        raise ConfigError(f"{section} is missing required key {key!r}")
    return raw[key]


def _require_mapping(raw: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{section}' must be a mapping, got {raw!r}")
    return raw


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {section}: {', '.join(unknown)}. "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _resolve(value: Any, base_dir: Path | None) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"path must be a string, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve()
    return path
