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
CONVERTER_SOURCES = ("agibot", "libero", "openx", "robocasa", "robomind")
SOURCE_TYPES = LEROBOT_SOURCES + CONVERTER_SOURCES
DEST_TYPES = LEROBOT_SOURCES

_TOP_LEVEL_KEYS = {"name", "source", "steps", "dest", "runtime"}
_SOURCE_KEYS = {"type", "path", "args"}
_DEST_KEYS = {"type", "path"}
_RUNTIME_KEYS = {"workers", "threads_per_ffmpeg", "preset", "crf"}


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


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    source: SourceConfig
    dest: DestConfig
    steps: tuple[Any, ...]
    runtime: RuntimeConfig


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

    source = _parse_source(_require(raw, "source", "config"), base_dir)
    dest = _parse_dest(_require(raw, "dest", "config"), base_dir)
    steps = _parse_steps(raw.get("steps") or [])
    runtime = _parse_runtime(raw.get("runtime") or {})

    if source.path == dest.path:
        raise ConfigError(
            f"source.path and dest.path are the same ({source.path}); "
            "the pipeline never modifies its input in place"
        )

    return PipelineConfig(
        name=name, source=source, dest=dest, steps=steps, runtime=runtime
    )


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

    return RuntimeConfig(
        workers=positive_int("workers"),
        threads_per_ffmpeg=positive_int("threads_per_ffmpeg"),
        preset=preset,
        crf=crf,
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
