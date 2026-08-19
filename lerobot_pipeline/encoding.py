"""Named encoder settings, loaded from ``configs/encoding/*.yaml``.

Encoding is the one part of preprocessing that is expensive to get wrong and
impossible to recover after the fact, so the settings live in files rather than in
code: a run can be reproduced by naming a profile, and a new one can be added
without touching the pipeline.

A profile is a partial override. Anything it does not mention keeps the value
derived from the source video, so a profile that only sets ``crf`` still mirrors
the source's codec.
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .video_ops import EncodingParams

PROFILE_DIR = Path(__file__).resolve().parent / "configs" / "encoding"

# Every field a profile is allowed to set, and the type it is coerced to. Keeping
# this explicit means a typo in a profile file is an error rather than a silently
# ignored key.
_FIELDS: dict[str, type] = {
    "codec": str,
    "preset": str,
    "crf": int,
    "gop": int,
    "pix_fmt": str,
    "bframes": int,
    "profile": str,
    "sc_threshold": int,
}


# What LeRobot's own video writer does, for the settings it has no knob for. It
# emits no B-frames, pins no profile and leaves scene-cut detection alone. A rule
# asking for exactly that is already met; a rule asking for anything else -- three
# B-frames, High profile -- describes an ffmpeg command line and can only be applied
# by the transform stage.
_WRITER_LEAVES_AS = {"profile": None, "bframes": 0, "sc_threshold": None}


def unwritable_settings(overrides: Mapping[str, Any] | None) -> list[str]:
    """The settings in ``overrides`` that LeRobot's writer cannot be asked for.

    Empty means a converter can write this encoding itself, which decides where the
    resize happens: before the write, or as a transcode afterwards.
    """
    if not overrides:
        return []
    return sorted(
        f"{key}={overrides[key]!r}"
        for key, writer in _WRITER_LEAVES_AS.items()
        if key in overrides and overrides[key] != writer
    )


class EncodingProfileError(ValueError):
    """Raised for an unknown profile name or a malformed profile file."""


def available_profiles() -> list[str]:
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Return the overrides a profile applies, as a plain dict.

    ``source`` is either the name of a file in ``configs/encoding`` or the settings
    themselves -- as a mapping from a pipeline config, or as the JSON of one, which
    is the shape they take crossing a converter's command line.
    """
    if isinstance(source, str) and source.strip().startswith("{"):
        try:
            source = json.loads(source)
        except json.JSONDecodeError as exc:
            raise EncodingProfileError(
                f"could not parse {source!r} as encoding settings: {exc}"
            ) from exc
    if isinstance(source, Mapping):
        return _validate(source, origin="inline encoding settings")
    return _validate(_read_profile_file(source), origin=f"encoding profile {source!r}")


def apply_profile(
    encoding: EncodingParams, overrides: Mapping[str, Any] | None
) -> EncodingParams:
    """Layer a profile's overrides on top of source-derived settings."""
    if not overrides:
        return encoding
    return replace(encoding, **dict(overrides))


def _read_profile_file(name: str) -> Mapping[str, Any]:
    import yaml

    if not isinstance(name, str) or not name:
        raise EncodingProfileError("encoding profile name must be a non-empty string")
    if Path(name).name != name:
        raise EncodingProfileError(
            f"encoding profile name {name!r} must not contain a path; "
            f"available: {', '.join(available_profiles())}"
        )

    path = PROFILE_DIR / f"{name}.yaml"
    if not path.is_file():
        raise EncodingProfileError(
            f"unknown encoding profile {name!r}. "
            f"available: {', '.join(available_profiles())}"
        )

    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, Mapping):
        raise EncodingProfileError(f"{path} must contain a mapping")
    return loaded


def _validate(raw: Mapping[str, Any], origin: str) -> dict[str, Any]:
    unknown = sorted(set(raw) - set(_FIELDS))
    if unknown:
        raise EncodingProfileError(
            f"{origin}: unknown key(s) {', '.join(unknown)}. "
            f"allowed: {', '.join(sorted(_FIELDS))}"
        )

    overrides: dict[str, Any] = {}
    for name, value in raw.items():
        # an explicit null means "do not pass the flag at all"; that is a real
        # setting, not an absent one
        if value is None:
            overrides[name] = None
            continue
        try:
            overrides[name] = _FIELDS[name](value)
        except (TypeError, ValueError) as exc:
            raise EncodingProfileError(
                f"{origin}: {name} must be {_FIELDS[name].__name__}, got {value!r}"
            ) from exc
    return overrides
