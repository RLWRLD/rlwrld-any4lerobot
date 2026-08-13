"""Named processing conventions.

A profile is the answer to "how does this collection get built" -- state layout,
video geometry, encoding, output version -- in one file, separate from *which*
dataset is being built and from *where* this particular run reads and writes.

That separation is what makes switching conventions cheap. The RLDX-1 datasets were
laid out for a checkpoint whose per-embodiment projector was trained on that exact
slot order; moving to a different order later means writing a second profile, not
editing every dataset spec.

Same pattern as ``encoding.py``: a name resolves to a file under ``configs/``, and
an inline mapping is accepted where a name would be.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(__file__).resolve().parent / "configs" / "profiles"

_TOP_LEVEL = {"state", "video", "dest", "builders", "note"}
_STATE = {"build_layout_as"}
_VIDEO = {"resize", "encoding"}
_DEST = {"version", "uri"}


class ProfileError(ValueError):
    """Raised for a malformed or unknown profile."""


def available_profiles() -> list[str]:
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a profile name, or validate an inline mapping."""
    if isinstance(source, Mapping):
        return _validate(dict(source), "<inline profile>")

    import yaml

    path = PROFILE_DIR / f"{source}.yaml"
    if Path(source).name != source or not path.is_file():
        raise ProfileError(
            f"unknown profile {source!r}. available: {', '.join(available_profiles())}"
        )
    return _validate(yaml.safe_load(path.read_text()) or {}, str(path))


def _validate(raw: Any, origin: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{origin}: profile must be a mapping")
    _reject(raw, _TOP_LEVEL, origin)
    _reject(raw.get("state") or {}, _STATE, f"{origin}.state")
    _reject(raw.get("video") or {}, _VIDEO, f"{origin}.video")
    _reject(raw.get("dest") or {}, _DEST, f"{origin}.dest")

    layouts = (raw.get("state") or {}).get("build_layout_as") or {}
    if not isinstance(layouts, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in layouts.items()
    ):
        raise ProfileError(
            f"{origin}.state.build_layout_as maps the layout a dataset declares to "
            "the layout to actually build it with; both sides must be layout names"
        )

    builders = raw.get("builders") or {}
    if not isinstance(builders, Mapping) or not all(
        isinstance(value, Mapping) for value in builders.values()
    ):
        raise ProfileError(f"{origin}.builders must map builder name -> flags")

    resize = (raw.get("video") or {}).get("resize")
    if resize is not None and not isinstance(resize, Mapping):
        raise ProfileError(f"{origin}.video.resize must be a step mapping")
    if isinstance(resize, Mapping) and "type" not in resize:
        raise ProfileError(f"{origin}.video.resize is missing 'type'")
    return dict(raw)


def _reject(raw: Any, allowed: set[str], where: str) -> None:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{where} must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ProfileError(
            f"{where}: unknown key(s) {', '.join(unknown)}. "
            f"allowed: {', '.join(sorted(allowed))}"
        )
