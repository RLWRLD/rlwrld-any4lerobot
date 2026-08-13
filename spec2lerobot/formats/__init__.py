"""Readers for raw source layouts, selected by name from a dataset spec.

One reader per *file format*, not per dataset. Adding ActionNet or AgiBot is a YAML
file; adding a format that nobody has read before is code. That is the whole reason
this set exists: it grows with the number of ways robot data is stored (a handful),
not with the number of datasets (35 and counting).

A reader answers three questions about a source directory:

* which episodes are in it (``episode_ids``);
* what each episode's prompt is (``prompts``);
* what one episode contains (``read_episode``) -- per-frame source features already
  lined up with the video, plus the video files to carry over.

Readers never build ``observation.state`` or ``action``. Those are assembled later
from the layout, by ``lerobot_pipeline``'s ``state_layout`` step, so that the same
rule applies whether a dataset came through a reader or was already LeRobot.
"""

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


class FormatError(ValueError):
    """Raised when a source cannot be read as the format its spec claims."""


class EpisodeSkipped(Exception):
    """Raised when one episode is unusable; carries the reason for logs."""


def register_format(name: str):
    def decorator(cls):
        cls.format_name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_formats() -> list[str]:
    return sorted(_REGISTRY)


def build_reader(spec, root):
    """Instantiate the reader a dataset spec names."""
    from . import hdf5_episodes  # noqa: F401  -- registers on import

    if spec.source is None:
        raise FormatError(
            f"{spec.id} has no source: section, so its raw files cannot be read"
        )
    cls = _REGISTRY.get(spec.source.format)
    if cls is None:
        raise FormatError(
            f"unknown source format {spec.source.format!r}. "
            f"available: {', '.join(available_formats())}"
        )
    return cls(spec, root)
