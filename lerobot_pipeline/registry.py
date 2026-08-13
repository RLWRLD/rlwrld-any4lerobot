"""Step registry and the video-step contract.

Steps are referenced from config by registered name only -- arbitrary import paths
are deliberately not supported, so a typo fails config validation instead of
importing something unexpected.

A video step returns a *plan* rather than performing work. Keeping planning pure
lets the runner chain several steps into a single ffmpeg filter chain (one decode/
encode pass), settle the output shape before any work starts, and detect no-ops.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class UnknownStepError(KeyError):
    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class VideoPlan:
    """ffmpeg filters to apply to one video key, and the resulting ``(H, W)``."""

    filters: tuple[str, ...]
    out_shape: tuple[int, int]


@runtime_checkable
class VideoStep(Protocol):
    kind: str

    def applies_to(self, key: str) -> bool: ...

    def plan(self, shape: tuple[int, int]) -> VideoPlan | None:
        """Return the plan for a video key currently sized ``shape``, or None for a no-op."""


_REGISTRY: dict[str, type] = {}


def register_step(name: str):
    def decorator(cls):
        cls.config_name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_steps() -> list[str]:
    return sorted(_REGISTRY)


def build_step(spec: Mapping[str, Any]):
    """Instantiate one step from its config mapping (``{"type": ..., **params}``)."""
    params = dict(spec)
    type_name = params.pop("type", None)
    if type_name is None:
        raise ValueError(f"step is missing required key 'type': {dict(spec)!r}")

    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise UnknownStepError(
            f"unknown step type {type_name!r}. "
            f"available steps: {', '.join(available_steps())}"
        )

    try:
        return cls(**params)
    except TypeError as exc:
        raise ValueError(f"invalid parameters for step {type_name!r}: {exc}") from exc


def compose_video_plans(
    steps: Sequence[Any] | Iterable[Any],
    key: str,
    shape: tuple[int, int],
) -> VideoPlan | None:
    """Fold every applicable video step into a single filter chain for one video key."""
    filters: list[str] = []
    current = shape

    for step in steps:
        if getattr(step, "kind", None) != "video" or not step.applies_to(key):
            continue
        plan = step.plan(current)
        if plan is None:
            continue
        filters.extend(plan.filters)
        current = plan.out_shape

    if not filters:
        return None
    return VideoPlan(tuple(filters), current)


@runtime_checkable
class TableStep(Protocol):
    """A step that rewrites parquet columns and leaves the video alone.

    Distinct from a frame step, which changes how many frames there are and so has
    to re-cut the video too. A table step preserves the row count, which is what
    lets the runner hard-link every mp4 straight through -- the reason this is the
    kind that exists so far and the frame kind is still reserved.
    """

    kind: str

    def apply(self, root, out) -> None: ...
