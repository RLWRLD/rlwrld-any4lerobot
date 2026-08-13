"""Lining a robot stream up with a camera stream.

A source that samples its robot and its cameras on separate clocks has to decide
which robot sample belongs to which video frame. That decision is an algorithm, not
a path or a key, so it is the one part of reading a dataset that cannot be written
in YAML. It is named instead: a spec says ``strategy: nearest_timestamp_dedup`` and
the code behind that name is shared by every dataset that needs it.

Strategies return the robot-sample indices to emit, one per kept video frame, in
video order. Anything they drop is dropped from both sides at once.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np


class ClockError(ValueError):
    """Raised when the two streams cannot be lined up at all."""


_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}


def register_clock(name: str):
    def decorator(func):
        _REGISTRY[name] = func
        return func

    return decorator


def available_clocks() -> list[str]:
    return sorted(_REGISTRY)


def align(strategy: str, data_times: Any, image_times: Any) -> np.ndarray:
    """Robot-sample indices to emit, one per kept video frame."""
    func = _REGISTRY.get(strategy)
    if func is None:
        raise ClockError(
            f"unknown clock strategy {strategy!r}. "
            f"available: {', '.join(available_clocks())}"
        )
    return func(data_times, image_times)


def parse_times(values, fmt: str | None) -> np.ndarray:
    """Video frame times as seconds since the epoch.

    ``fmt`` is a ``strptime`` pattern for sources that write wall-clock strings;
    without one the values are already numeric.
    """
    if fmt is None:
        return np.asarray(values, dtype=np.float64)
    return np.array(
        [
            datetime.strptime(str(value).split(".")[0], fmt)
            .replace(tzinfo=timezone.utc)
            .timestamp()
            for value in values
        ],
        dtype=np.float64,
    )


@register_clock("index")
def index(data_times, image_times) -> np.ndarray:
    """One robot sample per video frame, in order. For single-clock sources."""
    count = min(len(data_times), len(image_times))
    if count == 0:
        raise ClockError("one of the two streams is empty")
    return np.arange(count, dtype=np.int64)


def _nearest(candidate: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """For every time in ``ref``, the index of the nearest time in ``candidate``.

    Ported verbatim from the reference converter (``FFTAI/fourier-lerobot``,
    ``scripts/convert_hdf5_to_lerobot.py``) so that a rebuilt dataset lines up with
    what the upstream training pipeline was built against. ``candidate`` must be
    sorted ascending.

    A robot sample is claimed by at most one video frame: when the nearest one is
    already taken the next is used instead, and if that is taken too the frame is
    dropped. So the result can be shorter than ``ref``.
    """
    closest_indices: list[int] = []
    already_matched: set[int] = set()
    for time in ref:
        idx = int(np.searchsorted(candidate, time, side="left"))
        if idx > 0 and (
            idx == candidate.size
            or np.fabs(time - candidate[idx - 1]) < np.fabs(time - candidate[idx])
        ):
            idx -= 1
        if idx not in already_matched:
            closest_indices.append(idx)
            already_matched.add(idx)
        elif idx + 1 not in already_matched and idx + 1 < candidate.size:
            closest_indices.append(idx + 1)
            already_matched.add(idx + 1)
    return np.array(closest_indices, dtype=np.int64)


@register_clock("nearest_timestamp_dedup")
def nearest_timestamp_dedup(data_times, image_times) -> np.ndarray:
    """Nearest robot sample per video frame, each sample claimed at most once.

    Two filters run first, both the reference converter's: keep only robot samples
    whose timestamp advances, and only video frames recorded before the last robot
    sample. The second typically drops the final frame, so an emitted episode can be
    one row shorter than its mp4 -- the reference converter copies the mp4 unchanged
    and leaves that trailing frame unreferenced, and so do we.
    """
    data_times = np.asarray(data_times, dtype=np.float64)
    image_times = np.asarray(image_times, dtype=np.float64)
    if data_times.size == 0:
        raise ClockError("no robot samples")
    if image_times.size == 0:
        raise ClockError("no video frames")

    usable = np.where(np.diff(data_times) > 0)[0]
    if usable.size == 0:
        raise ClockError("no usable robot timestamps")
    image_times = image_times[image_times < data_times[-1]]
    if image_times.size == 0:
        raise ClockError("no video frames inside the robot time range")

    # _nearest indexes the filtered array; map back to the full one. The two
    # coincide whenever the timestamps are strictly increasing, which is what every
    # episode inspected so far looks like.
    return usable[_nearest(data_times[usable], image_times)]
