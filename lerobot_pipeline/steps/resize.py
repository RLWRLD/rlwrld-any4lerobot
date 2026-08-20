"""Aspect-ratio preserving resize used by RLDX-family models.

The geometry function is kept byte-for-byte equivalent to the reference logic the
model team trained with. Only an explicit area assertion is added on top, because
the reference `max(m, ...)` guard silently breaks the area bound for degenerate
aspect ratios.
"""

import math
from collections.abc import Sequence

from ..registry import VideoPlan, register_step

DEFAULT_MAX_AREA = 256**2
DEFAULT_MULTIPLE = 32

# libswscale's own names, because both paths that resize end up in libswscale: the
# transform stage through ffmpeg's `scale` filter, and openx2lerobot through PyAV's
# reformatter, whose Interpolation members are these words uppercased.
#
# bicubic is the default here because it is what was shipped -- ffmpeg's `scale`
# defaults to it and video_rules asked PyAV for it by name, so the two agreed by
# coincidence rather than by declaration. Naming it is the point: a filter that is
# implicit on one path and explicit on the other is one edit away from silently
# building half a collection differently from the other half.
RESIZE_FILTERS = ("bicubic", "bilinear", "lanczos", "sinc", "area", "gauss", "bicublin")
DEFAULT_FILTER = "bicubic"


class UnknownFilterError(ValueError):
    """Raised when a resize names a resampler libswscale does not have."""


class AreaBoundError(ValueError):
    """Raised when the computed crop exceeds the requested area bound."""


def resize_preserve_aspect_area_then_crop(
    h: int,
    w: int,
    max_area: int = DEFAULT_MAX_AREA,
    multiple: int = DEFAULT_MULTIPLE,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``((resize_h, resize_w), (crop_h, crop_w))`` for a source of size ``h x w``.

    Downscale only, aspect ratio preserved, area bounded by ``max_area``, and both
    crop dimensions multiples of ``multiple``.
    """
    # downscale only (no upscaling)
    smax = min(1.0, math.sqrt(max_area / (h * w)))

    short, long_ = (h, w) if h <= w else (w, h)

    # make the shorter side a multiple of `multiple`, as large as possible under area bound
    short_r = max(multiple, int((short * smax) // multiple) * multiple)
    s = short_r / short

    # preserve aspect ratio on resize (floor to keep area <= max_area)
    long_r = int(long_ * s)

    # assign back to (H, W)
    h_r, w_r = (short_r, long_r) if h <= w else (long_r, short_r)

    # crop down to multiples of `multiple` (doesn't increase area)
    h_c = h_r - (h_r % multiple)
    w_c = w_r - (w_r % multiple)

    if h_c * w_c > max_area:
        raise AreaBoundError(
            f"source {h}x{w} resolves to a {h_c}x{w_c} crop "
            f"({h_c * w_c} px) which exceeds max_area={max_area}. "
            "This happens when the short side is already <= "
            f"multiple={multiple}; such a source cannot be preprocessed."
        )

    return (h_r, w_r), (h_c, w_c)


@register_step("resize_preserve_aspect_area")
class ResizePreserveAspectArea:
    """Downscale each video so the frame area fits ``max_area`` while keeping its
    aspect ratio, then centre-crop both sides to multiples of ``multiple``."""

    kind = "video"

    def __init__(
        self,
        max_area: int = DEFAULT_MAX_AREA,
        multiple: int = DEFAULT_MULTIPLE,
        keys: Sequence[str] | None = None,
        filter: str = DEFAULT_FILTER,
    ):
        self.max_area = int(max_area)
        self.multiple = int(multiple)
        self.keys = tuple(keys) if keys else None
        if filter not in RESIZE_FILTERS:
            raise UnknownFilterError(
                f"unknown resize filter {filter!r}; "
                f"expected one of {', '.join(RESIZE_FILTERS)}"
            )
        self.filter = filter

    def applies_to(self, key: str) -> bool:
        return self.keys is None or key in self.keys

    def plan(self, shape: tuple[int, int]) -> VideoPlan | None:
        h, w = shape
        (h_r, w_r), (h_c, w_c) = resize_preserve_aspect_area_then_crop(
            h, w, self.max_area, self.multiple
        )

        filters: list[str] = []
        if (h_r, w_r) != (h, w):
            # flags= is passed even when it names the default: this filter decides how
            # much detail survives a downscale, and a setting that reads as absent is
            # a setting nobody reviews. dlr_edan's video came out 0.79x the delivered
            # size on the shipped filter, against 0.98x for datasets not resized at all.
            filters.append(f"scale={w_r}:{h_r}:flags={self.filter}")
        if (h_c, w_c) != (h_r, w_r):
            # ffmpeg's crop filter centres by default
            filters.append(f"crop={w_c}:{h_c}")

        if not filters:
            return None
        return VideoPlan(tuple(filters), (h_c, w_c))
