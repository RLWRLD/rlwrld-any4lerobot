"""The video rules this converter is handed, resolved to what it needs.

openx2lerobot does not transcode an existing file: it decodes RLDS frames and hands
them to LeRobot's own video writer. So the two decisions the rest of the pipeline
expresses as a transform stage -- what size the frames are, and how they are encoded
-- have to be made *here*, before the write. Doing them afterwards would decode and
re-encode what this converter just produced, a second lossy generation that the
delivered datasets do not have.

The rules arrive as inputs rather than being decided here: a step mapping (or the
name of one) for the resize, the name of a file in
``lerobot_pipeline/configs/encoding`` (or an inline mapping) for the encoder, and a
name for the channel order. Naming a different set converts the same source a
different way, with no code change.
"""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# openx_rlds.py is run by path, not as a module, so the repo root is only on the
# path when the run happens to start there. Put it on deliberately: the geometry and
# the encoder settings must come from the same definitions the transform stage uses,
# and a second copy of either would drift.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from lerobot_pipeline.encoding import load_profile, unwritable_settings  # noqa: E402
from lerobot_pipeline.registry import build_step, compose_video_plans  # noqa: E402


class VideoRuleError(ValueError):
    """Raised for a rule that cannot be resolved into something to run."""


def parse_rule(value: str | Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A rule given on the command line, as a mapping.

    Accepts JSON (what the pipeline passes, so parameters survive) or a bare name
    (what a person types, which then takes the rule's own defaults).
    """
    if value is None or isinstance(value, Mapping):
        return value
    text = value.strip()
    if not text:
        return None
    if not text.startswith("{"):
        return {"type": text}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VideoRuleError(f"could not parse {value!r} as JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise VideoRuleError(f"a rule must be a mapping, got {parsed!r}")
    return parsed


# Cameras Open X-Embodiment reads as BGR, and the datasets they belong to.
#
# The upstream standardisation transforms flip these to RGB, with the comment "flip
# image & wrist_image from bgr to rgb". The delivered copies were written without that
# flip, so their colours are the source bytes read as RGB -- red and blue exchanged
# against what the camera saw. Measured on utaustin_mutex against the RLDS source:
# the delivered frames correlate 0.99 with the source read as-is and 0.73 with it
# flipped, and the delivered channel means track the source's exactly.
#
# The list lives here rather than inside the transforms because *whether* to flip is
# a rule a run is given. The transforms no longer flip; this does, when asked to.
BGR_CAMERAS = {
    "berkeley_autolab_ur5": ("hand_image",),
    "stanford_hydra_dataset_converted_externally_to_rlds": ("image", "wrist_image"),
    "utaustin_mutex": ("image", "wrist_image"),
    "berkeley_fanuc_manipulation": ("image", "wrist_image"),
    "fmb_dataset": (
        "image_wrist_1",
        "image_wrist_2",
        "image_side_1",
        "image_side_2",
    ),
}

CHANNEL_RULES = ("as_source", "bgr_to_rgb")


def flips_channels(channels, dataset_name: str, key: str) -> bool:
    """Whether frames from camera ``key`` should have their channels reversed.

    ``as_source`` writes the bytes in the order the RLDS file stores them, which is
    what the delivered copies did. ``bgr_to_rgb`` reverses the cameras listed in
    :data:`BGR_CAMERAS`, which is what Open X-Embodiment's own transforms do and what
    makes the colours match the scene. Every camera not on that list is unaffected by
    either, so most datasets convert the same way under both.
    """
    rule = parse_rule(channels)
    if not rule:
        return False
    name = rule.get("type")
    if name not in CHANNEL_RULES:
        raise VideoRuleError(
            f"unknown channel rule {name!r}; expected one of {', '.join(CHANNEL_RULES)}"
        )
    return name == "bgr_to_rgb" and key in BGR_CAMERAS.get(dataset_name, ())


def target_shape(
    resize: Mapping[str, Any] | None, key: str, shape: tuple[int, int]
) -> tuple[int, int]:
    """The ``(height, width)`` frames for ``key`` should be written at.

    The geometry comes from the same step objects the transform stage uses, so the
    two paths cannot drift apart: only *when* the resize happens differs.
    """
    if not resize:
        return shape
    step = build_step(dict(resize))
    plan = compose_video_plans([step], f"observation.images.{key}", shape)
    return shape if plan is None else plan.out_shape


def resize_filter(resize: Mapping[str, Any] | None) -> str:
    """Which resampler ``resize`` asks for, as a libswscale name.

    Read off the same step object ``target_shape`` builds its geometry from, so the
    filter cannot disagree with the one the transform stage puts in its ``scale``
    filter chain -- the two paths differ in *when* they resize and in nothing else.

    One value for the whole dataset. It used to take the camera and its geometry as
    well, because the profile could name a rule that picked per downscale factor;
    the collection is one filter now and the question no longer has a per-camera
    answer.
    """
    from lerobot_pipeline.steps.resize import DEFAULT_FILTER

    if not resize:
        return DEFAULT_FILTER
    return getattr(build_step(dict(resize)), "filter", DEFAULT_FILTER)


def resize_frame(frame, shape: tuple[int, int], filter: str | None = None):
    """Downscale ``frame`` to ``shape``, centre-cropping whatever the scale leaves over.

    Through libswscale, which is the resizer ffmpeg's ``scale`` filter *is* -- not a
    library that also offers something called bicubic. The transform stage resizes
    with ``scale``, so this is the same rule applied by the same code rather than an
    approximation of it, and PyAV is already here for the encoder.

    The distinction is not academic. OpenCV's INTER_CUBIC has no scale-dependent
    prefilter, so it keeps detail swscale would have low-passed away, and the encoder
    pays for it.

    Which resampler is the profile's to say -- ``video.resize.filter`` -- and this
    reads it rather than naming one, so the value cannot disagree with the one the
    transform stage puts in its ``scale`` filter chain.

    The measurement behind the profile's default lives in
    ``verification/records/resize-filter-sweep.md``. In short: no filter is flat across
    scale factors -- every one comes out larger the gentler the downscale, and that
    offset survives whichever is chosen -- so the question is which stays inside
    ``SIZE_TOLERANCE`` everywhere. sinc does, at 10.0% against a 15% bound.

    An earlier table here reached the opposite conclusion and named bicubic. Two things
    were wrong with it. It measured per-episode ratios paired by filename, and the
    rebuild does not write episodes in the delivered order, so most of what it compared
    was two different episodes; and it never tried lanczos. Redone on total bytes per
    camera, bicubic misses dlr_edan by 18.9% against a 15% bound.
    """
    import av
    from av.video.reformatter import Interpolation

    from lerobot_pipeline.steps.resize import DEFAULT_FILTER, UnknownFilterError

    height, width = shape
    if frame.shape[:2] == (height, width):
        return frame

    name = (filter or DEFAULT_FILTER).upper()
    interpolation = getattr(Interpolation, name, None)
    if interpolation is None:
        # PyAV exposes a subset of libswscale's names, and which subset depends on the
        # build -- SPLINE is missing from the one in the node image. Failing here beats
        # falling back, which would resize a whole collection with a filter nobody asked
        # for and leave no trace of it.
        raise UnknownFilterError(
            f"PyAV in this build has no Interpolation.{name}; "
            f"available: {', '.join(sorted(m.name.lower() for m in Interpolation))}"
        )

    scale = max(height / frame.shape[0], width / frame.shape[1])
    scaled_h = max(height, round(frame.shape[0] * scale))
    scaled_w = max(width, round(frame.shape[1] * scale))
    picture = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(
        width=scaled_w, height=scaled_h, interpolation=interpolation
    )
    resized = picture.to_ndarray(format="rgb24")

    top = (scaled_h - height) // 2
    left = (scaled_w - width) // 2
    return resized[top : top + height, left : left + width]


# encoding profile key -> the field LeRobot's RGBEncoderConfig calls it.
_ENCODER_FIELDS = {
    "codec": "vcodec",
    "gop": "g",
    "crf": "crf",
    "preset": "preset",
    "pix_fmt": "pix_fmt",
}


def rgb_encoder(encoding: str | Mapping[str, Any] | None):
    """LeRobot's encoder config for this rule, or ``None`` to keep its own defaults.

    A rule is a partial override: anything it leaves unset keeps LeRobot's default,
    which is what ``lerobot_av1_default`` means -- it names the writer's own settings
    so that "keep them" is something a run can say rather than something it omits.
    """
    overrides = load_profile(encoding) if encoding is not None else None
    if not overrides:
        return None

    unsatisfiable = unwritable_settings(overrides)
    if unsatisfiable:
        raise VideoRuleError(
            f"encoding rule asks for {', '.join(unsatisfiable)}, which LeRobot's "
            "video writer cannot be told to do. Those settings belong to the ffmpeg "
            "transform stage; an encoding this converter writes with has to be one "
            "the writer can actually produce."
        )

    from lerobot.configs.video import rgb_encoder_defaults

    config = rgb_encoder_defaults()
    for key, field in _ENCODER_FIELDS.items():
        value = overrides.get(key)
        if value is not None:
            setattr(config, field, value)
    return config
