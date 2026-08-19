"""The resize and encoding rules this converter is handed, resolved to what it needs.

openx2lerobot does not transcode an existing file: it decodes RLDS frames and hands
them to LeRobot's own video writer. So the two decisions the rest of the pipeline
expresses as a transform stage -- what size the frames are, and how they are encoded
-- have to be made *here*, before the write. Doing them afterwards would decode and
re-encode what this converter just produced, a second lossy generation that the
delivered datasets do not have.

Both rules arrive as inputs rather than being decided here: a step mapping (or the
name of one) for the resize, and the name of a file in
``lerobot_pipeline/configs/encoding`` (or an inline mapping) for the encoder. Naming
a different pair converts the same source a different way, with no code change.
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


def resize_frame(frame, shape: tuple[int, int]):
    """Downscale ``frame`` to ``shape``, centre-cropping whatever the scale leaves over.

    Through libswscale, which is the resizer ffmpeg's ``scale`` filter *is* -- not a
    library that also offers something called bicubic. The transform stage resizes
    with ``scale``, so this is the same rule applied by the same code rather than an
    approximation of it, and PyAV is already here for the encoder.

    The distinction is not academic. OpenCV's INTER_CUBIC has no scale-dependent
    prefilter, so it keeps detail swscale would have low-passed away, and the encoder
    pays for it.

    Bicubic within swscale is measured too, not assumed. The target is the 0.96-1.00x
    that cameras which are *not* resized come out at, that being the encoder build
    difference on its own:

        filter            ucsd (2.5x down)   taco_play (1.2x down)
        swscale BICUBIC         0.86           1.01 / 1.04    64/64 episodes
        swscale SINC            0.97           1.10 / 1.15    14/64
        cv2 INTER_CUBIC         1.03           1.13 / 1.14    46/64

    No filter hits the target on both, and the reason is not the filter: the gentler
    the downscale the larger everything comes out, and that offset survives whichever
    one is chosen. So the question is which stays inside tolerance everywhere, and
    only bicubic does -- which is also the one ffmpeg would have used.
    """
    import av
    from av.video.reformatter import Interpolation

    height, width = shape
    if frame.shape[:2] == (height, width):
        return frame

    scale = max(height / frame.shape[0], width / frame.shape[1])
    scaled_h = max(height, round(frame.shape[0] * scale))
    scaled_w = max(width, round(frame.shape[1] * scale))
    picture = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(
        width=scaled_w, height=scaled_h, interpolation=Interpolation.BICUBIC
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
