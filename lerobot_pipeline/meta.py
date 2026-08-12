"""LeRobot dataset metadata: reading, video discovery and shape patching.

Nothing here hardcodes a per-version directory layout. ``info.json`` carries its
own ``video_path`` template, so turning that template into a glob handles v2.1
(one video per episode) and v3.0 (videos concatenated into chunk files) through
the same code path -- and keeps working if the layout changes again.
"""

import json
import re
from pathlib import Path
from typing import Any

from .video_ops import EncodingParams

INFO_RELPATH = Path("meta") / "info.json"

# LeRobot writes a very short keyframe interval on purpose: training samples
# random frames, and a long GOP makes every read walk back to the last keyframe.
LEROBOT_GOP = 2

# Re-encoding must stay within the source's codec family. Silently turning an
# AV1 dataset into H.264 (or handing libsvtav1 an x264 preset name) is exactly
# the kind of failure that only shows up much later.
_CODEC_PROFILES: dict[str, tuple[str, str, int]] = {
    # source codec -> (encoder, preset, crf)
    "h264": ("libx264", "fast", 18),
    "avc1": ("libx264", "fast", 18),
    "hevc": ("libx265", "fast", 23),
    "h265": ("libx265", "fast", 23),
    "av1": ("libsvtav1", "8", 30),
}

_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


class MetadataError(ValueError):
    """Raised when dataset metadata is missing, malformed or inconsistent."""


def load_info(root: str | Path) -> dict[str, Any]:
    path = Path(root) / INFO_RELPATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise MetadataError(f"no LeRobot info.json at {path}") from exc
    except json.JSONDecodeError as exc:
        raise MetadataError(f"could not parse {path}: {exc}") from exc


def write_info(info: dict[str, Any], root: str | Path) -> None:
    path = Path(root) / INFO_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=4))


def video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in (info.get("features") or {}).items()
        if feature.get("dtype") == "video"
    ]


def feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(_feature(info, key)["shape"])


def template_to_glob(template: str, video_key: str) -> str:
    """Turn a LeRobot path template into a glob for one video key.

    Segments that only encode chunk/episode indices collapse to ``*``; the
    ``{video_key}`` segment is substituted so other cameras are not matched.
    """
    segments = []
    for segment in template.split("/"):
        if "{video_key}" in segment:
            segments.append(segment.replace("{video_key}", video_key))
            continue

        matches = list(_PLACEHOLDER.finditer(segment))
        if matches:
            # collapse everything through the last placeholder, keep any suffix
            # such as the file extension
            segments.append("*" + segment[matches[-1].end() :])
        else:
            segments.append(segment)
    return "/".join(segments)


def video_files(root: str | Path, info: dict[str, Any], key: str) -> list[Path]:
    template = info.get("video_path")
    if not template:
        raise MetadataError(
            f"info.json has no 'video_path' template, so the files for video key "
            f"{key!r} cannot be located"
        )
    return sorted(Path(root).glob(template_to_glob(template, key)))


def patch_video_feature(
    info: dict[str, Any], key: str, shape: tuple[int, int]
) -> dict[str, Any]:
    """Record a video key's new ``(H, W)`` in ``info``, in place."""
    feature = _feature(info, key)
    if feature.get("dtype") != "video":
        raise MetadataError(
            f"feature {key!r} has dtype {feature.get('dtype')!r}, not 'video'"
        )

    height, width = shape
    feature["shape"] = [height, width, *list(feature.get("shape", []))[2:]]

    nested = feature.get("info")
    if isinstance(nested, dict):
        if "video.height" in nested:
            nested["video.height"] = height
        if "video.width" in nested:
            nested["video.width"] = width

    return info


def encoding_from_info(info: dict[str, Any], key: str) -> EncodingParams:
    """Derive encoder settings that mirror the source video for one key."""
    nested = _feature(info, key).get("info") or {}

    source_codec = nested.get("video.codec")
    if not source_codec:
        raise MetadataError(
            f"info.json does not record 'video.codec' for {key!r}; refusing to "
            "guess an encoder for re-encoding"
        )

    profile = _CODEC_PROFILES.get(str(source_codec).lower())
    if profile is None:
        raise MetadataError(
            f"unsupported source codec {source_codec!r} for {key!r}. "
            f"supported: {', '.join(sorted(_CODEC_PROFILES))}"
        )

    codec, preset, crf = profile
    return EncodingParams(
        codec=codec,
        preset=preset,
        crf=crf,
        gop=LEROBOT_GOP,
        pix_fmt=nested.get("video.pix_fmt") or "yuv420p",
    )


def _feature(info: dict[str, Any], key: str) -> dict[str, Any]:
    features = info.get("features") or {}
    if key not in features:
        raise MetadataError(
            f"unknown feature {key!r}; dataset has: {', '.join(sorted(features))}"
        )
    return features[key]
