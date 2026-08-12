import copy
import json

import pytest

from lerobot_pipeline.meta import (
    MetadataError,
    encoding_from_info,
    feature_shape,
    load_info,
    patch_video_feature,
    template_to_glob,
    video_files,
    video_keys,
    write_info,
)

V21_VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V30_VIDEO_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

CAM = "observation.images.egocentric"
WRIST = "observation.images.wrist"


def _info(video_template=V21_VIDEO_TEMPLATE, codebase_version="v2.1"):
    return {
        "codebase_version": codebase_version,
        "fps": 30,
        "video_path": video_template,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            CAM: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": 30,
                },
            },
            WRIST: {
                "dtype": "video",
                "shape": [240, 320, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.height": 240, "video.width": 320, "video.codec": "h264"},
            },
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
        },
    }


def _dataset(tmp_path, info):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


# --- reading -----------------------------------------------------------------


def test_info_is_read_from_the_meta_directory(tmp_path):
    root = _dataset(tmp_path, _info())
    assert load_info(root)["codebase_version"] == "v2.1"


def test_missing_info_json_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MetadataError) as exc:
        load_info(tmp_path / "empty")
    assert "info.json" in str(exc.value)


def test_only_video_features_are_reported_as_video_keys():
    assert video_keys(_info()) == [CAM, WRIST]


def test_feature_shape_is_returned_as_a_tuple():
    assert feature_shape(_info(), CAM) == (480, 640, 3)


def test_round_trips_through_disk(tmp_path):
    root = _dataset(tmp_path, _info())
    info = load_info(root)
    info["fps"] = 15
    write_info(info, root)
    assert load_info(root)["fps"] == 15


# --- version-agnostic file discovery ----------------------------------------


def test_v21_template_becomes_a_glob_for_one_video_key():
    assert template_to_glob(V21_VIDEO_TEMPLATE, CAM) == f"videos/*/{CAM}/*.mp4"


def test_v30_template_becomes_a_glob_for_one_video_key():
    assert template_to_glob(V30_VIDEO_TEMPLATE, CAM) == f"videos/{CAM}/*/*.mp4"


def test_finds_v21_per_episode_video_files(tmp_path):
    root = _dataset(tmp_path, _info())
    for episode in range(3):
        path = root / "videos" / "chunk-000" / CAM / f"episode_{episode:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "videos" / "chunk-000" / WRIST).mkdir(parents=True)
    (root / "videos" / "chunk-000" / WRIST / "episode_000000.mp4").write_bytes(b"x")

    found = video_files(root, _info(), CAM)
    assert len(found) == 3
    assert all(path.parent.name == CAM for path in found)


def test_finds_v30_chunked_video_files(tmp_path):
    info = _info(V30_VIDEO_TEMPLATE, "v3.0")
    root = _dataset(tmp_path, info)
    for chunk in range(2):
        path = root / "videos" / CAM / f"chunk-{chunk:03d}" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert len(video_files(root, info, CAM)) == 2


def test_a_dataset_without_videos_reports_no_video_keys():
    info = _info()
    info["video_path"] = None
    info["features"] = {"observation.state": {"dtype": "float32", "shape": [7]}}
    assert video_keys(info) == []


def test_video_key_without_a_path_template_is_an_error():
    info = _info()
    info["video_path"] = None
    with pytest.raises(MetadataError):
        video_files("/nowhere", info, CAM)


# --- patching ----------------------------------------------------------------


def test_patching_updates_the_feature_shape_and_keeps_the_channel_count():
    info = patch_video_feature(copy.deepcopy(_info()), CAM, (192, 256))
    assert info["features"][CAM]["shape"] == [192, 256, 3]


def test_patching_updates_the_nested_video_info_block():
    info = patch_video_feature(copy.deepcopy(_info()), CAM, (192, 256))
    assert info["features"][CAM]["info"]["video.height"] == 192
    assert info["features"][CAM]["info"]["video.width"] == 256


def test_patching_one_key_leaves_the_others_untouched():
    info = patch_video_feature(copy.deepcopy(_info()), CAM, (192, 256))
    assert info["features"][WRIST]["shape"] == [240, 320, 3]


def test_patching_preserves_unrelated_video_info_fields():
    info = patch_video_feature(copy.deepcopy(_info()), CAM, (192, 256))
    assert info["features"][CAM]["info"]["video.codec"] == "h264"
    assert info["features"][CAM]["info"]["video.fps"] == 30


def test_patching_an_unknown_key_is_an_error():
    with pytest.raises(MetadataError):
        patch_video_feature(_info(), "observation.images.nope", (192, 256))


def test_patching_a_non_video_feature_is_an_error():
    with pytest.raises(MetadataError):
        patch_video_feature(_info(), "observation.state", (192, 256))


# --- encoding parameters mirrored from the source ---------------------------


def test_encoding_mirrors_the_source_codec_and_pixel_format():
    encoding = encoding_from_info(_info(), CAM)
    assert encoding.codec == "libx264"
    assert encoding.pix_fmt == "yuv420p"


def test_av1_sources_are_re_encoded_with_an_av1_encoder_not_silently_switched():
    info = _info()
    info["features"][CAM]["info"]["video.codec"] = "av1"
    encoding = encoding_from_info(info, CAM)
    assert encoding.codec == "libsvtav1"


def test_av1_gets_a_numeric_preset_because_svtav1_rejects_x264_preset_names():
    info = _info()
    info["features"][CAM]["info"]["video.codec"] = "av1"
    assert encoding_from_info(info, CAM).preset.isdigit()


def test_unknown_source_codec_fails_loudly_rather_than_defaulting():
    info = _info()
    info["features"][CAM]["info"]["video.codec"] = "theora"
    with pytest.raises(MetadataError) as exc:
        encoding_from_info(info, CAM)
    assert "theora" in str(exc.value)


def test_gop_defaults_to_the_short_interval_lerobot_uses_for_random_access():
    assert encoding_from_info(_info(), CAM).gop == 2


def test_missing_codec_information_is_an_error_not_a_guess():
    info = _info()
    info["features"][CAM]["info"].pop("video.codec")
    with pytest.raises(MetadataError):
        encoding_from_info(info, CAM)
