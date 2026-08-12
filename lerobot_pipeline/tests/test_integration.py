"""End-to-end checks against real encoded video.

Everything else in the suite is pure-function; these are the tests that would
catch a filter chain that ffmpeg rejects, or an output whose real resolution
disagrees with the metadata we wrote.
"""

import json
import subprocess

import pytest

from lerobot_pipeline.meta import load_info
from lerobot_pipeline.registry import build_step
from lerobot_pipeline.transform import materialize, plan_transform

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg

CAM = "observation.images.egocentric"
SOURCE_H, SOURCE_W = 480, 640
TARGET_H, TARGET_W = 192, 256
FRAMES = 30


def _write_video(path, height=SOURCE_H, width=SOURCE_W, frames=FRAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate=30:duration={frames / 30}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _decoded_frame_bytes(path) -> int:
    """Total RGB24 bytes of the decoded video -- frames * height * width * 3."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    return len(result.stdout)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "src"
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            CAM: {
                "dtype": "video",
                "shape": [SOURCE_H, SOURCE_W, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": SOURCE_H,
                    "video.width": SOURCE_W,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": 30,
                },
            },
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"not-really-parquet")

    for episode in range(2):
        _write_video(root / "videos" / "chunk-000" / CAM / f"episode_{episode:06d}.mp4")

    return root


def _run(dataset, dest):
    step = build_step({"type": "resize_preserve_aspect_area"})
    materialize(plan_transform(dataset, dest, [step]))
    return dest


def test_output_video_really_has_the_target_resolution(dataset, tmp_path):
    dest = _run(dataset, tmp_path / "out")
    video = dest / "videos" / "chunk-000" / CAM / "episode_000000.mp4"

    assert _decoded_frame_bytes(video) == FRAMES * TARGET_H * TARGET_W * 3


def test_source_video_is_left_untouched(dataset, tmp_path):
    source = dataset / "videos" / "chunk-000" / CAM / "episode_000000.mp4"
    before = source.read_bytes()

    _run(dataset, tmp_path / "out")

    assert source.read_bytes() == before
    assert _decoded_frame_bytes(source) == FRAMES * SOURCE_H * SOURCE_W * 3


def test_no_frames_are_lost(dataset, tmp_path):
    dest = _run(dataset, tmp_path / "out")
    video = dest / "videos" / "chunk-000" / CAM / "episode_000000.mp4"

    assert _decoded_frame_bytes(video) // (TARGET_H * TARGET_W * 3) == FRAMES


def test_metadata_agrees_with_the_encoded_video(dataset, tmp_path):
    dest = _run(dataset, tmp_path / "out")
    info = load_info(dest)

    assert info["features"][CAM]["shape"] == [TARGET_H, TARGET_W, 3]
    assert info["features"][CAM]["info"]["video.height"] == TARGET_H
    assert info["features"][CAM]["info"]["video.width"] == TARGET_W


def test_every_episode_is_converted(dataset, tmp_path):
    dest = _run(dataset, tmp_path / "out")
    videos = sorted((dest / "videos" / "chunk-000" / CAM).glob("*.mp4"))
    assert len(videos) == 2
    for video in videos:
        assert _decoded_frame_bytes(video) == FRAMES * TARGET_H * TARGET_W * 3


def test_untouched_files_are_hardlinked_not_duplicated(dataset, tmp_path):
    dest = _run(dataset, tmp_path / "out")
    source = dataset / "data" / "chunk-000" / "episode_000000.parquet"
    copied = dest / "data" / "chunk-000" / "episode_000000.parquet"

    assert copied.stat().st_ino == source.stat().st_ino


def test_a_dataset_already_at_the_target_size_is_not_re_encoded(dataset, tmp_path):
    """Second pass over an already-processed dataset must be a pure hard-link."""
    first = _run(dataset, tmp_path / "out1")
    second = _run(first, tmp_path / "out2")

    original = first / "videos" / "chunk-000" / CAM / "episode_000000.mp4"
    result = second / "videos" / "chunk-000" / CAM / "episode_000000.mp4"
    assert result.stat().st_ino == original.stat().st_ino


def test_a_failed_run_leaves_no_destination_behind(dataset, tmp_path):
    dest = tmp_path / "out"
    plan = plan_transform(dataset, dest, [build_step({"type": "resize_preserve_aspect_area"})])

    corrupted = plan.transcodes[0].src
    corrupted.write_bytes(b"this is not a video")

    with pytest.raises(Exception):
        materialize(plan)

    assert not dest.exists()
