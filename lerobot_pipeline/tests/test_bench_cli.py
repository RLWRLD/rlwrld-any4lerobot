import json
from pathlib import Path

import pytest

from lerobot_pipeline.bench import measure, parse_args

from .conftest import requires_ffmpeg

CAM = "observation.images.egocentric"


def test_config_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_sample_size_has_a_small_default():
    args = parse_args(["--config", "a.yaml"])
    assert 1 <= args.sample <= 50


def test_sample_size_is_overridable():
    assert parse_args(["--config", "a.yaml", "--sample", "8"]).sample == 8


def test_presets_are_parsed_as_a_comma_separated_sweep():
    args = parse_args(["--config", "a.yaml", "--presets", "ultrafast,veryfast,fast"])
    assert args.presets == ["ultrafast", "veryfast", "fast"]


def test_no_preset_sweep_by_default():
    assert parse_args(["--config", "a.yaml"]).presets is None


def test_config_path_is_a_path():
    assert parse_args(["--config", "a.yaml"]).config == Path("a.yaml")


# --- measurement -------------------------------------------------------------


@requires_ffmpeg
def test_measure_reports_real_throughput_for_a_sample(tmp_path):
    import subprocess

    root = tmp_path / "src"
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
                },
            }
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    for episode in range(2):
        path = root / "videos" / "chunk-000" / CAM / f"episode_{episode:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=0.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", str(path),
            ],
            check=True,
            capture_output=True,
        )

    summary = measure(root, sample_size=2, workdir=tmp_path / "bench")

    assert summary.files == 2
    assert summary.frames > 0
    assert summary.fps > 0
    assert summary.wall_clock_s > 0


@requires_ffmpeg
def test_measure_leaves_no_output_behind(tmp_path):
    import subprocess

    root = tmp_path / "src"
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            CAM: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.height": 480, "video.width": 640, "video.codec": "h264"},
            }
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    path = root / "videos" / CAM / "episode_000000.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=0.3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", str(path),
        ],
        check=True,
        capture_output=True,
    )

    workdir = tmp_path / "bench"
    measure(root, sample_size=1, workdir=workdir)
    assert not workdir.exists()


def test_projection_counts_only_files_that_are_actually_re_encoded(tmp_path):
    """Video keys already at the target size are hard-linked, so counting every
    video file would over-estimate the full run."""
    from lerobot_pipeline.bench import count_transcodes
    from lerobot_pipeline.registry import build_step

    root = tmp_path / "src"
    def camera(h, w):
        return {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264"},
        }

    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {CAM: camera(480, 640), "observation.images.wrist": camera(192, 256)},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    for key in (CAM, "observation.images.wrist"):
        for episode in range(3):
            path = root / "videos" / key / f"episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    steps = [build_step({"type": "resize_preserve_aspect_area"})]
    assert count_transcodes(root, steps) == 3
