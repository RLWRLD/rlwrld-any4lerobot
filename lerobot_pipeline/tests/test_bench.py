import pytest

from lerobot_pipeline.bench import BenchSummary, collect_videos, summarize
from lerobot_pipeline.video_ops import Parallelism


# --- discovery ---------------------------------------------------------------


def test_finds_matching_videos_recursively(tmp_path):
    (tmp_path / "ep1" / "top").mkdir(parents=True)
    (tmp_path / "ep2" / "top").mkdir(parents=True)
    wanted = [
        tmp_path / "ep1" / "top" / "rgb.mp4",
        tmp_path / "ep2" / "top" / "rgb.mp4",
    ]
    for path in wanted:
        path.write_bytes(b"x")
    (tmp_path / "ep1" / "top" / "depth.mkv").write_bytes(b"x")

    assert collect_videos(tmp_path, ("rgb.mp4",)) == sorted(wanted)


def test_multiple_patterns_are_unioned_without_duplicates(tmp_path):
    (tmp_path / "a").mkdir()
    rgb = tmp_path / "a" / "rgb.mp4"
    depth = tmp_path / "a" / "depth.mkv"
    rgb.write_bytes(b"x")
    depth.write_bytes(b"x")

    found = collect_videos(tmp_path, ("rgb.mp4", "*.mp4", "depth.mkv"))
    assert found == sorted([rgb, depth])


def test_no_matches_returns_empty(tmp_path):
    assert collect_videos(tmp_path, ("rgb.mp4",)) == []


# --- summary maths -----------------------------------------------------------

PARALLELISM = Parallelism(workers=8, threads=1)


def test_throughput_is_frames_over_wall_clock():
    summary = summarize(
        frames=90_000, in_pixels=90_000 * 480 * 640, files=100,
        wall_clock_s=45.0, parallelism=PARALLELISM, cores=8,
    )
    assert summary.fps == pytest.approx(2000.0)


def test_per_core_throughput_is_what_makes_architectures_comparable():
    summary = summarize(
        frames=90_000, in_pixels=1, files=100,
        wall_clock_s=45.0, parallelism=PARALLELISM, cores=8,
    )
    assert summary.fps_per_core == pytest.approx(250.0)


def test_megapixels_per_second_normalises_for_source_resolution():
    """fps alone is misleading across datasets: decoding 1080p costs more than VGA."""
    summary = summarize(
        frames=1000, in_pixels=1000 * 480 * 640, files=1,
        wall_clock_s=10.0, parallelism=PARALLELISM, cores=8,
    )
    assert summary.megapixels_per_s == pytest.approx(480 * 640 * 100 / 1e6)


def test_zero_wall_clock_does_not_divide_by_zero():
    summary = summarize(
        frames=10, in_pixels=10, files=1,
        wall_clock_s=0.0, parallelism=PARALLELISM, cores=8,
    )
    assert summary.fps == 0.0
    assert summary.fps_per_core == 0.0


def test_summary_records_the_parallelism_actually_used():
    summary = summarize(
        frames=10, in_pixels=10, files=1,
        wall_clock_s=1.0, parallelism=Parallelism(workers=4, threads=2), cores=8,
    )
    assert (summary.workers, summary.threads, summary.cores) == (4, 2, 8)


def test_summary_is_json_serialisable():
    import json

    summary = summarize(
        frames=10, in_pixels=10, files=1,
        wall_clock_s=1.0, parallelism=PARALLELISM, cores=8,
    )
    assert isinstance(summary, BenchSummary)
    assert json.loads(json.dumps(summary.as_dict()))["fps"] == 10.0
