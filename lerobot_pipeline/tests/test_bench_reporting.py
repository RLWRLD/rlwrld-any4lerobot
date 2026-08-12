import pytest

from lerobot_pipeline.bench import (
    format_summary,
    project_seconds,
    sample,
    summarize,
)
from lerobot_pipeline.video_ops import Parallelism

PARALLELISM = Parallelism(workers=8, threads=1)


def _summary(frames=1000, wall_clock_s=1.0):
    return summarize(
        frames=frames,
        in_pixels=frames * 480 * 640,
        files=10,
        wall_clock_s=wall_clock_s,
        parallelism=PARALLELISM,
        cores=8,
    )


# --- projection --------------------------------------------------------------


def test_projects_a_full_run_from_the_sample_throughput():
    assert project_seconds(_summary(frames=1000, wall_clock_s=1.0), 60_000) == pytest.approx(60.0)


def test_projection_of_a_zero_throughput_sample_is_zero_not_infinite():
    assert project_seconds(_summary(wall_clock_s=0.0), 60_000) == 0.0


# --- sampling ----------------------------------------------------------------


def test_sampling_more_than_available_returns_everything(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"x" * (index + 1))
        paths.append(path)

    assert sample(paths, 10) == paths


def test_sampling_spreads_across_file_sizes_rather_than_taking_the_smallest(tmp_path):
    """A sample of only tiny episodes would badly under-estimate the real run."""
    paths = []
    for index in range(10):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"x" * (index + 1) * 100)
        paths.append(path)

    chosen = sample(paths, 3)
    sizes = [path.stat().st_size for path in chosen]
    assert len(chosen) == 3
    assert max(sizes) > min(sizes) * 2


def test_sampling_returns_distinct_files(tmp_path):
    paths = []
    for index in range(10):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"x" * (index + 1))
        paths.append(path)

    chosen = sample(paths, 4)
    assert len(set(chosen)) == 4


# --- reporting ---------------------------------------------------------------


def test_report_states_throughput_and_the_parallelism_used():
    text = format_summary(_summary())
    assert "8 workers" in text
    assert "fps" in text


def test_report_includes_a_projection_when_the_total_is_known():
    text = format_summary(_summary(frames=1000, wall_clock_s=1.0), total_frames=600_000)
    assert "projected" in text
    assert "10.0 min" in text


def test_report_omits_the_projection_when_the_total_is_unknown():
    assert "projected" not in format_summary(_summary())
