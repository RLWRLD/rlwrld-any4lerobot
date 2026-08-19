import os
import threading

import pytest

from .helpers import FPS, requires_ffmpeg, write_concatenated_video
from .downgrade import module as dg

CAM = "observation.images.egocentric"
WRIST = "observation.images.wrist"


def _record(episode_index, start, end):
    """One v3.0 episode row, carrying only the columns video planning reads."""
    record = {"episode_index": episode_index}
    for key in (CAM, WRIST):
        record[f"videos/{key}/chunk_index"] = 0
        record[f"videos/{key}/file_index"] = 0
        record[f"videos/{key}/from_timestamp"] = start
        record[f"videos/{key}/to_timestamp"] = end
    return record


def test_minus_one_workers_means_one_per_core():
    assert dg.resolve_workers(-1) == os.cpu_count()


def test_an_explicit_worker_count_is_taken_as_given():
    assert dg.resolve_workers(3) == 3


def test_a_worker_count_below_one_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        dg.resolve_workers(0)


@pytest.mark.parametrize("workers", [1, 4])
def test_a_failing_job_aborts_the_run(workers):
    def run(job):
        if job == 3:
            raise RuntimeError("job 3 failed")

    with pytest.raises(RuntimeError, match="job 3 failed"):
        dg.work_through(list(range(10)), run, workers, "test")


def test_nothing_to_do_is_not_an_error():
    dg.work_through([], lambda job: None, -1, "test")


def test_the_jobs_really_do_run_at_the_same_time():
    """Four jobs that only finish if all four are in flight together.

    Without this the determinism tests above would still pass on an
    implementation that quietly ignored ``workers`` and ran everything serially.
    """
    gate = threading.Barrier(4, timeout=10)

    dg.work_through(list(range(4)), lambda job: gate.wait(), 4, "test")


def test_one_worker_really_does_run_them_one_at_a_time():
    gate = threading.Barrier(4, timeout=1)

    with pytest.raises(threading.BrokenBarrierError):
        dg.work_through(list(range(4)), lambda job: gate.wait(), 1, "test")


def test_the_command_line_takes_a_worker_count(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["prog", "--repo-id", "test/dataset", "--workers", "8"]
    )

    assert dg.parse_args().workers == 8


def test_the_command_line_leaves_the_worker_count_to_the_machine(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--repo-id", "test/dataset"])

    assert dg.parse_args().workers == -1


def _touch_sources(root, video_keys):
    """The concatenated files the segments are cut out of, empty but present."""
    for key in video_keys:
        source = root / "videos" / key / "chunk-000" / "file-000.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"")


def test_every_episode_and_camera_is_its_own_segment(tmp_path):
    records = [_record(0, 0.0, 1.0), _record(1, 1.0, 2.5), _record(2, 2.5, 3.0)]
    _touch_sources(tmp_path / "in", [CAM, WRIST])

    segments = dg.video_segments(
        tmp_path / "in", tmp_path / "out", records, [CAM, WRIST]
    )

    assert len(segments) == 6


def test_a_segment_names_its_episode_file_and_its_window(tmp_path):
    _touch_sources(tmp_path / "in", [CAM])

    segments = dg.video_segments(
        tmp_path / "in", tmp_path / "out", [_record(7, 1.25, 3.5)], [CAM]
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.src == tmp_path / "in" / "videos" / CAM / "chunk-000" / "file-000.mp4"
    assert (
        segment.dst
        == tmp_path / "out" / "videos" / "chunk-000" / CAM / "episode_000007.mp4"
    )
    assert (segment.start, segment.end) == (1.25, 3.5)


def test_segments_run_forwards_through_the_file_they_are_cut_from(tmp_path):
    """Keeps the readers near each other, serially and in a pool alike."""
    records = [_record(0, 4.0, 5.0), _record(1, 0.0, 1.0), _record(2, 2.0, 3.0)]
    _touch_sources(tmp_path / "in", [CAM])

    segments = dg.video_segments(tmp_path / "in", tmp_path / "out", records, [CAM])

    assert [segment.start for segment in segments] == [0.0, 2.0, 4.0]


def test_no_two_segments_write_to_the_same_file(tmp_path):
    """What makes the list safe to work through in parallel."""
    records = [_record(index, index, index + 1) for index in range(5)]
    _touch_sources(tmp_path / "in", [CAM, WRIST])

    segments = dg.video_segments(
        tmp_path / "in", tmp_path / "out", records, [CAM, WRIST]
    )

    assert len({segment.dst for segment in segments}) == len(segments)


def test_a_missing_source_file_is_reported_before_any_cutting_starts(tmp_path):
    _touch_sources(tmp_path / "in", [CAM])

    with pytest.raises(FileNotFoundError, match="wrist"):
        dg.video_segments(
            tmp_path / "in", tmp_path / "out", [_record(0, 0.0, 1.0)], [CAM, WRIST]
        )


EPISODE_SECONDS = 1.0


def _video_dataset(root, episodes, video_keys):
    """A v3.0 source: one concatenated file per camera, plus its episode rows."""
    for key in video_keys:
        write_concatenated_video(
            root / "videos" / key / "chunk-000" / "file-000.mp4",
            seconds=episodes * EPISODE_SECONDS,
        )
    return [
        _record(index, index * EPISODE_SECONDS, (index + 1) * EPISODE_SECONDS)
        for index in range(episodes)
    ]


def _episode_files(root):
    return sorted(path.relative_to(root) for path in root.rglob("*.mp4"))


@requires_ffmpeg
def test_workers_do_not_change_what_the_video_conversion_writes(tmp_path):
    source = tmp_path / "in"
    records = _video_dataset(source, episodes=6, video_keys=[CAM, WRIST])

    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    dg.convert_videos(source, serial, records, [CAM, WRIST], workers=1)
    dg.convert_videos(source, parallel, records, [CAM, WRIST], workers=4)

    assert _episode_files(serial) == _episode_files(parallel)
    assert _episode_files(serial) != []
    for relative in _episode_files(serial):
        assert (serial / relative).read_bytes() == (parallel / relative).read_bytes()


ROWS_PER_EPISODE = 4


def _data_dataset(root, files, episodes_per_file):
    """A v3.0 source split over several consolidated data files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    episode = 0
    first_row = 0
    for file_index in range(files):
        rows = episodes_per_file * ROWS_PER_EPISODE
        path = root / dg.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "index": list(range(first_row, first_row + rows)),
                    "observation.state": [[float(row)] for row in range(rows)],
                }
            ),
            path,
        )
        for _ in range(episodes_per_file):
            records.append(
                {
                    "episode_index": episode,
                    "dataset_from_index": first_row,
                    "dataset_to_index": first_row + ROWS_PER_EPISODE,
                    "data/chunk_index": 0,
                    "data/file_index": file_index,
                }
            )
            episode += 1
            first_row += ROWS_PER_EPISODE
    return records


def _episode_parquets(root):
    return sorted(path.relative_to(root) for path in root.rglob("*.parquet"))


def test_workers_do_not_change_what_the_data_conversion_writes(tmp_path):
    source = tmp_path / "in"
    records = _data_dataset(source, files=3, episodes_per_file=2)

    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    dg.convert_data(source, serial, records, workers=1)
    dg.convert_data(source, parallel, records, workers=4)

    assert len(_episode_parquets(serial)) == 6
    assert _episode_parquets(serial) == _episode_parquets(parallel)
    for relative in _episode_parquets(serial):
        assert (serial / relative).read_bytes() == (parallel / relative).read_bytes()


EPISODES = 6


def _v30_dataset(root):
    """A whole v3.0 dataset: info, tasks, episode rows, data and video."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    video_records = _video_dataset(root, EPISODES, [CAM])
    data_records = _data_dataset(root, files=2, episodes_per_file=EPISODES // 2)
    records = [
        {
            **data,
            **video,
            "tasks": ["do the thing"],
            "length": ROWS_PER_EPISODE,
            "stats/observation.state/mean": [0.5],
            "stats/observation.state/std": [0.1],
            "stats/observation.state/min": [0.0],
            "stats/observation.state/max": [1.0],
            "stats/observation.state/count": [ROWS_PER_EPISODE],
        }
        for data, video in zip(data_records, video_records)
    ]

    episodes = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), episodes)

    tasks = root / "meta" / "tasks.parquet"
    pq.write_table(
        pa.table({"task": ["do the thing"], "task_index": [0]}).drop_columns([]),
        tasks,
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": FPS,
        "total_episodes": EPISODES,
        "total_frames": EPISODES * ROWS_PER_EPISODE,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "data_path": dg.DEFAULT_DATA_PATH,
        "video_path": dg.DEFAULT_VIDEO_PATH,
        "splits": {"train": f"0:{EPISODES}"},
        "features": {
            CAM: {
                "dtype": "video",
                "shape": [64, 64, 3],
                "names": ["height", "width", "channel"],
                "fps": FPS,
                "info": {"video.height": 64, "video.width": 64, "video.fps": FPS},
            },
            "observation.state": {"dtype": "float32", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def _tree(root):
    return sorted(
        (path.relative_to(root), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    )


@requires_ffmpeg
def test_a_whole_downgrade_lands_in_the_same_place_however_many_workers(tmp_path):
    serial = _v30_dataset(tmp_path / "serial")
    parallel = _v30_dataset(tmp_path / "parallel")

    dg.convert_dataset(repo_id="test/dataset", root=serial, workers=1)
    dg.convert_dataset(repo_id="test/dataset", root=parallel, workers=4)

    assert (serial / "meta" / "episodes.jsonl").exists()
    assert len(list((serial / "videos").rglob("*.mp4"))) == EPISODES
    assert _tree(serial) == _tree(parallel)
