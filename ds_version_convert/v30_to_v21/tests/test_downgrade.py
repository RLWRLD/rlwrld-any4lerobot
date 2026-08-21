import os
import threading
from pathlib import Path

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


def test_an_episode_parquet_says_what_datasets_would_have_said(tmp_path):
    """The writer changed; what a v2.1 reader sees must not have.

    ``_split_data_file`` writes each episode with pyarrow now, instead of building a
    ``datasets.Dataset`` for it and writing through that. The two differ in row-group
    layout and file size, which no reader looks at. They must not differ in the
    columns, the types, the values, or the feature description in the schema
    metadata, which is everything a reader does look at.
    """

    import pyarrow.parquet as pq
    from datasets import Dataset

    source = tmp_path / "in"
    records = _data_dataset(source, files=1, episodes_per_file=2)
    dg.convert_data(source, tmp_path / "out", records, workers=1)

    table = pq.read_table(
        source / dg.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    )
    through_datasets = tmp_path / "through_datasets.parquet"
    Dataset(table.slice(0, ROWS_PER_EPISODE)).to_parquet(through_datasets)

    written = pq.read_table(sorted((tmp_path / "out").rglob("*.parquet"))[0])
    expected = pq.read_table(through_datasets)

    assert written.schema.names == expected.schema.names
    assert written.schema.types == expected.schema.types
    assert written.schema.metadata == expected.schema.metadata
    assert written.to_pydict() == expected.to_pydict()


def _wide_data_dataset(root, rows_per_episode, width, episodes):
    """A v3.0 source whose columns are shaped like a real one's.

    ``_data_dataset`` writes one float per row, which is too little for the writer
    settings to show up in: a dictionary of four sequential values costs nothing
    either way. Robot state and action columns are tens of floats wide and repeat
    almost nothing, which is where dictionary encoding turns into dead weight.
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = rows_per_episode * episodes
    rng = np.random.default_rng(0)
    path = root / dg.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "index": pa.array(np.arange(rows, dtype="int64")),
                "action": pa.array(rng.random((rows, 7)).tolist()),
                "observation.state": pa.array(rng.random((rows, width)).tolist()),
            }
        ),
        path,
    )
    return [
        {
            "episode_index": episode,
            "dataset_from_index": episode * rows_per_episode,
            "dataset_to_index": (episode + 1) * rows_per_episode,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for episode in range(episodes)
    ]


def test_an_episode_parquet_is_not_bigger_than_datasets_would_have_written(tmp_path):
    """The writer settings are mirrored to hold the size, so hold them to it.

    pyarrow's own defaults dictionary-encode the float columns, and robot state and
    action data does not repeat enough for that to pay. Dropping the mirrored
    settings from ``_parquet_writing`` would not fail any of the checks above --
    every one of them still passes on a fatter file.
    """

    import pyarrow.parquet as pq
    from datasets import Dataset

    source = tmp_path / "in"
    records = _wide_data_dataset(source, rows_per_episode=60, width=30, episodes=2)
    dg.convert_data(source, tmp_path / "out", records, workers=1)

    table = pq.read_table(
        source / dg.DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    )
    through_datasets = tmp_path / "through_datasets.parquet"
    Dataset(table.slice(0, 60)).to_parquet(through_datasets)

    written = sorted((tmp_path / "out").rglob("*.parquet"))[0]
    ratio = written.stat().st_size / through_datasets.stat().st_size
    assert ratio < 1.05, f"episode parquet is {ratio:.2f}x what datasets would write"


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


class TestWholeDatasetStats:
    """`meta/stats.json`, aggregated from the per-episode entries already in hand.

    v2.1 carries it and the delivered RLDX-1 copies have one; the downgrade wrote
    `episodes_stats.jsonl` and nothing else, so every rebuild came out without it.
    """

    def _episodes(self):
        """Two episodes of one 1-wide feature, with numbers whose aggregate is known."""
        return [
            {"f": {"mean": [1.0], "std": [0.0], "min": [1.0], "max": [1.0],
                   "count": [1], "q50": [1.0]}},
            {"f": {"mean": [3.0], "std": [0.0], "min": [3.0], "max": [3.0],
                   "count": [3], "q50": [3.0]}},
        ]

    def test_counts_and_extremes_are_exact(self):
        out = dg.aggregate_episode_stats(self._episodes())["f"]
        assert out["count"].tolist() == [4]
        assert out["min"].tolist() == [1.0]
        assert out["max"].tolist() == [3.0]

    def test_the_mean_is_weighted_by_the_count(self):
        """One row at 1 and three at 3 average 2.5, not 2."""
        out = dg.aggregate_episode_stats(self._episodes())["f"]
        assert out["mean"].tolist() == [2.5]

    def test_the_std_is_pooled_and_not_averaged(self):
        """Both episodes have std 0 and the pool does not. How far a group spreads
        about its own mean says nothing about how far that mean sits from everyone
        else's -- averaging instead put `index` out by 7,168 on cmu_stretch."""
        out = dg.aggregate_episode_stats(self._episodes())["f"]
        # E[x^2] - mean^2 = (1 + 27)/4 - 6.25 = 0.75
        assert out["std"].tolist() == pytest.approx([0.75**0.5])

    def test_a_quantile_is_a_count_weighted_mean(self):
        """Not lerobot's envelope, which takes the min of the lower quantiles and the
        max of the upper. The delivered copies used the weighted mean; measured
        against cmu_stretch's own stats.json, 4.2e-09 this way against 1.2e+04."""
        out = dg.aggregate_episode_stats(self._episodes())["f"]
        assert out["q50"].tolist() == [2.5]

    def test_a_quantile_only_some_episodes_have_is_left_out(self):
        episodes = self._episodes()
        del episodes[0]["f"]["q50"]
        assert "q50" not in dg.aggregate_episode_stats(episodes)["f"]

    def test_each_feature_is_weighted_by_its_own_count(self):
        """Image statistics come from a hundred sampled frames while the vectors cover
        every row, so one weight for the whole episode is wrong for one of them."""
        episodes = [
            {"vec": {"mean": [0.0], "std": [0.0], "min": [0.0], "max": [0.0], "count": [1]},
             "img": {"mean": [0.0], "std": [0.0], "min": [0.0], "max": [0.0], "count": [100]}},
            {"vec": {"mean": [1.0], "std": [0.0], "min": [1.0], "max": [1.0], "count": [99]},
             "img": {"mean": [1.0], "std": [0.0], "min": [1.0], "max": [1.0], "count": [100]}},
        ]
        out = dg.aggregate_episode_stats(episodes)
        assert out["vec"]["mean"].tolist() == pytest.approx([0.99])
        assert out["img"]["mean"].tolist() == pytest.approx([0.5])

    def test_a_feature_only_one_episode_has_still_aggregates(self):
        episodes = self._episodes()
        episodes[0]["only_here"] = {"mean": [5.0], "std": [0.0], "min": [5.0],
                                    "max": [5.0], "count": [2]}
        out = dg.aggregate_episode_stats(episodes)
        assert out["only_here"]["count"].tolist() == [2]

    def test_no_episodes_is_not_an_error(self):
        assert dg.aggregate_episode_stats([]) == {}


def test_the_quantiles_are_no_longer_dropped():
    """They used to be filtered out to match the v2.1 schema. v2.1 does not require
    them but the delivered copies carry them, and they cannot be recovered later
    without reading every frame again -- droid has 27 million of them."""
    assert set(dg.LEGACY_STATS_KEYS) == {"mean", "std", "min", "max", "count"}
    source = Path(dg.__file__).read_text()
    assert "k in LEGACY_STATS_KEYS" not in source
