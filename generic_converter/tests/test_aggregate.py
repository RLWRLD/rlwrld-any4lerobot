import pandas as pd
import pytest
from lerobot.datasets import aggregate as upstream

from generic_converter.pipeline import BatchedParquetWriter, plan_destinations

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def test_sources_that_fit_share_one_destination():
    assert plan_destinations([10, 10, 10], cap_mb=100) == [0, 0, 0]


def test_a_destination_stops_taking_sources_at_the_cap():
    assert plan_destinations([40, 40, 40, 40], cap_mb=100) == [0, 0, 1, 1]


def test_a_source_larger_than_the_cap_still_gets_a_destination():
    """It cannot be split, so it goes somewhere on its own rather than looping."""
    assert plan_destinations([500, 10], cap_mb=100) == [0, 1]


def test_nothing_to_place_is_not_an_error():
    assert plan_destinations([], cap_mb=100) == []


def _source(tmp_path, index, rows=4):
    """One temp dataset's data file, as the aggregation would find it."""
    frame = pd.DataFrame(
        {
            "episode_index": [index] * rows,
            "index": range(index * rows, index * rows + rows),
            "observation.state": [[float(row)] * 3 for row in range(rows)],
        }
    )
    path = tmp_path / f"src{index:03d}.parquet"
    frame.to_parquet(path)
    return frame, path


def _files(root):
    return sorted(path.relative_to(root) for path in root.rglob("*.parquet"))


def _write_the_upstream_way(sources, root, cap_mb):
    idx = {"chunk": 0, "file": 0}
    placements = []
    for frame, path in sources:
        idx, dst = upstream.append_or_create_parquet_file(
            frame, path, idx, cap_mb, 1000, DATA_PATH,
            aggr_root=root, one_row_group_per_episode=True,
        )
        placements.append(dst)
    return placements


def _write_in_batches(sources, root, cap_mb):
    writer = BatchedParquetWriter()
    idx = {"chunk": 0, "file": 0}
    placements = []
    for frame, path in sources:
        idx, dst = writer.append(
            frame, path, idx, cap_mb, 1000, DATA_PATH,
            aggr_root=root, one_row_group_per_episode=True,
        )
        placements.append(dst)
    writer.flush()
    return placements


@pytest.mark.parametrize("cap_mb", [100, 0.006])
def test_batching_writes_the_same_files_as_appending_one_at_a_time(tmp_path, cap_mb):
    sources = [_source(tmp_path, index) for index in range(8)]
    one_at_a_time, batched = tmp_path / "upstream", tmp_path / "batched"

    upstream_places = _write_the_upstream_way(sources, one_at_a_time, cap_mb)
    batched_places = _write_in_batches(sources, batched, cap_mb)

    assert _files(one_at_a_time) == _files(batched) != []
    assert upstream_places == batched_places
    for relative in _files(one_at_a_time):
        assert (one_at_a_time / relative).read_bytes() == (
            batched / relative
        ).read_bytes()


def test_a_flush_with_nothing_buffered_is_not_an_error():
    BatchedParquetWriter().flush()
