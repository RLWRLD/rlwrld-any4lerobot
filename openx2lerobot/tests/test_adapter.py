from pathlib import Path

import pytest

from adapter import OpenXAdapter, episode_chunks, read_raw_dir


class TestReadRawDir:
    def test_a_version_directory_is_recognised_as_one(self):
        assert read_raw_dir(Path("/raw/cmu_stretch/0.1.0")) == (
            "cmu_stretch",
            "0.1.0",
            Path("/raw"),
        )

    def test_without_a_version_the_directory_is_the_dataset(self):
        """What the pipeline passes: the mirror stops above the version directory so
        that `aws s3 sync` leaves it in place, and tfds finds the only one there."""
        assert read_raw_dir(Path("/raw/cmu_stretch")) == ("cmu_stretch", "", Path("/raw"))

    def test_a_directory_that_merely_contains_digits_is_not_a_version(self):
        assert read_raw_dir(Path("/raw/fractal20220817_data"))[0] == "fractal20220817_data"


class TestEpisodeChunks:
    def test_it_splits_into_equal_runs(self):
        assert episode_chunks(200, 100) == [(0, 100), (100, 200)]

    def test_the_last_chunk_is_short_rather_than_padded(self):
        assert episode_chunks(250, 100) == [(0, 100), (100, 200), (200, 250)]

    def test_fewer_episodes_than_a_chunk_is_one_chunk(self):
        assert episode_chunks(50, 100) == [(0, 50)]

    def test_no_episodes_is_no_work(self):
        assert episode_chunks(0, 100) == []

    def test_a_chunk_size_below_one_is_refused(self):
        """Zero would divide the dataset into infinitely many empty tasks."""
        with pytest.raises(ValueError):
            episode_chunks(10, 0)


def _adapter(tmp_path, episodes=250, per_task=100, **kwargs):
    adapter = OpenXAdapter(
        raw_dir=Path("/raw/cmu_stretch"),
        output_path=tmp_path / "out",
        episodes_per_task=per_task,
        **kwargs,
    )
    adapter._count_episodes = lambda: episodes
    return adapter


class TestLoadTasks:
    def test_one_task_per_chunk(self, tmp_path):
        assert len(_adapter(tmp_path).load_tasks()) == 3

    def test_tasks_are_in_episode_order(self, tmp_path):
        """aggregate_tasks concatenates in list order, so this order is the order the
        episodes come out in -- and the whole point is to match a single-process run."""
        splits = [task.metadata["split"] for task in _adapter(tmp_path).load_tasks()]

        assert splits == ["train[0:100]", "train[100:200]", "train[200:250]"]

    def test_each_task_writes_to_its_own_directory(self, tmp_path):
        paths = {task.output_path for task in _adapter(tmp_path).load_tasks()}

        assert len(paths) == 3
        assert all(str(tmp_path) in str(path) for path in paths)

    def test_task_names_carry_the_episodes_they_hold(self, tmp_path):
        first = _adapter(tmp_path).load_tasks()[0]

        assert "cmu_stretch" in first.local_repo_id
        assert "0" in first.local_repo_id and "99" in first.local_repo_id

    def test_max_episodes_stops_early(self, tmp_path):
        """For smoke tests: convert the first N rather than the whole mirror."""
        tasks = _adapter(tmp_path, max_episodes=120).load_tasks()

        assert [task.metadata["split"] for task in tasks] == [
            "train[0:100]",
            "train[100:120]",
        ]

    def test_a_source_with_no_episodes_is_an_error_rather_than_an_empty_run(
        self, tmp_path
    ):
        with pytest.raises(ValueError):
            _adapter(tmp_path, episodes=0).load_tasks()


class TestLocalConfig:
    """How the conversion's workers are sized and started."""

    def test_minus_one_workers_fills_the_machine(self):
        from generic_converter.pipeline import local_config

        config = local_config(task_count=10, workers=-1, cpus_per_task=1)

        assert config["tasks"] == 10
        assert config["workers"] >= 1

    def test_a_worker_count_is_taken_as_given(self):
        from generic_converter.pipeline import local_config

        assert local_config(10, 4, 1)["workers"] == 4

    def test_the_start_method_is_left_to_datatrove_unless_asked_for(self):
        from generic_converter.pipeline import local_config

        assert "start_method" not in local_config(10, 4, 1)

    def test_a_named_start_method_reaches_the_executor(self):
        """openx asks for spawn: it has TensorFlow loaded, and workers forked from
        that can inherit a lock held by a thread they do not have."""
        from generic_converter.pipeline import local_config

        assert local_config(10, 4, 1, "spawn")["start_method"] == "spawn"
