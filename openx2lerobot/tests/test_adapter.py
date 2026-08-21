import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import adapter
from adapter import OpenXAdapter, episode_chunks, read_raw_dir, stack_steps, to_numpy
from generic_converter import ConversionTask


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


class TestFileFormat:
    """Which reader a mirror needs.

    tfds writes two formats and offers a different API for each: tfrecord through
    as_dataset, array_record through as_data_source, and asking for the wrong one
    fails with an assertion about download_and_prepare that says nothing about the
    format. So the format is read from the metadata instead of inferred from that
    failure. bc_z's mirror is the collection's only array_record one, 1024 shards of it.
    """

    def _info(self, root: Path, file_format: str | None):
        root.mkdir(parents=True, exist_ok=True)
        payload = {"name": "demo", "version": "1.0.0"}
        if file_format is not None:
            payload["fileFormat"] = file_format
        (root / "dataset_info.json").write_text(json.dumps(payload))
        return root

    def test_a_tfrecord_mirror_says_so(self, tmp_path):
        root = self._info(tmp_path / "demo" / "1.0.0", "tfrecord")
        assert adapter.file_format(root) == "tfrecord"

    def test_an_array_record_mirror_says_so(self, tmp_path):
        root = self._info(tmp_path / "demo" / "1.0.0", "array_record")
        assert adapter.file_format(root) == "array_record"

    def test_an_older_mirror_without_the_field_reads_as_tfrecord(self, tmp_path):
        """The field arrived after the format did; absent means the original one."""
        root = self._info(tmp_path / "demo" / "1.0.0", None)
        assert adapter.file_format(root) == "tfrecord"

    def test_no_metadata_at_all_is_not_guessed_at(self, tmp_path):
        assert adapter.file_format(tmp_path / "nothing") is None

    def test_the_format_does_not_change_the_plan(self, tmp_path):
        """Both readers take the same ``train[a:b]``, so a mirror's format is a
        question for the worker that reads it and not for the task arithmetic."""
        raw = self._info(tmp_path / "bc_z" / "1.0.1", "array_record")
        subject = OpenXAdapter(raw_dir=raw, output_path=tmp_path / "out")
        subject._count_episodes = lambda: 250

        splits = [task.metadata["split"] for task in subject.load_tasks()]
        assert splits == ["train[0:100]", "train[100:200]", "train[200:250]"]


class TestStackSteps:
    """The batch tf.data was doing for free.

    `as_dataset` gives an episode whose steps are a nested tf.data.Dataset, and the
    tfrecord reader turns that into a trajectory with `.batch(cardinality())`.
    `as_data_source` gives a sequence of per-step dicts instead, so the same
    trajectory has to be assembled here -- and it has to come out the same shape,
    because the standardization transform that reads it is shared.
    """

    def test_a_field_becomes_one_array_over_time(self):
        steps = [{"action": np.array([1.0, 2.0])}, {"action": np.array([3.0, 4.0])}]
        assert stack_steps(steps)["action"].tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_nesting_is_kept_rather_than_flattened(self):
        """bc_z's action is a dict of six fields, and its transform slices into them
        by name: trajectory["action"]["future/xyz_residual"][:, :3]."""
        steps = [
            {"action": {"future/xyz_residual": np.zeros(3)}},
            {"action": {"future/xyz_residual": np.ones(3)}},
        ]
        stacked = stack_steps(steps)
        assert stacked["action"]["future/xyz_residual"].shape == (2, 3)

    def test_a_language_instruction_survives_as_something_decodable(self):
        """save_episode reads the task as traj["task"][0].decode(), so whatever the
        steps hold has to still be bytes once stacked."""
        steps = [{"language_instruction": b"pick up the cup"}] * 3
        assert stack_steps(steps)["language_instruction"][0].decode() == "pick up the cup"

    def test_a_scalar_per_step_becomes_a_column(self):
        steps = [{"is_last": np.bool_(False)}, {"is_last": np.bool_(True)}]
        assert stack_steps(steps)["is_last"].tolist() == [False, True]

    def test_an_episode_with_no_steps_is_an_error(self):
        """Rather than an empty trajectory that fails later inside the transform, where
        the shapes no longer say which episode it came from."""
        with pytest.raises(ValueError, match="no steps"):
            stack_steps([])

    def test_it_reads_a_sequence_that_only_supports_length_and_indexing(self):
        """Which is all as_data_source promises: steps decode when indexed."""

        class Lazy:
            def __init__(self, frames):
                self.frames = frames

            def __len__(self):
                return len(self.frames)

            def __getitem__(self, index):
                return self.frames[index]

        lazy = Lazy([{"action": np.zeros(2)}, {"action": np.ones(2)}])
        assert stack_steps(lazy)["action"].shape == (2, 2)


class _Tensor:
    """Something with .numpy(), which is all this walk knows about a tensor."""

    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class TestToNumpy:
    def test_a_tensor_becomes_its_array(self):
        assert to_numpy(_Tensor(np.zeros(3))).shape == (3,)

    def test_it_reaches_into_nested_dicts(self):
        out = to_numpy({"observation": {"image": _Tensor(np.ones(2))}})
        assert out["observation"]["image"].tolist() == [1.0, 1.0]

    def test_an_array_is_left_alone(self):
        """Half of what the transform returns never went through TensorFlow: it comes
        out of the reader as an array and stays one."""
        array = np.arange(3)
        assert to_numpy({"task": array})["task"] is array

    def test_bytes_are_left_alone(self):
        assert to_numpy({"task": b"lift"})["task"] == b"lift"


def _standardize(traj, dataset_name):
    """Stand-in for openx_rlds.standardize_trajectory, which imports TensorFlow.

    Adds what the real one adds, and returns one field as a tensor so that the
    conversion afterwards has something to do.
    """
    traj["proprio"] = _Tensor(np.zeros((traj["action"].shape[0], 8)))
    traj["task"] = traj.pop("language_instruction")
    return traj


class _Builder:
    """as_data_source's half of a tfds builder, and a refusal for the other half."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.asked_for = None

    def as_data_source(self, split=None):
        self.asked_for = split
        return self.episodes

    def as_dataset(self, split=None, read_config=None):
        raise AssertionError("an array_record mirror cannot be read as a stream")


class _Stream:
    """The tf.data chain the stream reader builds, recorded rather than executed.

    Eager where tf.data is lazy, which the assertions here do not depend on. What
    they do depend on is that the chain stays bounded: a buffered element is a whole
    decoded episode, and worker memory is what caps parallelism.
    """

    def __init__(self, episodes, calls):
        self.episodes = episodes
        self.calls = calls

    def filter(self, predicate):
        return _Stream([e for e in self.episodes if predicate(e)], self.calls)

    def map(self, transform):
        return _Stream([transform(e) for e in self.episodes], self.calls)

    def prefetch(self, size):
        self.calls["prefetch"] = size
        return self

    def as_numpy_iterator(self):
        return iter(self.episodes)


class _StreamBuilder:
    """as_dataset's half, for a tfrecord mirror -- which is every source but bc_z."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.calls = {}

    def as_dataset(self, split=None, read_config=None):
        self.calls["split"] = split
        self.calls["read_config"] = read_config
        return _Stream(self.episodes, self.calls)


def _task(split: str) -> ConversionTask:
    return ConversionTask(
        input_path=Path("/raw"),
        output_path=Path("/out"),
        local_repo_id="chunk",
        metadata={"split": split},
    )


def _episode(steps, **fields):
    return {"steps": steps, **fields}


def _steps(count=2, **fields):
    return [
        {
            "action": np.full(3, float(index)),
            "language_instruction": b"lift the lid",
            "observation": {"image": np.zeros((2, 2, 3), np.uint8)},
            **fields,
        }
        for index in range(count)
    ]


@pytest.fixture
def without_tensorflow(monkeypatch):
    """openx_rlds imports TensorFlow at module level, so the reader is exercised
    against a stand-in for it. What is under test is this module's plumbing -- which
    reader is chosen, what shape reaches the transform, what is left afterwards --
    and none of that is TensorFlow's to answer."""
    module = types.ModuleType("openx_rlds")
    module.standardize_trajectory = _standardize
    module.transform_raw_dataset = lambda episode, dataset_name: episode
    monkeypatch.setitem(sys.modules, "openx_rlds", module)

    # The stream reader also asks tfds for a ReadConfig, to carry the read bounds.
    # Same argument as above: which bound to ask for is this module's to decide, and
    # honouring it is tfds's, so the stand-in keeps the request and honours nothing.
    tfds = types.ModuleType("tensorflow_datasets")
    tfds.ReadConfig = lambda **options: options
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", tfds)
    return module


class TestWhichReaderRuns:
    def _mirror(self, tmp_path, name, file_format):
        raw = tmp_path / name / "1.0.0"
        raw.mkdir(parents=True)
        (raw / "dataset_info.json").write_text(json.dumps({"fileFormat": file_format}))
        return raw

    def _subject(self, tmp_path, name, file_format, episodes):
        raw = self._mirror(tmp_path, name, file_format)
        subject = OpenXAdapter(raw_dir=raw, output_path=tmp_path / "out")
        builder = _Builder(episodes)
        subject._builder = lambda: builder
        return subject, builder

    def test_an_array_record_mirror_is_read_by_index(
        self, tmp_path, without_tensorflow
    ):
        """The stream reader on this mirror is the failure this replaces: tfds asserts
        that download_and_prepare has not been run, naming a file rather than a format."""
        subject, _ = self._subject(
            tmp_path, "bc_z", "array_record", [_episode(_steps())]
        )
        assert len(list(subject.load_subset(_task("train[0:1]")))) == 1

    def test_the_slice_reaches_the_data_source_unchanged(
        self, tmp_path, without_tensorflow
    ):
        """A chunk is only a chunk if the reader honours it; otherwise every task
        converts the whole dataset and the concatenation is 1024 copies."""
        subject, builder = self._subject(
            tmp_path, "bc_z", "array_record", [_episode(_steps())]
        )
        list(subject.load_subset(_task("train[300:400]")))
        assert builder.asked_for == "train[300:400]"

    def test_a_tfrecord_mirror_still_reads_as_a_stream(
        self, tmp_path, without_tensorflow
    ):
        subject, _ = self._subject(tmp_path, "cmu_stretch", "tfrecord", [])
        with pytest.raises(AssertionError, match="cannot be read as a stream"):
            list(subject.load_subset(_task("train[0:1]")))

    def test_a_mirror_with_no_metadata_reads_as_a_stream(
        self, tmp_path, without_tensorflow
    ):
        """Which is what every mirror but bc_z is, so an unreadable dataset_info.json
        should not silently switch readers."""
        raw = tmp_path / "cmu_stretch" / "1.0.0"
        raw.mkdir(parents=True)
        subject = OpenXAdapter(raw_dir=raw, output_path=tmp_path / "out")
        builder = _Builder([])
        subject._builder = lambda: builder
        with pytest.raises(AssertionError, match="cannot be read as a stream"):
            list(subject.load_subset(_task("train[0:1]")))


class TestReadingByStream:
    def _subject(self, tmp_path, episodes):
        raw = tmp_path / "cmu_stretch" / "1.0.0"
        raw.mkdir(parents=True)
        (raw / "dataset_info.json").write_text(json.dumps({"fileFormat": "tfrecord"}))
        subject = OpenXAdapter(raw_dir=raw, output_path=tmp_path / "out")
        builder = _StreamBuilder(episodes)
        subject._builder = lambda: builder
        return subject, builder

    def test_the_prefetch_is_bounded(self, tmp_path, without_tensorflow):
        """A buffered element here is not a frame, it is an episode decoded whole --
        300 MB for toto. AUTOTUNE sizes its buffer for throughput and cannot know that
        twenty-two workers are doing the same on one machine, which is how a worker
        came to cost 7.75 GB. Worker memory is the cap on parallelism, so an unbounded
        buffer here costs cores."""
        subject, builder = self._subject(tmp_path, [_episode(_steps())])
        list(subject.load_subset(_task("train[0:1]")))
        assert builder.calls["prefetch"] == 1

    def test_one_shard_is_read_at_a_time(self, tmp_path, without_tensorflow):
        """The same argument one layer down: interleaving four shards buffers four
        shards' worth to hand back the one episode the consumer asked for."""
        subject, builder = self._subject(tmp_path, [_episode(_steps())])
        list(subject.load_subset(_task("train[0:1]")))
        assert builder.calls["read_config"] == {"interleave_cycle_length": 1}

    def test_the_slice_reaches_the_stream_unchanged(self, tmp_path, without_tensorflow):
        """A chunk is only a chunk if the reader honours it -- the same guarantee the
        by-index reader has a test for, and this reader had none."""
        subject, builder = self._subject(tmp_path, [_episode(_steps())])
        list(subject.load_subset(_task("train[300:400]")))
        assert builder.calls["split"] == "train[300:400]"


class TestReadingByIndex:
    def _subject(self, tmp_path, name, episodes):
        raw = tmp_path / name / "1.0.0"
        raw.mkdir(parents=True)
        (raw / "dataset_info.json").write_text(json.dumps({"fileFormat": "array_record"}))
        subject = OpenXAdapter(raw_dir=raw, output_path=tmp_path / "out")
        subject._builder = lambda: _Builder(episodes)
        return subject

    def test_the_transform_is_handed_a_trajectory_not_a_step(
        self, tmp_path, without_tensorflow, monkeypatch
    ):
        """The one thing the two readers could disagree about: `as_dataset` batches the
        steps before the transform sees them, so this reader has to as well."""
        seen = {}

        def spy(traj, dataset_name):
            seen["action"] = traj["action"].shape
            return _standardize(traj, dataset_name)

        without_tensorflow.standardize_trajectory = spy
        subject = self._subject(tmp_path, "bc_z", [_episode(_steps(count=7))])
        list(subject.load_subset(_task("train[0:1]")))
        assert seen["action"] == (7, 3)

    def test_nothing_a_tensor_is_yielded(self, tmp_path, without_tensorflow):
        """The writer takes arrays. The tfrecord reader gets there with
        as_numpy_iterator over the whole stream; this one converts an episode."""
        subject = self._subject(tmp_path, "bc_z", [_episode(_steps())])
        [episode] = list(subject.load_subset(_task("train[0:1]")))
        assert isinstance(episode["steps"]["proprio"], np.ndarray)

    def test_an_episode_survives_the_round_trip_intact(
        self, tmp_path, without_tensorflow
    ):
        subject = self._subject(tmp_path, "bc_z", [_episode(_steps(count=4))])
        [episode] = list(subject.load_subset(_task("train[0:1]")))
        traj = episode["steps"]
        assert traj["action"].shape == (4, 3)
        assert traj["observation"]["image"].shape == (4, 2, 2, 3)
        assert traj["task"][0].decode() == "lift the lid"

    def test_kukas_failed_episodes_are_dropped_here_too(
        self, tmp_path, without_tensorflow
    ):
        """kuka is the one source that ships failures alongside successes. tf.data did
        the filtering as a stream operation, so it had to be rewritten -- and getting it
        wrong converts episodes the delivered copies do not contain."""
        subject = self._subject(
            tmp_path,
            "kuka",
            [
                _episode(_steps(), success=np.bool_(True)),
                _episode(_steps(), success=np.bool_(False)),
                _episode(_steps(), success=np.bool_(True)),
            ],
        )
        assert len(list(subject.load_subset(_task("train[0:3]")))) == 2

    def test_every_other_dataset_keeps_every_episode(
        self, tmp_path, without_tensorflow
    ):
        """No `success` field to read, so the predicate must not look for one."""
        subject = self._subject(tmp_path, "bc_z", [_episode(_steps())] * 3)
        assert len(list(subject.load_subset(_task("train[0:3]")))) == 3


class TestEpisodesPerTask:
    """Task size derived from the dataset rather than written down once.

    `episodes_per_task: 8` was measured on toto -- 902 episodes at 326 frames each,
    113 tasks -- and applied to the collection as a constant. On language_table it
    means 55,278 tasks of 120 frames: each one pays a process start and a TensorFlow
    import to do almost nothing, and a 64-core node sat 92% idle with the parent
    pinned at 100% of one core and no task finished in fourteen minutes.

    What the measurement actually found is a frame count, not an episode count.
    """

    def test_it_reproduces_the_measured_optimum_on_the_machine_it_was_measured_on(self):
        """toto, 22 workers: the value this rule replaces."""
        from adapter import episodes_per_task

        assert episodes_per_task(frames_per_episode=326, episodes=902, workers=22) == 8

    def test_short_episodes_are_grouped_far_more_than_long_ones(self):
        from adapter import episodes_per_task

        language_table = episodes_per_task(frames_per_episode=16, episodes=442_226, workers=64)
        furniture_bench = episodes_per_task(frames_per_episode=774, episodes=5_100, workers=64)
        assert language_table > 100
        assert furniture_bench <= 4
        assert language_table > furniture_bench * 20

    def test_a_task_carries_roughly_the_measured_frame_count(self):
        from adapter import episodes_per_task

        for per_episode, episodes in ((16, 442_226), (12, 209_880), (139, 39_350)):
            chosen = episodes_per_task(per_episode, episodes, workers=64)
            assert 1_500 <= chosen * per_episode <= 4_000, (per_episode, chosen)

    def test_a_small_dataset_is_split_finely_enough_to_fill_the_machine(self):
        """viola is 135 episodes of 510 frames. Grouping by frames alone would make
        27 tasks for 64 workers, so most of the machine would idle."""
        from adapter import episodes_per_task

        assert episodes_per_task(frames_per_episode=510, episodes=135, workers=64) == 1

    def test_never_zero(self):
        from adapter import episodes_per_task

        assert episodes_per_task(frames_per_episode=100_000, episodes=2, workers=64) == 1

    def test_a_pinned_value_is_not_second_guessed(self):
        from adapter import episodes_per_task

        assert episodes_per_task(16, 442_226, workers=64, pinned=8) == 8

    def test_an_unknown_frame_count_falls_back_to_the_declared_default(self):
        """A spec without delivered counts cannot be derived from."""
        from adapter import episodes_per_task

        assert episodes_per_task(None, 442_226, workers=64, default=8) == 8
        assert episodes_per_task(0, 442_226, workers=64, default=8) == 8


class TestTaskSizeReachesLoadTasks:
    def _for(self, tmp_path, episodes, **kwargs):
        adapter = OpenXAdapter(
            raw_dir=Path("/raw/cmu_stretch"), output_path=tmp_path / "out", **kwargs
        )
        adapter._count_episodes = lambda: episodes
        return adapter

    def test_the_size_is_derived_when_nothing_pins_it(self, tmp_path):
        """language_table's shape: 442,226 episodes of 16 frames on 64 workers.
        The constant made 55,278 tasks; the derived size makes a few thousand."""
        tasks = self._for(tmp_path, 442_226, frames_per_episode=16, workers=64).load_tasks()
        assert 2_000 < len(tasks) < 4_000

    def test_a_pinned_size_still_wins(self, tmp_path):
        tasks = self._for(
            tmp_path, 442_226, episodes_per_task=8, frames_per_episode=16, workers=64
        ).load_tasks()
        assert len(tasks) == 55_279

    def test_without_a_frame_count_it_falls_back_rather_than_guessing(self, tmp_path):
        tasks = self._for(tmp_path, 250, workers=64).load_tasks()
        assert len(tasks) == 3          # the standing default of 100 per task
