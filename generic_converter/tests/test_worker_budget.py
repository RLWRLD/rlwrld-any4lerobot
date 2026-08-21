"""How many workers a machine can hold, and what happens when it holds none.

Both of this converter's hangs looked identical from outside the process: parent
alive, no error, no output, load average 0.00. A forkserver deadlock held a 48-core
node for fifteen hours; OOM-killed workers left three nodes in `do_wait` for over an
hour each. Neither is visible in a return code, because there is no return.
"""

import os

import pytest

from generic_converter import pipeline


class TestWorkerBudget:
    def test_memory_caps_below_the_core_count(self, monkeypatch):
        """The case that was hit: 48 cores, 185 GB, and workers at 4.74 GB each."""
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        assert pipeline.worker_budget(1) == 30      # 185 // 6, not 48

    def test_cores_still_cap_a_machine_with_memory_to_spare(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 512 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        assert pipeline.worker_budget(1) == 8

    def test_cpus_per_task_divides_the_cores_before_memory_is_consulted(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 512 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        assert pipeline.worker_budget(4) == 12

    def test_a_dataset_that_needs_more_can_ask_for_it(self, monkeypatch):
        """toto's episodes are ~301 MB where bc_z's are ~20 MB; one number cannot
        serve both, so the measured default is overridable."""
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.setenv("ANY4LEROBOT_WORKER_MEMORY_GB", "16")
        assert pipeline.worker_budget(1) == 11      # 185 // 16

    def test_never_zero(self, monkeypatch):
        """A machine smaller than one worker still has to try, and a run that asks
        for zero workers is a hang of its own."""
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        assert pipeline.worker_budget(1) == 1

    def test_unreadable_memory_falls_back_to_cores(self, monkeypatch):
        """Rather than guessing low and halving throughput everywhere."""
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        monkeypatch.setattr(pipeline, "available_memory", lambda: None)
        assert pipeline.worker_budget(1) == 16

    def test_an_explicit_worker_count_is_never_second_guessed(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 8 * 1024**3)
        assert pipeline.local_config(10, 40, 1)["workers"] == 40

    def test_minus_one_asks_the_machine(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        assert pipeline.local_config(10, -1, 1)["workers"] == 30


class TestAvailableMemory:
    def test_a_container_limit_wins_over_the_host(self, monkeypatch):
        """A container is what runs, and its limit can be far below the host's."""
        monkeypatch.setattr(pipeline.Path, "read_text",
                            lambda self: "34359738368\n")     # 32 GB
        assert pipeline.available_memory() == 32 * 1024**3

    def test_an_unlimited_cgroup_defers_to_the_host(self, monkeypatch):
        """cgroup v2 writes the word `max`, which is not a number of bytes."""
        monkeypatch.setattr(pipeline.Path, "read_text", lambda self: "max\n")
        got = pipeline.available_memory()
        assert got is None or got > 0

    def test_a_machine_with_neither_reports_nothing(self, monkeypatch):
        """Rather than raising into a converter that only wanted a worker count."""
        def refuse(self):
            raise OSError("no cgroup here")

        monkeypatch.setattr(pipeline.Path, "read_text", refuse)
        monkeypatch.setattr(pipeline.os, "sysconf",
                            lambda name: (_ for _ in ()).throw(ValueError(name)))
        assert pipeline.available_memory() is None


class TestStallWatchdog:
    def test_a_tree_that_grows_is_not_stalled(self, tmp_path):
        """The watchdog must not abort a slow run: one large episode can take
        minutes, and a false abort costs a re-run."""
        import time

        watched = tmp_path / "out"
        watched.mkdir()
        watchdog = pipeline.watch_for_stall([watched], seconds=2)
        try:
            for i in range(6):
                (watched / f"file{i}").write_text("x" * (i + 1))
                time.sleep(0.5)
            assert True     # still here: os._exit was not called
        finally:
            watchdog.stop()

    def test_it_watches_every_path_it_is_given(self, tmp_path):
        """Output and the executor's own logging dir both count as progress -- a run
        that is logging but not yet writing output is working."""
        import time

        out, logs = tmp_path / "out", tmp_path / "logs"
        out.mkdir(); logs.mkdir()
        watchdog = pipeline.watch_for_stall([out, logs], seconds=2)
        try:
            for i in range(6):
                (logs / f"log{i}").write_text("y")
                time.sleep(0.5)
            assert True
        finally:
            watchdog.stop()

    def test_a_stalled_tree_is_aborted(self, tmp_path, monkeypatch):
        """The other half of the contract, and it had no test: a watchdog that never
        fires is the fifteen-hour hang it was written for."""
        import time

        killed = []
        monkeypatch.setattr(pipeline.os, "_exit", killed.append)

        watched = tmp_path / "out"
        watched.mkdir()
        (watched / "one").write_text("x")
        watchdog = pipeline.watch_for_stall([watched], seconds=1)
        try:
            deadline = time.monotonic() + 20
            while not killed and time.monotonic() < deadline:
                time.sleep(0.2)
            assert killed and killed[0] == 75
        finally:
            watchdog.stop()

    def test_a_stopped_watchdog_does_not_fire(self, tmp_path, monkeypatch):
        """A watchdog outlives the code that starts it: a daemon thread looping on
        ``os._exit`` cannot be called off, so the one armed for the executor was still
        armed during the upload afterwards, where writing nothing for twenty minutes
        is normal. It also survived into the rest of this test suite and killed the
        pytest process at exit 75, which is how it was found."""
        import time

        killed = []
        monkeypatch.setattr(pipeline.os, "_exit", killed.append)

        watched = tmp_path / "out"
        watched.mkdir()
        (watched / "one").write_text("x")
        watchdog = pipeline.watch_for_stall([watched], seconds=1)
        watchdog.stop()

        assert not watchdog.thread.is_alive()
        time.sleep(7)           # past the interval it would next have woken on
        assert killed == []


class TestStartMethodReachesTheManager:
    """`start_method` alone reaches the pool and not the manager.

    datatrove's LocalPipelineExecutor.run does:

        mg = multiprocess.Manager()                        # default context: fork
        ctx = multiprocess.get_context(self.start_method)  # honoured here only

    A manager forked from a parent holding TensorFlow inherits locks no thread in it
    owns, then serves the queue the pool workers wait on. py-spy on a 64-core node
    building toto: parent in pool.py:861 next, child in managers.py:176 serve_forever
    via popen_fork, load 0.00, output frozen. No error, no CPU, no end.
    """

    def test_it_sets_the_process_default(self):
        import multiprocess

        from generic_converter.pipeline import apply_start_method

        before = multiprocess.get_start_method(allow_none=True)
        try:
            assert apply_start_method("spawn") == "spawn"
            assert multiprocess.get_start_method() == "spawn"
        finally:
            if before:
                multiprocess.set_start_method(before, force=True)

    def test_it_forces_past_an_already_fixed_default(self):
        """The default being wrong is the bug, so an existing choice is not deferred to."""
        import multiprocess

        from generic_converter.pipeline import apply_start_method

        before = multiprocess.get_start_method(allow_none=True)
        try:
            multiprocess.set_start_method("fork", force=True)
            assert apply_start_method("spawn") == "spawn"
        finally:
            if before:
                multiprocess.set_start_method(before, force=True)

    def test_no_request_leaves_the_default_alone(self):
        """ray and single-worker runs do not go through this and must not be changed."""
        import multiprocess

        from generic_converter.pipeline import apply_start_method

        before = multiprocess.get_start_method(allow_none=True)
        assert apply_start_method(None) is None
        assert multiprocess.get_start_method(allow_none=True) == before


class TestOomFailsRatherThanHangs:
    """An OOM-killed worker hangs the run; it has to fail it instead.

    multiprocess.Pool hands work out through a queue guarded by a lock. A worker the
    kernel kills while it holds that lock never releases it, so every surviving worker
    blocks in Queue.get and the parent blocks in pool.next waiting for results that
    cannot arrive. Measured on toto, 2026-08-20: one kill, 42 workers left waiting,
    128 GB resident, output frozen at 1,472 MB, load average 0.00, nothing in any log.
    """

    def test_the_counter_is_read_from_vmstat(self, tmp_path, monkeypatch):
        from generic_converter import pipeline

        stat = tmp_path / "vmstat"
        stat.write_text("pgfault 12345\noom_kill 7\npgmajfault 8\n")
        monkeypatch.setattr(pipeline, "Path", lambda *a, **k: stat)
        assert pipeline.oom_count() == 7

    def test_a_machine_without_the_counter_is_not_an_error(self, tmp_path, monkeypatch):
        """Older kernels and some containers do not expose it. The watchdog then
        declines to run rather than failing every build."""
        from generic_converter import pipeline

        stat = tmp_path / "vmstat"
        stat.write_text("pgfault 12345\n")
        monkeypatch.setattr(pipeline, "Path", lambda *a, **k: stat)
        assert pipeline.oom_count() is None

    def test_the_watchdog_declines_when_the_counter_is_absent(self, monkeypatch):
        from generic_converter import pipeline

        monkeypatch.setattr(pipeline, "oom_count", lambda: None)
        assert pipeline.watch_for_oom(poll=1) is None

    def test_the_exit_code_is_distinct_from_the_stall_code(self):
        """A scheduler has to tell "the kernel took a worker, retry smaller" from
        "no progress and nobody knows why"."""
        from generic_converter.pipeline import EXIT_WORKER_KILLED

        assert EXIT_WORKER_KILLED == 76


class TestWorkerMemoryFollowsTheEpisode:
    """A constant per worker was the second wrong answer.

    With 6 GB assumed, a 64-core 247 GB node took 41 workers for toto, whose workers
    reach 9.3 GB. Fitted on the two datasets that were actually measured.
    """

    def test_it_covers_both_measurements(self):
        from generic_converter.pipeline import worker_memory

        # jaco_play 9.9 GB / 976 episodes, workers measured at 4.74 GB
        assert worker_memory(10_100_000) > 4.74 * 1024**3
        # toto 137.1 GB / 902 episodes, a worker OOM-killed at 9.28 GB
        assert worker_memory(152_000_000) > 9.28 * 1024**3

    def test_a_bigger_episode_costs_more(self):
        from generic_converter.pipeline import worker_memory

        assert worker_memory(152_000_000) > worker_memory(10_100_000)

    def test_toto_gets_fewer_workers_than_cores(self, monkeypatch):
        """The whole point: 64 cores and 247 GB is 22 workers for toto, not 41."""
        from generic_converter import pipeline

        monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 64)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 247 * 1024**3)
        assert pipeline.worker_budget(1, 152_000_000) < 30
        assert pipeline.worker_budget(1, 10_100_000) > 35

    def test_an_unknown_episode_keeps_the_old_constant(self, monkeypatch):
        """Adapters that cannot say must not be silently given a huge budget."""
        from generic_converter import pipeline

        assert pipeline.worker_memory(None) == pipeline.WORKER_MEMORY_BYTES

    def test_the_override_still_wins(self, monkeypatch):
        """The escape hatch has to beat the model, not be averaged with it."""
        from generic_converter import pipeline

        monkeypatch.setenv("ANY4LEROBOT_WORKER_MEMORY_GB", "3")
        assert pipeline.worker_memory(152_000_000) == 3 * 1024**3


class TestSharingAMachine:
    """The orchestrator runs up to ``max_datasets`` builds at once, each its own
    process, and each one sized itself against the whole machine -- which is the
    overcommit this budget exists to prevent, one level up from where it looks."""

    def test_builds_sharing_a_machine_each_get_a_share_of_its_memory(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        monkeypatch.setenv("ANY4LEROBOT_MEMORY_SHARE", "3")

        assert pipeline.worker_budget(1) == 10      # (185 / 3) // 6, not 30

    def test_a_run_with_the_machine_to_itself_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        monkeypatch.setenv("ANY4LEROBOT_MEMORY_SHARE", "1")

        assert pipeline.worker_budget(1) == 30

    def test_no_share_declared_is_the_whole_machine(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        monkeypatch.delenv("ANY4LEROBOT_MEMORY_SHARE", raising=False)

        assert pipeline.worker_budget(1) == 30

    def test_a_share_too_small_for_one_worker_still_gets_one(self, monkeypatch):
        """Asking for zero workers is a hang of its own, so the floor holds here too."""
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 8 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        monkeypatch.setenv("ANY4LEROBOT_MEMORY_SHARE", "4")

        assert pipeline.worker_budget(1) == 1

    def test_a_nonsense_share_is_ignored_rather_than_crashing_the_build(self, monkeypatch):
        """It arrives from the environment, so it can be anything."""
        monkeypatch.setattr(os, "cpu_count", lambda: 48)
        monkeypatch.setattr(pipeline, "available_memory", lambda: 185 * 1024**3)
        monkeypatch.delenv("ANY4LEROBOT_WORKER_MEMORY_GB", raising=False)
        for value in ("0", "-2", "half", ""):
            monkeypatch.setenv("ANY4LEROBOT_MEMORY_SHARE", value)
            assert pipeline.worker_budget(1) == 30
