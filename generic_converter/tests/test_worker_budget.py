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
        pipeline.watch_for_stall([watched], seconds=2)
        for i in range(6):
            (watched / f"file{i}").write_text("x" * (i + 1))
            time.sleep(0.5)
        assert True     # still here: os._exit was not called

    def test_it_watches_every_path_it_is_given(self, tmp_path):
        """Output and the executor's own logging dir both count as progress -- a run
        that is logging but not yet writing output is working."""
        import time

        out, logs = tmp_path / "out", tmp_path / "logs"
        out.mkdir(); logs.mkdir()
        pipeline.watch_for_stall([out, logs], seconds=2)
        for i in range(6):
            (logs / f"log{i}").write_text("y")
            time.sleep(0.5)
        assert True
