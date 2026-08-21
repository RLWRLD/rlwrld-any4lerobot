import os
import shutil
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import get_random_str, get_timestamp
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

from .adapter import BaseAdapter
from .utils import (
    ConversionTask,
    setup_logger,
    unique_strings,
)


class SaveLeRobotDataset(PipelineStep):
    name = "Save Temp LeRobotDataset"

    def __init__(self, tasks: list[ConversionTask], adapter: BaseAdapter):
        super().__init__()
        self.tasks = tasks
        self.adapter = adapter
        self.type = f"{adapter.dataset_type}2lerobot"

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = setup_logger()
        task = self.tasks[rank]

        if task.output_path.exists():
            shutil.rmtree(task.output_path)

        dataset = self.adapter.create_dataset(task)

        logger.info(
            f"start processing for {task.input_path}, saving to {task.output_path}"
        )
        raw_dataset = self.adapter.load_subset(task)
        saved_episodes = 0
        for episode_index, episode_data in enumerate(raw_dataset):
            with self.track_time("saving episode"):
                saved = self.adapter.save_episode(
                    dataset,
                    episode_data,
                    task,
                )
                status = "skipped" if saved is False else "process done"
                logger.info(
                    f"{status} for {dataset.repo_id}, episode {episode_index}, "
                    f"len {self.adapter.get_episode_length(episode_data)}"
                )
                if saved is not False:
                    saved_episodes += 1
        dataset.finalize()
        if saved_episodes == 0:
            logger.info(
                f"no episodes saved for {dataset.repo_id}; deleting temp output"
            )
            shutil.rmtree(task.output_path, ignore_errors=True)


# What one conversion worker costs, measured rather than assumed. furniture_bench
# workers were OOM-killed at 4.74 GB resident on 2026-08-20; 6 GB is that plus room,
# and ANY4LEROBOT_WORKER_MEMORY_GB overrides it for a dataset that needs more.
# What a worker costs: a fixed base, plus a term in the size of the episode it holds.
# Two measurements, both taken from OOM investigations on 2026-08-20:
#
#   jaco_play    9.9 GB source /   976 episodes =  10.1 MB an episode -> 4.74 GB resident
#   toto       137.1 GB source /   902 episodes = 152.0 MB an episode -> 9.28 GB resident
#
# A constant cannot fit those, and 6 GB -- the constant that used to be here -- sits
# between them, which is the worst place: too many workers for toto, too few for
# jaco_play. Fitting the two gives a 4.4 GB base, which is TensorFlow plus the
# interpreter, and about 32x the episode, which is the decode and the prefetch depth.
#
# Two points is a thin fit, so it is biased high by WORKER_MEMORY_MARGIN. Guessing low
# wedges a node for an hour; guessing high costs throughput on one dataset.
WORKER_MEMORY_BASE = int(4.4 * 1024**3)
WORKER_MEMORY_PER_SOURCE_BYTE = 32
WORKER_MEMORY_MARGIN = 1.25
# What to assume when the caller cannot say how big an episode is.
WORKER_MEMORY_BYTES = 6 * 1024**3

# How long the whole run may make no progress before it is called stalled. Generous:
# a single large episode can take minutes, and a false abort costs a re-run.
STALL_SECONDS = 20 * 60

# Distinct from the stall code, so a scheduler can tell "the kernel took a worker,
# retry smaller" from "this made no progress and nobody knows why".
EXIT_WORKER_KILLED = 76


def available_memory() -> int | None:
    """This machine's usable memory, in bytes, or ``None`` if it cannot be read.

    The cgroup limit comes first because a container is what actually runs, and its
    limit can be far below the host's. ``/sys/fs/cgroup/memory.max`` reads ``max``
    when unlimited, which is what sends this on to the host figure.
    """
    try:
        text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if text.isdigit():
            return int(text)
    except OSError:
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def worker_memory(episode_bytes: int | None = None) -> int:
    """What one worker will need, in bytes.

    ``episode_bytes`` is the source size of one episode. A worker holds one, so it is
    the only part of the cost that varies between datasets, and it varies fifteenfold
    across this collection.
    """
    override = os.environ.get("ANY4LEROBOT_WORKER_MEMORY_GB")
    if override:
        return int(float(override) * 1024**3)
    if not episode_bytes:
        return WORKER_MEMORY_BYTES
    return int((WORKER_MEMORY_BASE + WORKER_MEMORY_PER_SOURCE_BYTE * episode_bytes)
               * WORKER_MEMORY_MARGIN)


def worker_budget(cpus_per_task: int, episode_bytes: int | None = None) -> int:
    """How many workers this machine can actually hold.

    Cores were the wrong question. A worker holds one whole episode, so what limits it
    is memory per *worker*, not memory per core -- and the two only agree when episodes
    are small. On 2026-08-20 a 48-core, 185 GB node ran 48 workers at 4.74 GB each,
    asked for 226 GB, and the kernel killed twelve of them.

    A constant per worker was the second wrong answer. With 6 GB assumed, a 64-core,
    247 GB node took 41 workers for toto, whose workers reach 9.3 GB, and the kernel
    killed one -- while it held the pool's task queue lock, which left the other 42
    blocked in ``Queue.get`` and the parent in ``pool.next`` with the machine idle. So
    the estimate now takes the episode size, and ``oom_since`` below makes the kill
    fail the run rather than hang it, because no estimate is going to be right always.
    """
    cores = max(1, (os.cpu_count() or 1) // cpus_per_task)
    memory = available_memory()
    if memory is None:
        return cores
    return max(1, min(cores, memory // _memory_share() // worker_memory(episode_bytes)))


def _memory_share() -> int:
    """How many builds are sharing this machine, or 1 when nothing says.

    A batch runs its builds at once, each in its own process, so each one asking
    ``available_memory`` gets the whole machine and the budget above is computed that
    many times over. Three datasets on a 185 GB node then plan for 555 GB, which is
    the overcommit this budget exists to prevent, arrived at one level up from where
    it looks. The orchestrator declares the divisor because it is the only thing that
    knows how many builds it started.

    Cores are deliberately not divided the same way. Oversubscribed CPU degrades --
    the measurement that picked the worker count left 34% of it idle anyway -- while
    oversubscribed memory does not degrade, it wedges the node.

    The value crosses a process boundary as text, so anything that is not a positive
    integer is read as "nothing said" rather than failing a build over it.
    """
    try:
        share = int(os.environ.get("ANY4LEROBOT_MEMORY_SHARE", ""))
    except ValueError:
        return 1
    return share if share >= 1 else 1


class Stalled(RuntimeError):
    """Raised when a run stops making progress instead of finishing or failing."""


class WorkerKilled(RuntimeError):
    """Raised when the kernel took a worker, instead of waiting for it to answer."""


def oom_count() -> int | None:
    """How many processes the kernel has OOM-killed since boot, or ``None``.

    ``/proc/vmstat``'s ``oom_kill`` is the whole machine's counter and needs no root,
    which the ``dmesg`` this replaces did. It counts kills anywhere on the host, so a
    reading is evidence and not proof -- but on a node doing one conversion, a kill
    during the run is the conversion's.
    """
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


class Watchdog:
    """A running watchdog and its off switch.

    Both watchdogs here end the process outright, so the window they are armed for
    has to be exactly the window in which not ending it is worse: the executor and
    the aggregation, where no progress means a hang. Afterwards the run has already
    succeeded and the work left -- an upload, a delete -- can legitimately write
    nothing for a long time, so a watchdog still armed there would kill a finished
    run. Without an off switch there is no way to say that: a daemon thread looping
    on ``os._exit`` cannot be called off by the code that started it.
    """

    def __init__(self, thread: threading.Thread, stop: threading.Event):
        self.thread = thread
        self._stop = stop

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        self.thread.join(timeout=timeout)


def stop_watchdogs(watchdogs: Sequence["Watchdog | None"]) -> None:
    """Disarm every watchdog that started. ``None`` is a watchdog that declined to."""
    for watchdog in watchdogs:
        if watchdog is not None:
            watchdog.stop()


def watch_for_oom(poll: int = 10):
    """Abort the process as soon as the kernel kills anything.

    An OOM-killed worker does not fail the run, it hangs it, and the mechanism is
    worth stating because it is not obvious: ``multiprocess.Pool`` hands work out
    through a queue guarded by a lock, and a worker killed while holding that lock
    never releases it. Every surviving worker then blocks in ``Queue.get``, the parent
    blocks in ``pool.next`` waiting for results that cannot come, and the machine goes
    to load 0.00 with no error anywhere. Measured on toto: one kill, 42 workers left
    waiting, 128 GB resident, output frozen at 1,472 MB.

    ``watch_for_stall`` catches this too, but only after twenty minutes of nothing.
    The kill is knowable in ten seconds, and the difference is a node-hour each time.
    """
    baseline = oom_count()
    if baseline is None:
        return None

    stop = threading.Event()

    def watch():
        while not stop.wait(poll):
            now = oom_count()
            if now is None or now <= baseline:
                continue
            setup_logger().error(
                f"the kernel OOM-killed {now - baseline} process(es) during this run. "
                "A worker killed while holding the pool's task-queue lock leaves every "
                "other worker blocked in Queue.get and the parent in pool.next, which "
                "is a hang and not a failure -- so this fails it. Re-run with fewer "
                "workers: lower `workers` in the environment, or set "
                "ANY4LEROBOT_WORKER_MEMORY_GB above the peak a worker reached."
            )
            os._exit(EXIT_WORKER_KILLED)

    thread = threading.Thread(target=watch, name="oom-watchdog", daemon=True)
    thread.start()
    return Watchdog(thread, stop)


def watch_for_stall(watched: Sequence[Path], seconds: int = STALL_SECONDS):
    """Abort the process if nothing under ``watched`` changes for ``seconds``.

    This exists because the two hangs this converter has produced both looked like
    success from the outside: the parent alive, no error, no output, load average
    0.00. A forkserver deadlock held a 48-core node for fifteen hours; an OOM-killed
    worker left three nodes in ``do_wait`` for over an hour each. Neither is
    detectable from a return code, because there is no return.

    Progress is measured on the output tree rather than on the executor's own
    bookkeeping, so it does not depend on which executor ran or on datatrove's
    internals. A run that is working writes files; one that is stuck does not.

    ``os._exit`` after printing, deliberately. The parent is blocked in ``waitpid``
    and cannot be interrupted by an exception raised on another thread, so a clean
    unwind is not available -- and a stalled converter that stays alive is the whole
    problem being fixed.
    """

    def footprint() -> tuple[int, int]:
        files = size = 0
        for root in watched:
            for path in root.rglob("*"):
                try:
                    if path.is_file():
                        files += 1
                        size += path.stat().st_size
                except OSError:
                    continue
        return files, size

    stop = threading.Event()

    def watch() -> None:
        last, since = footprint(), time.monotonic()
        while not stop.wait(min(60, max(5, seconds // 20))):
            now = footprint()
            if now != last:
                last, since = now, time.monotonic()
                continue
            if time.monotonic() - since < seconds:
                continue
            print(
                f"\nconverter stalled: nothing written under "
                f"{', '.join(str(p) for p in watched)} for {seconds // 60} minutes.\n"
                "A worker is most likely gone and the parent is waiting for it. The "
                "usual cause is memory: check `dmesg -T | grep -i \"out of memory\"`, "
                "and if a worker was killed, lower the worker count or raise "
                "ANY4LEROBOT_WORKER_MEMORY_GB.",
                file=sys.stderr, flush=True,
            )
            os._exit(75)   # EX_TEMPFAIL: the work is fine, the machine was not

    thread = threading.Thread(target=watch, name="stall-watchdog", daemon=True)
    thread.start()
    return Watchdog(thread, stop)


def local_config(
    task_count: int,
    workers: int,
    cpus_per_task: int,
    start_method: str | None = None,
    episode_bytes: int | None = None,
) -> dict:
    """How a local run is sized, and how its workers are started.

    ``workers: -1`` means "as many as this machine can hold", which is a memory
    question and not a core count -- see ``worker_budget``.

    ``start_method`` is worth naming rather than leaving to the default when the
    parent process has already loaded a library that keeps threads. datatrove
    forkservers by default, which inherits whatever state the parent had when the
    server was started -- and a converter that has imported TensorFlow can hand its
    workers a lock held by a thread that does not exist in them. That deadlock does
    not fail: it waits. One was found holding a 48-core instance for fifteen hours
    with fifteen seconds of CPU used and no task started.

    Overcommitting memory produces a second hang of the same shape, which is why the
    budget above exists and why ``run_converter`` watches for a stall: the kernel
    kills a worker, the parent waits in ``do_wait`` for a child that is gone, and
    load average sits at 0.00 with nothing in any log to say why. Three nodes were
    found like that, between 66 and 107 minutes each.
    """
    resolved = (worker_budget(cpus_per_task, episode_bytes) if workers == -1
                else workers)
    config = {"tasks": task_count, "workers": resolved}
    if start_method:
        config["start_method"] = start_method
    return config


def apply_start_method(start_method: str | None) -> str | None:
    """Make ``start_method`` the process default, not just the worker pool's.

    ``LocalPipelineExecutor(start_method=...)`` reaches exactly one of the two things
    datatrove starts. Reading its ``run``::

        mg = multiprocess.Manager()                        # default context: fork
        ctx = multiprocess.get_context(self.start_method)  # honoured here only
        with ctx.Pool(self.workers) as pool:

    So the pool respects the request and the manager does not, and the manager is
    forked from a parent that has TensorFlow loaded -- which is the case the argument
    was added for. A fork copies one thread and every lock, so any mutex another
    thread held at that moment is locked forever in the child. The manager then serves
    the queue and the lock the pool workers need, so they block, so ``imap_unordered``
    never yields, so the parent waits in ``pool.next()``.

    That is a deadlock with no error and no CPU. Found by ``py-spy`` on a 64-core node
    building toto: parent in ``pool.py:861 next``, child in
    ``managers.py:176 serve_forever`` reached through ``popen_fork``, load average
    0.00, 118 GB resident, output frozen at 1,430 MB.

    Setting the default covers both. Returns what was set, so a caller can log it.
    """
    if not start_method:
        return None

    import multiprocess

    if multiprocess.get_start_method(allow_none=True) != start_method:
        # force: datatrove or an import may have fixed it already, and the default
        # being wrong is exactly the bug.
        multiprocess.set_start_method(start_method, force=True)
    return multiprocess.get_start_method()


def run_converter(
    adapter: BaseAdapter,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    resume_dir: str | None = None,
    debug: bool = False,
    local_repo_id: str | None = None,
    hub_repo_id: str | None = None,
    push_to_hub: bool = False,
    cleanup_temp: bool = True,
    extra_tags: Sequence[str] | None = None,
    start_method: str | None = None,
) -> Path:
    tasks = adapter.load_tasks()
    output_path = adapter.output_path

    if not tasks:
        raise ValueError(
            "No conversion tasks found. Provide a non-empty tasks file or matching source files."
        )
    if cpus_per_task < 1:
        raise ValueError("--cpus-per-task must be >= 1")

    output_path.mkdir(parents=True, exist_ok=True)

    if debug:
        executor = "local"
        workers = 1
        tasks = tasks[:2]
        push_to_hub = False

    match executor:
        case "local":
            from datatrove.executor import LocalPipelineExecutor

            executor_cls, executor_config = (
                LocalPipelineExecutor,
                local_config(len(tasks), workers, cpus_per_task, start_method,
                             adapter.episode_bytes()),
            )
        case "ray":
            import ray
            from datatrove.executor import RayPipelineExecutor
            from ray.runtime_env import RuntimeEnv

            runtime_env = RuntimeEnv(env_vars=_build_ray_env_vars())
            ray.init(runtime_env=runtime_env)
            executor_cls, executor_config = (
                RayPipelineExecutor,
                {
                    "tasks": len(tasks),
                    "workers": workers,
                    "cpus_per_task": cpus_per_task,
                    "tasks_per_job": tasks_per_job,
                },
            )
        case _:
            raise ValueError(f"Executor {executor} not supported")

    if resume_dir:
        logging_dir = str(resume_dir)
    else:
        logging_dir = str(Path.cwd() / "logs" / f"{get_timestamp()}_{get_random_str()}")

    if executor == "local":
        applied = apply_start_method(start_method)
        if applied:
            setup_logger().info(
                f"multiprocess default start method: {applied} "
                "(the manager datatrove forks is not covered by start_method alone)"
            )

    # Armed for the executor and the aggregation only. Past here the run has
    # succeeded, and the upload below can go twenty minutes without writing a local
    # byte -- which a stall watchdog cannot tell from a hang.
    watchdogs = (watch_for_oom(), watch_for_stall([output_path, Path(logging_dir)]))
    try:
        executor_cls(
            pipeline=[SaveLeRobotDataset(tasks, adapter)],
            **executor_config,
            logging_dir=logging_dir,
        ).run()
        aggregate_tasks(
            tasks,
            output_path,
            aggr_repo_id=local_repo_id,
        )
    finally:
        stop_watchdogs(watchdogs)

    if cleanup_temp:
        logger = setup_logger()
        logger.info("Delete temp data_dir")
        shutil.rmtree(adapter.temp_output_path, ignore_errors=True)

    if push_to_hub:
        if hub_repo_id is None:
            raise ValueError("--repo-id is required when --push-to-hub is set")

        tags = unique_strings(
            [
                "LeRobot",
                adapter.dataset_type,
                adapter.robot_type,
                *adapter.tags,
                *(extra_tags or []),
            ]
        )
        LeRobotDataset(
            repo_id=hub_repo_id,
            root=output_path,
        ).push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=False,
        )

    return output_path


def _build_ray_env_vars() -> dict[str, str]:
    env_vars = {
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
        "SVT_LOG": "1",
    }
    pythonpath = _build_ray_pythonpath()
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath
    return env_vars


def _build_ray_pythonpath() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    paths: list[str] = []

    def add_path(path_value: str | Path):
        path = Path(path_value).expanduser()
        try:
            path = path.resolve()
        except OSError:
            return
        if not path.exists():
            return
        path_str = str(path)
        if path_str not in paths:
            paths.append(path_str)

    add_path(repo_root)
    add_path(Path.cwd())
    for path in sys.path:
        if path:
            add_path(path)
    for path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if path:
            add_path(path)

    return os.pathsep.join(paths)


def aggregate_tasks(
    tasks: list[ConversionTask],
    output_dir: Path,
    aggr_repo_id: str | None = None,
    batched: bool = True,
):
    logger = setup_logger()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    roots = [task.output_path for task in tasks if task.output_path.exists()]
    if not roots:
        raise ValueError("No temporary datasets were produced; nothing to aggregate.")

    resolved_aggr_repo_id = aggr_repo_id or output_dir.name

    logger.info(
        f"aggregate {len(roots)} temporary datasets into {output_dir} as {resolved_aggr_repo_id}"
    )
    _aggregate_datasets_with_normalized_arrays(
        batched=batched,
        repo_ids=[None] * len(roots),
        roots=roots,
        aggr_repo_id=resolved_aggr_repo_id,
        aggr_root=output_dir,
    )
    logger.info(f"aggregation complete: {output_dir}")


def plan_destinations(sizes: Sequence[float], cap_mb: float) -> list[int]:
    """Which destination file each source belongs in, decided before any is written.

    The rule is the one the upstream fold applies -- a destination takes its
    first source whatever its size, and stops once another one would put it over
    ``cap_mb`` -- but measured against the sum of the source sizes rather than
    the size of the destination as built so far. That number does not exist
    until the file has been written, and writing it once is the whole point.

    Planning on the sum is safe because a concatenation is never bigger than its
    parts: measured on taco_play it comes to 0.9991 of the sum for video and
    0.9916 for parquet, both sharing one container's worth of overhead instead
    of one each. So a file planned to sit under the cap stays under it.
    """

    destinations: list[int] = []
    current = 0
    filled = 0.0
    for size in sizes:
        if destinations and filled + size >= cap_mb:
            current += 1
            filled = 0.0
        destinations.append(current)
        filled += size
    return destinations


class _OpenParquet(NamedTuple):
    """A destination parquet being filled, and how to write it when it is full."""

    path: Path
    frames: list
    filled_mb: float
    contains_images: bool
    hf_features: object
    one_row_group_per_episode: bool


class BatchedParquetWriter:
    """Collects what each destination parquet will hold and writes it once.

    Stands in for ``append_or_create_parquet_file``, which is not the append its
    name suggests: parquet keeps its index in a footer covering the whole file,
    so "appending" means reading the destination back, concatenating in memory
    and writing all of it out again. Doing that once per source turns one 22 MB
    file into 1.4 GB of writing (measured on taco_play: 130 sources, one
    destination).

    Nothing here decides anything differently -- the same sources land in the
    same destinations in the same order. Only the writing is deferred, until the
    destination is complete. So the files come out byte for byte identical, and
    the memory held is one destination's worth, which the size cap bounds.

    Both callers of the upstream function -- the frame data and the episode
    metadata -- keep their own destination sequence, so buffers are kept per
    path template.
    """

    def __init__(self):
        # path template -> the destination currently being filled
        self._open: dict[str, _OpenParquet] = {}

    def append(
        self,
        df,
        src_path: Path,
        idx: dict,
        max_mb: float,
        chunk_size: int,
        default_path: str,
        contains_images: bool = False,
        aggr_root: Path | None = None,
        hf_features=None,
        concatenate: bool = True,
        one_row_group_per_episode: bool = False,
    ) -> tuple[dict, tuple[int, int]]:
        from lerobot.datasets import aggregate as aggregate_module

        if aggr_root is None:
            raise ValueError("aggr_root must be provided.")

        buffered = self._open.get(default_path)
        src_mb = aggregate_module.get_parquet_file_size_in_mb(src_path)

        if buffered is not None and (
            not concatenate or buffered.filled_mb + src_mb >= max_mb
        ):
            self._write(default_path)
            idx["chunk"], idx["file"] = aggregate_module.update_chunk_file_indices(
                idx["chunk"], idx["file"], chunk_size
            )
            buffered = None

        dst_chunk, dst_file = idx["chunk"], idx["file"]
        if buffered is None:
            self._open[default_path] = _OpenParquet(
                path=aggr_root
                / default_path.format(chunk_index=dst_chunk, file_index=dst_file),
                frames=[df],
                filled_mb=src_mb,
                contains_images=contains_images,
                hf_features=hf_features,
                one_row_group_per_episode=one_row_group_per_episode,
            )
        else:
            buffered.frames.append(df)
            self._open[default_path] = buffered._replace(
                filled_mb=buffered.filled_mb + src_mb
            )

        return idx, (dst_chunk, dst_file)

    def _write(self, default_path: str) -> None:
        import pandas as pd
        from lerobot.datasets import aggregate as aggregate_module

        pending = self._open.pop(default_path)
        combined = (
            pending.frames[0]
            if len(pending.frames) == 1
            else pd.concat(pending.frames, ignore_index=True)
        )

        pending.path.parent.mkdir(parents=True, exist_ok=True)
        if pending.contains_images:
            aggregate_module.to_parquet_with_hf_images(
                combined, pending.path, features=pending.hf_features
            )
        elif pending.one_row_group_per_episode:
            aggregate_module.to_parquet_one_row_group_per_episode(combined, pending.path)
        else:
            combined.to_parquet(pending.path)

    def flush(self) -> None:
        """Write whatever destinations are still open. Safe to call twice."""
        for default_path in list(self._open):
            self._write(default_path)


class PlannedVideoFiles:
    """Works out which sources make up each destination video, then builds them.

    ``concatenate_video_files`` is not an append either. mp4 keeps its index --
    the moov atom, holding the offset and timestamp of every frame -- for the
    whole file, and this repo's writer puts it at the front, so growing a file
    means writing a new one and moving the old one aside. Calling it once per
    source therefore rewrites everything accumulated so far, every time:
    measured on real taco_play video, appending 24 sources of 4.2 MB each wrote
    1,267 MB to produce a 101 MB file, and the per-append cost climbed from
    0.04s to 0.30s while the amount of new video stayed the same.

    ffmpeg's concat demuxer takes a list. Given one, it streams every source
    through once and writes moov at the end -- 130 sources, 548 MB, in 1.89s
    against the 108.8s the same work takes an append at a time. So the planning
    pass records the list, and ``build`` hands each destination over whole.

    Nothing about the layout changes: the same sources land in the same
    destination in the same order, at the same timestamps.
    """

    def __init__(self):
        # destination path -> the sources that make it up, in order
        self._sources: dict[Path, list[Path]] = {}

    def plan(
        self,
        src_meta,
        dst_meta,
        videos_idx,
        video_files_size_in_mb,
        chunk_size,
        concatenate_videos=True,
    ):
        for video_idx in videos_idx.values():
            video_idx["episode_duration"] = 0
            video_idx["src_to_offset"] = {}
            video_idx["src_to_dst"] = {}
            video_idx.setdefault("dst_file_durations", {})
            video_idx.setdefault("filled_mb", {})

        for key in videos_idx:
            self._plan_key(
                key,
                src_meta,
                dst_meta,
                videos_idx[key],
                video_files_size_in_mb,
                chunk_size,
                concatenate_videos,
            )
        return videos_idx

    def _plan_key(
        self,
        key,
        src_meta,
        dst_meta,
        video_idx,
        video_files_size_in_mb,
        chunk_size,
        concatenate_videos,
    ):
        from lerobot.datasets import aggregate as aggregate_module

        pairs = sorted(
            {
                (chunk, file)
                for chunk, file in zip(
                    src_meta.episodes[f"videos/{key}/chunk_index"],
                    src_meta.episodes[f"videos/{key}/file_index"],
                    strict=False,
                )
            }
        )

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        dst_file_durations = video_idx["dst_file_durations"]
        filled_mb = video_idx["filled_mb"]

        for src_chunk_idx, src_file_idx in pairs:
            src_path = src_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
                video_key=key, chunk_index=src_chunk_idx, file_index=src_file_idx
            )
            src_duration = aggregate_module.get_video_duration_in_s(src_path)
            src_size = aggregate_module.get_file_size_in_mb(src_path)

            dst_key = (chunk_idx, file_idx)
            started = dst_key in filled_mb
            if started and (
                not concatenate_videos
                or filled_mb[dst_key] + src_size >= video_files_size_in_mb
            ):
                chunk_idx, file_idx = aggregate_module.update_chunk_file_indices(
                    chunk_idx, file_idx, chunk_size
                )
                dst_key = (chunk_idx, file_idx)
                started = False

            dst_path = dst_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
                video_key=key, chunk_index=chunk_idx, file_index=file_idx
            )
            offset = dst_file_durations.get(dst_key, 0) if started else 0

            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = offset
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            self._sources.setdefault(dst_path, []).append(src_path)

            filled_mb[dst_key] = filled_mb.get(dst_key, 0) + src_size
            dst_file_durations[dst_key] = offset + src_duration
            video_idx["episode_duration"] += src_duration

        video_idx["chunk"] = chunk_idx
        video_idx["file"] = file_idx
        return video_idx

    def build(self, workers: int = -1) -> None:
        """Build every destination, each from its whole source list, at once.

        ``-1`` is one per core here, not ``worker_budget``'s "as many as memory
        holds": these are threads waiting on an ffmpeg concat, not processes holding
        a decoded episode. The sentinel is shared with the converter's worker count
        and the unit is not.
        """
        from lerobot.datasets.aggregate import concatenate_video_files

        if not self._sources:
            return

        destinations = list(self._sources.items())
        resolved = os.cpu_count() or 1 if workers == -1 else max(1, workers)

        def build_one(item):
            dst_path, sources = item
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if len(sources) == 1:
                shutil.copy(str(sources[0]), str(dst_path))
                return
            concatenate_video_files(sources, dst_path, compatibility_check=True)

        with ThreadPoolExecutor(max_workers=min(resolved, len(destinations))) as pool:
            for _ in pool.map(build_one, destinations):
                pass
        self._sources.clear()


def _aggregate_datasets_with_normalized_arrays(batched: bool = True, **kwargs) -> None:
    """Aggregate, with this repo's corrections to how upstream writes files.

    ``batched=False`` leaves the file writing as upstream does it, one source at
    a time. Nothing in production wants that -- it is how the tests get a
    reference to compare the batched output against.
    """

    from lerobot.datasets import aggregate as aggregate_module

    original_aggregate_videos = aggregate_module.aggregate_videos
    original_read_parquet = aggregate_module.pd.read_parquet
    original_writer = aggregate_module.to_parquet_one_row_group_per_episode
    original_update_meta_data = aggregate_module.update_meta_data
    original_append = aggregate_module.append_or_create_parquet_file
    original_finalize = aggregate_module.finalize_aggregation

    def read_normalized_arrays(*args, **kwargs):
        return _normalize_array_values(original_read_parquet(*args, **kwargs))

    def write_normalized_arrays(df, path):
        return original_writer(_normalize_array_values(df), path)

    parquet_writer = BatchedParquetWriter()
    videos = PlannedVideoFiles()

    def finalize_once_everything_is_written(*args, **kwargs):
        """The source loop only planned; the files get built here, at the end."""
        parquet_writer.flush()
        videos.build()
        return original_finalize(*args, **kwargs)

    aggregate_module.pd.read_parquet = read_normalized_arrays
    aggregate_module.to_parquet_one_row_group_per_episode = write_normalized_arrays
    aggregate_module.update_meta_data = _update_meta_data_without_fragmenting
    if batched:
        aggregate_module.aggregate_videos = videos.plan
        aggregate_module.append_or_create_parquet_file = parquet_writer.append
        aggregate_module.finalize_aggregation = finalize_once_everything_is_written
    else:
        aggregate_module.aggregate_videos = _aggregate_videos_by_key_parallel
    try:
        aggregate_datasets(**kwargs)
    finally:
        aggregate_module.aggregate_videos = original_aggregate_videos
        aggregate_module.pd.read_parquet = original_read_parquet
        aggregate_module.to_parquet_one_row_group_per_episode = original_writer
        aggregate_module.update_meta_data = original_update_meta_data
        aggregate_module.append_or_create_parquet_file = original_append
        aggregate_module.finalize_aggregation = original_finalize


def _aggregate_videos_by_key_parallel(
    src_meta,
    dst_meta,
    videos_idx,
    video_files_size_in_mb,
    chunk_size,
    concatenate_videos=True,
):
    from concurrent.futures import ThreadPoolExecutor

    for video_idx in videos_idx.values():
        video_idx["episode_duration"] = 0
        video_idx["src_to_offset"] = {}
        video_idx["src_to_dst"] = {}
        if "dst_file_durations" not in video_idx:
            video_idx["dst_file_durations"] = {}

    def aggregate_key(key):
        return (
            key,
            _aggregate_video_key(
                key,
                src_meta,
                dst_meta,
                videos_idx[key],
                video_files_size_in_mb,
                chunk_size,
                concatenate_videos,
            ),
        )

    keys = list(videos_idx)
    if not keys:
        return videos_idx

    max_workers = min(len(keys), os.cpu_count() or len(keys))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, video_idx in executor.map(aggregate_key, keys):
            videos_idx[key] = video_idx

    return videos_idx


def _aggregate_video_key(
    key,
    src_meta,
    dst_meta,
    video_idx,
    video_files_size_in_mb,
    chunk_size,
    concatenate_videos,
):
    from lerobot.datasets import aggregate as aggregate_module

    unique_chunk_file_pairs = {
        (chunk, file)
        for chunk, file in zip(
            src_meta.episodes[f"videos/{key}/chunk_index"],
            src_meta.episodes[f"videos/{key}/file_index"],
            strict=False,
        )
    }
    unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

    chunk_idx = video_idx["chunk"]
    file_idx = video_idx["file"]
    dst_file_durations = video_idx["dst_file_durations"]

    for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
        src_path = src_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
            video_key=key,
            chunk_index=src_chunk_idx,
            file_index=src_file_idx,
        )
        dst_path = dst_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
            video_key=key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

        src_duration = aggregate_module.get_video_duration_in_s(src_path)
        dst_key = (chunk_idx, file_idx)

        if not dst_path.exists():
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_path), str(dst_path))
            dst_file_durations[dst_key] = src_duration
            video_idx["episode_duration"] += src_duration
            continue

        src_size = aggregate_module.get_file_size_in_mb(src_path)
        dst_size = aggregate_module.get_file_size_in_mb(dst_path)

        if not concatenate_videos or dst_size + src_size >= video_files_size_in_mb:
            chunk_idx, file_idx = aggregate_module.update_chunk_file_indices(
                chunk_idx, file_idx, chunk_size
            )
            dst_key = (chunk_idx, file_idx)
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            dst_path = dst_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_path), str(dst_path))
            dst_file_durations[dst_key] = src_duration
        else:
            current_dst_duration = dst_file_durations.get(dst_key, 0)
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = (
                current_dst_duration
            )
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            aggregate_module.concatenate_video_files(
                [dst_path, src_path],
                dst_path,
                compatibility_check=True,
            )
            dst_file_durations[dst_key] = current_dst_duration + src_duration

        video_idx["episode_duration"] += src_duration

    video_idx["chunk"] = chunk_idx
    video_idx["file"] = file_idx

    return video_idx


def _update_meta_data_without_fragmenting(df, dst_meta, meta_idx, data_idx, videos_idx):
    import pandas as pd

    df["meta/episodes/chunk_index"] = (
        df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    )
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]

    data_src_to_dst = data_idx.get("src_to_dst", {})
    if data_src_to_dst:
        orig_data_chunk = df["data/chunk_index"].copy()
        orig_data_file = df["data/file_index"].copy()
        mapping_index = pd.MultiIndex.from_tuples(
            list(data_src_to_dst.keys()),
            names=["chunk_index", "file_index"],
        )
        mapping_df = pd.DataFrame(
            list(data_src_to_dst.values()),
            index=mapping_index,
            columns=["dst_chunk", "dst_file"],
        )
        row_index = pd.MultiIndex.from_arrays(
            [orig_data_chunk, orig_data_file],
            names=["chunk_index", "file_index"],
        )
        reindexed = mapping_df.reindex(row_index)
        reindexed[["dst_chunk", "dst_file"]] = reindexed[
            ["dst_chunk", "dst_file"]
        ].fillna({"dst_chunk": data_idx["chunk"], "dst_file": data_idx["file"]})
        df["data/chunk_index"] = reindexed["dst_chunk"].to_numpy()
        df["data/file_index"] = reindexed["dst_file"].to_numpy()
    else:
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]

    for key, video_idx in videos_idx.items():
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        orig_chunks = df[orig_chunk_col].copy()
        orig_files = df[orig_file_col].copy()

        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})
        row_index = pd.MultiIndex.from_arrays(
            [orig_chunks, orig_files],
            names=["chunk_index", "file_index"],
        )

        if src_to_dst:
            src_keys = list(src_to_dst)
            mapping_index = pd.MultiIndex.from_tuples(
                src_keys,
                names=["chunk_index", "file_index"],
            )
            mapping_df = pd.DataFrame(
                [
                    (
                        *src_to_dst[src_key],
                        src_to_offset.get(src_key, 0.0),
                    )
                    for src_key in src_keys
                ],
                index=mapping_index,
                columns=["dst_chunk", "dst_file", "offset"],
            )
            reindexed = mapping_df.reindex(row_index)
            df[orig_chunk_col] = (
                reindexed["dst_chunk"]
                .fillna(video_idx["chunk"])
                .astype(orig_chunks.dtype, copy=False)
                .to_numpy()
            )
            df[orig_file_col] = (
                reindexed["dst_file"]
                .fillna(video_idx["file"])
                .astype(orig_files.dtype, copy=False)
                .to_numpy()
            )
            offsets = reindexed["offset"].fillna(0.0).to_numpy(dtype=float)
            df[f"videos/{key}/from_timestamp"] += offsets
            df[f"videos/{key}/to_timestamp"] += offsets
        elif src_to_offset:
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            mapping_series = pd.Series(src_to_offset, dtype=float)
            offsets = mapping_series.reindex(row_index).fillna(0.0).to_numpy()
            df[f"videos/{key}/from_timestamp"] += offsets
            df[f"videos/{key}/to_timestamp"] += offsets
        else:
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = (
                df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]
            )

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info.total_frames
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info.total_frames
    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes

    return df


def _normalize_array_values(df):
    import pandas as pd

    df = df.copy()
    for column in df.columns:
        if _has_array_values(df[column]):
            df[column] = pd.Series(
                [_normalize_array_value(value) for value in df[column]],
                dtype=object,
                index=df.index,
            )
    return df


def _normalize_array_value(value):
    import numpy as np

    if isinstance(value, np.ndarray) and value.ndim > 1:
        return [_normalize_array_value(item) for item in value]
    return value


def _has_array_values(series) -> bool:
    import numpy as np

    for value in series.head(32):
        if isinstance(value, np.ndarray):
            return True
    return False
