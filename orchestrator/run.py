"""fetch -> build -> publish, one batch at a time.

The stages do not overlap. Overlapping them was measured against the collection and
buys about 1.3 hours on a full pass -- 27.6 TB of source at 6.31 GB/s hidden behind
3.9 hours of encoding. What it costs is capacity prediction, backpressure and
reclaim ordering, every one of which deletes data when it is wrong. Not a trade
worth making, and keeping the stages apart also keeps a throughput measurement of
the encoder free of network I/O, which is half the reason the machine exists.

Within a stage, datasets are processed together, because one dataset does not keep a
large machine busy: work is parallelised per file, workers beyond the file count sit
idle, and 14 of the 36 datasets have fewer than a thousand episodes. See batch.py
for how a batch is chosen.

Everything that touches the world is injectable -- ``sync`` and ``run`` -- so the
ordering, the skipping and above all the deleting are testable without a network,
an encoder or a terabyte of disk.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import transfer
from .steps import Record, Steps, now


@dataclass(frozen=True)
class Outcome:
    dataset: str
    step: str
    # ok: it ran and succeeded. failed: it ran and did not. skipped: it was not
    # this dataset's turn, which is not a problem to be reported as one.
    status: str
    detail: str = ""

    def __str__(self) -> str:
        line = f"{self.status:<8}{self.step:<9}{self.dataset}"
        return f"{line}  {self.detail}" if self.detail else line


def spec_sha(spec) -> str:
    """The dataset's spec file, hashed. An edited spec makes an old build stale."""
    from dataset_registry.schema import DATASETS_DIR

    path = Path(DATASETS_DIR) / f"{spec.id}.yaml"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def dest_uri(profile: str | None, dataset: str) -> str | None:
    """Where a dataset built under ``profile`` is published."""
    if not profile:
        return None
    from lerobot_pipeline.profiles import ProfileError, load_profile

    try:
        template = (load_profile(profile).get("dest") or {}).get("uri")
    except ProfileError:
        return None
    if not template:
        return None
    return template.format(profile=profile, dataset=dataset)


# -- the three steps ----------------------------------------------------------


def fetch(env, spec, steps: Steps, *, dry_run: bool = False, sync=None) -> Outcome:
    """Bring the source onto this machine, unless it is already here.

    A dataset this machine stages by hand is left alone entirely, and no record is
    written for it. That absence is what later makes it undeletable: reclaim only
    ever removes paths a step recorded creating.
    """
    sync = sync or transfer.sync
    if env.is_staged(spec):
        return Outcome(spec.id, "fetch", "skipped", "source is staged on this machine")

    uri = spec.foundry_uri
    if not uri:
        return Outcome(spec.id, "fetch", "skipped", "no foundry mirror declared")
    if steps.done(spec.id, "fetch", source_uri=uri):
        return Outcome(spec.id, "fetch", "skipped", "already fetched")

    destination = env.raw_path(spec)
    if dry_run:
        return Outcome(spec.id, "fetch", "skipped", f"would sync {uri}")

    started = now()
    try:
        sync(uri, destination, nic_rate=env.nic_rate)
    except transfer.TransferError as exc:
        steps.write(
            Record(
                step="fetch",
                dataset=spec.id,
                status="failed",
                started=started,
                finished=now(),
                source_uri=uri,
                error=str(exc),
            )
        )
        return Outcome(spec.id, "fetch", "failed", str(exc))

    steps.write(
        Record(
            step="fetch",
            dataset=spec.id,
            status="ok",
            started=started,
            finished=now(),
            source_uri=uri,
            created=(str(destination),),
            bytes=transfer.tree_bytes(destination),
        )
    )
    return Outcome(spec.id, "fetch", "ok", str(destination))


def build(
    env,
    spec,
    steps: Steps,
    *,
    env_source: str,
    workers: int | None = None,
    batch_size: int = 1,
    dry_run: bool = False,
    run=None,
) -> Outcome:
    """Run the existing pipeline over one dataset.

    Nothing about how a dataset is processed is repeated here -- the pipeline
    already takes ``--env`` and ``--dataset`` and reads the rest from the registry
    and the profile. This decides only whether it is worth starting.

    ``batch_size`` is how many builds are running on this machine at once, declared
    because nothing downstream can find it out. The converter sizes its workers
    against the memory it can see, and what it can see is the whole node however many
    of these are started -- so a batch of three planned for three times the machine.
    """
    run = run or subprocess.run
    problems = spec.buildable()
    if problems:
        return Outcome(spec.id, "build", "skipped", problems[0])

    sha, profile = spec_sha(spec), env.profile
    if steps.done(spec.id, "build", spec_sha=sha, profile=profile):
        return Outcome(spec.id, "build", "skipped", "already built")

    source = env.raw_path(spec)
    if not source.exists():
        return Outcome(spec.id, "build", "skipped", f"no source at {source}")

    command = [
        sys.executable,
        "-m",
        "lerobot_pipeline.run",
        "--env",
        env_source,
        "--dataset",
        spec.id,
        "--overwrite",
    ]
    if workers:
        command += ["--workers", str(workers)]
    if dry_run:
        return Outcome(spec.id, "build", "skipped", "would run " + " ".join(command[1:]))

    started = now()
    completed = run(
        command,
        capture_output=True,
        text=True,
        check=False,
        **_shared_machine_env(batch_size),
    )
    if completed.returncode != 0:
        error = (completed.stderr or "").strip()[-2000:] or "the pipeline exited non-zero"
        steps.write(
            Record(
                step="build",
                dataset=spec.id,
                status="failed",
                started=started,
                finished=now(),
                spec_sha=sha,
                profile=profile,
                error=error,
            )
        )
        return Outcome(spec.id, "build", "failed", error.splitlines()[-1] if error else "")

    destination = env.out_path(spec)
    steps.write(
        Record(
            step="build",
            dataset=spec.id,
            status="ok",
            started=started,
            finished=now(),
            spec_sha=sha,
            profile=profile,
            created=(str(destination),),
            bytes=transfer.tree_bytes(destination),
        )
    )
    return Outcome(spec.id, "build", "ok", str(destination))


def publish(env, spec, steps: Steps, *, dry_run: bool = False, sync=None) -> Outcome:
    sync = sync or transfer.sync
    uri = dest_uri(env.profile, spec.id)
    if not uri:
        return Outcome(
            spec.id, "publish", "skipped", "the profile declares no dest.uri"
        )
    if steps.done(spec.id, "publish", dest_uri=uri):
        return Outcome(spec.id, "publish", "skipped", "already published")

    source = env.out_path(spec)
    if not source.exists():
        return Outcome(spec.id, "publish", "skipped", "nothing built to publish")
    if dry_run:
        return Outcome(spec.id, "publish", "skipped", f"would sync to {uri}")

    started = now()
    try:
        sync(source, uri)
    except transfer.TransferError as exc:
        steps.write(
            Record(
                step="publish",
                dataset=spec.id,
                status="failed",
                started=started,
                finished=now(),
                dest_uri=uri,
                error=str(exc),
            )
        )
        return Outcome(spec.id, "publish", "failed", str(exc))

    steps.write(
        Record(
            step="publish",
            dataset=spec.id,
            status="ok",
            started=started,
            finished=now(),
            dest_uri=uri,
            bytes=transfer.tree_bytes(source),
        )
    )
    return Outcome(spec.id, "publish", "ok", uri)


# -- reclaiming ---------------------------------------------------------------


def reclaim(
    env,
    spec,
    steps: Steps,
    known: Sequence,
    *,
    keep: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Delete what this dataset no longer needs, and nothing else.

    Three things have to hold before a path goes, and each rules out a different
    accident:

    * a step recorded creating it -- so a source staged by hand, which no step
      created, cannot be reached from here at all;
    * the step that consumes it has succeeded -- so a failed build never costs the
      source it failed on;
    * no other dataset still needs it -- the AgiBot dexhand and gripper subsets are
      one tree read twice, and deleting it after the first build would strand the
      second.
    """
    if keep or dry_run:
        return []

    removed: list[str] = []
    for step in ("fetch", "build"):
        for path in steps.reclaimable(spec.id, step):
            if _still_needed(env, spec, path, steps, known):
                continue
            target = Path(path)
            if not target.exists():
                continue
            shutil.rmtree(target, ignore_errors=True)
            removed.append(path)
    return removed


def _still_needed(env, spec, path: str, steps: Steps, known: Sequence) -> bool:
    """Whether another dataset sharing this exact path has yet to consume it."""
    from lerobot_pipeline.env import EnvError

    for other in known:
        if other.id == spec.id:
            continue
        try:
            shares = str(env.raw_path(other)) == path or str(env.out_path(other)) == path
        except EnvError:
            # this environment cannot place that dataset at all, so it is not
            # sharing anything; treating it as a sharer would strand this path
            continue
        if shares and not steps.done(other.id, "build"):
            return True
    return False


# -- a batch ------------------------------------------------------------------


def process(
    env,
    specs: Sequence,
    steps: Steps,
    *,
    env_source: str,
    only: Iterable[str] | None = None,
    keep: bool = False,
    dry_run: bool = False,
    sync=None,
    run=None,
    known: Sequence | None = None,
) -> list[Outcome]:
    """One batch, one stage at a time.

    The stage boundary is a barrier on purpose: every build in the batch finishes
    before any publish starts, so an encoder measurement is never taken while an
    upload is running.
    """
    wanted = tuple(only) if only is not None else ("fetch", "build", "publish")
    specs = list(specs)
    outcomes: list[Outcome] = []

    if "fetch" in wanted:
        outcomes += _spread(
            specs, lambda spec: fetch(env, spec, steps, dry_run=dry_run, sync=sync)
        )

    if "build" in wanted:
        share = _worker_share(env, len(specs))
        outcomes += _spread(
            specs,
            lambda spec: build(
                env,
                spec,
                steps,
                env_source=env_source,
                workers=share,
                batch_size=len(specs),
                dry_run=dry_run,
                run=run,
            ),
        )

    if "publish" in wanted:
        outcomes += _spread(
            specs, lambda spec: publish(env, spec, steps, dry_run=dry_run, sync=sync)
        )

    for spec in specs:
        for path in reclaim(
            env, spec, steps, known or specs, keep=keep, dry_run=dry_run
        ):
            outcomes.append(Outcome(spec.id, "reclaim", "ok", path))
    return outcomes


def _spread(specs: Sequence, work) -> list[Outcome]:
    """Run one step over every dataset in the batch at once.

    Threads rather than processes: every step here waits on a subprocess, so what is
    being overlapped is other people's CPU time, not this interpreter's.
    """
    if len(specs) <= 1:
        return [work(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        return list(pool.map(work, specs))


def _shared_machine_env(batch_size: int) -> dict:
    """The subprocess keywords that tell one build how much of the node is its own.

    Nothing when the build has the machine to itself, so the common path starts the
    pipeline with the environment it inherited and nothing to explain.
    """
    if batch_size <= 1:
        return {}
    return {"env": {**os.environ, "ANY4LEROBOT_MEMORY_SHARE": str(batch_size)}}


def _worker_share(env, batch_size: int) -> int | None:
    """The worker budget split across a batch, slightly oversubscribed.

    Slightly, because x264 at this output size cannot saturate the threads it is
    given -- the bench measured 16x16 marginally *faster* than 16x1 -- so leaving a
    little overlap absorbs the straggler tail rather than causing contention.
    """
    budget = env.runtime.get("workers")
    if not budget or batch_size <= 1:
        return budget or None
    return max(1, -(-int(budget) // batch_size))
