"""The sequential loop, and the two things it must never get wrong: deleting a
source something else still needs, and deleting a source it did not download."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import load  # noqa: E402
from lerobot_pipeline.env import Environment  # noqa: E402
from orchestrator.run import (  # noqa: E402
    Outcome,
    build,
    fetch,
    process,
    publish,
    reclaim,
)
from orchestrator.steps import Record, Steps  # noqa: E402


@pytest.fixture
def env(tmp_path):
    return Environment(
        name="test",
        profile="rldx1",
        raw_root=tmp_path / "raw",
        out_root=tmp_path / "out",
        state_root=tmp_path / "state",
    )


@pytest.fixture
def steps(env):
    return Steps(env.state_root)


class FakeSync:
    """Stands in for a transfer, creating the destination as a real one would."""

    def __init__(self, fails: bool = False):
        self.fails = fails
        self.calls: list[tuple[str, str]] = []

    def __call__(self, source, dest, *, dry_run=False, **kwargs):
        self.calls.append((str(source), str(dest)))
        if self.fails:
            from orchestrator.transfer import TransferError

            raise TransferError("fatal error: Access Denied")
        if str(dest).startswith("s3://"):
            return
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.bin").write_bytes(b"x" * 32)


class FakeRun:
    """Stands in for the pipeline, producing an output tree as a real run would."""

    def __init__(self, returncode: int = 0, creates: Path | None = None):
        self.returncode = returncode
        self.creates = creates
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        if self.returncode == 0 and self.creates is not None:
            self.creates.mkdir(parents=True, exist_ok=True)
            (self.creates / "meta").mkdir(exist_ok=True)
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def ok(step: str, dataset: str, **fields) -> Record:
    return Record(step=step, dataset=dataset, status="ok", **fields)


# -- fetch --------------------------------------------------------------------


def test_fetch_pulls_the_foundry_mirror_into_this_machines_raw_path(env, steps):
    spec = load("action_net")
    sync = FakeSync()

    outcome = fetch(env, spec, steps, sync=sync)

    assert outcome.status == "ok"
    assert sync.calls == [(spec.foundry_uri, str(env.raw_path(spec)))]


def test_fetch_records_the_path_it_created_so_it_can_be_reclaimed(env, steps):
    spec = load("action_net")

    fetch(env, spec, steps, sync=FakeSync())

    record = steps.read("action_net", "fetch")
    assert record.created == (str(env.raw_path(spec)),)
    assert record.source_uri == spec.foundry_uri


def test_fetch_skips_a_dataset_with_no_mirror_rather_than_failing(env, steps):
    """oneuniverse_simul is derived data with no upstream to mirror, so it will
    never have an address to fetch from. That is data not delivered, not a broken
    run, and the whole collection must not stop for it."""
    sync = FakeSync()

    outcome = fetch(env, load("oneuniverse_simul"), steps, sync=sync)

    assert outcome.status == "skipped"
    assert sync.calls == []


def test_fetch_leaves_a_hand_staged_source_alone(tmp_path, steps):
    staged = tmp_path / "somewhere" / "action_net"
    staged.mkdir(parents=True)
    env = Environment(
        name="test",
        profile="rldx1",
        raw_root=tmp_path / "raw",
        out_root=tmp_path / "out",
        state_root=tmp_path / "state",
        paths={"action_net": str(staged)},
    )
    sync = FakeSync()

    outcome = fetch(env, load("action_net"), steps, sync=sync)

    assert outcome.status == "skipped"
    assert sync.calls == []
    assert steps.read("action_net", "fetch") is None


def test_fetch_does_not_repeat_a_completed_transfer(env, steps):
    spec = load("action_net")
    sync = FakeSync()
    fetch(env, spec, steps, sync=sync)

    fetch(env, spec, steps, sync=sync)

    assert len(sync.calls) == 1


def test_fetch_repeats_when_the_mirror_it_recorded_is_not_the_current_one(env, steps):
    spec = load("action_net")
    steps.write(ok("fetch", "action_net", source_uri="s3://old/prefix/"))
    sync = FakeSync()

    fetch(env, spec, steps, sync=sync)

    assert len(sync.calls) == 1


def test_a_failed_fetch_records_the_failure_and_claims_nothing(env, steps):
    outcome = fetch(env, load("action_net"), steps, sync=FakeSync(fails=True))

    assert outcome.status == "failed"
    record = steps.read("action_net", "fetch")
    assert record.status == "failed"
    assert "Access Denied" in record.error
    assert record.created == ()


# -- build --------------------------------------------------------------------


def test_build_runs_the_pipeline_for_one_dataset(env, steps):
    spec = load("action_net")
    env.raw_path(spec).mkdir(parents=True)
    run = FakeRun()

    outcome = build(env, spec, steps, env_source="ec2", run=run)

    assert outcome.status == "ok"
    command = run.calls[0]
    assert command[1:4] == ["-m", "lerobot_pipeline.run", "--env"]
    assert "action_net" in command


def test_build_divides_the_worker_budget_across_a_batch(env, steps):
    spec = load("action_net")
    env.raw_path(spec).mkdir(parents=True)
    run = FakeRun()

    build(env, spec, steps, env_source="ec2", workers=12, run=run)

    command = run.calls[0]
    assert command[command.index("--workers") + 1] == "12"


def test_build_refuses_a_dataset_whose_layout_was_never_recovered(env, steps):
    """galaxea's 26-slot action has no recovered source. Better to say so than to
    spend hours emitting zeros that look like a dataset."""
    spec = load("galaxea")
    env.raw_path(spec).mkdir(parents=True)
    run = FakeRun()

    outcome = build(env, spec, steps, env_source="ec2", run=run)

    assert outcome.status == "skipped"
    assert run.calls == []


def test_build_skips_when_there_is_no_source_on_disk(env, steps):
    run = FakeRun()

    outcome = build(env, load("action_net"), steps, env_source="ec2", run=run)

    assert outcome.status == "skipped"
    assert run.calls == []


def test_a_failed_build_records_the_failure(env, steps):
    spec = load("action_net")
    env.raw_path(spec).mkdir(parents=True)

    outcome = build(env, spec, steps, env_source="ec2", run=FakeRun(returncode=1))

    assert outcome.status == "failed"
    assert steps.read("action_net", "build").status == "failed"


# -- publish ------------------------------------------------------------------


def test_publish_uploads_the_built_dataset_to_the_profiles_destination(env, steps):
    spec = load("action_net")
    env.out_path(spec).mkdir(parents=True)
    sync = FakeSync()

    outcome = publish(env, spec, steps, sync=sync)

    assert outcome.status == "ok"
    source, dest = sync.calls[0]
    assert source == str(env.out_path(spec))
    assert dest.startswith("s3://") and dest.endswith("/action_net/")


def test_publish_skips_when_nothing_was_built(env, steps):
    sync = FakeSync()

    outcome = publish(env, load("action_net"), steps, sync=sync)

    assert outcome.status == "skipped"
    assert sync.calls == []


# -- reclaim ------------------------------------------------------------------


def test_a_fetched_source_is_removed_once_it_has_been_built(env, steps):
    spec = load("action_net")
    source = env.raw_path(spec)
    source.mkdir(parents=True)
    steps.write(ok("fetch", "action_net", created=(str(source),)))
    steps.write(ok("build", "action_net"))

    removed = reclaim(env, spec, steps, [spec])

    assert removed == [str(source)]
    assert not source.exists()


def test_a_source_two_datasets_share_survives_until_both_are_built(env, steps):
    """agibot_dexhand and agibot_gripper are one tree read twice with different
    flags. Deleting it after the first build would break the second."""
    dexhand, gripper = load("agibot_dexhand"), load("agibot_gripper")
    shared = env.raw_path(dexhand)
    assert shared == env.raw_path(gripper)
    shared.mkdir(parents=True)
    for dataset in ("agibot_dexhand", "agibot_gripper"):
        steps.write(ok("fetch", dataset, created=(str(shared),)))
    steps.write(ok("build", "agibot_dexhand"))

    removed = reclaim(env, dexhand, steps, [dexhand, gripper])

    assert removed == []
    assert shared.exists()

    steps.write(ok("build", "agibot_gripper"))
    assert reclaim(env, dexhand, steps, [dexhand, gripper]) == [str(shared)]


def test_keep_removes_nothing(env, steps):
    spec = load("action_net")
    source = env.raw_path(spec)
    source.mkdir(parents=True)
    steps.write(ok("fetch", "action_net", created=(str(source),)))
    steps.write(ok("build", "action_net"))

    assert reclaim(env, spec, steps, [spec], keep=True) == []
    assert source.exists()


def test_a_hand_staged_source_is_never_removed(tmp_path, steps):
    """It has no fetch record, so there is nothing recorded as created and nothing
    to delete -- the protection is the absence of a claim, not a check."""
    staged = tmp_path / "somewhere" / "action_net"
    staged.mkdir(parents=True)
    env = Environment(
        name="test",
        profile="rldx1",
        raw_root=tmp_path / "raw",
        out_root=tmp_path / "out",
        state_root=tmp_path / "state",
        paths={"action_net": str(staged)},
    )
    spec = load("action_net")
    steps.write(ok("build", "action_net"))
    steps.write(ok("publish", "action_net"))

    assert reclaim(env, spec, steps, [spec]) == []
    assert staged.exists()


# -- a batch ------------------------------------------------------------------


def test_a_batch_runs_fetch_then_build_then_publish(env, steps):
    spec = load("action_net")
    sync, run = FakeSync(), FakeRun(creates=env.out_path(spec))

    outcomes = process(env, [spec], steps, env_source="ec2", sync=sync, run=run)

    order = [
        outcome.step
        for outcome in outcomes
        if outcome.status == "ok" and outcome.step != "reclaim"
    ]
    assert order == ["fetch", "build", "publish"]


def test_a_batch_reclaims_the_source_once_it_has_been_published(env, steps):
    spec = load("action_net")
    run = FakeRun(creates=env.out_path(spec))

    outcomes = process(env, [spec], steps, env_source="ec2", sync=FakeSync(), run=run)

    reclaimed = {outcome.detail for outcome in outcomes if outcome.step == "reclaim"}
    assert reclaimed == {str(env.raw_path(spec)), str(env.out_path(spec))}
    assert not env.raw_path(spec).exists()


def test_one_datasets_failure_does_not_stop_the_others(env, steps):
    specs = [load("action_net"), load("galaxea")]
    sync, run = FakeSync(), FakeRun()

    outcomes = process(env, specs, steps, env_source="ec2", sync=sync, run=run)

    built = {o.dataset: o.status for o in outcomes if o.step == "build"}
    assert built["action_net"] == "ok"
    assert built["galaxea"] == "skipped"


def test_a_dry_run_transfers_nothing_and_records_nothing(env, steps):
    sync, run = FakeSync(), FakeRun()

    process(env, [load("action_net")], steps, env_source="ec2", sync=sync, run=run, dry_run=True)

    assert steps.read("action_net", "fetch") is None
    assert run.calls == []


def test_outcomes_carry_the_dataset_and_step_they_describe(env, steps):
    outcome = fetch(env, load("droid"), steps, sync=FakeSync())

    assert isinstance(outcome, Outcome)
    assert (outcome.dataset, outcome.step) == ("droid", "fetch")


class TestTimings:
    """Durations come out of the records rather than out of a wrapper script.

    Every step already writes `started`, `finished` and, for fetch, `bytes` -- a
    record that cannot say when it ran cannot answer whether a re-run got slower.
    The report only reads them, which is why it works on records written by a run
    nobody was watching.
    """

    def _steps(self, tmp_path):
        from orchestrator.steps import Record, Steps

        steps = Steps(tmp_path)
        steps.write(Record(
            step="fetch", dataset="viola", status="ok",
            started="2026-08-20T00:00:00+00:00",
            finished="2026-08-20T00:01:40+00:00",   # 100s
            bytes=11_200_000_000,                    # 112 MB/s
        ))
        steps.write(Record(
            step="build", dataset="viola", status="ok",
            started="2026-08-20T00:02:00+00:00",
            finished="2026-08-20T00:07:00+00:00",   # 300s
        ))
        return steps

    def test_the_fetch_rate_is_bytes_over_its_own_seconds(self, tmp_path):
        from orchestrator.__main__ import render_timings

        class Spec:
            id = "viola"
            delivered_episodes = 135

        out = render_timings(self._steps(tmp_path), [Spec()])
        assert "112.0" in out, out
        assert "11.2 GB" in out, out

    def test_build_is_reported_per_episode(self, tmp_path):
        from orchestrator.__main__ import render_timings

        class Spec:
            id = "viola"
            delivered_episodes = 135

        out = render_timings(self._steps(tmp_path), [Spec()])
        assert "0.45" in out, out          # 135 episodes / 300s

    def test_a_step_that_never_ran_is_a_dash_not_a_zero(self, tmp_path):
        """A missing publish must not read as an instant one."""
        from orchestrator.__main__ import render_timings

        class Spec:
            id = "viola"
            delivered_episodes = 135

        out = render_timings(self._steps(tmp_path), [Spec()])
        assert out.rstrip().splitlines()[2].rstrip().endswith("-"), out

    def test_a_record_missing_a_stamp_yields_no_duration(self, tmp_path):
        """Records written before `started` existed must not crash the report."""
        from orchestrator.__main__ import _seconds
        from orchestrator.steps import Record

        assert _seconds(Record(step="fetch", dataset="v", status="ok",
                               finished="2026-08-20T00:00:00+00:00")) is None
        assert _seconds(None) is None
