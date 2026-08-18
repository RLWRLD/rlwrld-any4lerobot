import pytest

from orchestrator.steps import STEPS, Record, StepError, Steps


@pytest.fixture
def steps(tmp_path):
    return Steps(tmp_path / "state")


def ok(step: str, dataset: str = "action_net", **fields) -> Record:
    return Record(step=step, dataset=dataset, status="ok", **fields)


def test_a_step_that_never_ran_has_no_record(steps):
    assert steps.read("action_net", "fetch") is None


def test_a_written_record_reads_back_unchanged(steps):
    record = ok("fetch", source_uri="s3://bucket/x/", created=("/scratch/raw/a",), bytes=17)

    steps.write(record)

    assert steps.read("action_net", "fetch") == record


def test_a_rewritten_record_replaces_the_previous_one(steps):
    steps.write(Record(step="fetch", dataset="action_net", status="failed", error="boom"))
    steps.write(ok("fetch", bytes=5))

    stored = steps.read("action_net", "fetch")
    assert stored.status == "ok"
    assert stored.error is None


def test_writing_leaves_no_temporary_files_behind(steps, tmp_path):
    steps.write(ok("fetch"))

    written = sorted(p.name for p in (tmp_path / "state" / "action_net").iterdir())
    assert written == ["fetch.json"]


def test_datasets_and_steps_do_not_share_a_file(steps):
    steps.write(ok("fetch", "a"))
    steps.write(ok("build", "a"))
    steps.write(ok("fetch", "b"))

    assert steps.read("a", "fetch") is not None
    assert steps.read("a", "build") is not None
    assert steps.read("b", "build") is None


def test_an_unreadable_record_counts_as_not_done_rather_than_raising(steps, tmp_path):
    """Safe direction: a step we cannot read is a step we redo, and whose output we
    do not delete. Redoing is idempotent; deleting is not."""
    steps.write(ok("fetch"))
    (tmp_path / "state" / "action_net" / "fetch.json").write_text("{not json")

    assert steps.read("action_net", "fetch") is None
    assert steps.done("action_net", "fetch") is False


# -- done ---------------------------------------------------------------------


def test_done_is_false_without_a_record(steps):
    assert steps.done("action_net", "fetch") is False


def test_done_is_true_for_a_successful_step(steps):
    steps.write(ok("fetch"))

    assert steps.done("action_net", "fetch") is True


def test_done_is_false_for_a_failed_step(steps):
    steps.write(Record(step="fetch", dataset="action_net", status="failed", error="boom"))

    assert steps.done("action_net", "fetch") is False


def test_done_requires_the_recorded_inputs_to_still_match(steps):
    steps.write(ok("build", spec_sha="aaa", profile="rldx1"))

    assert steps.done("action_net", "build", spec_sha="aaa", profile="rldx1") is True
    assert steps.done("action_net", "build", spec_sha="bbb", profile="rldx1") is False
    assert steps.done("action_net", "build", spec_sha="aaa", profile="other") is False


# -- reclaimable --------------------------------------------------------------


def test_nothing_is_reclaimable_until_the_next_step_succeeds(steps):
    steps.write(ok("fetch", created=("/scratch/raw/action_net",)))

    assert steps.reclaimable("action_net", "fetch") == ()


def test_what_a_step_created_is_reclaimable_once_the_next_step_succeeds(steps):
    steps.write(ok("fetch", created=("/scratch/raw/action_net",)))
    steps.write(ok("build"))

    assert steps.reclaimable("action_net", "fetch") == ("/scratch/raw/action_net",)


def test_a_failed_next_step_reclaims_nothing(steps):
    steps.write(ok("fetch", created=("/scratch/raw/action_net",)))
    steps.write(Record(step="build", dataset="action_net", status="failed", error="boom"))

    assert steps.reclaimable("action_net", "fetch") == ()


def test_a_step_that_created_nothing_reclaims_nothing(steps):
    """A dataset staged by hand is never created by fetch, so it has no created
    paths and cannot be deleted -- the guarantee is structural, not a rule."""
    steps.write(ok("fetch", created=()))
    steps.write(ok("build"))

    assert steps.reclaimable("action_net", "fetch") == ()


def test_the_last_step_reclaims_nothing_because_nothing_follows_it(steps):
    steps.write(ok("publish", created=("/scratch/out/action_net",)))

    assert steps.reclaimable("action_net", "publish") == ()


def test_build_output_is_reclaimable_once_publish_succeeds(steps):
    steps.write(ok("build", created=("/scratch/out/action_net",)))
    steps.write(ok("publish"))

    assert steps.reclaimable("action_net", "build") == ("/scratch/out/action_net",)


# -- validation ---------------------------------------------------------------


def test_an_unknown_step_name_is_rejected(steps):
    with pytest.raises(StepError):
        steps.write(Record(step="upload", dataset="a", status="ok"))


def test_an_unknown_status_is_rejected(steps):
    with pytest.raises(StepError):
        steps.write(Record(step="fetch", dataset="a", status="maybe"))


def test_the_step_order_is_fetch_build_publish():
    assert STEPS == ("fetch", "build", "publish")


# --- a step is only done while what it made is still there --------------------


def test_a_step_whose_output_was_deleted_is_not_done(steps, tmp_path):
    """`reclaim` deletes a source once its build has succeeded, and leaves the fetch
    record saying ok. Anything asking again is then told the source is already there
    when it is not -- which is how a rebuild came to run against a missing
    directory."""
    gone = tmp_path / "raw" / "taco_play"
    steps.write(ok("fetch", created=(str(gone),)))

    assert not steps.done("action_net", "fetch")


def test_a_step_is_done_while_its_output_is_there(steps, tmp_path):
    here = tmp_path / "raw" / "taco_play"
    here.mkdir(parents=True)
    steps.write(ok("fetch", created=(str(here),)))

    assert steps.done("action_net", "fetch")


def test_a_step_that_created_nothing_is_still_done(steps):
    """A hand-staged source is fetched by nobody, so the record claims no paths and
    there is nothing to have gone missing."""
    steps.write(ok("fetch", created=()))

    assert steps.done("action_net", "fetch")


def test_every_recorded_path_has_to_survive_not_just_one(steps, tmp_path):
    here = tmp_path / "a"
    here.mkdir()
    steps.write(ok("fetch", created=(str(here), str(tmp_path / "b"))))

    assert not steps.done("action_net", "fetch")


def test_a_vanished_build_leaves_its_source_alone(steps, tmp_path):
    """The source may only go once the build that consumed it is real. If the build
    output has been removed the source is needed again, not reclaimable."""
    source = tmp_path / "raw" / "a"
    source.mkdir(parents=True)
    steps.write(ok("fetch", created=(str(source),)))
    steps.write(ok("build", created=(str(tmp_path / "out" / "a"),)))

    assert steps.reclaimable("action_net", "fetch") == ()
