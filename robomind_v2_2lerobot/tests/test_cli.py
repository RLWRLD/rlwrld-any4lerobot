"""0 개 변환은 실패다.

v1 은 `h5_<embodiment>/` 를 못 찾으면 아무 것도 yield 하지 않고 에러 없이 끝났고,
에피소드가 전부 skip 되면 출력 디렉토리를 rmtree 하고 정상 종료했다. "0 개 변환됨" 이
성공처럼 보였다. 이 테스트가 그 회귀를 막는다.
"""

import gc
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures import write_episode
from lerobot.datasets import LeRobotDataset

from robomind_v2_h5 import NothingConverted, plan_tasks


def test_tasks_are_grouped_by_embodiment_and_task(tmp_path):
    write_episode(tmp_path, "tienyi", "task_a", "0001_000000")
    write_episode(tmp_path, "tienyi", "task_a", "0002_000000")
    write_episode(tmp_path, "tienyi", "task_b", "0003_000000")

    grouped = plan_tasks([tmp_path])

    assert set(grouped) == {("tienyi", "task_a"), ("tienyi", "task_b")}
    assert len(grouped[("tienyi", "task_a")]) == 2


def test_five_roots_group_into_one_embodiment(tmp_path):
    """Franka 5 개 repo — task 는 다르지만 embodiment 는 하나다."""
    roots = []
    for part in range(1, 6):
        root = tmp_path / f"part_{part}"
        write_episode(root, "franka", f"task_{part}", "0509_130459")
        roots.append(root)

    grouped = plan_tasks(roots)

    assert {embodiment for embodiment, _ in grouped} == {"franka"}
    assert len(grouped) == 5


def test_an_empty_source_raises_rather_than_succeeding(tmp_path):
    from robomind_v2_h5 import main

    with pytest.raises(NothingConverted, match="no episodes"):
        main(src_paths=[tmp_path], output_path=tmp_path / "out", debug=True)


def test_an_unknown_embodiment_raises(tmp_path):
    """디렉토리 이름에 대응하는 config 가 없으면 조용히 넘어가지 않는다."""
    from robomind_v2_h5 import main

    write_episode(tmp_path, "no_such_robot", "task", "0001_000000")

    with pytest.raises(NothingConverted, match="no config"):
        main(src_paths=[tmp_path], output_path=tmp_path / "out", debug=True)


def test_short_episodes_alone_raise(tmp_path):
    """모든 에피소드가 min_frames 미달이면 결과가 0 개다 — 실패여야 한다."""
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)

    with pytest.raises(NothingConverted):
        main(
            src_paths=[tmp_path],
            output_path=tmp_path / "out",
            min_frames=1000,
            debug=True,
        )


def test_a_rate_that_rounds_to_zero_is_skipped_rather_than_divided_by(tmp_path, caplog):
    """`convert_task` takes a task's fps from its first surviving episode via
    `round(episode.fps)`, with no floor: below 0.5 Hz that rounds to 0. The
    later drift check (`abs(episode.fps - fps) / fps`) would then divide by
    it for every subsequent episode -- but empirically, before the fix, the
    crash actually happens one step earlier and as a different exception:
    opening the dataset itself at `fps=0` raises `ValueError` from LeRobot's
    own `DatasetInfo` (`fps must be positive, got 0`), on the first episode,
    before a second episode is ever looked at. Same root cause either way --
    nothing stops a rate that rounds to 0 from being used as the task's fps --
    just a stricter downstream guard than the one this fix adds. The fix
    skips the zero-rounding episode like any other unusable one, so the
    second, normal-rate episode becomes the task's first surviving episode
    instead, and neither exception is reached.

    Two episodes are used deliberately, matching the two-episode shape the
    review finding described: a second episode still exists, so success here
    means the fix actually prevents the crash rather than the task merely
    running out of episodes to divide with (`test_short_episodes_alone_raise`
    above already covers the all-skipped/`NothingConverted` case; this test is
    about the survivor, not the absence of one).
    """
    from robomind_v2_h5 import main

    # 2 frames / 10 seconds = 0.2 Hz -> round() = 0. min_frames=2 lets this
    # short, deliberately-slow episode reach the fps check instead of being
    # rejected earlier for being short.
    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=2, seconds=10)
    # A normal-rate episode right behind it, so the task actually succeeds and
    # the skip shows up as a smaller episode count rather than as NothingConverted.
    write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2)

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(
            src_paths=[tmp_path], output_path=output_path, min_frames=2, debug=True
        )

    assert result == output_path
    assert "measured rate 0.200 Hz rounds to 0" in caplog.text

    # The skip is visible in the count: one episode was written, not two, and
    # the survivor -- not the zero-rate one -- is what set the dataset up.
    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 6
    assert info["fps"] == 3


def test_running_as_a_script_emits_its_own_log_records(tmp_path):
    """The module used to call `logging.basicConfig(...)` at import time.
    That is a no-op once the root logger already has a handler -- which, in
    a real run, it does by the time this module is imported: something
    pulled in via `robomind_v2_utils.lerobot_utils` attaches one first. The
    effective root level then silently stayed at the default WARNING, so
    every `logging.info(...)` call -- including the one telling an operator
    how many episodes were written -- never reached the console.

    Every other test in this file calls `main()` in-process, which can't
    exercise this: `if __name__ == "__main__":` only runs when the file is
    actually executed as a script, which is exactly the situation that was
    broken. So this test launches it as a real subprocess -- still against a
    synthetic fixture tree, no S3, no real data -- and checks its own
    stderr, the way an operator watching a real run would.
    """
    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)
    output_path = tmp_path / "out"
    script = Path(__file__).resolve().parent.parent / "robomind_v2_h5.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--src-paths", str(tmp_path),
            "--output-path", str(output_path),
            "--min-frames", "1",
            "--debug",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr

    # The configured format ("<asctime> - <LEVEL> - <message>"), not the bare
    # "INFO:root:..." that Python's own unconfigured-logging fallback would
    # print -- evidence this came from the entry point's own handler rather
    # than happening to slip out some other way.
    timestamped = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    assert re.search(timestamped + r"INFO - wrote .*trajectory\.hdf5", result.stderr)
    assert re.search(timestamped + r"INFO - wrote 1 episode\(s\), skipped 0, across 1 task\(s\)", result.stderr)


def test_a_malformed_config_fails_before_any_task_runs(tmp_path, monkeypatch):
    """main validates every discovered embodiment's config before handing any
    task to a worker (see the C1(a) finding) -- a bad YAML fails the whole run
    immediately, naming the embodiment and the reason, rather than only
    surfacing once some worker happens to draw that embodiment's first task
    (which, under Ray, would cost only that one task -- see TasksFailed --
    while every other embodiment quietly kept going).
    """
    from robomind_v2_h5 import main
    from robomind_v2_utils import configs

    write_episode(tmp_path, "probe_bad", "task", "0001_000000")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "probe_bad.yaml").write_text("robot_type: probe_bad\n")  # no cameras/streams
    monkeypatch.setattr(configs, "CONFIG_DIR", config_dir)

    output_path = tmp_path / "out"
    with pytest.raises(NothingConverted, match="probe_bad"):
        main(src_paths=[tmp_path], output_path=output_path, debug=True)

    assert not output_path.exists(), "no task should have run at all"


def test_mixed_resolution_within_a_task_skips_the_mismatched_episode(tmp_path, caplog):
    """Real evidence this closes: both 720x1280 and 480x640 occur within a single
    release task. build_features fixes a task's video shape from whichever episode
    happens to create the dataset; a later episode's own different resolution must
    be skipped rather than handed to add_frame -- the older converter had a
    workaround for exactly this case (see the C1(b) finding).

    Before this fix, nothing compared a later episode's shape against the one the
    dataset was created with: this episode would have gone straight to add_frame
    and save_episode with a video feature declared at the wrong resolution, which
    is one of the four routes the C1 finding traced into a task that dies mid-write
    with nothing catching it.
    """
    from robomind_v2_h5 import main

    write_episode(
        tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2, resolution=(32, 32)
    )
    write_episode(
        tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2, resolution=(48, 64)
    )

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "shape mismatch" in caplog.text

    # Loadable, and honest about how much of it is real: only the first
    # episode's resolution survived, not two episodes silently merged.
    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1


def test_a_fatal_write_failure_fails_the_task_and_quarantines_its_output(tmp_path, monkeypatch):
    """A failure raised by save_episode() itself -- once it may already have
    committed part of an episode to the writer's own bookkeeping, before video
    encoding or the metadata write that follows it can still fail -- cannot be
    safely shrugged off as "skip this one episode" (see convert_task's own
    comment on exactly that split). This proves the whole chain: the failing
    task's own output is renamed rather than left looking finished, main fails
    the run even though a second, unrelated task succeeds, and that other
    task's real output survives untouched.

    It also proves the quarantined copy is not a husk: before the fix in
    ``convert_task``'s outer ``except``, the abandoned ``RoboMINDv2Dataset``
    object -- reachable only through this test's own caught exception at this
    point -- got garbage-collected at some later, uncontrolled moment, and
    ``DatasetWriter.__del__``'s safety net called ``finalize()`` on it *after*
    ``_quarantine`` had already renamed its directory away.
    ``_flush_metadata_buffer``'s own ``mkdir(parents=True, exist_ok=True)``
    against the writer's still-stale ``root`` then recreated the just-renamed
    live directory, splitting the dataset across a resurrected husk at the
    live path and a ``.failed`` copy now missing ``meta/episodes/`` -- neither
    half loadable alone. This test used to pass anyway, by accident: pytest's
    own logging plugin keeps every emitted ``LogRecord`` alive for the whole
    test, and ``main``'s old error-log line embedded the caught exception
    *object* itself as a format argument -- pinning its traceback, and
    everything the traceback references (including the dataset), alive well
    past this point, so ``__del__`` never actually ran during the test. Run
    with ``pytest -p no:logging`` (no such plugin, nothing pins the exception
    once ``main`` is done with it), the old code failed the "live path is
    gone" assertion below. The fix makes this deterministic instead of
    accidental: ``convert_task`` now finalizes and forces every
    ``_finalized`` flag ``True`` before ``_quarantine`` ever runs, so
    whenever ``__del__`` actually fires -- immediately, or held alive by a log
    record for the rest of the test, as here -- it is a guaranteed no-op. The
    explicit ``gc.collect()`` below is not load-bearing for that reason, but
    keeps this test from ever again passing only because something happened
    to still be holding a reference.
    """
    from robomind_v2_h5 import RoboMINDv2Dataset, TasksFailed, main

    write_episode(tmp_path, "tienyi", "flaky_task", "0001_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "flaky_task", "0002_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "fine_task", "0003_000000", frames=6, seconds=2)

    original_save_episode = RoboMINDv2Dataset.save_episode
    calls_per_task: dict[str, int] = {}

    def flaky_save_episode(self, *args, **kwargs):
        # Keyed by the dataset's own repo_id (embodiment/task), not id(self):
        # this must fail on flaky_task's *second* episode regardless of which
        # of the two tasks main happens to process first -- and id(self) is
        # not safe for that. fine_task (one episode) finishes and its own
        # dataset object can be garbage-collected before flaky_task's is even
        # created; CPython is then free to reuse that exact memory address
        # for the new object, making id(flaky_task's dataset) ==
        # id(fine_task's dataset) and silently inheriting its leftover count
        # of 1. That raises on flaky_task's *first* episode instead of its
        # second -- caught only by actually running the *full* suite many
        # times over (rare: it needs fine_task's own dataset to actually be
        # collected and its address actually reused before this runs, which
        # depends on interpreter-wide allocator state from unrelated tests),
        # where it then wrote 0 episodes instead of 1 and failed this test's
        # own total_episodes assertion below -- a real flake this fixes, not
        # a hypothetical one.
        key = self.repo_id
        count = calls_per_task.get(key, 0) + 1
        calls_per_task[key] = count
        if count == 2:
            raise OSError("disk full (simulated)")
        return original_save_episode(self, *args, **kwargs)

    monkeypatch.setattr(RoboMINDv2Dataset, "save_episode", flaky_save_episode)

    output_path = tmp_path / "out"
    with pytest.raises(TasksFailed, match="tienyi/flaky_task"):
        main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    # See the docstring above: no longer load-bearing for correctness, but
    # keeps this test from passing merely because nothing happened to force
    # collection before the assertions below run.
    gc.collect()

    assert not (output_path / "tienyi" / "flaky_task").exists()
    failed_dir = output_path / "tienyi" / "flaky_task.failed"
    assert failed_dir.is_dir()

    # Loadable on its own: every piece LeRobotDataset needs is actually
    # present *inside the quarantined directory*, not split between it and a
    # husk left behind at the live path -- the failure mode this closes.
    failed_info = json.loads((failed_dir / "meta" / "info.json").read_text())
    assert failed_info["total_episodes"] == 1  # the one episode that saved before the failure
    assert list((failed_dir / "meta" / "episodes").rglob("*.parquet")), (
        "meta/episodes must live in the quarantined directory itself, not a husk left "
        "behind at the live path"
    )
    loaded = LeRobotDataset(repo_id="tienyi/flaky_task", root=failed_dir)
    assert loaded.meta.total_episodes == 1
    assert len(loaded) == failed_info["total_frames"]

    fine_info = json.loads((output_path / "tienyi" / "fine_task" / "meta" / "info.json").read_text())
    assert fine_info["total_episodes"] == 1

    summary = json.loads((output_path / "summary.json").read_text())
    assert summary["tasks_failed"] == 1
    # The bytes _quarantine preserves are real: the failed task's own written
    # episode must be counted here too, not reported as zero just because the
    # task did not finish cleanly (the I-D finding).
    flaky_tally = summary["by_embodiment"]["tienyi"]["tasks"]["flaky_task"]
    assert flaky_tally["written"] == 1


def test_a_per_episode_write_failure_is_recovered_and_the_task_continues(tmp_path, monkeypatch, caplog):
    """add_frame failing partway through an episode is recoverable at the
    single-episode level -- nothing has been committed to the writer's own
    bookkeeping yet at that point (save_episode is what does that), so
    clear_episode_buffer(delete_images=True) can safely discard the partial
    buffer and the task moves on to its next episode (see convert_task's own
    comment on exactly this split, and the C1(c) finding). This recovery path
    had no test at all before this one.

    The flaky episode is the first of two in the task; if clear_episode_buffer
    did not actually reset the buffer a fresh add_frame needs, the second
    episode's own save_episode would fail too (stacking a mismatched buffer),
    so the second episode actually landing is itself part of the proof.
    """
    from lerobot.datasets import LeRobotDataset
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=6, seconds=2)

    original_add_frame = LeRobotDataset.add_frame
    calls = {"n": 0}

    def flaky_add_frame(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # partway through the first episode's 6 frames
            raise OSError("scratch disk full (simulated)")
        return original_add_frame(self, *args, **kwargs)

    monkeypatch.setattr(LeRobotDataset, "add_frame", flaky_add_frame)

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "failed while adding frames" in caplog.text

    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1  # the flaky first episode was dropped, not the task

    summary = json.loads((output_path / "summary.json").read_text())
    tally = summary["by_embodiment"]["tienyi"]["tasks"]["task"]
    assert tally["written"] == 1
    assert tally["skipped"] == 1
    assert tally["reasons"] == {"write_failed": 1}


def test_the_dataset_shape_base_is_the_tasks_majority_not_the_first_episode(tmp_path):
    """fps became a median over a task's episodes (see the I1 finding); the
    camera shape did not, until now: the dataset used to be created from
    whichever episode happened to survive first, so a task could lose almost
    all of its episodes to a single odd-resolution one that happened to sort
    first in discovery order, and the run still exited 0 (the I-B finding).

    "0001" sorts first, so under the old rule it would have created the
    dataset at 48x64 and the other three, real 32x32 episodes would each have
    been skipped as a "mismatch" against it -- backwards from what actually
    happened (three real episodes outvoted by one odd one). The majority base
    must make the *odd* episode the one that gets skipped instead.
    """
    from robomind_v2_h5 import main

    write_episode(
        tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2, resolution=(48, 64)
    )
    for stamp in ("0002_000000", "0003_000000", "0004_000000"):
        write_episode(
            tmp_path, "tienyi", "task", stamp, frames=6, seconds=2, resolution=(32, 32)
        )

    output_path = tmp_path / "out"
    result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3  # the three 32x32 episodes, the majority
    assert info["features"]["observation.images.camera_top"]["shape"] == [32, 32, 3]

    summary = json.loads((output_path / "summary.json").read_text())
    tally = summary["by_embodiment"]["tienyi"]["tasks"]["task"]
    assert tally["written"] == 3
    assert tally["skipped"] == 1
    assert tally["reasons"] == {"resolution_mismatch": 1}


def test_a_task_that_mostly_skips_is_warned_about_but_not_failed(tmp_path, caplog):
    """A task whose skipped count exceeds what it wrote used to be invisible
    inside its embodiment's rolled-up total -- a task that lost 99% of its
    episodes could vanish inside an otherwise-healthy embodiment's sum (the
    I-B finding). This is logged as a warning, not failed outright: a skip is
    already a deliberate, reported outcome (not a crash), and majority-shape-
    basing (the test above) already fixes the single worst known cause of a
    task losing most of its episodes -- see _summarize's own docstring for
    the reasoning behind not also failing the task here.
    """
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)
    for stamp in ("0002_000000", "0003_000000"):
        write_episode(tmp_path, "tienyi", "task", stamp, frames=2, seconds=2)  # too short

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=3, debug=True)

    assert result == output_path  # not failed
    assert "skipped 2 episode(s) but only wrote 1" in caplog.text

    summary = json.loads((output_path / "summary.json").read_text())
    assert summary["tasks_failed"] == 0
    tally = summary["by_embodiment"]["tienyi"]["tasks"]["task"]
    assert tally["written"] == 1
    assert tally["skipped"] == 2


def test_summary_json_is_written_incrementally_and_a_run_gets_its_own_timestamped_copy(tmp_path):
    """Two defects this closes (the I-D finding): nothing used to be written
    until every task resolved, so a run killed partway through left no
    record at all of the tasks that had already finished; and the filename
    was fixed, so a second run into the same --output-path destroyed the
    first run's tally. Two tasks (so there is a "first" checkpoint to find
    still on disk once the run completes) prove the file exists incrementally
    rather than only at the very end, and a second call into the same output
    proves the first run's own timestamped copy survives it.
    """
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task_a", "0001_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "task_b", "0002_000000", frames=6, seconds=2)

    output_path = tmp_path / "out"
    main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    run_summaries = sorted(output_path.glob("summary-*.json"))
    assert len(run_summaries) == 1, "one run must leave exactly one timestamped copy"
    first_run_summary = run_summaries[0]
    first_run_content = first_run_summary.read_text()
    assert json.loads(first_run_content)["written"] == 2

    # A second run into the same --output-path must not destroy the first
    # run's own record.
    write_episode(tmp_path, "tienyi", "task_c", "0003_000000", frames=6, seconds=2)
    main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert first_run_summary.read_text() == first_run_content, (
        "a later run must not overwrite an earlier run's own timestamped summary"
    )
    run_summaries_after = sorted(output_path.glob("summary-*.json"))
    assert len(run_summaries_after) == 2, "the second run adds its own copy rather than replacing"

    # summary.json itself is the convenience "latest" pointer, and does move.
    latest = json.loads((output_path / "summary.json").read_text())
    assert latest["written"] == 3


def test_an_episode_whose_rate_drifts_from_the_task_median_is_skipped(tmp_path, caplog):
    """Real evidence this closes: one release task's episodes measured 26.94 Hz
    and 31.33 Hz, 16% apart -- the slower one used to set the dataset's rate and
    the faster one was only warned about, then written onto that wrong base
    anyway (see the I1 finding). Three same-rate episodes plus one that drifts
    past the 10% threshold: the outlier is skipped, not stretched onto a base
    it does not belong on.

    Spans of 20s here, not this suite's usual 2s: a `real` episode's own span
    bounds how precisely its rate can be known (``episode_fps``'s whole-second
    timestamps), and a per-episode tolerance now widens the flat 10% threshold
    for a short span (the I-A finding) -- at a 2s span that tolerance is 50%,
    which would swallow this test's own 50%-drifted outlier. 20s keeps that
    tolerance (5%) safely under both the flat 10% floor and the actual drift,
    so this test still demonstrates the flat threshold doing its job rather
    than accidentally exercising the span-aware one instead (see
    ``test_a_short_episodes_own_drift_tolerance_widens_with_its_span`` below
    for that one).
    """
    from robomind_v2_h5 import main

    for stamp in ("0001_000000", "0002_000000", "0003_000000"):
        write_episode(tmp_path, "tienyi", "task", stamp, frames=60, seconds=20)  # 3 Hz
    # 90 frames / 20 seconds = 4.5 Hz -- 50% faster than the 3 Hz base, well
    # past the 10% drift threshold (and its own 5% span tolerance).
    write_episode(tmp_path, "tienyi", "task", "0004_000000", frames=90, seconds=20)

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "drifts from" in caplog.text

    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["fps"] == 3
    assert info["total_episodes"] == 3  # the drifting fourth episode was skipped


def test_a_short_episodes_own_drift_tolerance_widens_with_its_span(tmp_path, caplog):
    """A `real` episode's own rate is frame_count / span, from a span rounded
    to whole seconds -- so the rate itself already carries roughly 1/span of
    quantization noise before an episode has drifted from its task at all
    (see reader.episode_fps and robomind_v2_h5._drift_tolerance). Real
    evidence this closes: across 19 readable real sample episodes, spans of
    8-34 seconds put that noise band at 2.9%-12.5% -- comparable to, or wider
    than, the flat 10% threshold on its own, so a flat threshold alone would
    skip a perfectly good short episode as "drift" that is really just
    measurement noise (the I-A finding).

    Three 10 Hz baseline episodes with long (20s) spans -- their own
    tolerance is a tight 5%, so they do not themselves widen what counts as
    agreement -- plus one boundary episode: 89 frames over an 8s span is
    11.125 Hz, an 11.25% drift from the 10 Hz median. That is past the flat
    10% threshold but inside the 12.5% (1/8) noise floor its own short span
    carries. The old, flat-only rule would have skipped it as drift; the
    span-aware one keeps it (contrast
    ``test_an_episode_whose_rate_drifts_from_the_task_median_is_skipped``
    above, whose outlier drifts 50% -- past even its own tolerance).
    """
    from robomind_v2_h5 import main

    for stamp in ("0001_000000", "0002_000000", "0003_000000"):
        write_episode(tmp_path, "tienyi", "task", stamp, frames=200, seconds=20)  # 10 Hz
    # 89 / 8 = 11.125 Hz -- 11.25% from the 10 Hz median, inside this
    # episode's own 12.5% (1/8) span tolerance but outside the flat 10%.
    write_episode(tmp_path, "tienyi", "task", "0004_000000", frames=89, seconds=8)

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "drifts from" not in caplog.text

    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4  # nothing was skipped for drift


def test_a_fractional_rate_is_not_spuriously_flagged_as_drift(tmp_path, caplog):
    """Rounding the task's own median to the integer fps a LeRobot dataset
    actually opens at can itself read as over 10% away from a fractional true
    rate: 10 frames / 3 s = 3.333 Hz rounds to 3, an 11.1% gap from that
    integer alone, with no episode actually disagreeing with any other. The
    drift check has to compare each episode against the unrounded median, not
    against the rounding that median happens to produce -- discovered only by
    running the fix through real Ray on non-round-number synthetic data, since
    every other test in this suite happens to use frame/second counts that
    divide evenly and so never exercises this.
    """
    from robomind_v2_h5 import main

    for stamp in ("0001_000000", "0002_000000", "0003_000000"):
        write_episode(tmp_path, "tienyi", "task", stamp, frames=10, seconds=3)  # 10/3 Hz exactly

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "drifts from" not in caplog.text

    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["fps"] == 3


def test_summary_json_reports_written_skipped_and_reasons_per_embodiment(tmp_path):
    """With 4,502 known-broken files in the corpus and nothing persisted beside
    the output before this, an operator had no way to reconcile what landed
    against what was expected (see the I7 finding). One written episode and one
    skipped-for-being-too-short episode in the same task should show up in this
    file -- unlike the log, which a real run under Ray never surfaces per
    episode in the first place (see the module logger's docstring).
    """
    from robomind_v2_h5 import main

    write_episode(tmp_path, "tienyi", "task", "0001_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "task", "0002_000000", frames=2, seconds=2)  # too short

    output_path = tmp_path / "out"
    main(src_paths=[tmp_path], output_path=output_path, min_frames=3, debug=True)

    summary = json.loads((output_path / "summary.json").read_text())
    assert summary["written"] == 1
    assert summary["skipped"] == 1
    assert summary["tasks_failed"] == 0

    tally = summary["by_embodiment"]["tienyi"]
    assert tally["written"] == 1
    assert tally["skipped"] == 1
    assert tally["reasons"] == {"too_short": 1}
    assert tally["failed_tasks"] == []


def test_ray_is_imported_before_the_source_tree_is_walked(tmp_path, monkeypatch):
    """A node bootstrapped without ray installed (see the I3 finding) should
    fail immediately, not after plan_tasks has walked however many hundred
    thousand objects a real source tree holds. Simulates "ray not installed" by
    making `import ray` raise, and proves discovery never ran by replacing
    plan_tasks with a stub that records whether it was called.
    """
    import robomind_v2_h5

    monkeypatch.setitem(sys.modules, "ray", None)
    called = []
    monkeypatch.setattr(
        robomind_v2_h5, "plan_tasks", lambda src_paths: called.append(src_paths) or {}
    )

    with pytest.raises(ImportError):
        robomind_v2_h5.main(src_paths=[tmp_path], output_path=tmp_path / "out", debug=False)

    assert called == [], "plan_tasks ran before the ray import was even attempted"


def test_max_node_memory_is_the_largest_live_node_not_the_cluster_total(tmp_path):
    """A task's memory reservation has to fit on *one* node -- Ray does not
    split a single task across several -- so main warns by comparing against
    the largest single node, not the cluster's summed total (the I-C
    finding): three 10 GiB nodes sum to 30 GiB, which could wrongly look
    like room for a 20 GiB task that in fact fits on none of them. Dead
    nodes are also excluded -- ray.nodes() keeps reporting ones that have
    left the cluster, still carrying their old resource figures.
    """
    from robomind_v2_h5 import _max_node_memory

    nodes = [
        {"Alive": True, "Resources": {"memory": 10 * 1024**3}},
        {"Alive": True, "Resources": {"memory": 10 * 1024**3}},
        {"Alive": False, "Resources": {"memory": 64 * 1024**3}},  # left the cluster
    ]

    assert _max_node_memory(nodes) == 10 * 1024**3


def test_max_node_memory_is_zero_for_an_empty_or_dead_cluster(tmp_path):
    from robomind_v2_h5 import _max_node_memory

    assert _max_node_memory([]) == 0
    assert _max_node_memory([{"Alive": False, "Resources": {"memory": 999}}]) == 0


def test_a_full_disk_writing_the_partial_marker_does_not_mask_the_real_failure(
    tmp_path, caplog
):
    """Found by actually reproducing a fatal write failure through real Ray on
    a quota-limited disk (see the evidence doc): a disk full enough to fail
    the original save_episode() write is exactly the condition most likely to
    also fail _write_partial_result's own write, right in convert_task's
    outer failure handler. Before this fix, that raised straight out of
    _write_partial_result, skipping the _quarantine call written right after
    it in the same handler and replacing the real "disk full" failure with an
    unrelated "could not write partial.json" one.
    """
    from robomind_v2_h5 import _write_partial_result

    real_dir = tmp_path / "tienyi" / "task"

    with caplog.at_level("ERROR"):
        # A file in the way of mkdir(parents=True, ...) is a simple, portable
        # way to make the write fail without actually filling a disk.
        (tmp_path / "tienyi").write_text("not a directory")
        _write_partial_result("tienyi", "task", real_dir, written=1, skipped=0, reasons={})

    assert "could not write partial.json" in caplog.text


def test_a_full_disk_writing_the_checkpoint_does_not_crash_the_run(tmp_path, monkeypatch):
    """Same discovery as the test above, for the other new write this wave
    added: a checkpoint that cannot currently be persisted (summary.json
    right beside a disk that just failed the actual conversion write) must
    not crash the whole run out from under whatever other tasks are still in
    flight.
    """
    from robomind_v2_h5 import TaskResult, _checkpoint_summary

    def raise_disk_full(path, summary):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("robomind_v2_h5._write_summary", raise_disk_full)

    output_path = tmp_path / "out"
    results = {("tienyi", "task"): TaskResult(written=1, skipped=0, reasons={})}
    summary = _checkpoint_summary(
        output_path, output_path / "summary-x.json", results, failures={}
    )

    assert summary["written"] == 1  # still computed and returned, not lost


def test_task_memory_bytes_scales_with_frame_count_not_just_cameras(tmp_path, monkeypatch):
    """The reservation used to price a camera, not a camera-*frame*: a fixed
    per-camera constant silently assumed every task runs about as many
    frames as the one episode it was measured on (the I-C finding). Doubling
    a task's own worst-case frame count (`reader.task_max_frames`) must
    double its reservation; a fixed-per-camera scheme would not move at all.
    """
    from robomind_v2_utils import configs
    from robomind_v2_h5 import _task_memory_bytes

    config = configs.load("tienyi")  # one camera

    small = _task_memory_bytes(config, save_depth=False, max_frames=100)
    large = _task_memory_bytes(config, save_depth=False, max_frames=200)
    assert large == pytest.approx(2 * small, rel=1e-6)

    with_depth = _task_memory_bytes(config, save_depth=True, max_frames=100)
    assert with_depth > small, "--save-depth must reserve more, not the same, per frame"


def test_no_embodiment_name_appears_in_the_converter():
    """변환 로직에 embodiment 이름이 등장하면 안 된다.

    부분문자열이 아니라 **단어 단위**로 본다. `ark` 는 dark·marks·remarkable 같은
    평범한 영어 단어에 들어 있어서, 부분문자열 검사는 산문에서 오탐을 낸다
    (실제로 "Marks" 가 걸린 적이 있다). 브랜드 표기도 같이 막는다 — 한 번은
    소문자 슬러그만 보다가 "UR5"/"AgileX" 가 통과했다.
    """
    slugs = [
        "agilex", "agilex_mobile", "ark", "ark_mobile", "franka", "franka_sim",
        "tienkung", "tienkung_sim", "tienyi", "tienyi_mobile", "ur_dex",
    ]
    brands = ["UR5", "AgileX", "Agilex", "Franka", "Tianyi", "ARX", "TienKung"]
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(n) for n in slugs + brands) + r")\b"
    )

    root = Path(__file__).resolve().parents[1]
    sources = [root / "robomind_v2_h5.py", *(root / "robomind_v2_utils").glob("*.py")]

    for path in sources:
        found = sorted(set(pattern.findall(path.read_text())))
        assert found == [], f"{path.name} names embodiments: {found}"
