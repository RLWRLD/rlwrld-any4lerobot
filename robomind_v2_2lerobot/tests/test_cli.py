"""0 개 변환은 실패다.

v1 은 `h5_<embodiment>/` 를 못 찾으면 아무 것도 yield 하지 않고 에러 없이 끝났고,
에피소드가 전부 skip 되면 출력 디렉토리를 rmtree 하고 정상 종료했다. "0 개 변환됨" 이
성공처럼 보였다. 이 테스트가 그 회귀를 막는다.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures import write_episode

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
    """
    from robomind_v2_h5 import RoboMINDv2Dataset, TasksFailed, main

    write_episode(tmp_path, "tienyi", "flaky_task", "0001_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "flaky_task", "0002_000000", frames=6, seconds=2)
    write_episode(tmp_path, "tienyi", "fine_task", "0003_000000", frames=6, seconds=2)

    original_save_episode = RoboMINDv2Dataset.save_episode
    calls_per_instance: dict[int, int] = {}

    def flaky_save_episode(self, *args, **kwargs):
        # Keyed by instance identity, not a global counter: this must fail on
        # this dataset's *second* episode regardless of which of the two tasks
        # main happens to process first.
        count = calls_per_instance.get(id(self), 0) + 1
        calls_per_instance[id(self)] = count
        if count == 2:
            raise OSError("disk full (simulated)")
        return original_save_episode(self, *args, **kwargs)

    monkeypatch.setattr(RoboMINDv2Dataset, "save_episode", flaky_save_episode)

    output_path = tmp_path / "out"
    with pytest.raises(TasksFailed, match="tienyi/flaky_task"):
        main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert not (output_path / "tienyi" / "flaky_task").exists()
    assert (output_path / "tienyi" / "flaky_task.failed").is_dir()

    fine_info = json.loads((output_path / "tienyi" / "fine_task" / "meta" / "info.json").read_text())
    assert fine_info["total_episodes"] == 1

    summary = json.loads((output_path / "summary.json").read_text())
    assert summary["tasks_failed"] == 1


def test_an_episode_whose_rate_drifts_from_the_task_median_is_skipped(tmp_path, caplog):
    """Real evidence this closes: one release task's episodes measured 26.94 Hz
    and 31.33 Hz, 16% apart -- the slower one used to set the dataset's rate and
    the faster one was only warned about, then written onto that wrong base
    anyway (see the I1 finding). Three same-rate episodes plus one that drifts
    past the 10% threshold: the outlier is skipped, not stretched onto a base
    it does not belong on.
    """
    from robomind_v2_h5 import main

    for stamp in ("0001_000000", "0002_000000", "0003_000000"):
        write_episode(tmp_path, "tienyi", "task", stamp, frames=6, seconds=2)  # ~3 Hz
    # 9 frames / 2 seconds = 4.5 Hz -- 50% faster than the 3 Hz base, well past
    # the 10% drift threshold.
    write_episode(tmp_path, "tienyi", "task", "0004_000000", frames=9, seconds=2)

    output_path = tmp_path / "out"
    with caplog.at_level("WARNING"):
        result = main(src_paths=[tmp_path], output_path=output_path, min_frames=1, debug=True)

    assert result == output_path
    assert "drifts from" in caplog.text

    info = json.loads((output_path / "tienyi" / "task" / "meta" / "info.json").read_text())
    assert info["fps"] == 3
    assert info["total_episodes"] == 3  # the drifting fourth episode was skipped


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
