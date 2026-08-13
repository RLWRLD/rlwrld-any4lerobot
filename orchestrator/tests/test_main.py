import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot_pipeline.env import Environment  # noqa: E402
from orchestrator.__main__ import main, plan, render_status, select  # noqa: E402
from orchestrator.steps import Record, Steps  # noqa: E402


@pytest.fixture
def env(tmp_path):
    return Environment(
        name="test",
        profile="rldx1",
        raw_root=tmp_path / "raw",
        out_root=tmp_path / "out",
        state_root=tmp_path / "state",
        batch={"max_datasets": 3, "target_episodes": 768},
    )


# -- choosing what to work on -------------------------------------------------


def test_naming_no_dataset_selects_the_whole_registry():
    assert len(select(None)) >= 36


def test_naming_datasets_selects_exactly_those():
    assert [spec.id for spec in select(["galaxea", "action_net"])] == [
        "galaxea",
        "action_net",
    ]


def test_an_unknown_dataset_name_is_an_error():
    with pytest.raises(SystemExit):
        select(["not_a_dataset"])


def test_a_dataset_with_plenty_of_work_is_planned_on_its_own(env):
    grouped = plan(env, select(["droid", "viola", "cmu_stretch"]))

    assert [tuple(spec.id for spec in batch) for batch in grouped] == [
        ("cmu_stretch", "viola"),
        ("droid",),
    ]


def test_small_datasets_scattered_through_the_registry_still_share_a_batch(env):
    """Registry order is alphabetical, which leaves the small datasets nowhere near
    each other -- viola, ucsd_kitchen and berkeley_fanuc are separated by the whole
    collection. Planning by size instead is the difference between three of them
    filling a machine together and each one running alone on 96 cores."""
    grouped = plan(
        env,
        select(
            [
                "viola",
                "droid",
                "ucsd_kitchen_dataset_converted_externally_to_rlds",
                "cmu_stretch",
            ]
        ),
    )

    sizes = sorted(len(batch) for batch in grouped)
    assert sizes == [1, 3]


def test_the_environments_batch_limits_are_what_plan_uses(tmp_path):
    env = Environment(
        name="test",
        state_root=tmp_path / "state",
        batch={"max_datasets": 1, "target_episodes": 1},
    )

    grouped = plan(env, select(["viola", "cmu_stretch"]))

    assert [len(batch) for batch in grouped] == [1, 1]


# -- status -------------------------------------------------------------------


def test_status_reports_a_dataset_nothing_has_happened_to(env):
    text = render_status(Steps(env.state_root), select(["action_net"]))

    assert "action_net" in text
    assert "fetch" in text and "build" in text and "publish" in text


def test_status_shows_what_has_been_recorded(env):
    steps = Steps(env.state_root)
    steps.write(Record(step="fetch", dataset="action_net", status="ok"))
    steps.write(Record(step="build", dataset="action_net", status="failed", error="boom"))

    text = render_status(steps, select(["action_net"]))

    assert "ok" in text
    assert "failed" in text


# -- the CLI ------------------------------------------------------------------


def test_an_unknown_environment_fails_with_a_message(capsys):
    assert main(["status", "--env", "no_such_env"]) != 0
    assert "no_such_env" in capsys.readouterr().err


def test_an_environment_without_a_state_root_says_what_is_missing(tmp_path, capsys):
    path = tmp_path / "bare.yaml"
    path.write_text("profile: rldx1\nraw_root: /raw\nout_root: /out\n")

    assert main(["status", "--env", str(path)]) != 0
    assert "state_root" in capsys.readouterr().err


def test_status_runs_against_a_real_environment_file(tmp_path, capsys):
    path = tmp_path / "demo.yaml"
    path.write_text(
        "profile: rldx1\nraw_root: /raw\nout_root: /out\n"
        f"state_root: {tmp_path / 'state'}\n"
    )

    assert main(["status", "--env", str(path), "--dataset", "action_net"]) == 0
    assert "action_net" in capsys.readouterr().out


def test_a_dry_run_reports_what_it_would_do_and_changes_nothing(tmp_path, capsys):
    state = tmp_path / "state"
    path = tmp_path / "demo.yaml"
    path.write_text(
        "profile: rldx1\nraw_root: /raw\nout_root: /out\n" f"state_root: {state}\n"
    )

    code = main(
        ["run", "--env", str(path), "--dataset", "action_net", "--dry-run"]
    )

    assert code == 0
    assert "would" in capsys.readouterr().out
    assert Steps(state).read("action_net", "fetch") is None
