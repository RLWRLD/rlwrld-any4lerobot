import json
from pathlib import Path

import pytest

from lerobot_pipeline import run as run_module
from lerobot_pipeline.config import parse_config
from lerobot_pipeline.run import parse_args, run_pipeline
from lerobot_pipeline.stages import StageSpec


def _config(tmp_path, source_type="lerobot_v21", dest_type="lerobot_v21", steps=None):
    return parse_config(
        {
            "name": "demo",
            "source": {"type": source_type, "path": str(tmp_path / "in")},
            "steps": steps if steps is not None else [{"type": "resize_preserve_aspect_area"}],
            "dest": {"type": dest_type, "path": str(tmp_path / "out")},
        }
    )


def _recorder(calls):
    """A stage executor that records the call and creates its output directory."""

    def execute(stage, config):
        calls.append((stage.kind, stage.input_path, stage.output_path))
        stage.output_path.mkdir(parents=True, exist_ok=True)
        (stage.output_path / "marker").write_text(stage.kind)
        return stage.output_path

    return {"convert": execute, "transform": execute, "version_convert": execute}


# --- orchestration -----------------------------------------------------------


def test_runs_every_planned_stage_in_order(tmp_path):
    calls = []
    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        workdir=tmp_path / "work",
        executors=_recorder(calls),
    )
    assert [kind for kind, _, _ in calls] == ["convert", "transform", "version_convert"]


def test_returns_the_destination_path(tmp_path):
    result = run_pipeline(
        _config(tmp_path), workdir=tmp_path / "work", executors=_recorder([])
    )
    assert result == tmp_path / "out"


def test_each_stage_consumes_the_previous_output(tmp_path):
    calls = []
    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        workdir=tmp_path / "work",
        executors=_recorder(calls),
    )
    for (_, _, produced), (_, consumed, _) in zip(calls, calls[1:]):
        assert consumed == produced


def test_intermediate_output_is_removed_after_a_successful_run(tmp_path):
    workdir = tmp_path / "work"
    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        workdir=workdir,
        executors=_recorder([]),
    )
    assert not workdir.exists()


def test_intermediate_output_can_be_kept_for_debugging(tmp_path):
    workdir = tmp_path / "work"
    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        workdir=workdir,
        executors=_recorder([]),
        keep_intermediate=True,
    )
    assert workdir.exists()


def test_an_existing_destination_is_refused_before_any_work_starts(tmp_path):
    (tmp_path / "out").mkdir(parents=True)
    calls = []

    with pytest.raises(FileExistsError):
        run_pipeline(
            _config(tmp_path, "openx", "lerobot_v21"),
            workdir=tmp_path / "work",
            executors=_recorder(calls),
        )
    assert calls == []


def test_overwrite_allows_replacing_an_existing_destination(tmp_path):
    (tmp_path / "out").mkdir(parents=True)
    run_pipeline(
        _config(tmp_path),
        workdir=tmp_path / "work",
        executors=_recorder([]),
        overwrite=True,
    )
    assert (tmp_path / "out" / "marker").exists()


def test_a_failing_stage_stops_the_run_and_keeps_no_destination(tmp_path):
    def boom(stage, config):
        raise RuntimeError("ffmpeg exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        run_pipeline(
            _config(tmp_path),
            workdir=tmp_path / "work",
            executors={"transform": boom},
        )
    assert not (tmp_path / "out").exists()


def test_a_failing_later_stage_still_cleans_the_work_directory(tmp_path):
    calls = []
    executors = dict(_recorder(calls))

    def boom(stage, config):
        raise RuntimeError("nope")

    executors["version_convert"] = boom
    workdir = tmp_path / "work"

    with pytest.raises(RuntimeError):
        run_pipeline(
            _config(tmp_path, "openx", "lerobot_v21"),
            workdir=workdir,
            executors=executors,
        )
    assert not workdir.exists()


# --- CLI ---------------------------------------------------------------------


def test_an_environment_is_required():
    """The dataset is a flag and the machine is a file; neither has a default worth
    guessing, so both have to be said."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_parses_the_environment_and_dataset():
    args = parse_args(["--env", "ec2", "--dataset", "action_net"])
    assert (args.env, args.dataset) == ("ec2", "action_net")


def test_all_rebuilds_the_collection():
    assert parse_args(["--env", "ec2", "--all"]).all is True


def test_overwrite_and_keep_intermediate_default_to_off():
    args = parse_args(["--env", "ec2", "--dataset", "action_net"])
    assert args.overwrite is False
    assert args.keep_intermediate is False


def test_runtime_overrides_are_exposed_on_the_command_line():
    args = parse_args(["--env", "ec2", "--dataset", "action_net",
                       "--workers", "4", "--threads-per-ffmpeg", "2"])
    assert (args.workers, args.threads_per_ffmpeg) == (4, 2)


def test_workdir_can_be_pointed_at_fast_local_storage():
    args = parse_args(["--env", "ec2", "--dataset", "action_net", "--workdir", "/tmp/w"])
    assert args.workdir == Path("/tmp/w")


def test_a_default_work_directory_is_derived_next_to_the_destination(tmp_path):
    calls = []
    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        executors=_recorder(calls),
        keep_intermediate=True,
    )
    intermediate = calls[0][2]
    assert intermediate.parent.parent == tmp_path


# --- runtime settings must not be silently ignored ---------------------------


def test_runtime_encoder_settings_reach_the_transform(tmp_path, monkeypatch):
    """runtime.preset/crf are accepted by the config; a run that parsed them and
    then dropped them would silently encode with the wrong settings."""
    from lerobot_pipeline import run as run_module

    captured = {}

    # **_ on purpose: this fake broke once when plan_transform gained a new
    # pass-through argument. The test cares about preset/crf, not the full signature.
    def fake_plan_transform(root, dest, steps, preset=None, crf=None, **_):
        captured.update(preset=preset, crf=crf)
        raise SystemExit("stop after planning")

    monkeypatch.setattr(run_module, "plan_transform", fake_plan_transform)

    config = parse_config(
        {
            "name": "demo",
            "source": {"type": "lerobot_v30", "path": str(tmp_path / "in")},
            "steps": [{"type": "resize_preserve_aspect_area"}],
            "dest": {"type": "lerobot_v30", "path": str(tmp_path / "out")},
            "runtime": {"preset": "veryfast", "crf": 23},
        }
    )
    stage = run_module.plan_stages(config, tmp_path / "work")[0]

    with pytest.raises(SystemExit):
        run_module.execute_transform(stage, config)

    assert captured == {"preset": "veryfast", "crf": 23}


# --- protecting the source ---------------------------------------------------


def test_version_conversion_reading_the_users_source_must_be_staged_first(tmp_path):
    """convert_dataset_v21_to_v30.py rewrites its --root in place, moving the
    original aside. It must never be pointed at the user's dataset."""
    from lerobot_pipeline.run import needs_staging

    config = _config(tmp_path, "lerobot_v21", "lerobot_v30", steps=[])
    from lerobot_pipeline.stages import plan_stages

    (stage,) = plan_stages(config, tmp_path / "work")
    assert stage.kind == "version_convert"
    assert needs_staging(stage, config)


def test_version_conversion_of_an_intermediate_needs_no_staging(tmp_path):
    from lerobot_pipeline.run import needs_staging
    from lerobot_pipeline.stages import plan_stages

    config = _config(tmp_path, "lerobot_v21", "lerobot_v30")
    stages = plan_stages(config, tmp_path / "work")
    assert not needs_staging(stages[-1], config)


def test_runtime_encoding_overrides_reach_the_transform(tmp_path, monkeypatch):
    """runtime.preset/crf are accepted by the config schema, so they must not be
    silently dropped on the way to the encoder."""
    from types import SimpleNamespace

    from lerobot_pipeline.run import execute_transform
    from lerobot_pipeline.stages import plan_stages

    captured = {}

    def fake_plan_transform(
        root, dest, steps, preset=None, crf=None, encoding_profile=None, **_
    ):
        captured.update(preset=preset, crf=crf, encoding_profile=encoding_profile)
        return SimpleNamespace(transcodes=(), links=(), info={}, root=root, dest=dest)

    monkeypatch.setattr("lerobot_pipeline.run.plan_transform", fake_plan_transform)
    monkeypatch.setattr("lerobot_pipeline.run.materialize", lambda plan, **kw: plan.dest)

    config = parse_config(
        {
            "name": "demo",
            "source": {"type": "lerobot_v21", "path": str(tmp_path / "in")},
            "steps": [{"type": "resize_preserve_aspect_area"}],
            "dest": {"type": "lerobot_v21", "path": str(tmp_path / "out")},
            "runtime": {
                "preset": "veryfast",
                "crf": 23,
                "encoding": "rldx1_reference",
            },
        }
    )
    execute_transform(plan_stages(config, tmp_path / "work")[0], config)

    assert captured["preset"] == "veryfast"
    assert captured["crf"] == 23
    # the named profile is resolved at config-parse time and handed over whole
    assert captured["encoding_profile"]["gop"] == 250


# --- CLI error reporting -----------------------------------------------------


def test_expected_errors_are_reported_cleanly_not_as_a_traceback(tmp_path, capsys):
    from lerobot_pipeline.run import main

    env_path = tmp_path / "demo.yaml"
    env_path.write_text(
        "profile: rldx1\n"
        f"raw_root: {tmp_path / 'raw'}\n"
        f"out_root: {tmp_path / 'out'}\n"
    )

    # a dataset whose columns were never recovered: the layout step refuses to be
    # built, and that has to read as a message rather than a stack trace
    assert main(["--env", str(env_path), "--dataset", "neural_robocurate"]) == 1
    message = capsys.readouterr().err
    assert "cannot be laid out" in message
    assert "Traceback" not in message


def test_an_unknown_environment_is_reported_cleanly(tmp_path, capsys):
    from lerobot_pipeline.run import main

    assert main(["--env", "no_such_env", "--dataset", "action_net"]) == 1
    assert "no_such_env" in capsys.readouterr().err


def test_an_unknown_dataset_lists_the_available_ones(tmp_path, capsys):
    from lerobot_pipeline.run import main

    env_path = tmp_path / "demo.yaml"
    env_path.write_text(
        f"profile: rldx1\nraw_root: {tmp_path}\nout_root: {tmp_path / 'out'}\n")
    assert main(["--env", str(env_path), "--dataset", "nope"]) == 1
    assert "action_net" in capsys.readouterr().err


# --- where a converter actually wrote --------------------------------------


def _converted(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    return root


def _convert_stage(tmp_path):
    return StageSpec(
        kind="convert",
        input_path=tmp_path / "in",
        output_path=tmp_path / "00_convert",
        detail={"source_type": "openx", "args": {}},
    )


def test_convert_follows_a_converter_that_wrote_a_directory_of_its_own(
    tmp_path, monkeypatch
):
    """openx_rlds.py writes <local-dir>/<name>_<version>_lerobot, not <local-dir>."""
    stage = _convert_stage(tmp_path)
    monkeypatch.setattr(
        run_module,
        "_run",
        lambda command, what: _converted(stage.output_path / "cmu_stretch__lerobot"),
    )

    produced = run_module.execute_convert(stage, _config(tmp_path, "openx"))

    assert produced == stage.output_path / "cmu_stretch__lerobot"


def test_convert_leaves_the_output_path_alone_when_the_dataset_is_already_there(
    tmp_path, monkeypatch
):
    stage = _convert_stage(tmp_path)
    monkeypatch.setattr(
        run_module, "_run", lambda command, what: _converted(stage.output_path)
    )

    assert run_module.execute_convert(stage, _config(tmp_path, "openx")) == stage.output_path


def test_convert_does_not_guess_between_two_datasets(tmp_path, monkeypatch):
    """Two candidates is a converter doing something unexpected -- report the flag
    that was passed and let the next stage fail on it, rather than pick one."""
    stage = _convert_stage(tmp_path)

    def two(command, what):
        _converted(stage.output_path / "a_lerobot")
        _converted(stage.output_path / "b_lerobot")

    monkeypatch.setattr(run_module, "_run", two)

    assert run_module.execute_convert(stage, _config(tmp_path, "openx")) == stage.output_path


def test_a_workdir_left_by_an_earlier_failure_is_cleared_first(tmp_path):
    """A stale stage directory must not be mistaken for this run's output."""
    workdir = tmp_path / "work"
    (workdir / "00_convert" / "leftover_lerobot" / "meta").mkdir(parents=True)

    run_pipeline(
        _config(tmp_path, "openx", "lerobot_v21"),
        workdir=workdir,
        executors=_recorder([]),
        keep_intermediate=True,
    )

    assert not (workdir / "00_convert" / "leftover_lerobot").exists()


class TestModalityIsAlwaysWritten:
    """`meta/modality.json` into the finished dataset, whatever built it.

    It used to be written by the `state_layout` step, which runs only when a config
    has table steps -- so the 27 OpenX datasets, whose converter writes
    `observation.state` itself and needs no layout step, came out without it. A
    rebuild with no modality.json cannot be read by the training stack at all, and
    nothing said so.
    """

    def _config(self, tmp_path, dataset, steps):
        return parse_config(
            {
                "name": "demo",
                "dataset": dataset,
                "source": {"type": "openx", "path": str(tmp_path / "in")},
                "steps": steps,
                "dest": {"type": "lerobot_v21", "path": str(tmp_path / "out")},
            }
        )

    def test_a_dataset_with_no_table_step_still_gets_one(self, tmp_path):
        """cmu_stretch is built by openx2lerobot and has no state_layout step. This is
        the case that was silently missing the file."""
        run_pipeline(
            self._config(tmp_path, "cmu_stretch", []),
            workdir=tmp_path / "work",
            executors=_recorder([]),
        )
        modality = json.loads((tmp_path / "out" / "meta" / "modality.json").read_text())
        assert modality["video"] == {
            "primary": {"original_key": "observation.images.image"}
        }

    def test_it_is_written_after_the_last_stage(self, tmp_path):
        """A version conversion writes into a fresh root and carries across only the
        files it knows about; modality.json is not one of them, so anything written
        before the conversion is left behind."""
        calls = []
        run_pipeline(
            self._config(tmp_path, "cmu_stretch", []),
            workdir=tmp_path / "work",
            executors=_recorder(calls),
        )
        assert calls[-1][0] == "version_convert"
        assert (tmp_path / "out" / "meta" / "modality.json").is_file()

    def test_a_run_with_no_dataset_named_writes_none(self, tmp_path):
        """A plain transform has no spec to derive it from, and inventing one would
        put a claim about the data into a file nobody checked."""
        config = parse_config(
            {
                "name": "demo",
                "source": {"type": "lerobot_v21", "path": str(tmp_path / "in")},
                "steps": [{"type": "resize_preserve_aspect_area"}],
                "dest": {"type": "lerobot_v30", "path": str(tmp_path / "out")},
            }
        )
        run_pipeline(config, workdir=tmp_path / "work", executors=_recorder([]))
        assert not (tmp_path / "out" / "meta" / "modality.json").exists()
