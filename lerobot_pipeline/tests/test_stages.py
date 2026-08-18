from pathlib import Path

import pytest

from lerobot_pipeline.config import parse_config
from lerobot_pipeline.stages import (
    StageError,
    converter_command,
    plan_stages,
    version_convert_command,
    version_convert_output,
)

WORKDIR = Path("/work")
RESIZE = [{"type": "resize_preserve_aspect_area"}]


def _config(source_type, dest_type, steps=RESIZE, args=None):
    source = {"type": source_type, "path": "/data/in"}
    if args:
        source["args"] = args
    return parse_config(
        {
            "name": "demo",
            "source": source,
            "steps": steps,
            "dest": {"type": dest_type, "path": "/data/out"},
        }
    )


def _kinds(stages):
    return [stage.kind for stage in stages]


# --- stage planning ----------------------------------------------------------


def test_same_version_with_video_steps_needs_no_version_conversion():
    """The whole point of the mp4 fast path: v2.1 in, v2.1 out, one pass."""
    stages = plan_stages(_config("lerobot_v21", "lerobot_v21"), WORKDIR)
    assert _kinds(stages) == ["transform"]


def test_the_final_stage_writes_to_the_configured_destination():
    stages = plan_stages(_config("lerobot_v21", "lerobot_v21"), WORKDIR)
    assert stages[-1].output_path == Path("/data/out")


def test_the_first_stage_reads_from_the_configured_source():
    stages = plan_stages(_config("lerobot_v21", "lerobot_v21"), WORKDIR)
    assert stages[0].input_path == Path("/data/in")


def test_transform_runs_before_version_conversion_so_less_data_is_converted():
    stages = plan_stages(_config("lerobot_v21", "lerobot_v30"), WORKDIR)
    assert _kinds(stages) == ["transform", "version_convert"]


def test_downgrading_to_v21_appends_a_version_conversion():
    stages = plan_stages(_config("lerobot_v30", "lerobot_v21"), WORKDIR)
    assert _kinds(stages) == ["transform", "version_convert"]
    assert stages[-1].detail == {"from": "lerobot_v30", "to": "lerobot_v21"}


def test_a_converter_source_produces_a_v30_dataset_first():
    stages = plan_stages(_config("openx", "lerobot_v30"), WORKDIR)
    assert _kinds(stages) == ["convert", "transform"]


def test_a_converter_source_targeting_v21_needs_all_three_stages():
    stages = plan_stages(_config("openx", "lerobot_v21"), WORKDIR)
    assert _kinds(stages) == ["convert", "transform", "version_convert"]


def test_without_steps_a_same_version_run_is_rejected_as_a_no_op():
    with pytest.raises(StageError):
        plan_stages(_config("lerobot_v21", "lerobot_v21", steps=[]), WORKDIR)


def test_without_steps_a_version_change_is_still_valid_work():
    stages = plan_stages(_config("lerobot_v21", "lerobot_v30", steps=[]), WORKDIR)
    assert _kinds(stages) == ["version_convert"]


def test_intermediate_stages_write_inside_the_work_directory():
    stages = plan_stages(_config("openx", "lerobot_v21"), WORKDIR)
    for stage in stages[:-1]:
        assert WORKDIR in stage.output_path.parents


def test_each_stage_reads_what_the_previous_one_wrote():
    stages = plan_stages(_config("openx", "lerobot_v21"), WORKDIR)
    for previous, following in zip(stages, stages[1:]):
        assert following.input_path == previous.output_path


def test_intermediate_paths_are_distinct():
    stages = plan_stages(_config("openx", "lerobot_v21"), WORKDIR)
    outputs = [stage.output_path for stage in stages]
    assert len(set(outputs)) == len(outputs)


def test_converter_arguments_are_carried_on_the_convert_stage():
    stages = plan_stages(_config("openx", "lerobot_v30", args={"fps": 5}), WORKDIR)
    assert stages[0].detail["args"] == {"fps": 5}


# --- version conversion commands ---------------------------------------------


def test_v21_to_v30_never_pushes_to_the_hub_by_accident():
    """That script defaults --push-to-hub to true."""
    cmd = version_convert_command("lerobot_v21", "lerobot_v30", Path("/w/ds"), "demo")
    assert cmd[cmd.index("--push-to-hub") + 1] == "false"


def test_v21_to_v30_targets_the_right_script_and_root():
    cmd = version_convert_command("lerobot_v21", "lerobot_v30", Path("/w/ds"), "demo")
    assert cmd[1].endswith("convert_dataset_v21_to_v30.py")
    assert cmd[cmd.index("--root") + 1] == "/w/ds"
    assert cmd[cmd.index("--repo-id") + 1] == "demo"


def test_v30_to_v21_targets_the_right_script():
    cmd = version_convert_command("lerobot_v30", "lerobot_v21", Path("/w/ds"), "demo")
    assert cmd[1].endswith("convert_dataset_v30_to_v21.py")


def test_v21_to_v30_leaves_the_result_at_the_same_root():
    assert version_convert_output("lerobot_v21", "lerobot_v30", Path("/w/ds")) == Path("/w/ds")


def test_v30_to_v21_also_leaves_the_result_at_the_same_root():
    """It builds ``{root}_v2.1`` and then swaps that into ``root``, so by the time
    the script returns the sibling is gone and only ``root`` holds the result."""
    assert version_convert_output("lerobot_v30", "lerobot_v21", Path("/w/ds")) == Path("/w/ds")


def test_an_unsupported_version_hop_is_rejected():
    with pytest.raises(StageError):
        version_convert_command("lerobot_v21", "lerobot_v21", Path("/w/ds"), "demo")


# --- converter commands ------------------------------------------------------


def test_openx_converter_uses_its_own_flag_names():
    cmd = converter_command("openx", Path("/data/in"), Path("/w/out"), {"repo_id": "x"})
    assert cmd[1].endswith("openx_rlds.py")
    assert cmd[cmd.index("--raw-dir") + 1] == "/data/in"
    assert cmd[cmd.index("--local-dir") + 1] == "/w/out"


def test_agibot_converter_uses_src_path_and_output_path():
    cmd = converter_command("agibot", Path("/data/in"), Path("/w/out"), {})
    assert cmd[cmd.index("--src-path") + 1] == "/data/in"
    assert cmd[cmd.index("--output-path") + 1] == "/w/out"


def test_extra_args_become_long_flags_with_underscores_converted():
    cmd = converter_command("agibot", Path("/in"), Path("/out"), {"eef_type": "gripper"})
    assert cmd[cmd.index("--eef-type") + 1] == "gripper"


def test_boolean_args_become_bare_flags():
    cmd = converter_command("openx", Path("/in"), Path("/out"), {"use_videos": True})
    assert "--use-videos" in cmd
    assert "True" not in cmd


def test_false_boolean_args_are_omitted():
    cmd = converter_command("openx", Path("/in"), Path("/out"), {"use_videos": False})
    assert "--use-videos" not in cmd


def test_list_args_are_expanded_into_repeated_values():
    cmd = converter_command("robomind", Path("/in"), Path("/out"), {"embodiments": ["a", "b"]})
    index = cmd.index("--embodiments")
    assert cmd[index + 1 : index + 3] == ["a", "b"]


def test_an_unknown_converter_is_rejected():
    with pytest.raises(StageError):
        converter_command("nope", Path("/in"), Path("/out"), {})
