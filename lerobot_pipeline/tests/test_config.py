from pathlib import Path

import pytest

from lerobot_pipeline.config import ConfigError, load_config, parse_config

MINIMAL = {
    "name": "demo",
    "source": {"type": "lerobot_v21", "path": "/data/in"},
    "steps": [{"type": "resize_preserve_aspect_area"}],
    "dest": {"type": "lerobot_v21", "path": "/data/out"},
}


def _config(**overrides):
    return parse_config({**MINIMAL, **overrides})


def test_parses_a_minimal_config():
    cfg = _config()
    assert cfg.name == "demo"
    assert cfg.source.type == "lerobot_v21"
    assert cfg.source.path == Path("/data/in")
    assert cfg.dest.path == Path("/data/out")


def test_steps_are_instantiated_from_the_registry():
    (step,) = _config().steps
    assert step.kind == "video"
    assert step.plan((480, 640)).out_shape == (192, 256)


def test_step_parameters_reach_the_step():
    cfg = _config(
        steps=[
            {"type": "resize_preserve_aspect_area", "max_area": 128**2, "multiple": 16}
        ]
    )
    assert cfg.steps[0].plan((480, 640)).out_shape == (96, 128)


def test_user_home_in_paths_is_expanded():
    cfg = _config(source={"type": "lerobot_v30", "path": "~/data/in"})
    assert "~" not in str(cfg.source.path)
    assert cfg.source.path.is_absolute()


def test_runtime_defaults_to_automatic():
    rt = _config().runtime
    assert rt.workers is None
    assert rt.threads_per_ffmpeg is None
    assert rt.preset is None
    assert rt.crf is None


def test_runtime_overrides_are_read():
    rt = _config(runtime={"workers": 8, "threads_per_ffmpeg": 2, "preset": "veryfast"}).runtime
    assert (rt.workers, rt.threads_per_ffmpeg, rt.preset) == (8, 2, "veryfast")


def test_steps_may_be_empty():
    assert _config(steps=[]).steps == ()


def test_unknown_source_type_lists_the_supported_ones():
    with pytest.raises(ConfigError) as exc:
        _config(source={"type": "openx_rlds", "path": "/data/in"})
    assert "openx" in str(exc.value)


def test_unknown_dest_type_is_rejected():
    with pytest.raises(ConfigError):
        _config(dest={"type": "rlds", "path": "/data/out"})


def test_unknown_step_type_is_a_config_error():
    with pytest.raises(ConfigError) as exc:
        _config(steps=[{"type": "resize_preserv_aspect_area"}])
    assert "resize_preserve_aspect_area" in str(exc.value)


def test_missing_required_section_is_reported():
    broken = {k: v for k, v in MINIMAL.items() if k != "dest"}
    with pytest.raises(ConfigError) as exc:
        parse_config(broken)
    assert "dest" in str(exc.value)


def test_unknown_top_level_key_is_rejected_so_typos_do_not_pass_silently():
    with pytest.raises(ConfigError) as exc:
        _config(step=[])
    assert "step" in str(exc.value)


def test_converter_args_on_a_lerobot_source_are_rejected():
    with pytest.raises(ConfigError) as exc:
        _config(source={"type": "lerobot_v21", "path": "/in", "args": {"fps": 5}})
    assert "args" in str(exc.value)


def test_converter_source_accepts_args():
    cfg = _config(source={"type": "openx", "path": "/in", "args": {"fps": 5}})
    assert cfg.source.args == {"fps": 5}


def test_source_and_dest_must_differ():
    with pytest.raises(ConfigError):
        _config(
            source={"type": "lerobot_v21", "path": "/data/same"},
            dest={"type": "lerobot_v21", "path": "/data/same"},
        )


def test_load_config_reads_yaml_from_disk(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(
        "name: demo\n"
        "source:\n"
        "  type: lerobot_v21\n"
        "  path: /data/in\n"
        "steps:\n"
        "  - type: resize_preserve_aspect_area\n"
        "    max_area: 65536\n"
        "dest:\n"
        "  type: lerobot_v21\n"
        "  path: /data/out\n"
    )
    cfg = load_config(path)
    assert cfg.name == "demo"
    assert cfg.steps[0].max_area == 65536
