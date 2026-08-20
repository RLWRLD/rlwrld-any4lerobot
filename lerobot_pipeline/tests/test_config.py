import json
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


# --- video work the converter does itself ------------------------------------


def _resolved(dataset: str):
    from lerobot_pipeline.env import Environment, build_config

    env = Environment(name="t", profile="rldx1", raw_root=Path("/raw"), out_root=Path("/out"))
    return build_config(env, dataset)


def test_openx_is_handed_the_resize_rule_instead_of_a_transform_stage():
    """openx2lerobot encodes the video itself, so a transform stage afterwards would
    re-encode what it just wrote. The rule goes to the converter instead."""
    config = _resolved("ucsd_kitchen_dataset_converted_externally_to_rlds")

    assert json.loads(config.source.args["resize"])["type"] == "resize_preserve_aspect_area"
    assert [getattr(step, "kind", None) for step in config.steps] == []


def test_the_resize_reaches_the_converter_with_the_profile_s_parameters():
    """A bare step name would take the step's own defaults; the profile's numbers are
    the ones the collection was built with."""
    config = _resolved("ucsd_kitchen_dataset_converted_externally_to_rlds")

    assert json.loads(config.source.args["resize"]) == {
        "type": "resize_preserve_aspect_area",
        "max_area": 65536,
        "multiple": 32,
        "filter": "bicubic",
    }


def test_openx_encodes_with_what_the_delivered_oxe_copies_carry():
    assert _resolved("cmu_stretch").source.args["encoding"] == "lerobot_av1_default"


def test_a_converter_that_passes_video_through_still_gets_a_transform_stage():
    """action_net is hdf5 with mp4 already in it -- there the resize is a transcode."""
    config = _resolved("action_net")

    assert "resize" not in config.source.args
    assert "video" in [getattr(step, "kind", None) for step in config.steps]


# --- asking for an encoding other than the delivered one ----------------------


def _asked_for(dataset: str, encoding):
    """``dataset`` built under an explicit encoding instead of the delivered one."""
    return parse_config(
        {
            "name": dataset,
            "dataset": dataset,
            "profile": "rldx1",
            "source": {"path": "/raw"},
            "dest": {"path": "/out"},
            "runtime": {"encoding": encoding},
        }
    )


def test_an_asked_for_encoding_reaches_the_converter():
    """The delivered encoding records what a source *is*; asking for a different one
    is an instruction, and an instruction that only reaches half the collection is
    worse than none. bridge_orig is delivered as H.264, which the converter cannot
    write -- asked for AV1, it can, so the converter does the work."""
    config = _asked_for("bridge_orig", "lerobot_av1_default")

    assert config.source.args["encoding"] == "lerobot_av1_default"
    assert [getattr(step, "kind", None) for step in config.steps] == []


def test_an_asked_for_encoding_the_writer_cannot_produce_moves_work_to_the_stage():
    """The other direction: cmu_stretch is delivered as AV1 and normally written by
    the converter, but H.264 with B-frames is an ffmpeg command line, so asking for
    it has to hand the video back to the transform stage."""
    config = _asked_for("cmu_stretch", "rldx1_reference")

    assert "encoding" not in config.source.args
    assert "video" in [getattr(step, "kind", None) for step in config.steps]


def test_the_asked_for_encoding_is_the_one_reported_as_applied():
    from lerobot_pipeline.plan import encoding_used

    # the codec, not the gop: random_access and the delivered AV1 share a gop of 2,
    # so that field would pass whichever of the two won
    assert encoding_used(_asked_for("cmu_stretch", "random_access"))["codec"] == "libx264"


def test_without_an_instruction_each_dataset_keeps_its_delivered_encoding():
    """No override anywhere: the collection reproduces what was delivered, which is
    two different encodings split by builder rather than one."""
    from lerobot_pipeline.plan import encoding_used

    assert encoding_used(_resolved("cmu_stretch"))["codec"] == "libsvtav1"
    assert encoding_used(_resolved("bridge_orig"))["bframes"] == 3


def test_one_setting_can_be_asked_for_on_its_own():
    """Changing the keyframe interval should not mean restating the codec, the crf
    and the pixel format alongside it. An inline mapping is a partial instruction:
    what it names is overridden and the rest stays as delivered."""
    from lerobot_pipeline.plan import encoding_used

    applied = encoding_used(_asked_for("cmu_stretch", {"gop": 50}))

    assert applied["gop"] == 50
    assert applied["codec"] == "libsvtav1"


def test_a_partial_instruction_still_reaches_the_converter():
    """A gop the writer can be asked for keeps the video work where it was, rather
    than adding a transcode because the instruction arrived in a different shape."""
    config = _asked_for("cmu_stretch", {"gop": 50})

    reached = json.loads(config.source.args["encoding"])
    assert (reached["gop"], reached["codec"]) == (50, "libsvtav1")
    assert [getattr(step, "kind", None) for step in config.steps] == []


class TestResizeFilter:
    """The resampler is a declared value, and one value for both resize paths.

    A dataset that downscales is resized either before LeRobot's writer
    (openx2lerobot) or after it through ffmpeg's `scale` filter (the transform
    stage), decided by whether the converter can produce the delivered encoding.
    Both end in libswscale, so both must be handed the same filter -- and before
    this it was explicit on one path and ffmpeg's default on the other, agreeing
    only because the default happened to be bicubic.
    """

    def test_the_scale_filter_names_the_resampler(self):
        from lerobot_pipeline.steps.resize import ResizePreserveAspectArea

        step = ResizePreserveAspectArea(max_area=65536, multiple=32, filter="lanczos")
        plan = step.plan((360, 640))
        assert any("flags=lanczos" in f for f in plan.filters), plan.filters

    def test_the_default_is_what_shipped(self):
        """bicubic, so declaring the key changes no output. ffmpeg's `scale` defaults
        to it and video_rules asked PyAV for it by name."""
        from lerobot_pipeline.steps.resize import ResizePreserveAspectArea

        plan = ResizePreserveAspectArea(max_area=65536, multiple=32).plan((360, 640))
        assert any("flags=bicubic" in f for f in plan.filters), plan.filters

    def test_an_unknown_filter_is_refused_at_config_time(self):
        """Not hours into a run, and not by silently falling back."""
        from lerobot_pipeline.steps.resize import (
            ResizePreserveAspectArea,
            UnknownFilterError,
        )

        with pytest.raises(UnknownFilterError, match="lanzcos"):
            ResizePreserveAspectArea(filter="lanzcos")  # transposed, as typed

    def test_a_resize_that_composes_to_nothing_emits_no_filter(self):
        """cmu_stretch is 128x128 at the source and already at the target size. That
        is why the filter went unnoticed: it never ran."""
        from lerobot_pipeline.steps.resize import ResizePreserveAspectArea

        assert ResizePreserveAspectArea(filter="lanczos").plan((128, 128)) is None

    def test_the_profile_declares_it(self):
        """Reading it off the shipped profile, not a fixture -- the point of the
        change is that the collection's own convention names the resampler."""
        import yaml

        root = Path(__file__).resolve().parents[2]
        profile = yaml.safe_load(
            (root / "lerobot_pipeline/configs/profiles/rldx1.yaml").read_text())
        assert profile["video"]["resize"]["filter"] == "bicubic"

    def test_both_paths_read_one_declaration(self):
        """openx2lerobot's resize_filter and the transform stage's scale filter must
        come from the same mapping, so a profile edit moves both or neither."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "openx2lerobot"))
        from video_rules import resize_filter

        from lerobot_pipeline.steps.resize import ResizePreserveAspectArea

        resize = {"type": "resize_preserve_aspect_area", "max_area": 65536,
                  "multiple": 32, "filter": "lanczos"}
        assert resize_filter(resize) == "lanczos"
        step = ResizePreserveAspectArea(**{k: v for k, v in resize.items()
                                           if k != "type"})
        assert any("flags=lanczos" in f for f in step.plan((360, 640)).filters)
