import pytest

from lerobot_pipeline.registry import (
    UnknownStepError,
    available_steps,
    build_step,
    compose_video_plans,
)


def _resize(**kwargs):
    return build_step({"type": "resize_preserve_aspect_area", **kwargs})


def test_resize_step_is_registered_under_its_config_name():
    assert "resize_preserve_aspect_area" in available_steps()


def test_unknown_step_type_reports_the_available_names():
    with pytest.raises(UnknownStepError) as exc:
        build_step({"type": "resize_preserv_aspect_area"})
    assert "resize_preserve_aspect_area" in str(exc.value)


def test_resize_step_is_a_video_step():
    assert _resize().kind == "video"


def test_plan_emits_only_a_scale_filter_when_crop_is_a_no_op():
    plan = _resize().plan((480, 640))
    assert plan.out_shape == (192, 256)
    assert plan.filters == ("scale=256:192",)


def test_plan_emits_scale_then_crop_when_the_resize_is_not_a_multiple():
    plan = _resize().plan((1280, 720))
    assert plan.out_shape == (320, 192)
    assert plan.filters == ("scale=192:341", "crop=192:320")


def test_plan_returns_none_when_the_source_is_already_at_the_target():
    assert _resize().plan((192, 256)) is None


def test_keys_restricts_which_video_keys_the_step_touches():
    step = _resize(keys=["observation.images.egocentric"])
    assert step.applies_to("observation.images.egocentric")
    assert not step.applies_to("observation.images.wrist")


def test_omitting_keys_applies_the_step_to_every_video_key():
    step = _resize()
    assert step.applies_to("observation.images.egocentric")
    assert step.applies_to("anything.at.all")


def test_custom_max_area_and_multiple_flow_through_to_the_plan():
    plan = _resize(max_area=128**2, multiple=16).plan((480, 640))
    assert plan.out_shape == (96, 128)


def test_compose_chains_filters_and_reports_the_final_shape():
    steps = [_resize(), _resize(max_area=128**2, multiple=16)]
    composed = compose_video_plans(steps, "observation.images.cam", (480, 640))
    assert composed.out_shape == (96, 128)
    assert composed.filters == (
        "scale=256:192",
        "scale=128:96",
    )


def test_compose_returns_none_when_no_step_changes_anything():
    assert compose_video_plans([_resize()], "cam", (192, 256)) is None


def test_compose_skips_steps_whose_keys_do_not_match():
    steps = [_resize(keys=["other"])]
    assert compose_video_plans(steps, "cam", (480, 640)) is None
