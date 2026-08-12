import pytest

from lerobot_pipeline.steps.resize import (
    AreaBoundError,
    resize_preserve_aspect_area_then_crop,
)

MAX_AREA = 256**2
MULTIPLE = 32


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1280, 720), ((341, 192), (320, 192))),
        ((720, 1280), ((192, 341), (192, 320))),
        ((480, 640), ((192, 256), (192, 256))),
        ((640, 480), ((256, 192), (256, 192))),
    ],
)
def test_matches_reference_implementation_verbatim(shape, expected):
    """Locks parity with the reference logic the model team trained with.

    Note: the reference snippet's own `# example` comment claims
    `(1280, 720) -> ((192, 256), (192, 256))`, which is wrong -- that is the output
    for `(480, 640)`. These values come from executing the reference code itself.
    """
    assert resize_preserve_aspect_area_then_crop(*shape) == expected


def test_landscape_480x640_maps_to_192x256():
    """The shape actually used for humanoid_everyday, per the reference ffmpeg script."""
    assert resize_preserve_aspect_area_then_crop(480, 640) == ((192, 256), (192, 256))


def test_never_upscales_a_small_image():
    _, crop = resize_preserve_aspect_area_then_crop(64, 64)
    assert crop == (64, 64)


def test_crop_dimensions_are_multiples_of_the_patch_size():
    for h, w in [(1080, 1920), (720, 1280), (480, 640), (240, 320), (600, 800)]:
        _, (h_c, w_c) = resize_preserve_aspect_area_then_crop(h, w)
        assert h_c % MULTIPLE == 0, (h, w, h_c)
        assert w_c % MULTIPLE == 0, (h, w, w_c)


def test_crop_area_stays_within_the_bound():
    for h, w in [(1080, 1920), (720, 1280), (480, 640), (1920, 1080), (600, 800)]:
        _, (h_c, w_c) = resize_preserve_aspect_area_then_crop(h, w)
        assert h_c * w_c <= MAX_AREA, (h, w, h_c * w_c)


def test_aspect_ratio_is_approximately_preserved():
    for h, w in [(1080, 1920), (720, 1280), (480, 640), (600, 800)]:
        _, (h_c, w_c) = resize_preserve_aspect_area_then_crop(h, w)
        assert abs((w_c / h_c) - (w / h)) < 0.2, (h, w, h_c, w_c)


def test_portrait_and_landscape_are_symmetric():
    (_, (h_c, w_c)) = resize_preserve_aspect_area_then_crop(1920, 1080)
    (_, (h_c2, w_c2)) = resize_preserve_aspect_area_then_crop(1080, 1920)
    assert (h_c, w_c) == (w_c2, h_c2)


def test_extreme_aspect_ratio_raises_instead_of_silently_exceeding_area():
    """The reference algorithm's `max(m, ...)` guard breaks the area bound when the
    short side is already <= the patch size. Fail loudly rather than emit bad data."""
    with pytest.raises(AreaBoundError):
        resize_preserve_aspect_area_then_crop(32, 4096)


def test_custom_max_area_and_multiple_are_honoured():
    _, (h_c, w_c) = resize_preserve_aspect_area_then_crop(
        480, 640, max_area=128**2, multiple=16
    )
    assert h_c % 16 == 0 and w_c % 16 == 0
    assert h_c * w_c <= 128**2
