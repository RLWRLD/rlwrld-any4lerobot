"""What tfds's numpy reader actually hands back for an RLDS episode.

The array_record reader in ``adapter`` rests on one claim about a library: that
``as_data_source`` gives an episode whose ``steps`` is a random-access sequence of
per-step dicts, where ``as_dataset`` gives a nested ``tf.data.Dataset``. Everything
``stack_steps`` does follows from that, and getting it wrong is the kind of mistake
that surfaces after a 56 GB fetch rather than in review.

So this asks tfds directly, through the same decoder a run uses. It needs no
array_record file and no builder: the decode path is per-record, and a record
deserialized by hand goes through exactly the code ``ArrayRecordDataSource`` calls.

It also needs no TensorFlow, which is why the images are handed to tfds already
encoded -- turning an ndarray into PNG bytes is the one step of the *write* path that
imports TensorFlow, and writing is not what a mirror makes us do.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

tfds_features = pytest.importorskip(
    "tensorflow_datasets.core.features",
    reason="the openx extra is not installed here",
)
PILImage = pytest.importorskip("PIL.Image")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import stack_steps  # noqa: E402

TASK = b"pick up the bowl"
STEPS = 5
HEIGHT, WIDTH = 4, 6


def _features():
    """bc_z's shape, in miniature.

    The details that matter are the ones a flat reader would get wrong: ``action`` is
    itself a dict, several observation names carry slashes, and the instruction the
    task string comes from is per-step text rather than episode metadata.
    """
    f = tfds_features
    return f.FeaturesDict(
        {
            "steps": f.Dataset(
                {
                    "action": f.FeaturesDict(
                        {
                            "future/xyz_residual": f.Tensor(
                                shape=(3,), dtype=np.float32
                            ),
                            "future/target_close": f.Tensor(
                                shape=(1,), dtype=np.int64
                            ),
                        }
                    ),
                    "observation": f.FeaturesDict(
                        {
                            "image": f.Image(shape=(HEIGHT, WIDTH, 3)),
                            "present/xyz": f.Tensor(shape=(3,), dtype=np.float32),
                            "natural_language_instruction": f.Text(),
                        }
                    ),
                    "is_last": f.Scalar(dtype=np.bool_),
                }
            ),
            "episode_id": f.Text(),
        }
    )


def _as_png(array: np.ndarray) -> io.BytesIO:
    buffer = io.BytesIO()
    PILImage.fromarray(array).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture(scope="module")
def images():
    return np.random.default_rng(0).integers(
        0, 255, size=(STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8
    )


@pytest.fixture(scope="module")
def trajectory(episode):
    return stack_steps(episode["steps"])


@pytest.fixture(scope="module")
def episode(images):
    """One episode, serialized and read back the way a worker reads it."""
    features = _features()
    written = {
        "episode_id": "ep0",
        "steps": [
            {
                "action": {
                    "future/xyz_residual": np.full(3, index, np.float32),
                    "future/target_close": np.array([index % 2], np.int64),
                },
                "observation": {
                    "image": _as_png(images[index]),
                    "present/xyz": np.full(3, index * 2, np.float32),
                    "natural_language_instruction": TASK.decode(),
                },
                "is_last": index == STEPS - 1,
            }
            for index in range(STEPS)
        ],
    }
    return features.deserialize_example_np(features.serialize_example(written))


class TestWhatTheReaderGets:
    def test_the_steps_are_a_sequence_rather_than_a_stream(self, episode):
        """The whole reason this path exists. A tf.data.Dataset would have neither."""
        assert len(episode["steps"]) == STEPS
        assert isinstance(episode["steps"][0], dict)

    def test_a_step_decodes_only_when_it_is_indexed(self, episode):
        """Which is what keeps an episode's memory cost the same as the stream reader's
        -- nothing is decoded by asking for the episode."""
        assert not isinstance(episode["steps"], (list, tuple))
        assert episode["steps"][0]["observation"]["image"].shape == (HEIGHT, WIDTH, 3)

    def test_an_image_arrives_decoded_as_an_array(self, episode):
        image = episode["steps"][0]["observation"]["image"]
        assert (image.dtype, image.ndim) == (np.uint8, 3)


class TestStackStepsAgainstTheRealDecoder:
    def test_a_nested_action_stays_nested(self, trajectory):
        """bc_z's transform reads action["future/xyz_residual"][:, :3]; a reader that
        flattened the names would raise, and one that dropped them would not."""
        assert trajectory["action"]["future/xyz_residual"].shape == (STEPS, 3)
        assert trajectory["action"]["future/target_close"].shape == (STEPS, 1)

    def test_the_steps_stay_in_order(self, trajectory):
        """Out-of-order frames are the failure that looks like a working conversion:
        the video plays, the shapes match, the trajectory is nonsense."""
        assert trajectory["action"]["future/xyz_residual"][:, 0].tolist() == [
            0.0, 1.0, 2.0, 3.0, 4.0
        ]
        assert trajectory["is_last"].tolist() == [False, False, False, False, True]

    def test_the_pixels_come_back_untouched(self, trajectory, images):
        """tfds decodes with OpenCV here, which is BGR, and reorders before handing the
        frame over. So this reader is not somewhere a channel order can quietly differ
        from the stream reader's -- video_rules.flips_channels is the only place that
        decides colour order, and it sees the same bytes either way."""
        np.testing.assert_array_equal(trajectory["observation"]["image"], images)

    def test_an_observation_name_with_a_slash_survives(self, trajectory):
        assert trajectory["observation"]["present/xyz"].shape == (STEPS, 3)

    def test_the_instruction_is_still_bytes_to_decode(self, trajectory):
        """save_episode reads the task as traj["task"][0].decode()."""
        stacked = trajectory["observation"]["natural_language_instruction"]
        assert stacked[0].decode() == TASK.decode()
