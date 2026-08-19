"""Aggregate the same temp datasets both ways and compare what comes out.

The batched writer is only allowed to change *when* files are written, never
what they contain. These build real LeRobot datasets, with real video, and
aggregate them with the batching on and off.
"""

import shutil
import subprocess

import numpy as np
import pytest

from generic_converter.pipeline import aggregate_tasks
from generic_converter.utils import ConversionTask

FPS = 10
SIZE = 64
CAM = "observation.images.cam"

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    CAM: {
        "dtype": "video",
        "shape": (SIZE, SIZE, 3),
        "names": ["height", "width", "channel"],
    },
}


def _temp_dataset(root, seed, episodes=1, frames=8):
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"local/src{seed}",
        root=root,
        fps=FPS,
        robot_type="test",
        features=FEATURES,
    )
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        for _ in range(frames):
            dataset.add_frame(
                {
                    "observation.state": np.zeros(6, dtype=np.float32),
                    "action": np.zeros(6, dtype=np.float32),
                    CAM: rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8),
                    "task": "do the thing",
                }
            )
        dataset.save_episode()
    return root


@pytest.fixture
def sources(tmp_path):
    tasks = []
    for index in range(5):
        root = tmp_path / "src" / f"task{index:02d}"
        _temp_dataset(root, index)
        tasks.append(
            ConversionTask(
                input_path=root, output_path=root, local_repo_id=f"local/src{index}"
            )
        )
    return tasks


def _relative_files(root):
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def _packet_md5(path):
    """The coded video, with the container stripped off."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-c", "copy", "-f", "md5", "-"],
        check=True, capture_output=True,
    )
    return out.stdout.decode().strip()


@requires_ffmpeg
def test_batching_produces_the_same_dataset(sources, tmp_path):
    one_at_a_time, batched = tmp_path / "upstream", tmp_path / "batched"
    aggregate_tasks(sources, one_at_a_time, aggr_repo_id="local/agg", batched=False)
    aggregate_tasks(sources, batched, aggr_repo_id="local/agg", batched=True)

    assert _relative_files(one_at_a_time) == _relative_files(batched) != []

    for relative in _relative_files(one_at_a_time):
        left, right = one_at_a_time / relative, batched / relative
        if left.suffix == ".mp4":
            # The container is laid out differently -- moov is built once from
            # the whole list rather than rebuilt per append -- but the coded
            # video has to be identical.
            assert _packet_md5(left) == _packet_md5(right), relative
        else:
            assert left.read_bytes() == right.read_bytes(), relative


def _concat_call_sizes(sources, out, monkeypatch, batched):
    """How many sources each concat call was handed."""
    from lerobot.datasets import aggregate as aggregate_module

    calls = []
    original = aggregate_module.concatenate_video_files

    def counted(paths, output, **kwargs):
        calls.append(len(paths))
        return original(paths, output, **kwargs)

    monkeypatch.setattr(aggregate_module, "concatenate_video_files", counted)
    aggregate_tasks(sources, out, aggr_repo_id="local/agg", batched=batched)
    monkeypatch.undo()
    return calls


@requires_ffmpeg
def test_batching_hands_every_source_to_one_concat_call(sources, tmp_path, monkeypatch):
    """The point of the change: one call per destination, not one per source."""
    calls = _concat_call_sizes(sources, tmp_path / "out", monkeypatch, batched=True)

    assert len(list((tmp_path / "out" / "videos").rglob("*.mp4"))) == 1
    assert calls == [5]


@requires_ffmpeg
def test_without_batching_each_source_rewrites_the_destination(
    sources, tmp_path, monkeypatch
):
    """What the comparison above is a reference against -- five sources, four
    appends, each one handed the destination it is growing plus one source."""
    calls = _concat_call_sizes(sources, tmp_path / "out", monkeypatch, batched=False)

    assert len(list((tmp_path / "out" / "videos").rglob("*.mp4"))) == 1
    assert calls == [2, 2, 2, 2]
