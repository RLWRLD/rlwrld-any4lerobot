"""Comparing a rebuild against the delivered copy.

The point of these is the asymmetry: state and action are held to exact equality
because they are copied floats, while video bytes are not, because two ffmpeg builds
given the same flags do not emit the same file. A comparator that got that backwards
would either pass everything or fail everything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_registry import load  # noqa: E402
from dataset_registry.compare import (  # noqa: E402
    SIZE_TOLERANCE,
    compare_episode,
    episode_prompts,
    report,
    run,
)

pd = pytest.importorskip("pandas")

ROWS = 20
WIDTH = 44


def write_dataset(root: Path, rows=ROWS, episodes=2, nudge=None, prompts=None):
    """A minimal LeRobot tree with just enough for the comparator."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    lines = []
    for index in range(episodes):
        rng = np.random.default_rng(index)
        state = rng.random((rows, WIDTH)).astype(np.float32)
        action = rng.random((rows, WIDTH)).astype(np.float32)
        if nudge is not None and index == 0:
            state[0, nudge] = np.float32(state[0, nudge] + 1.0)
        pd.DataFrame(
            {"observation.state": list(state), "action": list(action)}
        ).to_parquet(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
        task = (prompts or {}).get(index, f"task {index}")
        lines.append(json.dumps(
            {"episode_index": index, "tasks": [task], "length": rows}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def spec():
    return load("action_net")


class TestVectors:
    def test_identical_datasets_reproduce(self, spec, tmp_path):
        a = write_dataset(tmp_path / "a")
        b = write_dataset(tmp_path / "b")
        reports = run(spec, a, b, episodes=2, check_video=False)
        assert all(r.ok for r in reports)
        assert "identical" in reports[0].columns["observation.state"]

    def test_a_single_changed_float_is_caught(self, spec, tmp_path):
        """No tolerance on the vectors: one slot in one row is a failure."""
        a = write_dataset(tmp_path / "a", nudge=17)
        b = write_dataset(tmp_path / "b")
        reports = run(spec, a, b, episodes=2, check_video=False)
        assert not reports[0].ok and reports[1].ok
        assert "17" in reports[0].columns["observation.state"]

    def test_a_different_row_count_is_reported_as_clock_alignment(self, spec, tmp_path):
        a = write_dataset(tmp_path / "a", rows=ROWS - 1)
        b = write_dataset(tmp_path / "b")
        problems = run(spec, a, b, episodes=1, check_video=False)[0].problems
        assert any("clock alignment" in p for p in problems)


class TestAlignment:
    def test_a_prompt_mismatch_stops_the_comparison(self, spec, tmp_path):
        """Episodes carry no source id, so position is the only alignment available.
        A dropped episode shifts everything after it; comparing shifted pairs would
        be a wall of meaningless differences."""
        a = write_dataset(tmp_path / "a")
        b = write_dataset(tmp_path / "b", prompts={0: "a different task"})
        report0 = run(spec, a, b, episodes=1, check_video=False)[0]
        assert not report0.ok
        assert any("alignment" in p for p in report0.problems)
        # and it does not go on to report vector differences it cannot trust
        assert not report0.columns

    def test_prompts_are_read_from_episodes_jsonl(self, tmp_path):
        root = write_dataset(tmp_path / "a", prompts={1: "pick up the cup"})
        assert episode_prompts(root)[1] == "pick up the cup"


class TestMissing:
    def test_an_episode_missing_from_the_rebuild_is_reported(self, spec, tmp_path):
        a = write_dataset(tmp_path / "a", episodes=1)
        b = write_dataset(tmp_path / "b", episodes=2)
        reports = run(spec, a, b, episodes=2, check_video=False)
        assert reports[0].ok and not reports[1].ok
        assert "missing from the rebuilt" in reports[1].problems[0]


def test_size_tolerance_only_applies_to_video():
    """Documented as a constant so it is reviewable: video bytes are allowed to
    drift because the encoder build decides them, vectors are not."""
    assert 0 < SIZE_TOLERANCE < 0.5


def test_report_counts_the_failures(spec, tmp_path):
    a = write_dataset(tmp_path / "a", nudge=3)
    b = write_dataset(tmp_path / "b")
    text = report(run(spec, a, b, episodes=2, check_video=False))
    assert "1/2 episodes reproduce" in text and "1 differ" in text
