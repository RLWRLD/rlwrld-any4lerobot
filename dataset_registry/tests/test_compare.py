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
from dataset_registry import compare  # noqa: E402
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
        # draw a fixed size and slice, so that a shorter episode really is a prefix
        # of the longer one rather than a different draw
        rng = np.random.default_rng(index)
        state = rng.random((ROWS, WIDTH)).astype(np.float32)[:rows]
        action = rng.random((ROWS, WIDTH)).astype(np.float32)[:rows]
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

    def test_a_trimmed_tail_is_tolerated_and_the_prefix_still_checked(self, spec, tmp_path):
        """One row short with every shared row identical is a boundary effect, not a
        different frame selection -- exactly the case the delivered copy is expected
        to land in, since the script that produced it no longer exists."""
        a = write_dataset(tmp_path / "a", rows=ROWS - 1)
        b = write_dataset(tmp_path / "b")
        report0 = run(spec, a, b, episodes=1, check_video=False)[0]
        assert report0.ok
        assert report0.compared_rows == ROWS - 1
        assert report0.first_divergence is None

    def test_a_large_row_difference_is_a_wrong_strategy(self, spec, tmp_path):
        a = write_dataset(tmp_path / "a", rows=ROWS - 9)
        b = write_dataset(tmp_path / "b")
        problems = run(spec, a, b, episodes=1, check_video=False)[0].problems
        assert any("clock strategy" in p for p in problems)

    def test_divergence_row_separates_a_trim_from_a_reshuffle(self, spec, tmp_path):
        """Where the values first disagree is the diagnosis: a trimmed tail leaves
        the shared rows perfect, a wrong strategy diverges partway through."""
        a = write_dataset(tmp_path / "a", nudge=17)
        b = write_dataset(tmp_path / "b")
        assert run(spec, a, b, episodes=1, check_video=False)[0].first_divergence == 0


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


class TestKeyframeInterval:
    """Reading `-g` back off a file, which the stream header does not carry.

    This is what makes deferring the encoding work safe: the first run does not try
    to match GOP, and if the delivered copy turns out to differ, `compare` says so
    rather than leaving it to be noticed in a size ratio.
    """

    def test_the_first_frame_counts_as_a_keyframe(self, tmp_path, monkeypatch):
        """ffprobe appends a stray comma to the first row, so a plain equality test
        misses frame 0 -- and frame 0 is always a keyframe, which turns every
        interval into "only one keyframe found"."""
        from dataset_registry import compare

        monkeypatch.setattr(compare, "_ffprobe", lambda command, path: "1,\n0\n0\n1\n0\n")
        assert compare.keyframe_interval(tmp_path / "x.mp4") == 3

    def test_one_keyframe_means_unknown_rather_than_zero(self, tmp_path, monkeypatch):
        """An episode shorter than the interval has a single keyframe; that is not
        the same as a one-frame GOP."""
        from dataset_registry import compare

        monkeypatch.setattr(compare, "_ffprobe", lambda command, path: "1,\n0\n0\n0\n")
        assert compare.keyframe_interval(tmp_path / "x.mp4") is None

    def test_a_two_frame_gop_is_read_as_two(self, tmp_path, monkeypatch):
        """LeRobot's own writer default, which the AV1 datasets still carry."""
        from dataset_registry import compare

        monkeypatch.setattr(compare, "_ffprobe", lambda command, path: "1,\n0\n1\n0\n1\n")
        assert compare.keyframe_interval(tmp_path / "x.mp4") == 2

    def test_gop_is_among_the_compared_fields(self):
        """Guards the deferral: if this drops out, a 250-vs-2 difference would only
        show up indirectly, as a file-size ratio."""
        source = Path(compare_episode.__globals__["__file__"]).read_text()
        fields = source.split('for field_name in (')[1].split('):')[0]
        assert '"gop"' in fields


# --- pairing episodes by what is in them -------------------------------------


class TestPairDigests:
    def test_a_permutation_is_paired_back_up(self):
        """openx2lerobot writes in tfds read order, the delivered copy is in another
        -- every episode is present, none of them at the same index."""
        rebuilt = {"a": [0], "b": [1], "c": [2]}
        delivered = {"a": [2], "b": [0], "c": [1]}

        assert compare.pair_digests(rebuilt, delivered) == {0: 2, 1: 0, 2: 1}

    def test_repeated_episodes_are_paired_in_index_order(self):
        """Two of ucsd_kitchen's 150 episodes carry identical vectors; there is no
        way to tell them apart, so pair them in a fixed order rather than at random."""
        assert compare.pair_digests({"a": [3, 1]}, {"a": [9, 4]}) == {1: 4, 3: 9}

    def test_an_episode_the_delivered_copy_does_not_have_is_left_unpaired(self):
        assert compare.pair_digests({"a": [0], "b": [1]}, {"a": [0]}) == {0: 0}

    def test_a_different_number_of_copies_is_not_guessed_at(self):
        """One rebuilt episode against two delivered ones has no honest pairing."""
        assert compare.pair_digests({"a": [0]}, {"a": [0, 1]}) == {}


class TestUnpairedReport:
    def test_it_fails_the_run(self):
        assert not compare.unpaired_report(3).ok

    def test_it_is_not_counted_as_an_episode(self):
        text = compare.report([compare.unpaired_report(3)])

        assert "0/0 episodes" in text
        assert "3 rebuilt episode(s)" in text
