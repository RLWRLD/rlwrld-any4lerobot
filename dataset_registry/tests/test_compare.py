"""Comparing a rebuild against the delivered copy.

The point of these is the asymmetry: state and action are held to exact equality
because they are copied floats, while video bytes are not, because two ffmpeg builds
given the same flags do not emit the same file. A comparator that got that backwards
would either pass everything or fail everything.
"""

import json
import subprocess
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


class TestDeclaredCameras:
    """Which cameras a comparison is about.

    A delivered dataset can carry a camera its own meta/modality.json does not
    expose -- bridge_orig keeps two spare views, humanoid_everyday keeps the
    unresized original beside the resized one the training stack reads. Holding a
    rebuild to a camera nothing consumes fails it for a file no one opens.
    """

    def _tree(self, root: Path, cameras, declared=None):
        for camera in cameras:
            path = root / "videos" / "chunk-000" / camera
            path.mkdir(parents=True, exist_ok=True)
            (path / "episode_000000.mp4").write_bytes(b"")
        if declared is not None:
            (root / "meta").mkdir(parents=True, exist_ok=True)
            (root / "meta" / "modality.json").write_text(json.dumps({"video": {
                c: {"original_key": f"observation.images.{c}"} for c in declared}}))
        return root

    def test_the_cameras_a_dataset_exposes_are_read_from_modality(self, tmp_path):
        """Both spellings of the exposed camera, and neither spelling of the one
        that is not: the directory it lives in may be named either way."""
        root = self._tree(tmp_path / "d", ["wrist", "top"], declared=["top"])
        assert compare.declared_cameras(root) == {"top", "observation.images.top"}

    def test_a_dataset_without_modality_declares_nothing(self, tmp_path):
        root = self._tree(tmp_path / "d", ["wrist", "top"])
        assert compare.declared_cameras(root) is None

    def test_a_camera_directory_named_by_its_full_key_is_recognised(self, tmp_path):
        """Two naming conventions are in use. Most delivered datasets name the
        directory after the modality entry's original_key in full --
        observation.images.rgb_static -- while humanoid_everyday names it with the
        last segment alone. Matching only one of the two silently empties the other."""
        root = self._tree(
            tmp_path / "d",
            ["observation.images.top", "observation.images.wrist"],
            declared=["top"],
        )
        assert set(compare.episode_videos(root, 0, compare.declared_cameras(root))) == {
            "observation.images.top"
        }

    def test_an_undeclared_camera_is_left_out(self, tmp_path):
        root = self._tree(tmp_path / "d", ["wrist", "top"], declared=["top"])
        assert set(compare.episode_videos(root, 0, keep={"top"})) == {"top"}

    def test_no_filter_keeps_every_camera(self, tmp_path):
        root = self._tree(tmp_path / "d", ["wrist", "top"], declared=["top"])
        assert set(compare.episode_videos(root, 0)) == {"wrist", "top"}

    def test_a_comparison_is_about_the_cameras_the_delivered_copy_exposes(
        self, tmp_path, monkeypatch
    ):
        """The delivered copy is the target, so its modality file is the one that
        says what the comparison is about -- not the rebuild's, which would let a
        rebuild narrow its own examination."""
        rebuilt = self._tree(tmp_path / "r", ["wrist", "top"], declared=["top", "wrist"])
        delivered = self._tree(tmp_path / "d", ["wrist", "top"], declared=["top"])
        probed = []

        def fake_probe(path):
            probed.append(path.parent.name)
            return {"width": 1, "height": 1, "nb_read_frames": 1, "codec_name": "av1",
                    "profile": "Main", "pix_fmt": "yuv420p", "has_b_frames": 0,
                    "gop": 2, "bytes": 100}

        monkeypatch.setattr(compare, "probe", fake_probe)
        # the fixtures are empty files, so the pixel read has nothing to decode; this
        # test is about which cameras are reached, not about what is in them
        monkeypatch.setattr(compare, "pixel_agreement", lambda *a, **k: 1.0)
        summary, problems = compare.compare_video(rebuilt, delivered, 0)

        assert set(probed) == {"top"}
        assert set(summary) == {"top"}
        assert not problems


class TestPixels:
    """Whether the pictures agree, not just their file sizes.

    A size ratio cannot see a difference that does not change how well the frames
    compress. Exchanging the red and blue channels is exactly that: measured on
    utaustin_mutex, a rebuilt episode whose channels were reversed came within 1% of
    the delivered size and passed, while its frames correlated 0.74 against the ones
    it was meant to reproduce.
    """

    def _clip(self, path: Path, colour, frames=6, size=(64, 64)):
        """A tiny mp4 of one flat colour, so the comparison is about the pixels."""
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             f"color=c={colour}:s={size[0]}x{size[1]}:r=10:d={frames / 10}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)],
            check=True, capture_output=True)
        return path

    def test_the_same_picture_agrees(self, tmp_path):
        a = self._clip(tmp_path / "a.mp4", "red")
        b = self._clip(tmp_path / "b.mp4", "red")
        assert compare.pixel_agreement(a, b) > 0.99

    def test_reversed_channels_do_not_agree(self, tmp_path):
        """Red against blue is the shape the real defect took."""
        a = self._clip(tmp_path / "a.mp4", "0xFF4020")
        b = self._clip(tmp_path / "b.mp4", "0x2040FF")
        assert compare.pixel_agreement(a, b) < 0.99

    def test_frames_are_sampled_rather_than_all_decoded(self, tmp_path):
        """A long episode costs the same as a short one: only the sample is decoded."""
        a = self._clip(tmp_path / "a.mp4", "green", frames=200)
        assert len(compare.sample_frames(a, 4)) == 4

    def test_a_reversed_episode_is_reported_even_though_its_size_matches(self, tmp_path):
        """The whole point: two clips of the same flat colour with red and blue
        exchanged encode to nearly the same number of bytes, so only the pixels can
        tell them apart."""
        for root, colour in ((tmp_path / "r", "0xFF4020"), (tmp_path / "d", "0x2040FF")):
            self._clip(root / "videos" / "chunk-000" / "cam" / "episode_000000.mp4", colour)

        summary, problems = compare.compare_video(tmp_path / "r", tmp_path / "d", 0)

        assert any("frames agree" in p for p in problems), problems
        assert "PIXELS" in summary["cam"]
