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
from verification import compare  # noqa: E402
from verification.compare import (  # noqa: E402
    SIZE_TOLERANCE,
    compare_episode,
    episode_prompts,
    report,
    run,
)

pd = pytest.importorskip("pandas")

ROWS = 20
WIDTH = 44


def write_dataset(root: Path, rows=ROWS, episodes=2, nudge=None, prompts=None,
                  nudge_row=0, only=None, order=None):
    """A minimal LeRobot tree with just enough for the comparator."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    lines = []
    for index in range(episodes):
        if only is not None and index not in only:
            continue
        # draw a fixed size and slice, so that a shorter episode really is a prefix
        # of the longer one rather than a different draw
        # `order` decouples an episode's contents from the index it is written at, so
        # a fixture can hold the same episodes in a different order
        seed = index if order is None else order[index]
        rng = np.random.default_rng(seed)
        state = rng.random((ROWS, WIDTH)).astype(np.float32)[:rows]
        action = rng.random((ROWS, WIDTH)).astype(np.float32)[:rows]
        if nudge is not None and index == 0:
            state[nudge_row, nudge] = np.float32(state[nudge_row, nudge] + 1.0)
        pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            # episode_fingerprints groups on these, as the real files carry them
            "episode_index": [index] * len(state),
            "frame_index": list(range(len(state))),
        }).to_parquet(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
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
        from verification import compare

        monkeypatch.setattr(compare, "_ffprobe", lambda command, path: "1,\n0\n0\n1\n0\n")
        assert compare.keyframe_interval(tmp_path / "x.mp4") == 3

    def test_one_keyframe_means_unknown_rather_than_zero(self, tmp_path, monkeypatch):
        """An episode shorter than the interval has a single keyframe; that is not
        the same as a one-frame GOP."""
        from verification import compare

        monkeypatch.setattr(compare, "_ffprobe", lambda command, path: "1,\n0\n0\n0\n")
        assert compare.keyframe_interval(tmp_path / "x.mp4") is None

    def test_a_two_frame_gop_is_read_as_two(self, tmp_path, monkeypatch):
        """LeRobot's own writer default, which the AV1 datasets still carry."""
        from verification import compare

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
        monkeypatch.setattr(compare, "pixel_verdict", lambda *a, **k: (1.0, None))
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


class TestRenamedCamera:
    """A camera the rebuild wrote under a different name.

    Eight of the 27 delivered OXE datasets renamed their cameras to the modality
    aliases -- bc_z's feature is `observation.images.primary` where the RLDS key is
    `image` -- and nineteen kept the source keys. A rebuild derives the name from the
    source, so for those eight the directory will not match. Because undeclared rebuilt
    cameras are filtered out on purpose, that arrives looking like a camera the rebuild
    never wrote, which sends the reader after the wrong thing.
    """

    def _clip(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "color=c=red:s=64x64:r=10:d=0.6",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)],
            check=True, capture_output=True)

    def test_the_name_the_rebuild_used_is_reported(self, tmp_path):
        rebuilt, delivered = tmp_path / "r", tmp_path / "d"
        self._clip(rebuilt / "videos/chunk-000/observation.images.image/episode_000000.mp4")
        self._clip(delivered / "videos/chunk-000/observation.images.primary/episode_000000.mp4")
        (delivered / "meta").mkdir(parents=True, exist_ok=True)
        (delivered / "meta" / "modality.json").write_text(json.dumps(
            {"video": {"primary": {"original_key": "observation.images.primary"}}}))

        _, problems = compare.compare_video(rebuilt, delivered, 0)

        assert len(problems) == 1
        assert "observation.images.primary only in the delivered" in problems[0]
        assert "the rebuild has observation.images.image instead" in problems[0]

    def test_a_camera_that_really_is_absent_says_only_that(self, tmp_path):
        rebuilt, delivered = tmp_path / "r", tmp_path / "d"
        (rebuilt / "videos").mkdir(parents=True)
        self._clip(delivered / "videos/chunk-000/observation.images.primary/episode_000000.mp4")
        (delivered / "meta").mkdir(parents=True, exist_ok=True)
        (delivered / "meta" / "modality.json").write_text(json.dumps(
            {"video": {"primary": {"original_key": "observation.images.primary"}}}))

        _, problems = compare.compare_video(rebuilt, delivered, 0)

        assert "instead" not in problems[0]


class TestFingerprints:
    """The two questions an episode's contents can answer.

    `whole` is "is this the same episode"; `prefix` is the weaker "is this the same
    episode with its tail somewhere else". Keeping them apart is what lets a report
    distinguish an episode that is missing from one that is a row short.
    """

    def test_the_same_episode_prints_the_same(self, tmp_path):
        a = compare.episode_fingerprints(write_dataset(tmp_path / "a"))
        b = compare.episode_fingerprints(write_dataset(tmp_path / "b"))
        assert a[0] == b[0]

    def test_a_trimmed_tail_keeps_the_prefix_and_changes_the_whole(self, tmp_path):
        a = compare.episode_fingerprints(write_dataset(tmp_path / "a", rows=ROWS - 1))
        b = compare.episode_fingerprints(write_dataset(tmp_path / "b"))
        assert a[0].prefix == b[0].prefix
        assert a[0].whole != b[0].whole
        assert (a[0].rows, b[0].rows) == (ROWS - 1, ROWS)

    def test_a_change_inside_the_prefix_changes_both(self, tmp_path):
        """Which is the point: a rebuild that got the values wrong must not be
        mistaken for one that trimmed differently."""
        a = compare.episode_fingerprints(write_dataset(tmp_path / "a", nudge=17))
        b = compare.episode_fingerprints(write_dataset(tmp_path / "b"))
        assert a[0].prefix != b[0].prefix

    def test_a_change_past_the_prefix_leaves_the_prefix_alone(self, tmp_path):
        """So it is paired and then compared, rather than counted as absent."""
        a = compare.episode_fingerprints(
            write_dataset(tmp_path / "a", nudge=17, nudge_row=ROWS - 1))
        b = compare.episode_fingerprints(write_dataset(tmp_path / "b"))
        assert a[0].prefix == b[0].prefix and a[0].whole != b[0].whole


class TestPairing:
    """Step 1 of the funnel: how much of the delivered copy is there at all."""

    def _pair(self, a, b):
        return compare.pair_episodes(
            compare.episode_fingerprints(a), compare.episode_fingerprints(b))

    def test_identical_datasets_pair_exactly(self, tmp_path):
        p = self._pair(write_dataset(tmp_path / "a"), write_dataset(tmp_path / "b"))
        assert (len(p.exact), len(p.prefix_only)) == (2, 0)
        assert not p.rebuilt_only and not p.delivered_only
        assert p.moved == 0

    def test_a_trimmed_episode_is_paired_and_labelled_as_such(self, tmp_path):
        """Not absent. The episode is there; its row count is not the same."""
        p = self._pair(write_dataset(tmp_path / "a", rows=ROWS - 1),
                       write_dataset(tmp_path / "b"))
        assert (len(p.exact), len(p.prefix_only)) == (0, 2)
        assert not p.delivered_only

    def test_a_missing_episode_is_counted_against_the_delivered_copy(self, tmp_path):
        p = self._pair(write_dataset(tmp_path / "a", episodes=1),
                       write_dataset(tmp_path / "b", episodes=2))
        assert len(p.exact) == 1
        assert p.delivered_only == [1] and not p.rebuilt_only
        assert (p.rebuilt_total, p.delivered_total) == (1, 2)

    def test_an_episode_the_delivered_copy_lacks_is_counted_the_other_way(self, tmp_path):
        p = self._pair(write_dataset(tmp_path / "a", episodes=2),
                       write_dataset(tmp_path / "b", episodes=1))
        assert p.rebuilt_only == [1] and not p.delivered_only

    def test_rows_are_totalled_on_both_sides(self, tmp_path):
        p = self._pair(write_dataset(tmp_path / "a", rows=ROWS - 1),
                       write_dataset(tmp_path / "b"))
        assert (p.rebuilt_rows, p.delivered_rows) == (2 * (ROWS - 1), 2 * ROWS)

    def test_a_permutation_pairs_fully_and_is_counted_as_moved(self, tmp_path):
        """openx2lerobot writes in tfds read order, so nearly every episode moves.
        Worth knowing, and not a defect, so it is counted rather than failed."""
        p = self._pair(write_dataset(tmp_path / "a", order=[1, 0]),
                       write_dataset(tmp_path / "b", order=[0, 1]))
        assert p.exact == {0: 1, 1: 0}
        assert p.moved == 2
        assert not p.rebuilt_only and not p.delivered_only

    def test_episodes_in_the_same_place_are_not_counted_as_moved(self, tmp_path):
        assert self._pair(write_dataset(tmp_path / "a"),
                          write_dataset(tmp_path / "b")).moved == 0

    def test_a_value_wrong_in_the_first_rows_leaves_the_episode_unmatched(self, tmp_path):
        """Recorded because it shapes how the report has to be read: `absent` means
        "no episode with these contents", not "no episode at this index"."""
        p = self._pair(write_dataset(tmp_path / "a", nudge=17),
                       write_dataset(tmp_path / "b"))
        assert p.rebuilt_only == [0] and p.delivered_only == [0]


class TestDistributions:
    """Step 3: whether the two describe the same distribution, and the reason the
    question is asked twice."""

    def test_identical_datasets_have_no_gap(self, tmp_path):
        a, b = write_dataset(tmp_path / "a"), write_dataset(tmp_path / "b")
        gaps = compare.distribution_gap(compare.distribution(a), compare.distribution(b))
        assert set(gaps) == {"observation.state", "action"}
        assert max(gaps.values()) <= compare.DISTRIBUTION_TOLERANCE

    def test_a_changed_value_opens_a_gap(self, tmp_path):
        a = write_dataset(tmp_path / "a", nudge=17, episodes=1)
        b = write_dataset(tmp_path / "b", episodes=1)
        gaps = compare.distribution_gap(compare.distribution(a), compare.distribution(b))
        assert gaps["observation.state"] > compare.DISTRIBUTION_TOLERANCE

    def test_a_column_on_only_one_side_is_not_quietly_skipped(self):
        gaps = compare.distribution_gap({"action": {}}, {})
        assert gaps["action"] == float("inf")

    def test_restricting_to_shared_episodes_removes_the_effect_of_a_missing_one(
        self, tmp_path
    ):
        """The whole reason step 3 is asked twice. A rebuild that lost an episode has a
        genuinely different distribution overall, and an identical one over what the
        two copies share -- and only the second says whether the values are right."""
        a = write_dataset(tmp_path / "a", episodes=3, only={0, 1})
        b = write_dataset(tmp_path / "b", episodes=3)
        shared = {0, 1}
        overall = compare.distribution_gap(
            compare.distribution(a), compare.distribution(b))
        restricted = compare.distribution_gap(
            compare.distribution(a, episodes=shared),
            compare.distribution(b, episodes=shared))
        assert overall["action"] > compare.DISTRIBUTION_TOLERANCE
        assert restricted["action"] <= compare.DISTRIBUTION_TOLERANCE


class TestFunnel:
    """The three questions together, and what each one is allowed to fail on."""

    def test_a_faithful_rebuild_passes_every_step(self, spec, tmp_path):
        f = compare.measure(spec, write_dataset(tmp_path / "a"),
                            write_dataset(tmp_path / "b"), episodes=2, check_video=False)
        assert f.values_agree and f.distributions_agree
        assert len(f.pairing.exact) == 2

    def test_a_missing_episode_is_measured_and_does_not_fail_the_run(self, spec, tmp_path):
        """Which episodes a rebuild ends up with is decided outside this comparison, so
        the report counts them and still answers whether the rest is right."""
        f = compare.measure(spec, write_dataset(tmp_path / "a", episodes=2, only={0}),
                            write_dataset(tmp_path / "b", episodes=2),
                            episodes=2, check_video=False)
        assert f.pairing.delivered_only == [1]
        assert f.values_agree and f.distributions_agree

    def test_step_three_restricts_both_sides_to_the_shared_episodes(self, spec, tmp_path):
        """A rebuild carrying an episode the delivered copy does not have differs
        overall and agrees over what they share. Restricting only one side would leave
        the second number reading like the first, which is the mistake that makes the
        whole arrangement pointless."""
        f = compare.measure(spec, write_dataset(tmp_path / "a", episodes=2),
                            write_dataset(tmp_path / "b", episodes=2, only={0}),
                            episodes=2, check_video=False)
        assert f.pairing.rebuilt_only == [1]
        assert f.gap_overall["action"] > compare.DISTRIBUTION_TOLERANCE
        assert f.gap_shared["action"] <= compare.DISTRIBUTION_TOLERANCE
        assert f.distributions_agree

    def test_wrong_values_fail_the_run(self, spec, tmp_path):
        """Nothing pairs, so step 2 falls back to position and still reports what
        differs rather than going quiet."""
        f = compare.measure(spec, write_dataset(tmp_path / "a", nudge=17, episodes=1),
                            write_dataset(tmp_path / "b", episodes=1),
                            episodes=2, check_video=False)
        assert not f.values_agree
        assert [r for r in f.episodes if r.index >= 0]

    def test_the_report_names_every_step_and_ends_with_a_verdict(self, spec, tmp_path):
        text = compare.funnel_report(compare.measure(
            spec, write_dataset(tmp_path / "a"), write_dataset(tmp_path / "b"),
            episodes=2, check_video=False))
        for heading in ("1  declaration", "2  episodes", "3  sample",
                        "4  distributions"):
            assert heading in text
        # a step's numbers are not a verdict on their own; a reader should not have to
        # know that 1.5e-15 is a pass and 5.3e-02 is not
        assert text.rstrip().splitlines()[-1].startswith("verdict")
        assert "[ ok ]" in text

    def test_the_step_a_run_failed_is_named_in_the_verdict(self, spec, tmp_path):
        # past the prefix, so the episode still pairs and reaches the full comparison
        # rather than dropping out of the pairing as an episode nobody has
        measured = compare.measure(
            spec, write_dataset(tmp_path / "a", nudge=3, nudge_row=15),
            write_dataset(tmp_path / "b"), episodes=2, check_video=False)
        text = compare.funnel_report(measured)
        assert "[FAIL]" in text
        assert "sampled episodes differ" in text

    def test_the_episode_count_step_asserts_nothing(self, spec, tmp_path):
        """It is measured and not judged, so it must not print a pass either."""
        text = compare.funnel_report(compare.measure(
            spec, write_dataset(tmp_path / "a", episodes=3, only={0, 1}),
            write_dataset(tmp_path / "b", episodes=3), episodes=2, check_video=False))
        step = next(line for line in text.splitlines() if line.startswith("2  episodes"))
        assert "[ -- ]" in step
        assert "verdict          [ ok ]" in text

    def test_the_report_explains_a_gap_the_missing_episodes_account_for(
        self, spec, tmp_path
    ):
        text = compare.funnel_report(compare.measure(
            spec, write_dataset(tmp_path / "a", episodes=3, only={0, 1}),
            write_dataset(tmp_path / "b", episodes=3), episodes=2, check_video=False))
        assert "1 absent from the rebuild" in text
        assert "over what both copies hold identically, they agree" in text


INFO = {
    "codebase_version": "v2.1",
    "robot_type": "hello_stretch",
    "fps": 10,
    "total_episodes": 135,
    "total_frames": 25016,
    "splits": {"train": "0:135"},
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [128, 128, 3],
            "info": {"video.codec": "av1", "video.pix_fmt": "yuv420p"},
        },
        "observation.state": {"dtype": "float32", "shape": [8]},
    },
}


def write_info(root: Path, **overrides):
    """A delivered-shaped meta/info.json, with fields replaced at the top level."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({**INFO, **overrides}))
    return root


class TestDeclaration:
    """What the two copies say about themselves, before either is opened.

    The cheapest question in the funnel: two small JSON files decide whether the
    rebuild even claims to be the same dataset. Anything wrong here is wrong in every
    episode, so it is worth failing before a parquet is read.
    """

    def _compare(self, tmp_path, mine, theirs):
        write_info(tmp_path / "a", **mine)
        write_info(tmp_path / "b", **theirs)
        return compare.compare_declarations(tmp_path / "a", tmp_path / "b")

    def test_the_same_declaration_agrees(self, tmp_path):
        assert self._compare(tmp_path, {}, {}).agree

    def test_a_different_fps_is_a_difference(self, tmp_path):
        result = self._compare(tmp_path, {"fps": 20}, {})
        assert not result.agree
        assert result.differences["fps"] == (20, 10)

    def test_a_different_robot_is_a_difference(self, tmp_path):
        assert "robot_type" in self._compare(
            tmp_path, {"robot_type": "panda"}, {}).differences

    def test_a_nested_feature_field_is_reached(self, tmp_path):
        """The interesting fields are three levels down -- a codec or a pixel format
        sits under features.<name>.info -- so a shallow diff would miss all of them."""
        theirs = json.loads(json.dumps(INFO["features"]))
        theirs["observation.images.image"]["info"]["video.codec"] = "h264"
        result = self._compare(tmp_path, {}, {"features": theirs})
        assert result.differences[
            "features.observation.images.image.info.video.codec"] == ("av1", "h264")

    def test_a_renamed_camera_reads_as_a_field_each_side_lacks(self, tmp_path):
        """Eight of the delivered OXE datasets renamed their cameras to the modality
        aliases. That shows up here as one key on each side, which is why an absent
        field has to be reported rather than skipped as 'nothing to compare'."""
        mine = {"observation.images.primary": {"dtype": "video", "shape": [128, 128, 3]}}
        result = self._compare(tmp_path, {"features": mine}, {})
        assert not result.agree
        absent = [k for k, v in result.differences.items()
                  if compare.ABSENT in v]
        assert any("primary" in k for k in absent)
        assert any("observation.state" in k for k in absent)

    def test_the_count_fields_are_set_aside_rather_than_failed(self, tmp_path):
        """They follow from which episodes the rebuild ended up with, which the next
        step measures. Failing here as well would report one finding twice."""
        result = self._compare(
            tmp_path, {"total_episodes": 100, "total_frames": 20000}, {})
        assert result.agree
        assert set(result.counts) == {"total_episodes", "total_frames"}

    def test_a_split_range_is_a_count_field_too(self, tmp_path):
        result = self._compare(tmp_path, {"splits": {"train": "0:100"}}, {})
        assert result.agree
        assert "splits.train" in result.counts

    def test_neither_copy_having_one_is_not_a_finding(self, tmp_path):
        """A partial tree may carry no info.json, and inventing a failure for a file
        nobody wrote is noise."""
        for name in ("a", "b"):
            (tmp_path / name / "meta").mkdir(parents=True)
        result = compare.compare_declarations(tmp_path / "a", tmp_path / "b")
        assert result.agree
        assert not result.missing

    def test_one_copy_having_one_is_a_finding(self, tmp_path):
        (tmp_path / "a" / "meta").mkdir(parents=True)
        write_info(tmp_path / "b")
        result = compare.compare_declarations(tmp_path / "a", tmp_path / "b")
        assert not result.agree
        assert result.missing == ["rebuilt"]

    def test_unreadable_json_is_treated_as_absent(self, tmp_path):
        (tmp_path / "a" / "meta").mkdir(parents=True)
        (tmp_path / "a" / "meta" / "info.json").write_text("{ truncated")
        write_info(tmp_path / "b")
        assert compare.compare_declarations(
            tmp_path / "a", tmp_path / "b").missing == ["rebuilt"]

    def test_the_funnel_fails_on_a_declaration_alone(self, spec, tmp_path):
        """Every episode can be byte-identical and the rebuild still be the wrong
        dataset -- a different fps means the same rows played at a different speed."""
        write_dataset(tmp_path / "a")
        write_dataset(tmp_path / "b")
        write_info(tmp_path / "a", fps=20)
        write_info(tmp_path / "b")
        measured = compare.measure(spec, tmp_path / "a", tmp_path / "b",
                                   episodes=2, check_video=False)
        assert measured.values_agree and measured.distributions_agree
        assert not measured.agree
        assert "declare different datasets" in "; ".join(measured.reasons)


class TestSampleChoice:
    """Which paired episodes get compared in full.

    Taking the first N looked thorough and was not: the converter writes in chunks of
    twenty-five episodes, one worker to a chunk, so the front of the range is a single
    worker's output and every later chunk went unsampled.
    """

    def test_half_the_sample_comes_from_each_end(self):
        pairs = {index: index for index in range(100)}
        chosen = compare.choose_sample(pairs, 8)
        assert chosen == [0, 1, 2, 3, 96, 97, 98, 99]

    def test_a_defect_in_the_last_chunk_is_inside_the_sample(self):
        """The property the front-only sample did not have."""
        pairs = {index: index for index in range(135)}
        assert max(compare.choose_sample(pairs, 64)) == 134

    def test_everything_is_taken_when_there_is_less_than_asked_for(self):
        pairs = {index: index for index in range(5)}
        assert compare.choose_sample(pairs, 64) == [0, 1, 2, 3, 4]

    def test_an_odd_sample_puts_the_extra_at_the_front(self):
        pairs = {index: index for index in range(100)}
        assert compare.choose_sample(pairs, 5) == [0, 1, 2, 98, 99]

    def test_the_indices_come_from_the_pairing_and_not_from_a_range(self):
        """Unpaired episodes are not sampled: there is nothing to compare them to."""
        pairs = {3: 0, 9: 1, 40: 2}
        assert compare.choose_sample(pairs, 2) == [3, 40]

    def test_the_funnel_samples_both_ends(self, spec, tmp_path):
        measured = compare.measure(
            spec, write_dataset(tmp_path / "a", episodes=10),
            write_dataset(tmp_path / "b", episodes=10), episodes=4, check_video=False)
        assert measured.chosen == [0, 1, 8, 9]


def write_videos(root: Path, camera: str, episodes, size=1000, declared=True):
    """Video files of a given size, with no picture in them.

    The coverage check is deliberately readable from file metadata alone -- that is
    what makes it affordable over every episode rather than over a sample -- so a
    fixture for it does not need a decodable clip.
    """
    path = root / "videos" / "chunk-000" / camera
    path.mkdir(parents=True, exist_ok=True)
    for index in episodes:
        (path / f"episode_{index:06d}.mp4").write_bytes(b"\0" * size)
    if declared:
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "modality.json").write_text(json.dumps(
            {"video": {camera: {"original_key": f"observation.images.{camera}"}}}))
    return root


class TestVideoCoverage:
    """Every episode's video weighed, without opening any of them.

    The frame-by-frame comparison is only affordable on a sample, which leaves the
    rest of the dataset unlooked-at. A rebuild that wrote a tenth of its videos used
    to pass on that.
    """

    def _coverage(self, tmp_path, mine, theirs, size=1000, episodes=None,
                  their_episodes=None):
        """``mine``/``theirs`` are the videos on each side; the episode counts default
        to one video an episode, which is what a whole dataset looks like."""
        mine, theirs = list(mine), list(theirs)
        rebuilt = write_videos(tmp_path / "a", "top", mine, size=size)
        delivered = write_videos(tmp_path / "b", "top", theirs)
        return compare.compare_video_totals(
            rebuilt, delivered,
            len(mine) if episodes is None else episodes,
            len(theirs) if their_episodes is None else their_episodes,
        )

    def test_a_video_for_every_episode_on_both_sides_agrees(self, tmp_path):
        assert self._coverage(tmp_path, range(10), range(10)).agree

    def test_missing_videos_are_caught_even_though_no_frame_is_read(self, tmp_path):
        """Ten episodes and one video. The sampled comparison would have looked at
        that one video and said nothing about the other nine."""
        result = self._coverage(tmp_path, [0], range(10), episodes=10)
        assert not result.agree
        assert "1 files for 10 episodes" in result.problems[0]

    def test_a_shortfall_of_episodes_is_not_a_shortfall_of_videos(self, tmp_path):
        """A rebuild missing episodes is legitimately missing their videos too, and
        that is the episode step's finding rather than this one's."""
        rebuilt = write_videos(tmp_path / "a", "top", range(4))
        delivered = write_videos(tmp_path / "b", "top", range(10))
        assert compare.compare_video_totals(rebuilt, delivered, 4, 10).agree

    def test_the_same_episodes_with_videos_missing_still_fails(self, tmp_path):
        rebuilt = write_videos(tmp_path / "a", "top", range(4))
        delivered = write_videos(tmp_path / "b", "top", range(10))
        result = compare.compare_video_totals(rebuilt, delivered, 10, 10)
        assert not result.agree
        assert "4 files for 10 episodes" in result.problems[0]

    def test_a_camera_the_rebuild_never_wrote_is_named(self, tmp_path):
        rebuilt = write_videos(tmp_path / "a", "wrist", range(10), declared=False)
        delivered = write_videos(tmp_path / "b", "top", range(10))
        result = compare.compare_video_totals(rebuilt, delivered, 10, 10)
        assert not result.agree
        assert "the rebuild has none at all" in result.problems[0]

    def test_a_size_that_ffmpeg_cannot_explain_is_caught(self, tmp_path):
        result = self._coverage(tmp_path, range(10), range(10), size=2000)
        assert not result.agree
        assert "2.00x" in result.problems[0]

    def test_a_size_a_different_ffmpeg_build_explains_is_tolerated(self, tmp_path):
        assert self._coverage(tmp_path, range(10), range(10), size=1050).agree

    def test_the_size_is_the_mean_and_not_the_total(self, tmp_path):
        """Otherwise a rebuild missing half its episodes would read as a rebuild whose
        videos are half the size, and the two want different answers."""
        rebuilt = write_videos(tmp_path / "a", "top", range(5))
        delivered = write_videos(tmp_path / "b", "top", range(10))
        assert compare.compare_video_totals(rebuilt, delivered, 5, 10).agree

    def test_an_undeclared_camera_is_not_weighed(self, tmp_path):
        """The delivered copy decides which cameras the comparison is about; it can
        carry views nothing consumes."""
        rebuilt = write_videos(tmp_path / "a", "top", range(10))
        delivered = write_videos(tmp_path / "b", "top", range(10))
        write_videos(delivered, "spare", range(10), size=99, declared=False)
        result = compare.compare_video_totals(rebuilt, delivered, 10, 10)
        assert set(result.delivered) == {"top"}
        assert result.agree

    def test_coverage_failing_fails_the_run(self, spec, tmp_path):
        rebuilt = write_dataset(tmp_path / "a", episodes=4)
        delivered = write_dataset(tmp_path / "b", episodes=4)
        write_videos(rebuilt, "top", [0])
        write_videos(delivered, "top", range(4))
        measured = compare.measure(spec, rebuilt, delivered, episodes=0,
                                   check_video=True)
        assert not measured.values_agree
        assert "videos do not add up" in "; ".join(measured.reasons)


class TestPromptsOverEveryPair:
    """episodes.jsonl is read in full anyway, so there is nothing to save by
    checking only the sample -- and a task_index that shifted would survive it."""

    def test_matching_prompts_agree(self, tmp_path):
        write_dataset(tmp_path / "a", episodes=6)
        write_dataset(tmp_path / "b", episodes=6)
        result = compare.compare_prompts(
            tmp_path / "a", tmp_path / "b", {i: i for i in range(6)})
        assert result.agree
        assert result.pairs == 6

    def test_a_shifted_mapping_is_caught_outside_the_sample(self, tmp_path):
        """The prompt is wrong on episode 5 only, which a front-weighted sample of
        two would never have opened."""
        write_dataset(tmp_path / "a", episodes=6, prompts={5: "wrong"})
        write_dataset(tmp_path / "b", episodes=6)
        result = compare.compare_prompts(
            tmp_path / "a", tmp_path / "b", {i: i for i in range(6)})
        assert not result.agree
        assert result.mismatched == {5: ("wrong", "task 5")}

    def test_prompts_are_checked_against_the_paired_index(self, tmp_path):
        """A reordered rebuild carries its prompts reordered with it, so the check has
        to follow the pairing rather than the position."""
        write_dataset(tmp_path / "a", episodes=3,
                      prompts={0: "task 2", 2: "task 0"})
        write_dataset(tmp_path / "b", episodes=3)
        assert compare.compare_prompts(
            tmp_path / "a", tmp_path / "b", {0: 2, 1: 1, 2: 0}).agree

    def test_a_missing_episode_removes_its_prompt_without_failing(self, tmp_path):
        write_dataset(tmp_path / "a", episodes=3, only={0, 1})
        write_dataset(tmp_path / "b", episodes=3)
        result = compare.compare_prompts(
            tmp_path / "a", tmp_path / "b", {0: 0, 1: 1})
        assert result.agree
        assert result.delivered_only == {"task 2": 1}

    def test_a_wrong_prompt_fails_the_run(self, spec, tmp_path):
        measured = compare.measure(
            spec, write_dataset(tmp_path / "a", episodes=4, prompts={3: "wrong"}),
            write_dataset(tmp_path / "b", episodes=4), episodes=0, check_video=False)
        assert not measured.values_agree
        assert "different task prompts" in "; ".join(measured.reasons)


class TestTrimmedRows:
    def test_how_far_off_each_trimmed_episode_is_is_recorded(self, tmp_path):
        pairing = compare.pair_episodes(
            compare.episode_fingerprints(write_dataset(tmp_path / "a", rows=18)),
            compare.episode_fingerprints(write_dataset(tmp_path / "b")),
        )
        assert pairing.row_deltas == {0: -2, 1: -2}
        assert pairing.worst_row_delta == 2

    def test_an_exact_pair_contributes_no_delta(self, tmp_path):
        pairing = compare.pair_episodes(
            compare.episode_fingerprints(write_dataset(tmp_path / "a")),
            compare.episode_fingerprints(write_dataset(tmp_path / "b")),
        )
        assert pairing.row_deltas == {}
        assert pairing.worst_row_delta == 0

    def test_the_report_says_the_range_rather_than_only_the_count(self, spec, tmp_path):
        text = compare.funnel_report(compare.measure(
            spec, write_dataset(tmp_path / "a", rows=18),
            write_dataset(tmp_path / "b"), episodes=2, check_video=False))
        assert "2 differ in length (-2 to -2 rows)" in text

    def test_a_same_length_pair_is_not_called_a_trim(self, spec, tmp_path):
        """The prefix pass catches two things and only one is a trimmed tail. An
        episode of the right length whose values go wrong past row 8 is the more
        serious of the two, and reporting it as "a different row count" buries it."""
        pairing = compare.pair_episodes(
            compare.episode_fingerprints(
                write_dataset(tmp_path / "a", nudge=3, nudge_row=15)),
            compare.episode_fingerprints(write_dataset(tmp_path / "b")),
        )
        assert pairing.same_length == [0]
        text = compare.funnel_report(compare.measure(
            spec, tmp_path / "a", tmp_path / "b", episodes=2, check_video=False))
        assert "0 differ in length" in text
        assert "1 are the same length with values that go wrong further in" in text


class TestReportFile:
    """The record a run leaves behind.

    Runs happen on throwaway nodes, one dataset to a node, and the files are collected
    into one directory and committed. So the record has to carry enough to be read
    without the node, the dataset, or this file at that revision.
    """

    def _payload(self, spec, tmp_path):
        measured = compare.measure(
            spec, write_dataset(tmp_path / "a", episodes=4),
            write_dataset(tmp_path / "b", episodes=4), episodes=2, check_video=False)
        return compare.as_dict(
            measured, tmp_path / "a", tmp_path / "b",
            compare.settings(episodes=2, row_tolerance=2, check_video=False))

    def test_both_files_are_written_and_named_after_the_dataset(self, spec, tmp_path):
        payload = self._payload(spec, tmp_path)
        written = compare.write_report(payload, "the text", tmp_path / "out")
        assert [p.name for p in written] == ["action_net.json", "action_net.txt"]
        assert (tmp_path / "out" / "action_net.txt").read_text() == "the text\n"

    def test_the_json_survives_a_round_trip(self, spec, tmp_path):
        """numpy is all over the distribution statistics, and json refuses it."""
        payload = self._payload(spec, tmp_path)
        written = compare.write_report(payload, "t", tmp_path / "out")
        assert json.loads(written[0].read_text())["dataset"] == "action_net"

    def test_re_running_a_dataset_overwrites_its_own_record(self, spec, tmp_path):
        payload = self._payload(spec, tmp_path)
        compare.write_report(payload, "first", tmp_path / "out")
        compare.write_report(payload, "second", tmp_path / "out")
        assert len(list((tmp_path / "out").glob("action_net.*"))) == 2
        assert (tmp_path / "out" / "action_net.txt").read_text() == "second\n"

    def test_the_thresholds_that_decided_the_verdict_are_recorded(self, spec, tmp_path):
        """A verdict without its thresholds cannot be re-read later."""
        recorded = self._payload(spec, tmp_path)["settings"]
        assert recorded["pixel_agreement"] == compare.PIXEL_AGREEMENT
        assert recorded["distribution_tolerance"] == compare.DISTRIBUTION_TOLERANCE
        assert recorded["sample_episodes"] == 2

    def test_the_index_map_is_kept(self, spec, tmp_path):
        """Without it, a finding about 'episode 12' cannot be looked up again."""
        assert self._payload(spec, tmp_path)["episodes"]["pairs"] == {
            "0": 0, "1": 1, "2": 2, "3": 3}

    def test_the_statistics_are_kept_and_not_only_the_gap(self, spec, tmp_path):
        """The datasets are far too large to keep beside the record, so a number
        nobody can recompute is a number nobody can argue with."""
        detail = self._payload(spec, tmp_path)["distributions"]["detail"]
        stats = detail["overall"]["delivered"]["observation.state"]
        assert set(compare.DISTRIBUTION_STATS) <= set(stats)
        assert len(stats["mean"]) == WIDTH

    def test_every_sampled_episode_is_kept(self, spec, tmp_path):
        sample = self._payload(spec, tmp_path)["sample"]
        assert sample["chosen"] == [0, 3]
        assert [r["index"] for r in sample["reports"]] == [0, 3]

    def test_the_verdict_carries_its_reasons(self, spec, tmp_path):
        measured = compare.measure(
            spec, write_dataset(tmp_path / "a", nudge=2, nudge_row=15),
            write_dataset(tmp_path / "b"), episodes=2, check_video=False)
        payload = compare.as_dict(measured, tmp_path / "a", tmp_path / "b", {})
        assert payload["verdict"]["overall"] is False
        assert payload["verdict"]["reasons"] == ["sampled episodes differ"]

    def test_the_statistics_are_rounded_so_a_re_run_diffs_cleanly(self, spec, tmp_path):
        """The records are committed. A float64's full repr changes in its last digits
        when a rebuild sums in a different order, so an unrounded record re-diffs
        entirely on a run that found nothing new -- and a diff that always changes is
        a diff nobody reads."""
        stats = self._payload(spec, tmp_path)["distributions"]["detail"]
        mean = stats["overall"]["delivered"]["observation.state"]["mean"]
        # every recorded value is already its own rounding, and the raw statistic is
        # not -- so the rounding happened and re-running cannot change these digits
        assert all(v == float(f"{v:.{compare.RECORDED_DIGITS}g}") for v in mean)
        raw = compare.distribution(tmp_path / "b")["observation.state"]["mean"]
        assert any(v != float(f"{v:.{compare.RECORDED_DIGITS}g}") for v in raw.tolist())


class TestChannelDiagnosis:
    """Naming the cause once the frames have already disagreed.

    Channel order is deliberately not a check of its own. It was one difference among
    the many a wrong picture can be -- a time offset and a crop position are the next
    two -- and a dedicated test for each never ends, while the frame comparison
    catches all of them. So the reversal is consulted only after a failure, and only
    to say what it probably was.
    """

    def _clip(self, path: Path, colour, frames=6):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             f"color=c={colour}:s=64x64:r=10:d={frames / 10}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)],
            check=True, capture_output=True)
        return path

    def test_a_matching_pair_is_not_diagnosed(self, tmp_path):
        """Nothing to explain, so nothing is said -- and the reversal is never even
        computed, which is why it costs a passing run nothing."""
        clip = self._clip(tmp_path / "a.mp4", "red")
        score, diagnosis = compare.pixel_verdict(clip, clip)
        assert score > 0.99 and diagnosis is None

    def test_exchanged_channels_are_named(self, tmp_path):
        a = self._clip(tmp_path / "a.mp4", "0xFF4020")
        b = self._clip(tmp_path / "b.mp4", "0x2040FF")
        score, diagnosis = compare.pixel_verdict(a, b)
        assert score < compare.PIXEL_AGREEMENT
        assert diagnosis is not None and "red and blue" in diagnosis

    def test_a_picture_that_is_simply_wrong_gets_no_false_explanation(self, tmp_path):
        """Reversing it does not help either, so the report should not offer a cause
        it cannot support."""
        a = self._clip(tmp_path / "a.mp4", "0xFF4020")
        b = self._clip(tmp_path / "b.mp4", "0x20FF40")
        score, diagnosis = compare.pixel_verdict(a, b)
        assert score < compare.PIXEL_AGREEMENT
        assert diagnosis is None

    def test_the_diagnosis_reaches_the_reported_problem(self, tmp_path):
        for root, colour in ((tmp_path / "r", "0xFF4020"), (tmp_path / "d", "0x2040FF")):
            self._clip(root / "videos" / "chunk-000" / "cam" / "episode_000000.mp4",
                       colour)
        _, problems = compare.compare_video(tmp_path / "r", tmp_path / "d", 0)
        assert any("red and blue channels are exchanged" in p for p in problems), problems


def test_rounding_keeps_the_distinction_the_records_exist_for():
    """Significant digits, not decimal places: rounding 1.5e-15 to twelve decimals is
    a flat zero, and "the shared episodes agree to 1e-15" is the whole finding."""
    assert compare._jsonable(1.5e-15) == 1.5e-15
    assert compare._jsonable(float("inf")) == float("inf")
    assert compare._jsonable(0.1 + 0.2) == 0.3
    assert compare._jsonable({"a": [np.float64(1.5e-15)]}) == {"a": [1.5e-15]}
