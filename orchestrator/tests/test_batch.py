import pytest

from orchestrator.batch import BatchError, Candidate, batches


def small(name: str, episodes: int = 50) -> Candidate:
    return Candidate(dataset=name, episodes=episodes)


def test_a_dataset_with_enough_work_runs_alone():
    result = batches(
        [small("droid", 92233)], max_datasets=3, target_episodes=768
    )

    assert result == [("droid",)]


def test_small_datasets_are_grouped_until_the_target_is_reached():
    result = batches(
        [small("a", 300), small("b", 300), small("c", 300)],
        max_datasets=8,
        target_episodes=768,
    )

    # a + b is 600, still short; c takes it to 900 and closes the batch
    assert result == [("a", "b", "c")]


def test_a_batch_never_exceeds_max_datasets():
    result = batches(
        [small("a"), small("b"), small("c"), small("d")],
        max_datasets=2,
        target_episodes=10_000,
    )

    assert result == [("a", "b"), ("c", "d")]


def test_a_large_dataset_closes_the_batch_being_filled():
    result = batches(
        [small("a", 100), small("big", 5000), small("b", 100)],
        max_datasets=3,
        target_episodes=768,
    )

    assert result == [("a",), ("big",), ("b",)]


def test_a_partly_filled_batch_is_still_returned():
    result = batches([small("a", 100)], max_datasets=3, target_episodes=768)

    assert result == [("a",)]


def test_no_candidates_means_no_batches():
    assert batches([], max_datasets=3, target_episodes=768) == []


def test_an_unknown_episode_count_never_closes_a_batch_on_its_own():
    """A dataset whose episode count is missing counts as no work, so it is
    grouped rather than run alone -- max_datasets is what bounds it."""
    result = batches(
        [small("a", 0), small("b", 0), small("c", 0), small("d", 0)],
        max_datasets=3,
        target_episodes=768,
    )

    assert result == [("a", "b", "c"), ("d",)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_datasets": 0, "target_episodes": 768},
        {"max_datasets": -1, "target_episodes": 768},
        {"max_datasets": 3, "target_episodes": 0},
    ],
)
def test_meaningless_limits_are_rejected(kwargs):
    with pytest.raises(BatchError):
        batches([small("a")], **kwargs)


def test_negative_episode_counts_are_rejected():
    with pytest.raises(BatchError):
        batches([small("a", -1)], max_datasets=3, target_episodes=768)
