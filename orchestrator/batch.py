"""Which datasets get processed together.

A single dataset does not keep a large machine busy. The transform parallelises per
file, so `workers beyond the file count sit idle` -- and 14 of the 36 registry
datasets have fewer than a thousand episodes. Running those one at a time leaves
most of the machine idle for most of the run.

The rule is not "three at a time". It is *fill the machine*: accumulate datasets
until there is enough work to go round, and let anything that already has enough
work run on its own. That distinction is what keeps disk use flat -- the datasets
that get grouped are exactly the small ones, and the large ones, which are the only
ones big enough to matter, are never multiplied.

Episode count stands in for parallel work. Video files are roughly episodes times
cameras, and episodes is what the registry actually records.

Nothing here touches the disk, the network or the clock: it is a list in and a list
of lists out, so the awkward cases are reachable from a test.
"""

from collections.abc import Iterable
from dataclasses import dataclass


class BatchError(ValueError):
    """Raised for limits that cannot produce a sensible batching."""


@dataclass(frozen=True)
class Candidate:
    dataset: str
    # 0 for a dataset whose episode count is unknown; see batches()
    episodes: int = 0


def batches(
    candidates: Iterable[Candidate],
    *,
    max_datasets: int,
    target_episodes: int,
) -> list[tuple[str, ...]]:
    """Group ``candidates`` into batches, keeping their order.

    A dataset holding ``target_episodes`` or more on its own becomes its own batch,
    and closes whatever batch was being filled -- putting a large dataset alongside
    small ones would multiply the largest disk footprint for no gain, since it
    already saturates the workers by itself.

    A dataset with an unknown episode count contributes nothing towards the target,
    so it is grouped rather than run alone; ``max_datasets`` is what bounds it.
    """
    if max_datasets < 1:
        raise BatchError(f"max_datasets must be at least 1, got {max_datasets}")
    if target_episodes < 1:
        raise BatchError(f"target_episodes must be at least 1, got {target_episodes}")

    out: list[tuple[str, ...]] = []
    current: list[str] = []
    accumulated = 0

    def close() -> None:
        nonlocal current, accumulated
        if current:
            out.append(tuple(current))
            current, accumulated = [], 0

    for candidate in candidates:
        if candidate.episodes < 0:
            raise BatchError(
                f"{candidate.dataset}: episodes must not be negative, "
                f"got {candidate.episodes}"
            )
        if candidate.episodes >= target_episodes:
            close()
            out.append((candidate.dataset,))
            continue

        current.append(candidate.dataset)
        accumulated += candidate.episodes
        if len(current) >= max_datasets or accumulated >= target_episodes:
            close()

    close()
    return out
