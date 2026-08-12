"""What to fetch, in what order, and how many at once.

Pure decisions only -- no network, no filesystem beyond stat() for resume -- so
the choices that govern throughput are testable without AWS credentials.

Every constant here is the result of a measurement, not a guess. See README.md
in this directory for the numbers and the machines they came from.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Past this many concurrent transfers the client degraded and then failed
# outright: 128 concurrent transfers lost 433 of 604 objects on c8gn.48xlarge.
# A safety rail, not a tuning knob.
MAX_CONCURRENCY = 32

# Throughput one concurrent transfer sustained (Gbps), measured consistently
# across x86 and Graviton and across 25/50/100/600 Gbps NICs. Used to turn a
# bandwidth target into a concurrency.
GBPS_PER_TRANSFER = 3.0

# Used when the NIC rate cannot be determined. Deliberately low: overstating the
# target measurably slowed transfers, so under-driving is the safer error.
FALLBACK_TARGET_GBPS = 25

# Multipart part size. Small parts cap the achievable rate on request count:
# 8MB parts need ~9,400 GET/s to sustain 600 Gbps, above the ~5,500/s per-prefix
# guidance, and these datasets keep every object under one prefix. 64MB needs
# ~1,170/s. Measured: 8MB parts held ~17 Gbps where 64MB reached 89.
DEFAULT_PART_SIZE = 64 * 1024 * 1024


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int


@dataclass(frozen=True)
class FetchTask:
    obj: RemoteObject
    dest: Path | None  # None streams to nowhere, for measuring the network alone


@dataclass(frozen=True)
class FetchPlan:
    tasks: tuple[FetchTask, ...]
    skipped: tuple[RemoteObject, ...]
    concurrency: int
    target_gbps: float
    total_bytes: int
    discard: bool
    part_size: int


@dataclass(frozen=True)
class FetchSummary:
    objects: int
    total_bytes: int
    seconds: float
    concurrency: int
    gbps: float
    gigabytes_per_s: float
    nic_utilisation_pct: float | None

    def format(self) -> str:
        line = (
            f"{self.objects} object(s), {self.total_bytes / 1e12:.3f} TB in "
            f"{self.seconds:.1f}s at {self.gbps:,.1f} Gbps "
            f"({self.gigabytes_per_s:,.2f} GB/s, {self.concurrency} concurrent)"
        )
        if self.nic_utilisation_pct is not None:
            line += f"\n  NIC utilisation: {self.nic_utilisation_pct:.0f}%"
        return line


def resolve_concurrency(
    n_objects: int,
    target_gbps: float,
    requested: int | None = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> int:
    """Pick how many objects to transfer at once."""
    if requested is not None:
        if requested > max_concurrency:
            warnings.warn(
                f"concurrency {requested} is beyond the measured safe range "
                f"(<= {max_concurrency}); transfers degraded and then failed "
                "outright above this in testing",
                UserWarning,
                stacklevel=2,
            )
        return max(1, requested)

    if n_objects <= 0:
        return 1
    wanted = max(1, round(target_gbps / GBPS_PER_TRANSFER))
    return max(1, min(wanted, max_concurrency, n_objects))


def resolve_target_gbps(
    nic_gbps: float | None, requested: float | None = None
) -> float:
    """Clamp the declared bandwidth target to what the NIC can actually do."""
    if nic_gbps is None:
        return (
            float(requested) if requested is not None else float(FALLBACK_TARGET_GBPS)
        )
    if requested is None:
        return float(nic_gbps)
    if requested > nic_gbps:
        warnings.warn(
            f"target {requested} Gbps is above the NIC rate {nic_gbps} Gbps; "
            "declaring more than the NIC measurably slowed transfers "
            "(41.41 -> 37.42 Gbps in testing). Clamping to the NIC rate.",
            UserWarning,
            stacklevel=2,
        )
        return float(nic_gbps)
    return float(requested)


def plan_fetch(
    objects: Iterable[RemoteObject],
    dest_dir: str | Path | None,
    target_gbps: float | None = None,
    nic_gbps: float | None = None,
    concurrency: int | None = None,
    resume: bool = True,
    part_size: int = DEFAULT_PART_SIZE,
) -> FetchPlan:
    """Decide what to fetch, in what order, and how many at once.

    Largest objects first: with a fixed number of workers, handing out the big
    ones while everyone is still free keeps the run from ending on a straggler.
    """
    objects = list(objects)
    discard = dest_dir is None
    root = None if discard else Path(dest_dir)

    tasks: list[FetchTask] = []
    skipped: list[RemoteObject] = []
    for obj in sorted(objects, key=lambda o: -o.size):
        dest = None if root is None else root / obj.key
        # a partially written file is worse than a missing one: only an exact
        # size match counts as done
        if (
            resume
            and dest is not None
            and dest.exists()
            and dest.stat().st_size == obj.size
        ):
            skipped.append(obj)
            continue
        tasks.append(FetchTask(obj=obj, dest=dest))

    resolved_target = resolve_target_gbps(nic_gbps, target_gbps)
    return FetchPlan(
        tasks=tuple(tasks),
        skipped=tuple(skipped),
        concurrency=resolve_concurrency(len(tasks), resolved_target, concurrency),
        target_gbps=resolved_target,
        total_bytes=sum(t.obj.size for t in tasks),
        discard=discard,
        part_size=part_size,
    )


def summarize_fetch(
    total_bytes: int,
    seconds: float,
    concurrency: int,
    objects: int = 0,
    nic_gbps: float | None = None,
) -> FetchSummary:
    """Report GB/s alongside Gbps: storage specs are quoted in GB/s and network
    specs in Gbps, and comparing them is the whole game."""
    gbps = total_bytes * 8 / seconds / 1e9 if seconds > 0 else 0.0
    return FetchSummary(
        objects=objects,
        total_bytes=total_bytes,
        seconds=seconds,
        concurrency=concurrency,
        gbps=gbps,
        gigabytes_per_s=total_bytes / seconds / 1e9 if seconds > 0 else 0.0,
        nic_utilisation_pct=(gbps / nic_gbps * 100) if nic_gbps else None,
    )
