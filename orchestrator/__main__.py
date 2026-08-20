"""Download, preprocess and upload every dataset on this machine.

    python -m orchestrator run     --env ec2                  # all of it
    python -m orchestrator fetch   --env ec2                  # one stage
    python -m orchestrator build   --env ec2 --dataset droid  # one dataset
    python -m orchestrator status  --env ec2                  # how far it got

``run`` walks the registry in batches, and within each batch does fetch, then build,
then publish, then reclaims what is no longer needed. Every step records what it
did, so re-running picks up where the last one stopped and skips what is already
done -- which is also how a stage can be run by hand without confusing the others.
"""

import argparse
import sys
from collections.abc import Sequence

from .batch import Candidate, batches
from .run import Outcome, process
from .steps import STEPS, Steps


def select(names: Sequence[str] | None):
    """The dataset specs to work on, in the order given."""
    from dataset_registry import SpecError, available, load

    wanted = list(names) if names else available()
    specs = []
    for name in wanted:
        try:
            specs.append(load(name))
        except SpecError as exc:
            raise SystemExit(f"orchestrator: {exc}") from exc
    return specs


def plan(env, specs) -> list[list]:
    """Group the specs into batches, using this machine's limits.

    Smallest first, which is what makes the grouping do anything. The registry is
    alphabetical, so the small datasets are scattered the length of it -- viola,
    ucsd_kitchen and berkeley_fanuc have the whole collection between them -- and
    batching them in that order leaves each one running alone on a machine it
    cannot fill. Ordering by size puts them next to each other; the large ones then
    fall out as single-dataset batches on their own, which is what they want anyway.
    """
    by_id = {spec.id: spec for spec in specs}
    ordered = sorted(specs, key=lambda spec: (spec.delivered_episodes, spec.id))
    grouped = batches(
        [Candidate(spec.id, spec.delivered_episodes) for spec in ordered],
        max_datasets=env.max_datasets,
        target_episodes=env.target_episodes,
    )
    return [[by_id[name] for name in group] for group in grouped]


def render_status(steps: Steps, specs) -> str:
    width = max((len(spec.id) for spec in specs), default=7)
    header = f"{'dataset':<{width}}  " + "  ".join(f"{step:<8}" for step in STEPS)
    lines = [header, "-" * len(header)]
    for spec in specs:
        cells = []
        for step in STEPS:
            record = steps.read(spec.id, step)
            cells.append(f"{record.status if record else '-':<8}")
        lines.append(f"{spec.id:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def _seconds(record) -> float | None:
    """How long a step took, from the two stamps it already records."""
    if record is None or not record.started or not record.finished:
        return None
    from datetime import datetime

    return (datetime.fromisoformat(record.finished)
            - datetime.fromisoformat(record.started)).total_seconds()


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def render_timings(steps: Steps, specs) -> str:
    """How long each stage took, per dataset.

    Nothing new is measured: ``started``, ``finished`` and ``bytes`` are already in
    every step record, because a record that cannot say when it ran cannot answer
    whether a re-run got slower. This only reads them.

    ``fetch`` is the S3-to-instance leg and is the one with a rate worth printing --
    it is the only stage whose work is a known number of bytes. ``build`` is the
    instance's own time and is reported per episode, since a frame count is a
    property of the delivered copy rather than of what the converter was handed.
    """
    width = max((len(spec.id) for spec in specs), default=7)
    header = (f"{'dataset':<{width}}  {'source':>9}  {'fetch':>7}  {'MB/s':>7}  "
              f"{'build':>7}  {'ep/s':>6}  {'publish':>7}")
    lines = [header, "-" * len(header)]
    total_bytes = total_fetch = total_build = 0.0
    for spec in specs:
        fetch = steps.read(spec.id, "fetch")
        build = steps.read(spec.id, "build")
        publish = steps.read(spec.id, "publish")
        fetch_s, build_s = _seconds(fetch), _seconds(build)
        size = (fetch.bytes if fetch and fetch.bytes else None)
        rate = f"{size / fetch_s / 1e6:.1f}" if size and fetch_s else "-"
        episodes = spec.delivered_episodes or 0
        per = f"{episodes / build_s:.2f}" if build_s and episodes else "-"
        lines.append(
            f"{spec.id:<{width}}  "
            f"{(f'{size / 1e9:.1f} GB' if size else '-'):>9}  "
            f"{_clock(fetch_s):>7}  {rate:>7}  "
            f"{_clock(build_s):>7}  {per:>6}  {_clock(_seconds(publish)):>7}"
        )
        total_bytes += size or 0
        total_fetch += fetch_s or 0
        total_build += build_s or 0
    lines.append("-" * len(header))
    overall = f"{total_bytes / total_fetch / 1e6:.1f}" if total_fetch else "-"
    lines.append(
        f"{'total':<{width}}  {total_bytes / 1e9:6.1f} GB  "
        f"{_clock(total_fetch):>7}  {overall:>7}  {_clock(total_build):>7}"
    )
    if total_fetch and total_build:
        lines.append(
            f"\nfetch was {total_fetch / (total_fetch + total_build) * 100:.0f}% of "
            f"the wall clock, build {total_build / (total_fetch + total_build) * 100:.0f}%."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    from lerobot_pipeline.env import EnvError, load_env

    try:
        env = load_env(args.env)
    except EnvError as exc:
        print(f"orchestrator: {exc}", file=sys.stderr)
        return 2

    if env.state_root is None:
        print(
            f"orchestrator: {env.name} has no state_root, so there is nowhere to "
            "record how far each dataset got -- and without that a re-run would "
            "redo everything. Add one to the environment file.",
            file=sys.stderr,
        )
        return 2

    specs = select(args.dataset)
    steps = Steps(env.state_root)

    if args.command == "status":
        print(render_status(steps, specs))
        return 0

    if args.command == "timings":
        if args.json:
            import dataclasses
            import json

            print(json.dumps({
                spec.id: {
                    step: dataclasses.asdict(record) | {
                        "seconds": _seconds(record)}
                    for step in STEPS
                    if (record := steps.read(spec.id, step)) is not None
                }
                for spec in specs
            }, indent=2))
        else:
            print(render_timings(steps, specs))
        return 0

    only = STEPS if args.command == "run" else (args.command,)
    outcomes: list[Outcome] = []
    for batch in plan(env, specs):
        produced = process(
            env,
            batch,
            steps,
            env_source=args.env,
            only=only,
            keep=args.keep,
            dry_run=args.dry_run,
            known=specs,
        )
        for outcome in produced:
            print(outcome, flush=True)
        outcomes += produced

    failed = [outcome for outcome in outcomes if outcome.status == "failed"]
    print(f"\n{len(outcomes)} step(s), {len(failed)} failed")
    return 1 if failed else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description=(
            "Download source data, rebuild it as LeRobot, and publish the result. "
            "What a dataset is comes from dataset_registry, how it is processed "
            "from its profile, and where it lives from the environment."
        ),
    )
    parser.add_argument(
        "command",
        choices=("run", *STEPS, "status", "timings"),
        help=("run does every stage; the stage names do one; status and timings "
              "only report"),
    )
    parser.add_argument("--env", required=True, help="an environment name or path")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="a dataset in the registry; repeatable. Default: all of them",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="do not delete a source once it is built, or output once published",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would happen; transfer nothing and record nothing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=("timings only: the raw step records, for aggregating several nodes' "
              "runs into one table"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
