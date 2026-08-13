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
        choices=("run", *STEPS, "status"),
        help="run does every stage; the stage names do one; status only reports",
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
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
