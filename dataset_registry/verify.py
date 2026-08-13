"""Check a dataset spec against a real LeRobot dataset.

The layouts in ``datasets/*.yaml`` were recovered from delivered data rather than
read from a spec, so they are claims, not facts. This re-derives the claim: it reads
episodes from a converted dataset and confirms that every slot really does hold the
source column the spec says it does.

    uv run python -m dataset_registry.verify action_net /path/to/action_net
    uv run python -m dataset_registry.verify --all          # against delivered.path

Run it after rebuilding a dataset to prove the rebuild matches, and run it against
the delivered copy to prove the spec still describes it.
"""

import argparse
import sys
from pathlib import Path

from .schema import DatasetSpec, StateSpec, available, load

TOLERANCE = 1e-6


class VerificationError(RuntimeError):
    pass


def episode_frames(root: Path, limit: int):
    """Read a spread of episode parquet files as one table."""
    import pandas as pd

    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise VerificationError(f"no episode parquet files under {root}")
    step = max(1, len(files) // limit)
    chosen = files[::step][:limit]
    return pd.concat([pd.read_parquet(p) for p in chosen], ignore_index=True), chosen


def columns(df, name: str):
    import numpy as np

    if name not in df.columns:
        raise VerificationError(f"{name!r} is not in the dataset")
    return np.stack([np.atleast_1d(v) for v in df[name].values]).astype(np.float64)


def check_state(spec: StateSpec, df, side: str) -> list[str]:
    """Return a list of problems; empty means the spec holds."""
    import numpy as np

    flat = columns(df, "observation.state" if side == "state" else "action")
    problems: list[str] = []

    if flat.shape[1] != spec.width:
        problems.append(
            f"{side}: dataset is {flat.shape[1]} wide, spec says {spec.width}"
        )
        return problems

    cache: dict[str, "np.ndarray"] = {}
    for block in spec.blocks:
        if block.evidence == "constant":
            # the one evidence value that is itself a claim about the numbers
            values = flat[:, block.start : block.end]
            if not np.allclose(values, values[0], atol=1e-9):
                problems.append(
                    f"{side}.{block.name}: spec says constant, but slots "
                    f"{block.start}-{block.end - 1} vary"
                )
            continue
        if block.is_constant:
            # declared / inferred / unknown: nothing in the data to check against
            continue

        path = spec.source_features[block.feature][side]
        if path not in df.columns:
            # sources are often only carried for the observation side
            continue
        if path not in cache:
            cache[path] = columns(df, path)
        source = cache[path]

        for offset in range(block.sourced_width):
            got = flat[:, block.start + offset]
            want = source[:, block.src_start + offset]
            if not np.allclose(got, want, atol=TOLERANCE, equal_nan=True):
                problems.append(
                    f"{side}.{block.name}: slot {block.start + offset} != "
                    f"{path}[{block.src_start + offset}]"
                )
    return problems


def verify(spec: DatasetSpec, root: Path, episodes: int = 4) -> list[str]:
    if spec.state is None:
        return [f"{spec.id}: spec has no state layout to check"]
    df, files = episode_frames(root, episodes)
    print(f"  read {len(df)} rows from {len(files)} episode(s)")

    problems: list[str] = []
    for side in ("state", "action"):
        vector = spec.vector(side)
        if vector is None:
            continue
        print(f"  {side}: width {vector.width}, slots by evidence "
              f"{vector.evidence_counts()}")
        problems += check_state(vector, df, side)
    return problems


def check_upstream(spec: DatasetSpec) -> list[str]:
    """Confirm the pinned revision is still what the upstream repo reports.

    The upstream release is the dataset's identity, and foundry lays its mirrors out
    under that revision -- so this one online call re-checks the whole mapping
    without reading a single byte of any mirror.
    """
    import json
    import urllib.request

    if not spec.huggingface:
        return []
    url = f"https://huggingface.co/api/datasets/{spec.huggingface}"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            head = json.load(response).get("sha")
    except Exception as exc:  # noqa: BLE001 - network problems are not spec problems
        return [f"could not reach {url}: {exc}"]

    if spec.revision and head != spec.revision:
        return [
            f"upstream has moved: {spec.huggingface} is now at {head}, "
            f"spec pins {spec.revision} (fine if intentional -- the mirror still "
            "holds the pinned release)"
        ]
    for mirror in spec.mirrors:
        uri = mirror.get("uri") or ""
        if mirror.get("kind") == "foundry" and spec.revision not in uri:
            return [f"foundry mirror {uri} does not match revision {spec.revision}"]
    return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help=f"one of: {', '.join(available())}")
    parser.add_argument("root", nargs="?", type=Path, help="a converted LeRobot dataset")
    parser.add_argument(
        "--all", action="store_true", help="verify every spec against delivered.path"
    )
    parser.add_argument(
        "--upstream",
        action="store_true",
        help="only re-check the pinned revisions online; reads no dataset",
    )
    parser.add_argument("--episodes", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.all and not args.dataset:
        print("give a dataset name, or --all", file=sys.stderr)
        return 2

    if args.upstream:
        names = available() if args.all or not args.dataset else [args.dataset]
        stale = 0
        for name in names:
            spec = load(name)
            problems = check_upstream(spec)
            label = spec.huggingface or "(no upstream)"
            print(f"{spec.id:24s} {label:46s} {'OK' if not problems else ''}")
            for problem in problems:
                stale += 1
                print(f"  {problem}")
        return 1 if stale else 0

    targets = []
    if args.all:
        for name in available():
            spec = load(name)
            if spec.delivered_path:
                targets.append((spec, Path(spec.delivered_path)))
            else:
                print(f"{name}: no delivered.path, skipping")
    else:
        spec = load(args.dataset)
        root = args.root or (Path(spec.delivered_path) if spec.delivered_path else None)
        if root is None:
            print(f"{spec.id}: no root given and no delivered.path", file=sys.stderr)
            return 2
        targets.append((spec, root))

    failed = 0
    for spec, root in targets:
        print(f"\n=== {spec.id}  <-  {root}")
        if not root.exists():
            print("  unreachable, skipping")
            continue
        try:
            problems = verify(spec, root, args.episodes)
        except (VerificationError, KeyError) as exc:
            print(f"  ERROR {exc}")
            failed += 1
            continue
        if problems:
            failed += 1
            for problem in problems[:20]:
                print(f"  MISMATCH {problem}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
        else:
            print("  OK: every non-constant slot matches its declared source")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
