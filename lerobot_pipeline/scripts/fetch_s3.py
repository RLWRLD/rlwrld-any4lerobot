"""Fetch S3 objects fast, and diagnose why a transfer is slow.

    # measure the network alone (nothing touches the disk)
    python3 -m lerobot_pipeline.scripts.fetch_s3 s3://bucket/prefix/ --discard --sample 8

    # is the disk the ceiling, or the network? runs both and reports the gap
    python3 -m lerobot_pipeline.scripts.fetch_s3 s3://bucket/prefix/ \
        --dest /scratch/data --diagnose --sample 8

    # actually fetch
    python3 -m lerobot_pipeline.scripts.fetch_s3 s3://bucket/prefix/ --dest /scratch/data

Defaults come from measurements on this project's data -- see `lerobot_pipeline/fetch.py`.
"""

import argparse
import json
import sys
from pathlib import Path

from ..fetch import (
    detect_nic_gbps,
    execute_plan,
    list_objects,
    plan_fetch,
)


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, prefix


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot_pipeline.scripts.fetch_s3",
        description="Fetch S3 objects with one coordinated client, or diagnose a slow transfer.",
    )
    parser.add_argument("source", help="s3://bucket/prefix/")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--dest", type=Path, default=None, help="directory to write into")
    where.add_argument(
        "--discard",
        action="store_true",
        help="stream to nowhere: measures the network with the disk taken out of the path",
    )
    parser.add_argument("--suffix", default=None, help="only keys ending with this")
    parser.add_argument("--sample", type=int, default=0, help="use at most N objects (0 = all)")
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="concurrent transfers (default: derived from the NIC rate, capped at 32)",
    )
    parser.add_argument(
        "--target-gbps", type=float, default=None,
        help="bandwidth target (default: the detected NIC rate; never set above it)",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument(
        "--diagnose", action="store_true",
        help="fetch the sample twice -- to disk and to nowhere -- and report which limits you",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bucket, prefix = split_s3_uri(args.source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    nic = detect_nic_gbps()
    print(f"NIC rate: {f'{nic:g} Gbps' if nic else 'unknown (off EC2?)'}")

    objects = list_objects(bucket, prefix, args.suffix)
    if not objects:
        raise SystemExit(f"no objects under {args.source}")
    if args.sample:
        objects = sorted(objects, key=lambda o: -o.size)[: args.sample]
    total = sum(o.size for o in objects)
    print(f"{len(objects)} object(s), {total / 1e12:.3f} TB")

    def run(dest, label):
        plan = plan_fetch(
            objects, dest_dir=dest, target_gbps=args.target_gbps, nic_gbps=nic,
            concurrency=args.concurrency, resume=args.resume,
        )
        if plan.skipped:
            print(f"  ({len(plan.skipped)} already complete, skipped)")
        print(f"-- {label}: {plan.concurrency} concurrent, target {plan.target_gbps:g} Gbps")
        summary = execute_plan(plan, bucket=bucket, nic_gbps=nic)
        print(summary.format())
        return summary

    payload = {"nic_gbps": nic, "objects": len(objects), "total_bytes": total}

    if args.diagnose:
        if args.dest is None:
            raise SystemExit("--diagnose needs --dest so it can compare against the disk")
        net = run(None, "streaming to nowhere (network only)")
        disk = run(args.dest, "writing to disk")
        payload["network_only"] = net.as_dict() if hasattr(net, "as_dict") else vars(net)
        payload["to_disk"] = vars(disk)

        print("\n=== verdict ===")
        if net.gbps <= 0 or disk.gbps <= 0:
            print("could not measure both paths")
        else:
            penalty = (1 - disk.gbps / net.gbps) * 100
            print(f"disk costs {penalty:.0f}% ({net.gbps:,.1f} -> {disk.gbps:,.1f} Gbps)")
            if penalty > 15:
                print(
                    "  => the DISK is limiting you, not the network.\n"
                    "     Stream into your consumer instead of landing files:\n"
                    f"       aws s3 cp s3://{bucket}/<key> - | tar -x ..."
                )
            elif nic and net.gbps < nic * 0.6:
                print(
                    "  => neither the disk nor the NIC is the limit; the client is.\n"
                    "     Raise --concurrency (up to 32) and make sure --target-gbps\n"
                    "     equals the NIC rate, never more."
                )
            else:
                print("  => close to the NIC rate; little left to win here.")
        payload["verdict_penalty_pct"] = round((1 - disk.gbps / net.gbps) * 100, 1) if net.gbps else None
    else:
        dest = None if args.discard else args.dest
        summary = run(dest, "streaming to nowhere" if dest is None else "writing to disk")
        payload["result"] = vars(summary)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
