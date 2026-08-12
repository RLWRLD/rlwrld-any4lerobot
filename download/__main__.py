"""python3 -m download s3://bucket/prefix/ /scratch/data"""

import argparse
import sys

from . import download, nic_gbps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="download", description="Download everything under an S3 prefix."
    )
    parser.add_argument("source", help="s3://bucket/prefix/")
    parser.add_argument("dest", help="directory to write into")
    parser.add_argument(
        "--concurrency", type=int, default=None, help="override the derived value"
    )
    parser.add_argument(
        "--target-gbps", type=float, default=None,
        help="override the detected NIC rate (never set above it)",
    )
    args = parser.parse_args(argv)

    detected = nic_gbps()
    print(f"NIC: {f'{detected:g} Gbps' if detected else 'unknown'}")
    try:
        result = download(args.source, args.dest, args.target_gbps, args.concurrency)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
