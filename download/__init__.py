"""Download everything under an S3 prefix, as fast as the machine allows.

    from download import download
    download("s3://bucket/prefix/", "/scratch/data")

One CRT client drives every transfer: it owns the connection pool and the
bandwidth budget, which separate processes cannot share. Concurrency, bandwidth
target and part size are derived from the instance's NIC rate; the numbers behind
those defaults are in README.md.
"""

import os
import re
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Result", "download", "plan", "split_uri"]

# One transfer sustains roughly this much (Gbps), so concurrency = target / this.
_GBPS_PER_TRANSFER = 3.0
# Above this, transfers degrade and then start failing outright.
_MAX_CONCURRENCY = 32
# Small parts run out of request budget before they run out of bandwidth
# (8MB needs ~9,400 GET/s at 600 Gbps; the per-prefix guidance is ~5,500/s).
_PART_SIZE = 64 * 1024 * 1024
# Used only when the NIC rate is unknown. Low on purpose: overstating the target
# is slower than understating it.
_FALLBACK_GBPS = 25.0

_NIC_GBPS: dict[str, float] = {
    "c6id.16xlarge": 25.0, "c6id.32xlarge": 50.0,
    "c7gd.2xlarge": 15.0, "c7gn.16xlarge": 200.0,
    "c8g.48xlarge": 50.0, "c8gd.48xlarge": 50.0,
    "c8gn.48xlarge": 600.0, "c8in.96xlarge": 600.0,
    "c9gd.48xlarge": 100.0,
    "i7ie.48xlarge": 100.0, "i8g.48xlarge": 100.0,
    "m8idn.96xlarge": 600.0, "r8idn.96xlarge": 600.0,
}


@dataclass(frozen=True)
class Result:
    objects: int
    skipped: int
    bytes: int
    seconds: float
    concurrency: int
    target_gbps: float

    @property
    def gbps(self) -> float:
        return self.bytes * 8 / self.seconds / 1e9 if self.seconds > 0 else 0.0

    @property
    def gigabytes_per_s(self) -> float:
        return self.bytes / self.seconds / 1e9 if self.seconds > 0 else 0.0

    def __str__(self) -> str:
        skipped = f", {self.skipped} already present" if self.skipped else ""
        return (
            f"{self.objects} object(s){skipped}, {self.bytes / 1e12:.3f} TB in "
            f"{self.seconds:.1f}s — {self.gbps:,.1f} Gbps "
            f"({self.gigabytes_per_s:,.2f} GB/s, {self.concurrency} concurrent)"
        )


def split_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/a/b/`` -> ``("bucket", "a/b/")``"""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, prefix


def nic_gbps() -> float | None:
    """The NIC rate, from the link speed or the instance type. Needs no IAM."""
    speed = _read_link_speed()
    if speed:
        return speed
    return _NIC_GBPS.get(_instance_type() or "")


def plan(
    sizes: Iterable[tuple[str, int]],
    dest: str | Path,
    target_gbps: float | None = None,
    concurrency: int | None = None,
) -> tuple[list[tuple[str, Path, int]], list[str], float, int]:
    """Return ``(todo, skipped, target_gbps, concurrency)``.

    Largest objects first, so the run does not end waiting on one straggler.
    Objects already present at the right size are skipped; a size mismatch is
    refetched, because a truncated file is worse than a missing one.
    """
    root = Path(dest)
    todo: list[tuple[str, Path, int]] = []
    skipped: list[str] = []
    for key, size in sorted(sizes, key=lambda item: -item[1]):
        path = root / key
        if path.exists() and path.stat().st_size == size:
            skipped.append(key)
        else:
            todo.append((key, path, size))

    target = _resolve_target(target_gbps)
    if concurrency is None:
        concurrency = min(
            max(1, round(target / _GBPS_PER_TRANSFER)), _MAX_CONCURRENCY, max(1, len(todo))
        )
    return todo, skipped, target, max(1, concurrency)


def download(
    uri: str,
    dest: str | Path,
    target_gbps: float | None = None,
    concurrency: int | None = None,
) -> Result:
    """Download every object under ``uri`` into ``dest``."""
    bucket, prefix = split_uri(uri)
    todo, skipped, target, workers = plan(
        _list(bucket, prefix), dest, target_gbps, concurrency
    )
    if not todo:
        return Result(0, len(skipped), 0, 0.0, workers, target)

    started = time.perf_counter()
    _run(bucket, todo, target, workers)
    return Result(
        objects=len(todo),
        skipped=len(skipped),
        bytes=sum(size for _, _, size in todo),
        seconds=time.perf_counter() - started,
        concurrency=workers,
        target_gbps=target,
    )


# --- internals ---------------------------------------------------------------


def _resolve_target(requested: float | None) -> float:
    detected = nic_gbps()
    if detected is None:
        if requested is None:
            warnings.warn(
                f"could not determine the NIC rate; using {_FALLBACK_GBPS:g} Gbps, "
                "which will under-drive a faster NIC. Pass target_gbps to fix.",
                UserWarning,
                stacklevel=3,
            )
            return _FALLBACK_GBPS
        return float(requested)
    if requested is None:
        return float(detected)
    # asking for more than the NIC has makes transfers slower, not faster
    return float(min(requested, detected))


def _list(bucket: str, prefix: str) -> list[tuple[str, int]]:
    import boto3

    pages = boto3.client("s3").get_paginator("list_objects_v2")
    return [
        (item["Key"], item["Size"])
        for page in pages.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
        if item["Size"]
    ]


def _run(bucket, todo, target_gbps, workers) -> None:
    from awscrt.auth import AwsCredentialsProvider
    from awscrt.http import HttpHeaders, HttpRequest
    from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
    from awscrt.s3 import S3Client, S3RequestType

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    elg = EventLoopGroup()
    bootstrap = ClientBootstrap(elg, DefaultHostResolver(elg))
    client = S3Client(
        bootstrap=bootstrap,
        region=region,
        credential_provider=AwsCredentialsProvider.new_default_chain(bootstrap),
        throughput_target_gbps=target_gbps,
        part_size=_PART_SIZE,
    )
    # regional endpoint: the global one redirects outside us-east-1
    headers = HttpHeaders([("host", f"{bucket}.s3.{region}.amazonaws.com")])

    in_flight: list = []
    for key, path, _ in todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        in_flight.append(
            client.make_request(
                request=HttpRequest("GET", f"/{key}", headers),
                type=S3RequestType.GET_OBJECT,
                recv_filepath=str(path),
            ).finished_future
        )
        while len(in_flight) >= workers:
            in_flight.pop(0).result()
    for future in in_flight:
        future.result()


def _read_link_speed() -> float | None:
    """/sys/class/net/<nic>/speed, in Gbps. ENA often reports a sentinel."""
    nic = _default_nic()
    if not nic:
        return None
    try:
        gbps = float(Path(f"/sys/class/net/{nic}/speed").read_text().strip()) / 1000
    except (OSError, ValueError):
        return None
    return gbps if 0 < gbps <= 1600 else None


def _default_nic() -> str | None:
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 1 and fields[1] == "00000000":
                return fields[0]
    except OSError:
        return None
    return None


def _instance_type() -> str | None:
    try:
        import urllib.request

        token = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            ),
            timeout=2,
        ).read().decode()
        return urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=2,
        ).read().decode().strip()
    except Exception:  # noqa: BLE001 - not on EC2, or IMDS disabled
        return None


def parse_gigabit(text: str) -> float | None:
    """``"Up to 12.5 Gigabit"`` -> 12.5. Kept for adding rows to _NIC_GBPS."""
    match = re.search(r"([\d.]+)\s*Gigabit", text or "")
    return float(match.group(1)) if match else None
