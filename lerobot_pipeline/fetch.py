"""Fast S3 -> local fetch, tuned from measurements rather than guesses.

Every default here comes from a benchmark on this project's own data
(``s3://rlwrld-foundry-data/external/action_net/``, us-east-1):

* one transfer client peaks near 14 Gbps regardless of instance size, so
  concurrency is mandatory -- but it must live *inside one process*. Forking
  ``aws s3 cp`` per object was measured at 89 Gbps on a 600 Gbps NIC and started
  losing objects outright past 64 processes (128 lost 433 of 604).
* declaring a bandwidth target above the real NIC rate makes transfers *slower*
  (41.41 -> 37.42 Gbps when a 50 Gbps NIC was told 100 Gbps).
* landing bytes on local NVMe costs 30-43% versus streaming, and even 4 striped
  NVMe devices capped near 4.25 GB/s. Disk is usually the real ceiling.
* the stock CLI defaults reach ~3% of a large NIC; the CRT client is ~10x faster.
"""

import os
import re
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Past this many concurrent transfers the client degraded and then failed
# outright in testing. Treat it as a safety rail, not a tuning knob.
MAX_CONCURRENCY = 32

# Roughly the throughput one concurrent transfer sustained (Gbps), measured
# consistently across x86 and Graviton and across 25/50/600 Gbps NICs.
GBPS_PER_TRANSFER = 3.0

# Used when the NIC rate cannot be determined. Deliberately low: overstating the
# target is measurably worse than understating it.
FALLBACK_TARGET_GBPS = 25

# Multipart part size. Small parts cap the achievable rate on request count:
# 8MB parts need ~9,400 GET/s to sustain 600 Gbps, above the ~5,500/s per-prefix
# guidance, and this dataset keeps every object under one prefix. 64MB needs
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
        return float(requested) if requested is not None else float(FALLBACK_TARGET_GBPS)
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
    """Decide what to fetch, in what order, and how many at once."""
    objects = list(objects)
    discard = dest_dir is None
    root = None if discard else Path(dest_dir)

    tasks: list[FetchTask] = []
    skipped: list[RemoteObject] = []
    for obj in sorted(objects, key=lambda o: -o.size):
        dest = None if root is None else root / obj.key
        # a partially written file is worse than a missing one: only an exact
        # size match counts as done
        if resume and dest is not None and dest.exists() and dest.stat().st_size == obj.size:
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


# --- environment ------------------------------------------------------------


# NIC rate by instance type, for when the EC2 API is unreachable. Deliberately
# incomplete: an unknown type returns None and the caller warns, rather than
# guessing. `--target-gbps` is the escape hatch.
NIC_GBPS_BY_TYPE: dict[str, float] = {
    "c6id.16xlarge": 25.0, "c6id.32xlarge": 50.0,
    "c7gd.2xlarge": 15.0, "c7gn.16xlarge": 200.0,
    "c8g.48xlarge": 50.0, "c8gd.48xlarge": 50.0, "c8gn.48xlarge": 600.0,
    "c9gd.48xlarge": 100.0,
    "i7ie.48xlarge": 100.0, "i8g.48xlarge": 100.0,
    "m8idn.96xlarge": 600.0, "r8idn.96xlarge": 600.0,
    "c8in.96xlarge": 600.0,
}

# Linux reports link speed in Mbps. ENA often reports a sentinel instead of a
# real rate, so implausible values are discarded rather than believed.
_MAX_PLAUSIBLE_GBPS = 1600.0


def nic_gbps_from_sysfs(nic: str | None = None, read=None) -> float | None:
    """Read /sys/class/net/<nic>/speed. Needs no credentials and no IAM."""
    if read is None:
        if not nic:
            return None

        def read() -> str:
            return Path(f"/sys/class/net/{nic}/speed").read_text()

    try:
        mbps = float(str(read()).strip())
    except (OSError, ValueError, TypeError):
        return None
    gbps = mbps / 1000.0
    if gbps <= 0 or gbps > _MAX_PLAUSIBLE_GBPS:
        return None
    return gbps


def nic_gbps_from_table(instance_type: str | None) -> float | None:
    return NIC_GBPS_BY_TYPE.get(instance_type or "")


def _api_nic_gbps(instance_type: str) -> float | None:
    """Ask EC2. Requires ec2:DescribeInstanceTypes, which many instance roles
    lack -- so this is the last resort, never the only path."""
    try:
        import boto3

        info = boto3.client("ec2").describe_instance_types(
            InstanceTypes=[instance_type]
        )["InstanceTypes"][0]
        return parse_network_performance(info["NetworkInfo"]["NetworkPerformance"])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return None


def detect_nic_gbps(
    instance_type: str | None = None,
    nic: str | None = None,
    sysfs=None,
    api=None,
) -> float | None:
    """Determine the NIC rate, preferring sources that need no permissions.

    Order: sysfs link speed -> static table -> EC2 API. Returns None *and warns*
    when nothing answers: a silent fallback once configured a 100 Gbps instance
    for 25 Gbps and reported the resulting 2.9 GB/s as if it meant something.
    """
    from_sysfs = (sysfs or (lambda: nic_gbps_from_sysfs(nic)))()
    if from_sysfs:
        return from_sysfs

    instance_type = instance_type or _imds_instance_type()
    from_table = nic_gbps_from_table(instance_type)
    if from_table:
        return from_table

    if instance_type:
        from_api = (api or _api_nic_gbps)(instance_type)
        if from_api:
            return from_api

    warnings.warn(
        "could not determine the NIC rate"
        + (f" for {instance_type}" if instance_type else " (not on EC2?)")
        + f"; falling back to {FALLBACK_TARGET_GBPS} Gbps, which will under-drive a "
        "faster NIC. Pass --target-gbps explicitly to avoid guessing.",
        UserWarning,
        stacklevel=2,
    )
    return None


def parse_network_performance(text: str) -> float | None:
    """``"Up to 12.5 Gigabit"`` / ``"100 Gigabit"`` -> Gbps."""
    match = re.search(r"([\d.]+)\s*Gigabit", text or "")
    return float(match.group(1)) if match else None


def _imds_instance_type() -> str | None:
    try:
        import urllib.request

        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=2).read().decode().strip()
    except Exception:  # noqa: BLE001 - not on EC2, or IMDS disabled
        return None


# --- listing ----------------------------------------------------------------


def list_objects(bucket: str, prefix: str, suffix: str | None = None) -> list[RemoteObject]:
    import boto3

    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    found: list[RemoteObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if item["Size"] == 0:
                continue
            if suffix and not item["Key"].endswith(suffix):
                continue
            found.append(RemoteObject(key=item["Key"], size=item["Size"]))
    return found


# --- execution --------------------------------------------------------------


def execute_plan(plan: FetchPlan, bucket: str, nic_gbps: float | None = None) -> FetchSummary:
    """Run a plan with one CRT client coordinating every transfer.

    A single client is the point: it owns the connection pool and the bandwidth
    budget, which independent processes cannot share. Forking one CLI per object
    was measured far slower and unstable at high concurrency.
    """
    if not plan.tasks:
        return summarize_fetch(0, 0.0, plan.concurrency, 0, nic_gbps)

    runner = _crt_runner(plan) or _boto3_runner(plan)
    started = time.perf_counter()
    runner(bucket)
    seconds = time.perf_counter() - started

    return summarize_fetch(
        total_bytes=plan.total_bytes,
        seconds=seconds,
        concurrency=plan.concurrency,
        objects=len(plan.tasks),
        nic_gbps=nic_gbps,
    )


def crt_request_kwargs(task: FetchTask) -> dict:
    """Per-request kwargs for CRT.

    Discard mode writes to os.devnull via ``recv_filepath`` rather than an
    ``on_body`` callback: the callback crossed into Python for every chunk, which
    measured *slower* than letting CRT write a real file in C -- making the
    "network only" baseline slower than the disk path it was meant to isolate.
    """
    return {"recv_filepath": os.devnull if task.dest is None else str(task.dest)}


def _crt_runner(plan: FetchPlan):
    """The fast path: aws-crt's S3 client, which is what makes the CLI's `crt`
    transfer client ~10x the stock one. Returns None if awscrt is unavailable."""
    try:
        from awscrt.auth import AwsCredentialsProvider
        from awscrt.io import (
            ClientBootstrap,
            DefaultHostResolver,
            EventLoopGroup,
        )
        from awscrt.s3 import S3Client, S3RequestType
        from awscrt.http import HttpHeaders, HttpRequest
    except ImportError:
        return None

    def run(bucket: str) -> None:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        elg = EventLoopGroup()
        bootstrap = ClientBootstrap(elg, DefaultHostResolver(elg))
        client = S3Client(
            bootstrap=bootstrap,
            region=region,
            credential_provider=AwsCredentialsProvider.new_default_chain(bootstrap),
            throughput_target_gbps=plan.target_gbps,
            part_size=plan.part_size,
        )
        # regional endpoint: the global one redirects outside us-east-1
        host = f"{bucket}.s3.{region}.amazonaws.com"

        in_flight: list = []
        for task in plan.tasks:
            if task.dest is not None:
                task.dest.parent.mkdir(parents=True, exist_ok=True)
            request = HttpRequest(
                "GET", f"/{task.obj.key}", HttpHeaders([("host", host)])
            )
            in_flight.append(
                client.make_request(
                    request=request,
                    type=S3RequestType.GET_OBJECT,
                    **crt_request_kwargs(task),
                ).finished_future
            )
            # keep at most `concurrency` requests outstanding
            while len(in_flight) >= plan.concurrency:
                in_flight.pop(0).result()

        for future in in_flight:
            future.result()

    return run


def _boto3_runner(plan: FetchPlan):
    """Fallback when awscrt is missing. Threaded boto3 is CPU-bound on TLS in one
    process -- measured at roughly a fifth of the CRT path -- so warn loudly."""
    warnings.warn(
        "awscrt is not installed; falling back to threaded boto3, which was "
        "measured at a fraction of CRT throughput. Install `awscrt` for the fast path.",
        UserWarning,
        stacklevel=2,
    )

    def run(bucket: str) -> None:
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        import boto3

        client = boto3.client("s3")

        def fetch(task: FetchTask) -> None:
            body = client.get_object(Bucket=bucket, Key=task.obj.key)["Body"]
            if task.dest is None:
                shutil.copyfileobj(body, open(os.devnull, "wb"))
                return
            task.dest.parent.mkdir(parents=True, exist_ok=True)
            with open(task.dest, "wb") as handle:
                shutil.copyfileobj(body, handle, length=8 * 1024 * 1024)

        with ThreadPoolExecutor(max_workers=plan.concurrency) as pool:
            for future in [pool.submit(fetch, t) for t in plan.tasks]:
                future.result()

    return run


def fetch(
    bucket: str,
    prefix: str,
    dest_dir: str | Path | None,
    suffix: str | None = None,
    concurrency: int | None = None,
    target_gbps: float | None = None,
    resume: bool = True,
    part_size: int = DEFAULT_PART_SIZE,
) -> FetchSummary:
    """List, plan and fetch in one call."""
    nic = detect_nic_gbps()
    objects = list_objects(bucket, prefix, suffix)
    plan = plan_fetch(
        objects,
        dest_dir=dest_dir,
        target_gbps=target_gbps,
        nic_gbps=nic,
        concurrency=concurrency,
        resume=resume,
        part_size=part_size,
    )
    return execute_plan(plan, bucket=bucket, nic_gbps=nic)


def _objects_from_plan(plan: FetchPlan) -> Sequence[RemoteObject]:
    return [t.obj for t in plan.tasks]
