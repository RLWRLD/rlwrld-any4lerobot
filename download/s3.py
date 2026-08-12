"""Listing and execution against S3.

One client coordinates every transfer. That is the whole design: a single CRT
client owns the connection pool and the bandwidth budget, which independent
processes cannot share. Forking ``aws s3 cp`` per object reached only 89 Gbps on a
600 Gbps NIC and lost 433 of 604 objects at 128 processes.
"""

import os
import time
import warnings
from pathlib import Path

from .nic import default_nic, detect_nic_gbps
from .plan import (
    DEFAULT_PART_SIZE,
    FetchPlan,
    FetchSummary,
    FetchTask,
    RemoteObject,
    plan_fetch,
    summarize_fetch,
)


def list_objects(
    bucket: str, prefix: str, suffix: str | None = None
) -> list[RemoteObject]:
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


def crt_request_kwargs(task: FetchTask) -> dict:
    """Per-request kwargs for CRT.

    Discard mode writes to ``os.devnull`` via ``recv_filepath`` rather than an
    ``on_body`` callback. The callback crossed into Python for every body chunk and
    measured *slower* than letting CRT write a real file in C -- which made the
    "network only" baseline slower than the disk path it was meant to isolate.
    """
    return {"recv_filepath": os.devnull if task.dest is None else str(task.dest)}


def execute_plan(
    plan: FetchPlan, bucket: str, nic_gbps: float | None = None
) -> FetchSummary:
    """Run a plan and report what it achieved."""
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


def _crt_runner(plan: FetchPlan):
    """The fast path: aws-crt's S3 client, the same engine behind the AWS CLI's
    ``crt`` transfer client (~10x the stock one). None if awscrt is missing."""
    try:
        from awscrt.auth import AwsCredentialsProvider
        from awscrt.http import HttpHeaders, HttpRequest
        from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
        from awscrt.s3 import S3Client, S3RequestType
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
            # bound how many requests are outstanding at once
            while len(in_flight) >= plan.concurrency:
                in_flight.pop(0).result()

        for future in in_flight:
            future.result()

    return run


def _boto3_runner(plan: FetchPlan):
    """Fallback when awscrt is missing. Threaded boto3 is CPU-bound on TLS inside
    one process -- the stock CLI path measured ~3% of a large NIC -- so warn."""
    warnings.warn(
        "awscrt is not installed; falling back to threaded boto3, which measured "
        "at a fraction of CRT throughput. Install the `s3` extra for the fast path.",
        UserWarning,
        stacklevel=2,
    )

    def run(bucket: str) -> None:
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        import boto3

        client = boto3.client("s3")

        def fetch_one(task: FetchTask) -> None:
            body = client.get_object(Bucket=bucket, Key=task.obj.key)["Body"]
            if task.dest is None:
                with open(os.devnull, "wb") as sink:
                    shutil.copyfileobj(body, sink, length=8 * 1024 * 1024)
                return
            task.dest.parent.mkdir(parents=True, exist_ok=True)
            with open(task.dest, "wb") as handle:
                shutil.copyfileobj(body, handle, length=8 * 1024 * 1024)

        with ThreadPoolExecutor(max_workers=plan.concurrency) as pool:
            for future in [pool.submit(fetch_one, t) for t in plan.tasks]:
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
    nic = detect_nic_gbps(nic=default_nic())
    plan = plan_fetch(
        list_objects(bucket, prefix, suffix),
        dest_dir=dest_dir,
        target_gbps=target_gbps,
        nic_gbps=nic,
        concurrency=concurrency,
        resume=resume,
        part_size=part_size,
    )
    return execute_plan(plan, bucket=bucket, nic_gbps=nic)
