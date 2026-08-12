"""Determine the instance's NIC rate, preferring sources that need no permissions.

This module exists on its own because getting it wrong is silent and expensive.
On c9gd.48xlarge (100 Gbps) the instance role lacked ``ec2:DescribeInstanceTypes``,
detection fell back to 25 Gbps, that produced concurrency 8 instead of 32, and the
resulting 2.9 GB/s was reported as though it meant something. The measured rate on
the same machine with the right settings was 7.8 GB/s.

So: never depend on a single source, and never fall back quietly.
"""

import re
import warnings
from pathlib import Path

from .plan import FALLBACK_TARGET_GBPS

# NIC rate by instance type, for when the EC2 API is unreachable. Deliberately
# incomplete -- an unknown type returns None and the caller warns, rather than
# guessing. `--target-gbps` is the escape hatch, and adding a row here is cheap.
NIC_GBPS_BY_TYPE: dict[str, float] = {
    "c6id.16xlarge": 25.0,
    "c6id.32xlarge": 50.0,
    "c7gd.2xlarge": 15.0,
    "c7gn.16xlarge": 200.0,
    "c8g.48xlarge": 50.0,
    "c8gd.48xlarge": 50.0,
    "c8gn.48xlarge": 600.0,
    "c8in.96xlarge": 600.0,
    "c9gd.48xlarge": 100.0,
    "i7ie.48xlarge": 100.0,
    "i8g.48xlarge": 100.0,
    "m8idn.96xlarge": 600.0,
    "r8idn.96xlarge": 600.0,
}

# Linux reports link speed in Mbps. ENA often reports a sentinel (-1, 0, or
# 2^32-1) instead of a real rate, so implausible values are discarded.
_MAX_PLAUSIBLE_GBPS = 1600.0


def nic_gbps_from_sysfs(nic: str | None = None, read=None) -> float | None:
    """Read ``/sys/class/net/<nic>/speed``. Needs no credentials and no IAM."""
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


def parse_network_performance(text: str) -> float | None:
    """``"Up to 12.5 Gigabit"`` / ``"100 Gigabit"`` -> Gbps.

    Note that "Up to" means burstable: a long transfer can fall back to a lower
    baseline. Instances at 16xlarge and above generally quote a sustained rate.
    """
    match = re.search(r"([\d.]+)\s*Gigabit", text or "")
    return float(match.group(1)) if match else None


def detect_nic_gbps(
    instance_type: str | None = None,
    nic: str | None = None,
    sysfs=None,
    api=None,
) -> float | None:
    """Best-effort NIC rate: sysfs -> static table -> EC2 API.

    Returns None *and warns* when nothing answers, so an under-driven transfer is
    visible rather than being mistaken for the machine's ceiling.
    """
    from_sysfs = (sysfs or (lambda: nic_gbps_from_sysfs(nic)))()
    if from_sysfs:
        return from_sysfs

    instance_type = instance_type or imds_instance_type()
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


def _api_nic_gbps(instance_type: str) -> float | None:
    """Ask EC2. Requires ``ec2:DescribeInstanceTypes``, which many instance roles
    lack -- hence the last resort, never the only path."""
    try:
        import boto3

        info = boto3.client("ec2").describe_instance_types(
            InstanceTypes=[instance_type]
        )["InstanceTypes"][0]
        return parse_network_performance(info["NetworkInfo"]["NetworkPerformance"])
    except Exception:  # noqa: BLE001 - detection is best-effort by design
        return None


def imds_instance_type() -> str | None:
    """Instance type from IMDSv2. No credentials, no IAM, no boto3."""
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


def default_nic() -> str | None:
    """The interface carrying the default route, for the sysfs lookup."""
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 1 and fields[1] == "00000000":
                return fields[0]
    except OSError:
        return None
    return None
