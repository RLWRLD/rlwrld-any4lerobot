"""Download stage: pull source data from S3 as fast as the machine allows.

The pipeline is download -> preprocess (`lerobot_pipeline/`) -> upload. This
package owns the first step. See README.md for the measurements behind its
defaults, and `python3 -m download --help` for the CLI.
"""

from .nic import default_nic, detect_nic_gbps, nic_gbps_from_sysfs, nic_gbps_from_table
from .plan import (
    DEFAULT_PART_SIZE,
    FALLBACK_TARGET_GBPS,
    GBPS_PER_TRANSFER,
    MAX_CONCURRENCY,
    FetchPlan,
    FetchSummary,
    FetchTask,
    RemoteObject,
    plan_fetch,
    resolve_concurrency,
    resolve_target_gbps,
    summarize_fetch,
)
from .s3 import crt_request_kwargs, execute_plan, fetch, list_objects

__all__ = [
    "DEFAULT_PART_SIZE", "FALLBACK_TARGET_GBPS", "GBPS_PER_TRANSFER",
    "MAX_CONCURRENCY", "FetchPlan", "FetchSummary", "FetchTask", "RemoteObject",
    "crt_request_kwargs", "default_nic", "detect_nic_gbps", "execute_plan",
    "fetch", "list_objects", "nic_gbps_from_sysfs", "nic_gbps_from_table",
    "plan_fetch", "resolve_concurrency", "resolve_target_gbps", "summarize_fetch",
]
