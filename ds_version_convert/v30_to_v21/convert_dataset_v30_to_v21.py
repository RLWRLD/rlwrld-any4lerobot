"""Utilities to convert a LeRobot dataset from codebase version v3.0 back to v2.1.

The script mirrors :mod:`lerobot.datasets.v21.convert_dataset_v21_to_v30` but applies the reverse
transformations so an existing dataset created with the new consolidated file
layout can be ported back to the legacy per-episode structure.

Usage examples
--------------

Convert a dataset that already exists locally::

    python convert_dataset_v30_to_v21.py \
        --repo-id=lerobot/pusht \
        --root=/path/to/dataset

The two steps that take the time -- splitting the consolidated video and the
consolidated parquet back into one file per episode -- are lists of independent
jobs and run on a pool of threads. ``--workers`` sizes it; see the README.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import tqdm
from datasets import Dataset
from huggingface_hub import snapshot_download
from lerobot.datasets.io_utils import (
    INFO_PATH,
    STATS_PATH,
    load_info,
    load_tasks,
    write_json,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    serialize_dict,
    unflatten_dict,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

# -1 asks the machine how many cores it has, rather than writing a count into
# the file. A count written here is a count for one machine.
DEFAULT_WORKERS = -1

LEGACY_DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
MIN_VIDEO_DURATION = 1e-6
# What v2.1 requires of a per-episode statistics entry. It is a floor, not a ceiling:
# the delivered RLDX-1 copies carry five quantiles beside these, so a v2.1 file with
# ten keys is what the collection actually looks like.
#
# These used to be a filter, and dropping the rest was a one-way door. Whole-dataset
# quantiles can be computed from per-episode ones; per-episode ones cannot be computed
# from anything but the data, so throwing them away here meant a full second pass over
# every frame to get them back -- 27 million of them for droid.
LEGACY_STATS_KEYS = ("mean", "std", "min", "max", "count")


def resolve_workers(workers: int) -> int:
    """How many jobs to keep in flight; ``-1`` means one per core."""

    if workers == -1:
        return os.cpu_count() or 1
    if workers < 1:
        raise ValueError(f"workers must be -1 or a positive integer, got {workers}")
    return workers


def work_through(
    jobs: list[Any],
    run: Callable[[Any], Any],
    workers: int,
    desc: str,
) -> None:
    """Apply ``run`` to every job, ``workers`` of them at a time.

    Threads rather than processes. Every job this is used for spends its time
    either inside an ffmpeg subprocess or inside pyarrow, and both release the
    GIL for the whole of it; threads then cost nothing to start, share the page
    cache over the file they are all reading, and need nothing pickled.

    The jobs must be independent -- each writing a path no other one writes --
    because nothing here orders them. ``video_segments`` and
    ``_group_episodes_by_data_file`` are what establish that.
    """

    if not jobs:
        return

    workers = min(resolve_workers(workers), len(jobs))
    if workers == 1:
        for job in tqdm.tqdm(jobs, desc=desc):
            run(job)
        return

    # map() re-raises on iteration, so a failed job still aborts the conversion.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm.tqdm(
            executor.map(run, jobs), total=len(jobs), desc=f"{desc} x{workers}"
        ):
            pass


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in ``meta/episodes``."""

    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks parquet to legacy JSONL")
    tasks = load_tasks(root)
    tasks = tasks.sort_values("task_index")

    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for task, row in tasks.iterrows():
            writer.write(
                {
                    "task_index": int(row["task_index"]),
                    "task": _to_serializable(task),
                }
            )


def read_info(root: Path) -> dict[str, Any]:
    """v3.0's ``info.json`` as a plain dict.

    ``load_info`` returns lerobot's ``DatasetInfo``, which models the v3.0 schema and
    only the v3.0 schema: it has no ``total_chunks`` or ``total_videos`` field, and
    assigning one raises. v2.1 needs both, so the conversion edits a dict and writes
    that, rather than trying to hold a v2.1 file in a v3.0 type.
    """
    return load_info(root).to_dict()


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    info = read_info(root)
    logging.info("Converting info.json metadata to v2.1 schema")

    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21

    # Restore legacy layout templates.
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None

    # Remove v3-specific sizing hints which do not exist in v2.1.
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Restore per-feature metadata: camera entries already contain their own fps.
    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = (
        math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    )
    info["total_videos"] = total_episodes * len(video_keys)

    write_json(info, new_root / INFO_PATH)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def _split_data_file(
    root: Path,
    new_root: Path,
    key: tuple[int, int],
    records: list[dict[str, Any]],
) -> None:
    """Cut one consolidated parquet file into its episodes.

    The unit of work is the file, not the episode: the whole table is read once
    and every episode in it is sliced out of that one read.
    """

    chunk_idx, file_idx = key
    source_path = root / DEFAULT_DATA_PATH.format(
        chunk_index=chunk_idx, file_index=file_idx
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

    table = pq.read_table(source_path)
    records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
    file_offset = int(records[0]["dataset_from_index"])

    for record in records:
        episode_index = int(record["episode_index"])
        start = int(record["dataset_from_index"]) - file_offset
        stop = int(record["dataset_to_index"]) - file_offset
        length = stop - start

        if length <= 0:
            raise ValueError(
                "Invalid episode length computed during data conversion: "
                f"episode_index={episode_index}, length={length}"
            )

        episode_table = table.slice(start, length)

        dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
        dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
            episode_chunk=dest_chunk,
            episode_index=episode_index,
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        Dataset(episode_table).to_parquet(dest_path)


def convert_data(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    workers: int = DEFAULT_WORKERS,
) -> None:
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)

    # One file in flight per worker, so peak memory is that many tables at once
    # rather than the whole dataset.
    work_through(
        list(grouped.items()),
        lambda item: _split_data_file(root, new_root, item[0], item[1]),
        workers,
        "convert data files",
    )


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _validate_video_paths(src: Path, dst: Path) -> None:
    """Validate source and destination paths to prevent security issues."""

    # Convert to Path objects if they aren't already
    src = Path(src)
    dst = Path(dst)

    # Resolve paths to handle symlinks and normalize them
    try:
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path provided: {exc}") from exc

    # Check that source file exists and is a regular file
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source video file does not exist: {src_resolved}")

    if not src_resolved.is_file():
        raise ValueError(f"Source path is not a regular file: {src_resolved}")

    # Validate file extensions for video files
    valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if src_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(
            f"Source file does not have a valid video extension: {src_resolved}"
        )

    if dst_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(
            f"Destination file does not have a valid video extension: {dst_resolved}"
        )

    # Check for path traversal attempts in the original paths
    src_str = str(src)
    dst_str = str(dst)

    # Ensure paths don't contain null bytes or other control characters
    for path_str, name in [(src_str, "source"), (dst_str, "destination")]:
        if "\0" in path_str:
            raise ValueError(f"Path contains null bytes: {name} path")
        if any(ord(c) < 32 and c not in ["\t", "\n", "\r"] for c in path_str):
            raise ValueError(f"Path contains invalid control characters: {name} path")

    # Additional check: ensure resolved paths don't point to system directories
    system_dirs = {"/etc", "/sys", "/proc", "/dev", "/boot", "/root"}
    for resolved_path, name in [
        (src_resolved, "source"),
        (dst_resolved, "destination"),
    ]:
        path_str = str(resolved_path)
        for sys_dir in system_dirs:
            if path_str.startswith(sys_dir + "/") or path_str == sys_dir:
                raise ValueError(
                    f"Path points to system directory: {name} path {resolved_path}"
                )

    # Ensure the destination directory can be created safely
    try:
        dst_parent = dst_resolved.parent
        if not dst_parent.exists():
            # Check if we can create the parent directory structure
            dst_parent.resolve()
    except OSError as exc:
        raise ValueError(f"Cannot create destination directory: {exc}") from exc


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    # Validate paths to prevent security issues
    _validate_video_paths(src, dst)

    # Validate numeric parameters to prevent injection
    if not (0 <= start <= 86400):  # 24 hours max
        raise ValueError(f"Invalid start time: {start}")
    if not (0 <= end <= 86400):  # 24 hours max
        raise ValueError(f"Invalid end time: {end}")
    if start >= end:
        raise ValueError(f"Start time {start} must be less than end time {end}")

    duration = max(end - start, MIN_VIDEO_DURATION)

    # Validate duration is reasonable
    if duration > 3600:  # 1 hour max
        raise ValueError(f"Video segment duration too long: {duration} seconds")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build command with validated parameters
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "debug",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]

    try:
        # Use more secure subprocess call with explicit timeout
        result = subprocess.run(
            cmd,
            check=True,
            timeout=300,  # 5 minute timeout
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out while processing video '{src}' -> '{dst}'"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found; it is required for video conversion"
        ) from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


class Segment(NamedTuple):
    """One episode's slice of one camera's concatenated file."""

    src: Path
    dst: Path
    start: float
    end: float


def video_segments(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> list[Segment]:
    """Every cut the video conversion has to make, as a flat list.

    One entry per episode per camera. They share their sources read-only and
    each writes a path of its own, so the list can be worked through in any
    order, or all at once.

    It is still ordered by position within each source file, which is the order
    a serial run used to read them in and keeps a parallel run's readers near
    each other rather than scattered across the file.
    """

    segments: list[Segment] = []
    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if len(grouped) == 0:
            logging.info("No video metadata found for key '%s'; skipping", video_key)
            continue

        for (chunk_idx, file_idx), records in grouped.items():
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")

            records = sorted(
                records,
                key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]),
            )

            for record in records:
                episode_index = int(record["episode_index"])
                dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                segments.append(
                    Segment(
                        src=src_path,
                        dst=dest_path,
                        start=float(record[f"videos/{video_key}/from_timestamp"]),
                        end=float(record[f"videos/{video_key}/to_timestamp"]),
                    )
                )

    return segments


def convert_videos(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    workers: int = DEFAULT_WORKERS,
) -> None:
    if len(video_keys) == 0:
        logging.info("No video features detected; skipping video conversion")
        return

    logging.info("Converting concatenated MP4 files back to per-episode videos")

    def cut(segment: Segment) -> None:
        _extract_video_segment(
            segment.src, segment.dst, start=segment.start, end=segment.end
        )

    work_through(
        video_segments(root, new_root, episode_records, video_keys),
        cut,
        workers,
        "convert videos",
    )


def aggregate_episode_stats(
    per_episode: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The whole dataset's statistics, from the per-episode ones. No pass over data.

    Deliberately not lerobot's ``aggregate_stats``. That one takes a conservative
    envelope for the quantiles -- the minimum of the lower ones and the maximum of the
    upper -- and says so in its own comment: "bounds across the inputs, not global
    quantile estimates". The delivered RLDX-1 copies used a count-weighted mean, and
    reproducing them is the point. Measured against cmu_stretch's own ``stats.json``,
    every feature and every statistic: **4.2e-09** this way against **1.2e+04** for the
    envelope.

    ``std`` is the one that cannot be averaged. How far a group spreads about its own
    mean says nothing about how far that mean sits from everyone else's, so the pooled
    variance needs both terms:

        var = sum(n_i * (std_i^2 + mean_i^2)) / N - mean^2

    Averaging the per-episode values instead put ``index`` out by 7,168 and
    ``observation.state`` by 0.08.

    Weighted by each feature's **own** count, not by the episode's row count: image
    statistics are taken from a hundred sampled frames while the vectors cover every
    row, so one weight for the whole episode is wrong for one of them.
    """
    import numpy as np

    features = {feature for entry in per_episode for feature in entry}
    out: dict[str, dict[str, Any]] = {}
    for feature in sorted(features):
        parts = [entry[feature] for entry in per_episode if feature in entry]
        if not parts:
            continue
        counts = np.array(
            [np.asarray(p["count"], dtype="float64").ravel()[0] for p in parts]
        )
        total = counts.sum()
        if total <= 0:
            continue

        def stack(key: str):
            return np.stack([np.asarray(p[key], dtype="float64") for p in parts])

        def weighted(values):
            shaped = counts.reshape((-1,) + (1,) * (values.ndim - 1))
            return (values * shaped).sum(axis=0) / total

        means = stack("mean")
        mean = weighted(means)
        variance = weighted(stack("std") ** 2 + means**2) - mean**2
        aggregated: dict[str, Any] = {
            "min": stack("min").min(axis=0),
            "max": stack("max").max(axis=0),
            "mean": mean,
            # clamped because the two terms are of similar size when an episode's
            # spread dominates, and float64 can land a hair below zero
            "std": np.sqrt(np.maximum(variance, 0.0)),
            "count": np.array([int(total)]),
        }
        for key in sorted(parts[0]):
            if key.startswith("q") and key[1:].isdigit():
                if all(key in p for p in parts):
                    aggregated[key] = weighted(stack(key))
        out[feature] = aggregated
    return out


def convert_episodes_metadata(
    new_root: Path, episode_records: list[dict[str, Any]]
) -> None:
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    def _stats(stats: dict[str, Any]) -> dict[str, Any]:
        """One episode's statistics, every key of them.

        The quantiles used to be dropped here. v2.1 does not require them, but the
        delivered copies carry them and they cannot be recovered later without
        reading every frame again -- see :data:`LEGACY_STATS_KEYS`.
        """
        return {
            feature: values
            for feature, values in stats.items()
            if isinstance(values, dict) and values
        }

    gathered: list[dict[str, Any]] = []
    with (
        jsonlines.open(episodes_path, mode="w") as episodes_writer,
        jsonlines.open(stats_path, mode="w") as stats_writer,
    ):
        for record in sorted(
            episode_records, key=lambda rec: int(rec["episode_index"])
        ):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            serializable_episode = {
                key: _to_serializable(value) for key, value in legacy_episode.items()
            }
            episodes_writer.write(serializable_episode)

            stats_flat = {
                key: record[key] for key in record if key.startswith("stats/")
            }
            stats_nested = _stats(unflatten_dict(stats_flat).get("stats", {}))
            gathered.append(stats_nested)
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": serialize_dict(stats_nested),
                }
            )

    # v2.1 carries the whole-dataset statistics in their own file, and the delivered
    # copies have one. It is written here rather than in convert_info because this is
    # where the per-episode statistics are already in hand: aggregating them costs
    # nothing, while a second pass over the frames would cost everything.
    if gathered:
        write_json(serialize_dict(aggregate_episode_stats(gathered)),
                   new_root / STATS_PATH)


def copy_ancillary_directories(root: Path, new_root: Path) -> None:
    for subdir in ["images"]:
        source = root / subdir
        if source.exists():
            shutil.copytree(source, new_root / subdir, dirs_exist_ok=True)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
) -> None:
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root)

    if not root.exists():
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=V30,
            local_dir=root,
        )

    old_root = root.parent / f"{root.name}_{V30}"
    new_root = root.parent / f"{root.name}_{V21}"

    if old_root.is_dir():
        shutil.rmtree(old_root)
    if new_root.is_dir():
        shutil.rmtree(new_root)

    new_root.mkdir(parents=True, exist_ok=True)

    episode_records = load_episode_records(root)
    video_keys = [
        key
        for key, ft in read_info(root)["features"].items()
        if ft.get("dtype") == "video"
    ]

    convert_info(root, new_root, episode_records, video_keys)
    convert_tasks(root, new_root)
    convert_data(root, new_root, episode_records, workers=workers)
    convert_videos(root, new_root, episode_records, video_keys, workers=workers)
    convert_episodes_metadata(new_root, episode_records)
    copy_ancillary_directories(root, new_root)

    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the local dataset root directory. If not provided, the script will use the dataset from local.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="How many files to cut at once. -1 (the default) uses one per core; "
        "1 runs the conversion serially.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()
    convert_dataset(**vars(args))
