"""Plan and apply video steps to a LeRobot dataset.

Planning is separated from execution so the whole decision -- which files are
re-encoded, with which filters, and what the resulting metadata says -- is
settled and inspectable before a single frame is touched.

Only files a step actually changes are re-encoded. Everything else is hard-linked,
so the destination costs no extra disk and the source is never modified.
"""

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .meta import (
    INFO_RELPATH,
    encoding_from_info,
    feature_shape,
    load_info,
    patch_video_feature,
    video_files,
    video_keys,
    write_info,
)
from .registry import compose_video_plans
from .video_ops import (
    EncodingParams,
    Parallelism,
    build_ffmpeg_command,
    plan_parallelism,
    run_ffmpeg,
)


class TransformError(RuntimeError):
    """Raised when one or more files could not be processed."""


@dataclass(frozen=True)
class TranscodeJob:
    src: Path
    dst: Path
    filters: tuple[str, ...]
    encoding: EncodingParams


@dataclass(frozen=True)
class TransformPlan:
    root: Path
    dest: Path
    transcodes: tuple[TranscodeJob, ...] = ()
    links: tuple[tuple[Path, Path], ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


def plan_transform(
    root: str | Path,
    dest: str | Path,
    steps: Sequence[Any],
    preset: str | None = None,
    crf: int | None = None,
) -> TransformPlan:
    root = Path(root)
    dest = Path(dest)
    info = load_info(root)

    jobs: list[TranscodeJob] = []
    transcoded: set[Path] = set()

    for key in video_keys(info):
        height, width = feature_shape(info, key)[:2]
        composed = compose_video_plans(steps, key, (height, width))
        if composed is None:
            continue

        # codec and GOP stay mirrored from the source; only speed/quality knobs
        # are open to config overrides
        encoding = encoding_from_info(info, key)
        overrides = {
            name: value
            for name, value in (("preset", preset), ("crf", crf))
            if value is not None
        }
        if overrides:
            encoding = replace(encoding, **overrides)

        for src in video_files(root, info, key):
            jobs.append(
                TranscodeJob(
                    src=src,
                    dst=dest / src.relative_to(root),
                    filters=composed.filters,
                    encoding=encoding,
                )
            )
            transcoded.add(src)

        patch_video_feature(info, key, composed.out_shape)

    # longest-processing-time-first, across every key at once
    jobs.sort(key=lambda job: -job.src.stat().st_size)

    info_path = root / INFO_RELPATH
    links = tuple(
        (src, dest / src.relative_to(root))
        for src in sorted(root.rglob("*"))
        if src.is_file() and src != info_path and src not in transcoded
    )

    return TransformPlan(
        root=root, dest=dest, transcodes=tuple(jobs), links=links, info=info
    )


def materialize(
    plan: TransformPlan,
    overwrite: bool = False,
    parallelism: Parallelism | None = None,
) -> Path:
    """Write the planned dataset to ``plan.dest``.

    A partially written dataset is worse than none, so any failure removes the
    destination rather than leaving something that looks converted.
    """
    dest = plan.dest
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dest} already exists; pass overwrite=True to replace it"
            )
        shutil.rmtree(dest)

    if parallelism is None:
        parallelism = plan_parallelism(len(plan.transcodes), os.cpu_count() or 1)

    dest.mkdir(parents=True)
    try:
        for src, dst in plan.links:
            link_or_copy(src, dst)

        run_transcodes(plan.transcodes, parallelism)
        write_info(plan.info, dest)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    return dest


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``, copying only if the filesystem refuses."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run_transcodes(jobs: Sequence[TranscodeJob], parallelism: Parallelism) -> None:
    if not jobs:
        return

    for job in jobs:
        job.dst.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if parallelism.workers == 1:
        for job in jobs:
            failure = _transcode(job, parallelism.threads)
            if failure:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=parallelism.workers) as executor:
            futures = [
                executor.submit(_transcode, job, parallelism.threads) for job in jobs
            ]
            for future in futures:
                failure = future.result()
                if failure:
                    failures.append(failure)

    if failures:
        raise TransformError(
            f"{len(failures)} of {len(jobs)} file(s) failed:\n  "
            + "\n  ".join(failures[:20])
        )


def _transcode(job: TranscodeJob, threads: int) -> str | None:
    """Run one job. Returns an error description, or None on success."""
    try:
        run_ffmpeg(
            build_ffmpeg_command(job.src, job.dst, job.filters, job.encoding, threads)
        )
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return f"{job.src}: {exc}"
    return None
