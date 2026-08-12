"""CLI entry point: read one config, run its stages in order, keep the final output.

The orchestrator owns three guarantees:

* the destination is validated *before* any work starts, so a run either produces
  a complete dataset or leaves nothing behind;
* every stage consumes the previous stage's actual output, because the version
  conversion scripts do not all write where you would expect;
* the user's source directory is never handed to a script that rewrites its input
  in place.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, PipelineConfig, load_config
from .meta import MetadataError
from .stages import (
    StageError,
    StageSpec,
    converter_command,
    plan_stages,
    version_convert_command,
    version_convert_output,
)
from .transform import TransformError, link_or_copy, materialize, plan_transform
from .video_ops import plan_parallelism

Executor = Callable[[StageSpec, PipelineConfig], Path]


def needs_staging(stage: StageSpec, config: PipelineConfig) -> bool:
    """True when a stage would rewrite the user's own source directory.

    ``convert_dataset_v21_to_v30.py`` converts in place: it moves the original to
    ``{root}_old`` and puts the result at ``--root``. Pointed at the user's
    dataset that silently rearranges their data, so such a stage must read from a
    staged copy instead.
    """
    if stage.kind != "version_convert":
        return False
    return stage.input_path == config.source.path


def default_workdir(config: PipelineConfig) -> Path:
    dest = config.dest.path
    return dest.parent / f"{dest.name}_work"


def run_pipeline(
    config: PipelineConfig,
    workdir: str | Path | None = None,
    executors: Mapping[str, Executor] | None = None,
    keep_intermediate: bool = False,
    overwrite: bool = False,
) -> Path:
    workdir = Path(workdir) if workdir is not None else default_workdir(config)
    executors = dict(executors or DEFAULT_EXECUTORS)

    dest = config.dest.path
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"dest.path already exists: {dest}. "
                "Pass --overwrite to replace it, or choose another path."
            )
        shutil.rmtree(dest)

    stages = plan_stages(config, workdir)
    missing = sorted({stage.kind for stage in stages} - set(executors))
    if missing:
        raise KeyError(f"no executor registered for stage(s): {', '.join(missing)}")

    workdir.mkdir(parents=True, exist_ok=True)
    produced = config.source.path
    try:
        for stage in stages:
            stage = replace(stage, input_path=produced)
            produced = Path(executors[stage.kind](stage, config))
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        raise

    if produced != dest:
        # a stage wrote somewhere else (the version scripts pick their own path)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(dest))

    if not keep_intermediate:
        shutil.rmtree(workdir, ignore_errors=True)

    return dest


# --- stage executors ---------------------------------------------------------


def execute_convert(stage: StageSpec, config: PipelineConfig) -> Path:
    command = converter_command(
        source_type=stage.detail["source_type"],
        input_path=stage.input_path,
        output_path=stage.output_path,
        args=stage.detail.get("args") or {},
    )
    _run(command, f"converter for {stage.detail['source_type']}")
    return stage.output_path


def execute_transform(stage: StageSpec, config: PipelineConfig) -> Path:
    plan = plan_transform(
        stage.input_path,
        stage.output_path,
        config.steps,
        preset=config.runtime.preset,
        crf=config.runtime.crf,
        encoding_profile=config.runtime.encoding,
    )
    parallelism = plan_parallelism(
        file_count=len(plan.transcodes),
        cores=os.cpu_count() or 1,
        workers=config.runtime.workers,
        threads_per_ffmpeg=config.runtime.threads_per_ffmpeg,
    )
    # the destination was validated by run_pipeline before any stage ran
    return materialize(plan, overwrite=True, parallelism=parallelism)


def execute_version_convert(stage: StageSpec, config: PipelineConfig) -> Path:
    from_version = stage.detail["from"]
    to_version = stage.detail["to"]

    root = stage.input_path
    if needs_staging(stage, config):
        staged = stage.output_path.parent / f"{stage.output_path.name}_staged"
        _stage_copy(root, staged)
        root = staged

    command = version_convert_command(
        from_version=from_version,
        to_version=to_version,
        root=root,
        repo_id=config.name,
    )
    _run(command, f"version conversion {from_version} -> {to_version}")
    return version_convert_output(from_version, to_version, root)


def _stage_copy(src: Path, dst: Path) -> None:
    """Hard-link the dataset into ``dst`` so an in-place script cannot touch ``src``."""
    for path in sorted(src.rglob("*")):
        if path.is_file():
            link_or_copy(path, dst / path.relative_to(src))


def _run(command: Sequence[str], what: str) -> None:
    result = subprocess.run(list(command))
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed with exit code {result.returncode}: {' '.join(command)}"
        )


DEFAULT_EXECUTORS: dict[str, Executor] = {
    "convert": execute_convert,
    "transform": execute_transform,
    "version_convert": execute_version_convert,
}

EXPECTED_ERRORS = (
    ConfigError,
    StageError,
    MetadataError,
    TransformError,
    FileExistsError,
    FileNotFoundError,
)


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot_pipeline.run",
        description="Run a LeRobot preprocessing pipeline described by one config file.",
    )
    parser.add_argument("--config", type=Path, required=True, help="pipeline config YAML")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where intermediates go; point at fast local storage (default: next to dest)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace dest.path if it exists"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep stage outputs for debugging",
    )
    parser.add_argument("--workers", type=int, default=None, help="override runtime.workers")
    parser.add_argument(
        "--threads-per-ffmpeg", type=int, default=None, help="override runtime.threads_per_ffmpeg"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(parse_args(argv))
    except EXPECTED_ERRORS as exc:
        # these are user-facing problems (bad config, existing destination, a
        # failed subprocess), not defects worth a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _main(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    if args.workers is not None or args.threads_per_ffmpeg is not None:
        runtime = replace(
            config.runtime,
            workers=args.workers if args.workers is not None else config.runtime.workers,
            threads_per_ffmpeg=(
                args.threads_per_ffmpeg
                if args.threads_per_ffmpeg is not None
                else config.runtime.threads_per_ffmpeg
            ),
        )
        config = replace(config, runtime=runtime)

    dest = run_pipeline(
        config,
        workdir=args.workdir,
        keep_intermediate=args.keep_intermediate,
        overwrite=args.overwrite,
    )
    print(f"done: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
