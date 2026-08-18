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

from .config import ConfigError, PipelineConfig
from .env import EnvError, build_config, load_env
from .meta import MetadataError
from .stages import (
    StageError,
    StageSpec,
    converter_command,
    plan_stages,
    version_convert_command,
    version_convert_output,
)
from .steps.state_layout import LayoutError
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

    # A workdir left behind by an earlier failure is not a resume point: its stage
    # directories would be read as this run's output, and a converter that drops a
    # dataset into one of them would leave two side by side.
    shutil.rmtree(workdir, ignore_errors=True)
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


def dataset_root(path: Path) -> Path:
    """Where the dataset actually is, once a converter has written under ``path``.

    Not every converter treats its output flag as the dataset root.
    ``openx_rlds.py`` treats ``--local-dir`` as a directory to drop a dataset *into*
    and writes ``<local-dir>/<name>_<version>_lerobot``, so a stage that returned the
    flag it passed would hand the next stage a directory with no ``meta`` in it.

    A dataset is recognised by its ``meta`` directory. Anything else -- nothing
    written, or more than one candidate -- is left alone for the next stage to fail
    on, because guessing between two datasets is worse than the error.
    """
    if (path / "meta").is_dir() or not path.is_dir():
        return path
    written = [child for child in sorted(path.iterdir()) if (child / "meta").is_dir()]
    return written[0] if len(written) == 1 else path


def execute_convert(stage: StageSpec, config: PipelineConfig) -> Path:
    command = converter_command(
        source_type=stage.detail["source_type"],
        input_path=stage.input_path,
        output_path=stage.output_path,
        args=stage.detail.get("args") or {},
    )
    _run(command, f"converter for {stage.detail['source_type']}")
    return dataset_root(stage.output_path)


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


def execute_state_layout(stage: StageSpec, config: PipelineConfig) -> Path:
    """Apply the table steps: rewrite parquet, hard-link everything else."""
    produced = stage.input_path
    steps = stage.detail["steps"]
    for index, step in enumerate(steps):
        target = (
            stage.output_path
            if index == len(steps) - 1
            else stage.output_path.parent / f"{stage.output_path.name}_{index:02d}"
        )
        step.apply(produced, target)
        produced = target
    return produced


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
    "state_layout": execute_state_layout,
    "transform": execute_transform,
    "version_convert": execute_version_convert,
}

EXPECTED_ERRORS = (
    ConfigError,
    EnvError,
    StageError,
    LayoutError,
    MetadataError,
    TransformError,
    FileExistsError,
    FileNotFoundError,
)


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot_pipeline.run",
        description=(
            "Rebuild one dataset, or all of them, on this machine. What a dataset is "
            "comes from dataset_registry, how it is processed from its profile, and "
            "where it lives from the environment."
        ),
    )
    parser.add_argument("--env", required=True, help="an environment name or path")
    parser.add_argument("--dataset", default=None, help="a dataset in the registry")
    parser.add_argument(
        "--all",
        action="store_true",
        help="rebuild every buildable dataset, skipping the ones that are blocked",
    )
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
    from dataset_registry import available

    env = load_env(args.env)
    if args.all:
        return _run_all(env, args)
    if args.dataset is None:
        raise ConfigError(
            f"give --dataset (one of: {', '.join(available())}), or --all"
        )
    config = build_config(env, args.dataset)

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


def _run_all(env, args: argparse.Namespace) -> int:
    """Rebuild everything that can be, and say plainly what was skipped.

    A blocked dataset is not a reason to stop: most of the collection is
    independent, and a run over 36 sources should get as far as it can and then
    report, rather than falling over on the first hole.
    """
    from dataset_registry import available, load

    done, skipped, failed = [], [], []
    for name in available():
        problems = load(name).buildable()
        if problems:
            skipped.append((name, problems[0]))
            continue
        print(f"=== {name}")
        try:
            config = build_config(env, name)
            dest = run_pipeline(
                config,
                workdir=args.workdir,
                keep_intermediate=args.keep_intermediate,
                overwrite=args.overwrite,
            )
        except EXPECTED_ERRORS as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed.append((name, str(exc).splitlines()[0]))
            continue
        done.append((name, dest))

    print(f"\n{len(done)} rebuilt, {len(failed)} failed, {len(skipped)} skipped")
    for name, reason in failed:
        print(f"  FAILED  {name}: {reason}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
