"""Stage planning and command construction.

Every stage is a directory -> directory step, which is what the existing scripts
in this repo already are. That lets the pipeline reuse them untouched instead of
reimplementing conversion logic that is already in production use.

Two ordering rules are load-bearing:

* the converter runs first, because converters only ever emit v3.0;
* the transform runs *before* any version conversion, so the version converter
  handles already-shrunk video, and (for v2.1 sources) it runs while the dataset
  still has many small files to parallelise over.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

# (from, to) -> script, relative to the repo root
_VERSION_SCRIPTS: dict[tuple[str, str], str] = {
    ("lerobot_v21", "lerobot_v30"): "ds_version_convert/v21_to_v30/convert_dataset_v21_to_v30.py",
    ("lerobot_v30", "lerobot_v21"): "ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py",
}

# source type -> (script, input flag, output flag)
_CONVERTERS: dict[str, tuple[str, str, str]] = {
    # spec-driven and dataset-agnostic: it takes --dataset and reads the rest from
    # the registry. The others below still carry their dataset in their code.
    "spec": ("spec2lerobot", "--src-path", "--output-path"),
    "agibot": ("agibot2lerobot/agibot_h5.py", "--src-path", "--output-path"),
    "libero": ("libero2lerobot/libero_h5.py", "--src-paths", "--output-path"),
    "openx": ("openx2lerobot/openx_rlds.py", "--raw-dir", "--local-dir"),
    "robocasa": ("robocasa2lerobot/robocasa_h5.py", "--raw-dir", "--local-dir"),
    "robomind": ("robomind2lerobot/robomind_h5.py", "--src-path", "--output-path"),
}

CONVERTER_OUTPUT_VERSION = "lerobot_v30"


class StageError(ValueError):
    """Raised when a pipeline cannot be assembled from the config."""


@dataclass(frozen=True)
class StageSpec:
    kind: str
    input_path: Path
    output_path: Path
    detail: dict[str, Any] = field(default_factory=dict)


def plan_stages(config: PipelineConfig, workdir: str | Path) -> list[StageSpec]:
    workdir = Path(workdir)

    planned: list[tuple[str, dict[str, Any]]] = []
    version = config.source.type if config.source.is_lerobot else None

    if not config.source.is_lerobot:
        planned.append(
            (
                "convert",
                {"source_type": config.source.type, "args": dict(config.source.args)},
            )
        )
        version = CONVERTER_OUTPUT_VERSION

    table_steps = [s for s in config.steps if getattr(s, "kind", None) == "table"]
    video_steps = [s for s in config.steps if getattr(s, "kind", None) == "video"]

    # Table steps first: they rewrite parquet and hard-link the video, so a failure
    # costs seconds rather than a re-encode. They must also run before any version
    # conversion, so the columns are settled before the layout changes.
    if table_steps:
        planned.append(("state_layout", {"steps": tuple(table_steps)}))
    if video_steps:
        planned.append(("transform", {}))

    if version != config.dest.type:
        planned.append(("version_convert", {"from": version, "to": config.dest.type}))

    if not planned:
        raise StageError(
            f"nothing to do: source and dest are both {config.dest.type} and no "
            "steps are configured"
        )

    stages: list[StageSpec] = []
    input_path = config.source.path
    for index, (kind, detail) in enumerate(planned):
        is_last = index == len(planned) - 1
        output_path = (
            config.dest.path if is_last else workdir / f"{index:02d}_{kind}"
        )
        stages.append(
            StageSpec(
                kind=kind,
                input_path=input_path,
                output_path=output_path,
                detail=detail,
            )
        )
        input_path = output_path

    return stages


def version_convert_command(
    from_version: str, to_version: str, root: Path, repo_id: str
) -> list[str]:
    script = _VERSION_SCRIPTS.get((from_version, to_version))
    if script is None:
        raise StageError(
            f"no version conversion available from {from_version} to {to_version}. "
            f"supported: {', '.join(f'{a} -> {b}' for a, b in _VERSION_SCRIPTS)}"
        )

    command = [
        sys.executable,
        str(REPO_ROOT / script),
        "--repo-id",
        repo_id,
        "--root",
        str(root),
    ]
    if (from_version, to_version) == ("lerobot_v21", "lerobot_v30"):
        # that script defaults --push-to-hub to true; never upload implicitly
        command += ["--push-to-hub", "false"]
    return command


def version_convert_output(from_version: str, to_version: str, root: Path) -> Path:
    """Where the version conversion script leaves its result.

    The two scripts disagree: v21->v30 swaps the converted dataset into ``root``
    (keeping the original at ``{root}_old``), while v30->v21 writes a sibling.
    """
    if (from_version, to_version) == ("lerobot_v21", "lerobot_v30"):
        return root
    if (from_version, to_version) == ("lerobot_v30", "lerobot_v21"):
        return root.parent / f"{root.name}_v2.1"
    raise StageError(f"no version conversion from {from_version} to {to_version}")


def converter_command(
    source_type: str,
    input_path: Path,
    output_path: Path,
    args: Mapping[str, Any],
) -> list[str]:
    entry = _CONVERTERS.get(source_type)
    if entry is None:
        raise StageError(
            f"unknown converter source {source_type!r}. "
            f"available: {', '.join(sorted(_CONVERTERS))}"
        )

    script, input_flag, output_flag = entry
    # a package is run with -m so its relative imports resolve; the older
    # converters are single scripts and are run by path
    entry_point = (
        [sys.executable, "-m", script]
        if not script.endswith(".py")
        else [sys.executable, str(REPO_ROOT / script)]
    )
    command = [
        *entry_point,
        input_flag,
        str(input_path),
        output_flag,
        str(output_path),
    ]

    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command += [str(item) for item in value]
        else:
            command += [flag, str(value)]

    return command
