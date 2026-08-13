"""Print what a config resolves to, without running it.

Splitting facts, conventions and run settings into three files keeps each one small
and makes a convention switch a one-file edit, but it also means no single file
answers "what is this run about to do". This does.

    python -m lerobot_pipeline.plan --config lerobot_pipeline/configs/actionnet_rldx1.yaml

It reads no data and writes nothing. The first thing it reports is whether the
dataset can be rebuilt at all -- a spec with blocks whose source columns were never
recovered fails here rather than after two days of encoding.
"""

import argparse
import sys
from pathlib import Path

from .config import ConfigError, PipelineConfig, load_config
from .stages import StageError, plan_stages


def describe(config: PipelineConfig, workdir: Path | None = None) -> str:
    out: list[str] = []
    write = out.append

    write(f"run       {config.name}")
    if config.dataset is not None:
        spec = config.dataset
        write(f"dataset   {spec.id}  ({spec.name})")
        if spec.huggingface:
            write(f"upstream  {spec.huggingface} @ {spec.revision}")
        if spec.foundry_uri:
            write(f"mirror    {spec.foundry_uri}")
    write(f"profile   {config.profile or '(none)'}")
    write("")

    write(_buildability(config))
    write("")
    write(_stages(config, workdir))
    write("")
    write(_video(config))

    if config.dataset is not None:
        write("")
        write(_slots(config.dataset))
    return "\n".join(out)


def _buildability(config: PipelineConfig) -> str:
    if config.dataset is None:
        return "buildable no dataset named; the run is a plain transform"
    problems = config.dataset.buildable()
    if not problems:
        return "buildable yes"
    lines = ["buildable NO -- this dataset cannot be rebuilt from its source:"]
    lines += [f"            {problem}" for problem in problems]
    return "\n".join(lines)


def _stages(config: PipelineConfig, workdir: Path | None) -> str:
    lines = ["stages"]
    try:
        stages = plan_stages(config, workdir or Path("<workdir>"))
    except StageError as exc:
        return f"stages    cannot be planned: {exc}"
    for index, stage in enumerate(stages, start=1):
        detail = ""
        if stage.kind == "convert":
            detail = f"  source={stage.detail['source_type']}"
        elif stage.kind == "version_convert":
            detail = f"  {stage.detail['from']} -> {stage.detail['to']}"
        elif stage.kind == "state_layout":
            names = ", ".join(s.config_name for s in stage.detail["steps"])
            detail = f"  {names}"
        lines.append(f"  {index}. {stage.kind}{detail}")
        lines.append(f"     -> {stage.output_path}")
    return "\n".join(lines)


def _video(config: PipelineConfig) -> str:
    steps = [s for s in config.steps if getattr(s, "kind", None) == "video"]
    lines = ["video"]
    if not steps:
        lines.append("  (no video step; frames pass through unchanged)")
    for step in steps:
        params = {
            key: value
            for key, value in vars(step).items()
            if not key.startswith("_")
        }
        lines.append(f"  {step.config_name}  {params}")
    encoding = config.runtime.encoding
    lines.append(f"  encoding  {dict(encoding) if encoding else 'inherited from source'}")
    return "\n".join(lines)


def _slots(spec) -> str:
    """The full slot map: what each emitted column actually copies."""
    lines = []
    for side in ("state", "action"):
        vector = spec.vector(side)
        if vector is None:
            continue
        key = "observation.state" if side == "state" else "action"
        lines.append(
            f"{key}  width {vector.width}  layout {vector.layout}  "
            f"{vector.evidence_counts()}"
        )
        for block in vector.blocks:
            if block.feature is None:
                source = "zeros"
            else:
                path = vector.source_features[block.feature][side]
                source = f"{path}[{block.src_start}:{block.src_end}]"
                if block.pad:
                    source += f" + {block.pad} pad"
            lines.append(
                f"  [{block.start:2d}:{block.end:2d}] {block.name:<14} "
                f"{source:<46} {block.evidence}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--workdir", type=Path, default=None, help="only affects the paths shown"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(describe(config, args.workdir))
    return 0 if (config.dataset is None or not config.dataset.buildable()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
