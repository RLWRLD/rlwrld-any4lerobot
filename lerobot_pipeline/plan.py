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
    spec = config.dataset
    problems = spec.buildable()
    builder = spec.source.builder if spec.source else "(none)"
    if not problems:
        how = {
            "spec": "spec2lerobot reads the raw source; state_layout assembles",
            "none": "the source is already LeRobot; state_layout assembles",
        }.get(builder, f"{builder}2lerobot converts and writes the vectors itself")
        return f"buildable yes -- {how}"
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
    from .registry import compose_video_plans

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

    touched = _per_camera(config, steps, compose_video_plans)
    lines += touched
    encoding = config.runtime.encoding
    note = "" if any("re-encode" in line for line in touched) else \
        "  (not applied: nothing is re-encoded)"
    lines.append(f"  encoding  {dict(encoding) if encoding else 'inherited from source'}"
                 f"{note}")
    return "\n".join(lines)


def _per_camera(config: PipelineConfig, steps, compose) -> list[str]:
    """Per camera, whether the delivered copy was re-encoded -- and what this run does.

    The evidence is the delivered *codec*, not the delivered geometry. Geometry
    cannot settle it: a dataset whose delivered frames are already on the grid may
    have arrived that way or been resized to it, and using the delivered size as a
    stand-in for the source size assumes the answer.

    The codec does settle it. AV1 with a two-frame GOP is LeRobot's own writer
    default, which survives only if nothing re-encoded the file; H.264 High with a
    250-frame GOP is the rldx1_reference profile, which only our encoder produces.
    So a third of the collection is known to have been passed through untouched,
    whatever its dimensions.

    Where the source geometry is recorded the resize is also computed, which is the
    stronger check: it must land exactly on the delivered size.
    """
    spec = config.dataset
    if spec is None or not steps:
        return []

    lines = []
    for key in spec.cameras:
        delivered = spec.delivered_video.get(key) or {}
        codec, gop = delivered.get("codec"), delivered.get("gop")
        target = tuple(delivered["shape"][:2]) if delivered.get("shape") else None

        if codec == "av1" and gop == 2:
            was = "never re-encoded (AV1/GOP2 is LeRobot's own writer)"
        elif codec == "h264" and gop == 250:
            was = "re-encoded with rldx1_reference (H.264/GOP250 is ours)"
        else:
            was = f"delivered as {codec}/GOP{gop}; provenance unclear"

        shape = spec.camera_shape(key)
        if shape is None:
            lines.append(f"  {key:<22} delivered "
                         f"{target[1]}x{target[0]} -- {was}" if target else
                         f"  {key:<22} {was}")
            lines.append(f"  {'':<22} source geometry not recorded, so this run's "
                         "resize cannot be predicted here")
            continue

        height, width = shape[0], shape[1]
        plan = compose(steps, f"observation.images.{key}", (height, width))
        if plan is None:
            lines.append(f"  {key:<22} {width}x{height} unchanged -> hard-linked; "
                         f"{was}")
            continue
        out_h, out_w = plan.out_shape
        verdict = ""
        if target is not None:
            verdict = (" == delivered" if (out_h, out_w) == target
                       else f" != delivered {target[1]}x{target[0]}  MISMATCH")
        lines.append(f"  {key:<22} {width}x{height} -> {out_w}x{out_h}{verdict}, "
                     f"re-encoded; {was}")
    return lines


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


CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "datasets"


def summarise(paths: list[Path]) -> tuple[str, int]:
    """One line per config: can it be rebuilt, by what, and is the video touched."""
    from .registry import compose_video_plans

    from dataset_registry import SpecError, load

    lines = [f"{'dataset':<52} {'builder':<8} {'video':<27} {'raw source':<11} ready"]
    blocked = unlocated = 0
    for path in paths:
        try:
            config = load_config(path)
        except ConfigError as exc:
            # an unbuildable dataset fails at load, because the layout step refuses
            # to be constructed; report the count rather than the whole wall of holes
            try:
                spec = load(path.stem)
                builder = spec.source.builder if spec.source else "-"
                count = len(spec.buildable())
            except SpecError:
                builder, count = "-", 0
            reason = f"NO ({count})" if count else f"CONFIG ERROR: {str(exc).splitlines()[0]}"
            lines.append(f"{path.stem:<52} {builder:<8} {'-':<27} {'-':<11} {reason}")
            blocked += 1
            continue

        spec = config.dataset
        builder = (spec.source.builder if spec and spec.source else "-")
        problems = spec.buildable() if spec else []
        steps = [s for s in config.steps if getattr(s, "kind", None) == "video"]
        touched = _per_camera(config, steps, compose_video_plans)
        if not touched:
            video = "no video step"
        elif any("MISMATCH" in line for line in touched):
            video = "resize MISMATCHES delivered"
        elif any("== delivered" in line for line in touched):
            video = "re-encoded, size confirmed"
        elif any("never re-encoded" in line for line in touched):
            video = "delivered untouched (AV1)"
        else:
            video = "re-encoded (size unverified)"

        # a correct config still needs somewhere to point source.path
        located = "located" if (spec and (spec.huggingface or spec.foundry_uri)) \
            else "NOT FOUND"
        verdict = "yes" if not problems else f"NO ({len(problems)})"
        blocked += bool(problems)
        unlocated += located == "NOT FOUND" and not problems
        lines.append(f"{path.stem:<52} {builder:<8} {video:<27} {located:<11} {verdict}")

    ready = len(paths) - blocked
    lines.append("")
    lines.append(f"{ready}/{len(paths)} configs resolve with no missing source columns")
    if unlocated:
        lines.append(
            f"but {unlocated} of those have no raw source recorded -- the config is "
            "right and there is nowhere to point source.path yet"
        )
        lines.append(f"so {ready - unlocated}/{len(paths)} can actually be started today")
    return "\n".join(lines), blocked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--all", action="store_true",
                        help=f"summarise every config in {CONFIG_DIR.name}/")
    parser.add_argument(
        "--workdir", type=Path, default=None, help="only affects the paths shown"
    )
    args = parser.parse_args(argv)

    if args.all:
        text, blocked = summarise(sorted(CONFIG_DIR.glob("*.yaml")))
        print(text)
        return 1 if blocked else 0
    if args.config is None:
        print("give --config, or --all", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(describe(config, args.workdir))
    return 0 if (config.dataset is None or not config.dataset.buildable()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
