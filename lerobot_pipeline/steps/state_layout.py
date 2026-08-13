"""Assemble ``observation.state`` and ``action`` from a dataset's source features.

This is where a dataset's layout convention is actually applied. A converter emits
the source's own vectors -- ``observation.robot_joints``, ``observation.arm_joints``,
whatever the robot reports -- and this step copies them into the flat vector the
training stack reads, slot by slot, following the block order the layout declares.

Doing it here rather than inside each converter buys two things:

* datasets that arrive already in LeRobot form (humanoid_everyday, galaxea) go
  through the same code as datasets converted from raw, so the convention has one
  implementation rather than one per source;
* changing the convention later rewrites parquet only. The video is untouched and
  hard-linked through, so re-laying-out ActionNet costs minutes against the days a
  full reconversion of 2.49 TiB would take.

Statistics are permuted, not recomputed. Every slot is a copy of a source column, so
per-column statistics move with it exactly; only sourceless blocks need synthesising,
and those are zeros. That is what keeps the step from having to re-read 30,000
episodes.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..meta import INFO_RELPATH, MetadataError, load_info, write_info
from ..registry import register_step

STATE_KEY = "observation.state"
ACTION_KEY = "action"
MODALITY_RELPATH = Path("meta") / "modality.json"


class LayoutError(ValueError):
    """Raised when a dataset cannot be laid out as its spec describes."""


@dataclass(frozen=True)
class VectorPlan:
    """One emitted vector: which source column each slot copies."""

    key: str
    width: int
    # per slot: (source column name, index) or None for a slot that is always zero
    slots: tuple[tuple[str, int] | None, ...]

    @property
    def sources(self) -> list[str]:
        return sorted({name for name in (s[0] for s in self.slots if s)})


@register_step("state_layout")
class StateLayoutStep:
    kind = "table"

    def __init__(self, dataset: str | None = None, spec: Any = None):
        if (dataset is None) == (spec is None):
            raise ValueError("state_layout takes exactly one of `dataset` or `spec`")
        if spec is None:
            from dataset_registry import load

            spec = load(dataset)
        problems = spec.buildable()
        if problems:
            raise LayoutError(
                f"{spec.id} cannot be laid out from its source features:\n  "
                + "\n  ".join(problems)
            )
        self.spec = spec

    def plans(self) -> list[VectorPlan]:
        plans = []
        for side, key in (("state", STATE_KEY), ("action", ACTION_KEY)):
            vector = self.spec.vector(side)
            if vector is None:
                continue
            plans.append(
                VectorPlan(
                    key=key,
                    width=vector.width,
                    slots=tuple(vector.slot_map(side)),
                )
            )
        return plans

    # -- execution ------------------------------------------------------------

    def apply(self, root: str | Path, out: str | Path) -> None:
        root, out = Path(root), Path(out)
        info = load_info(root)
        plans = self.plans()
        _check_sources(info, plans)

        _mirror_tree(root, out)
        for path in _parquet_files(root):
            _rewrite(path, out / path.relative_to(root), plans)

        write_info(_patch_features(info, plans), out)
        write_modality(self.spec, out)


def _parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("data/**/*.parquet"))


def _mirror_tree(root: Path, out: Path) -> None:
    """Copy the dataset to ``out``, hard-linking everything that does not change.

    Only the episode parquet files are rewritten. Video is the bulk of a dataset and
    this step does not touch a frame of it, so it costs no disk and the source is
    never modified.
    """
    if out.exists():
        raise LayoutError(f"{out} already exists")
    out.mkdir(parents=True)
    for path in sorted(root.rglob("*")):
        target = out / path.relative_to(root)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.suffix != ".parquet" or "data" not in path.relative_to(root).parts:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.hardlink_to(path)
            except OSError:  # different filesystem, or a limit on link count
                shutil.copy2(path, target)


def _check_sources(info: dict[str, Any], plans: list[VectorPlan]) -> None:
    features = info.get("features") or {}
    for plan in plans:
        for name in plan.sources:
            if name not in features:
                raise LayoutError(
                    f"{plan.key} needs source feature {name!r}, which the dataset "
                    f"does not have. It carries: {', '.join(sorted(features))}"
                )


def _rewrite(path: Path, out_path: Path, plans: list[VectorPlan]) -> None:
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(path)
    for plan in plans:
        frame[plan.key] = list(_assemble(frame, plan, np))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)


def _assemble(frame, plan: VectorPlan, np):
    """The emitted vector, one row per frame."""
    columns = {
        name: np.stack([np.atleast_1d(v) for v in frame[name].to_numpy()])
        for name in plan.sources
    }
    rows = len(frame)
    out = np.zeros((rows, plan.width), dtype=np.float32)
    for slot, source in enumerate(plan.slots):
        if source is None:
            continue  # a body part this robot does not have: zeros by definition
        name, column = source
        out[:, slot] = columns[name][:, column]
    return out


def _patch_features(info: dict[str, Any], plans: list[VectorPlan]) -> dict[str, Any]:
    features = dict(info.get("features") or {})
    for plan in plans:
        features[plan.key] = {
            "dtype": "float32",
            "shape": [plan.width],
            "names": {"motors": [f"m{index}" for index in range(plan.width)]},
        }
    return {**info, "features": features}


def write_modality(spec, root: str | Path) -> Path:
    """``meta/modality.json`` -- the GR00T-style view the training stack reads.

    Derived from the spec rather than written by a converter, so it cannot disagree
    with the vector beside it. The delivered datasets expose state and action as one
    flat block even though the columns are in body-part order, because the training
    config asks for ``modality_keys=["state"]``; that is reproduced here.
    """
    modality: dict[str, Any] = {}
    for side, key in (("state", STATE_KEY), ("action", ACTION_KEY)):
        vector = spec.vector(side)
        if vector is None:
            continue
        block: dict[str, Any] = {"start": 0, "end": vector.width}
        if side == "action":
            block["absolute"] = True
        modality[side] = {side: block}

    modality["video"] = {
        camera: {"original_key": f"observation.images.{camera}"}
        for camera in spec.cameras
    }
    modality["annotation"] = {
        "human.action.task_description": {"original_key": "task_index"},
        "human.action.task_name": {},
        "human.validity": {},
    }

    path = Path(root) / MODALITY_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(modality, indent=4) + "\n")
    return path


__all__ = [
    "LayoutError",
    "MetadataError",
    "StateLayoutStep",
    "VectorPlan",
    "write_modality",
    "INFO_RELPATH",
]
