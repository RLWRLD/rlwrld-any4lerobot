"""Load and validate dataset specs.

One spec per dataset, in ``datasets/*.yaml``. A spec says three things:

* **where the data is** -- the raw source mirrored into foundry, and the already
  converted copy delivered on the training storage;
* **how its state and action vectors are laid out** -- which source columns land in
  which slots, and *how we know that*;
* **how its video was processed** -- camera key mapping plus the names of the shared
  resize step and encoding profile.

The layouts were recovered from delivered data rather than read from a spec (the
conversion code no longer exists), so every block carries an ``evidence`` field.
Validation is strict: unknown keys are errors, and slots must tile the vector
exactly. See ``verify.py`` for checking a spec against a real dataset.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REGISTRY_DIR / "datasets"

# How a block's mapping is known. Recorded per block because the confidence is not
# uniform even within one dataset.
EVIDENCE = {
    # matched column-by-column against the delivered data
    "measured",
    # named outright by the dataset's own meta/modality.json
    "declared",
    # follows from the conversion code's convention, not from the delivered bytes
    "inferred",
    # a block of identical constants: indistinguishable from its neighbours, and
    # for the same reason harmless to get wrong
    "constant",
    # width is known, composition is not
    "unknown",
}

# Evidence values a block may carry without naming a source column. `constant` is
# the only one that also asserts the values never vary -- the others just mean we
# have no source vector to point at.
SOURCELESS_EVIDENCE = {"constant", "declared", "inferred", "unknown"}

# What the raw source looks like inside its foundry prefix.
SOURCE_LAYOUTS = {
    "tar_archives",  # loose *.tar shards to untar
    "tar_gz_per_task",  # one *.tar.gz per task, LeRobot inside
    "hf_repo",  # an unpacked HuggingFace dataset repo
    "lerobot",  # a plain LeRobot dataset tree
}

_TOP_LEVEL = {"id", "name", "notes", "upstream", "mirrors", "delivered", "lerobot"}
_UPSTREAM = {"huggingface", "revision", "homepage", "license", "commercial_use"}
_MIRROR = {"kind", "uri", "layout", "objects", "bytes"}
MIRROR_KINDS = {"foundry", "naver", "other"}
_DELIVERED = {"path", "origin", "codebase_version", "episodes", "frames", "converted_by"}
_LEROBOT = {"robot_type", "fps", "embodiment_tag", "video", "state", "action", "features"}
_VIDEO = {"cameras", "resize", "encoding", "keeps_original"}
_STATE = {"width", "layout", "source_features", "blocks"}
_BLOCK = {"name", "slots", "source", "evidence", "note"}
_SOURCE = {"feature", "columns"}


class SpecError(ValueError):
    """Raised for a malformed dataset spec."""


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int
    evidence: str
    feature: str | None = None
    src_start: int | None = None
    src_end: int | None = None
    note: str | None = None

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def is_constant(self) -> bool:
        return self.feature is None


@dataclass(frozen=True)
class StateSpec:
    width: int
    layout: str
    # logical feature name -> {"state": <source path>, "action": <source path>}
    source_features: Mapping[str, Mapping[str, str]]
    blocks: tuple[Block, ...]

    def slot_map(self, side: str = "state") -> list[tuple[str, int] | None]:
        """Per slot, the ``(source path, column)`` it takes -- or ``None`` if constant.

        This is what a converter consumes: it is general enough for every layout in
        the collection, unlike a bare permutation, which only works when the source
        and target widths match.
        """
        if side not in ("state", "action"):
            raise SpecError(f"side must be 'state' or 'action', got {side!r}")
        slots: list[tuple[str, int] | None] = [None] * self.width
        for block in self.blocks:
            if block.is_constant:
                continue
            path = self.source_features[block.feature][side]
            for offset in range(block.width):
                slots[block.start + offset] = (path, block.src_start + offset)
        return slots

    def evidence_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.evidence] = counts.get(block.evidence, 0) + block.width
        return counts


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    raw: Mapping[str, Any] = field(repr=False)
    state: StateSpec | None = None
    # Most datasets use one layout for both vectors. Galaxea does not -- its state
    # is 18 wide and its action 26 -- so `action` overrides when present.
    action: StateSpec | None = None

    def vector(self, side: str) -> StateSpec | None:
        if side == "action":
            return self.action or self.state
        if side == "state":
            return self.state
        raise SpecError(f"side must be 'state' or 'action', got {side!r}")

    @property
    def revision(self) -> str | None:
        """The upstream HuggingFace revision -- the dataset's canonical identity."""
        return (self.raw.get("upstream") or {}).get("revision")

    @property
    def huggingface(self) -> str | None:
        return (self.raw.get("upstream") or {}).get("huggingface")

    @property
    def mirrors(self) -> list[Mapping[str, Any]]:
        return list(self.raw.get("mirrors") or [])

    def mirror(self, kind: str) -> Mapping[str, Any] | None:
        return next((m for m in self.mirrors if m.get("kind") == kind), None)

    @property
    def foundry_uri(self) -> str | None:
        mirror = self.mirror("foundry")
        return mirror.get("uri") if mirror else None

    @property
    def delivered_path(self) -> str | None:
        return (self.raw.get("delivered") or {}).get("path")

    @property
    def embodiment_tag(self) -> str | None:
        return (self.raw.get("lerobot") or {}).get("embodiment_tag")

    @property
    def cameras(self) -> Mapping[str, Any]:
        return ((self.raw.get("lerobot") or {}).get("video") or {}).get("cameras") or {}


def available() -> list[str]:
    return sorted(path.stem for path in DATASETS_DIR.glob("*.yaml"))


def load(name: str) -> DatasetSpec:
    import yaml

    path = DATASETS_DIR / f"{name}.yaml"
    if Path(name).name != name or not path.is_file():
        raise SpecError(f"unknown dataset {name!r}. available: {', '.join(available())}")
    return parse(yaml.safe_load(path.read_text()) or {}, origin=str(path))


def load_all() -> list[DatasetSpec]:
    return [load(name) for name in available()]


def parse(raw: Mapping[str, Any], origin: str = "<spec>") -> DatasetSpec:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{origin}: spec must be a mapping")
    _reject(raw, _TOP_LEVEL, origin)
    for key in ("id", "name"):
        if not raw.get(key):
            raise SpecError(f"{origin}: missing required key {key!r}")

    upstream = raw.get("upstream") or {}
    _reject(upstream, _UPSTREAM, f"{origin}.upstream")
    _reject(raw.get("delivered") or {}, _DELIVERED, f"{origin}.delivered")
    _parse_mirrors(raw.get("mirrors"), upstream.get("revision"), origin)

    lerobot = raw.get("lerobot") or {}
    _reject(lerobot, _LEROBOT, f"{origin}.lerobot")
    _reject(lerobot.get("video") or {}, _VIDEO, f"{origin}.lerobot.video")

    state_raw = lerobot.get("state")
    action_raw = lerobot.get("action")
    return DatasetSpec(
        id=raw["id"],
        name=raw["name"],
        raw=raw,
        state=_parse_state(state_raw, f"{origin}.lerobot.state") if state_raw else None,
        action=_parse_state(action_raw, f"{origin}.lerobot.action") if action_raw else None,
    )


def _parse_mirrors(raw: Any, revision: str | None, origin: str) -> None:
    """Mirrors are copies of the upstream release; the upstream revision is identity.

    Foundry lays external sources out as ``external/<name>/<revision>/``, so a
    foundry mirror whose path does not end in the upstream revision is pointing at a
    different release than the spec claims -- worth failing over, because everything
    downstream assumes the two agree.
    """
    if raw is None:
        return
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise SpecError(f"{origin}.mirrors must be a list")

    for index, mirror in enumerate(raw):
        where = f"{origin}.mirrors[{index}]"
        _reject(mirror, _MIRROR, where)
        kind = mirror.get("kind")
        if kind not in MIRROR_KINDS:
            raise SpecError(
                f"{where}.kind must be one of {', '.join(sorted(MIRROR_KINDS))}, "
                f"got {kind!r}"
            )
        layout = mirror.get("layout")
        if layout is not None and layout not in SOURCE_LAYOUTS:
            raise SpecError(
                f"{where}.layout must be one of {', '.join(sorted(SOURCE_LAYOUTS))}, "
                f"got {layout!r}"
            )
        uri = mirror.get("uri") or ""
        if kind == "foundry" and revision and revision not in uri:
            raise SpecError(
                f"{where}.uri does not contain the upstream revision {revision!r}; "
                "foundry stores external sources under external/<name>/<revision>/"
            )


def _parse_state(raw: Any, origin: str) -> StateSpec:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{origin} must be a mapping")
    _reject(raw, _STATE, origin)

    width = raw.get("width")
    if not isinstance(width, int) or width < 1:
        raise SpecError(f"{origin}.width must be a positive integer, got {width!r}")

    sources = raw.get("source_features") or {}
    for name, sides in sources.items():
        if not isinstance(sides, Mapping) or set(sides) != {"state", "action"}:
            raise SpecError(
                f"{origin}.source_features.{name} must map both 'state' and 'action' "
                f"to a source path, got {sides!r}"
            )

    blocks: list[Block] = []
    for index, entry in enumerate(raw.get("blocks") or []):
        where = f"{origin}.blocks[{index}]"
        _reject(entry, _BLOCK, where)
        start, end = _slots(entry.get("slots"), where)
        evidence = entry.get("evidence")
        if evidence not in EVIDENCE:
            raise SpecError(
                f"{where}.evidence must be one of {', '.join(sorted(EVIDENCE))}, "
                f"got {evidence!r}"
            )

        source = entry.get("source")
        feature = src_start = src_end = None
        if source is not None:
            _reject(source, _SOURCE, f"{where}.source")
            feature = source.get("feature")
            if feature not in sources:
                raise SpecError(
                    f"{where}.source.feature {feature!r} is not in source_features "
                    f"({', '.join(sources)})"
                )
            src_start, src_end = _slots(source.get("columns"), f"{where}.source")
            if (src_end - src_start) != (end - start):
                raise SpecError(
                    f"{where}: {end - start} slots but {src_end - src_start} source "
                    "columns"
                )
        elif evidence not in SOURCELESS_EVIDENCE:
            raise SpecError(
                f"{where}: a block with no source must be one of "
                f"{', '.join(sorted(SOURCELESS_EVIDENCE))}, got {evidence!r}"
            )

        blocks.append(
            Block(
                name=entry.get("name") or f"block_{index}",
                start=start,
                end=end,
                evidence=evidence,
                feature=feature,
                src_start=src_start,
                src_end=src_end,
                note=entry.get("note"),
            )
        )

    _check_tiling(blocks, width, origin)
    return StateSpec(
        width=width,
        layout=raw.get("layout") or "custom",
        source_features=sources,
        blocks=tuple(blocks),
    )


def _slots(value: Any, where: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise SpecError(f"{where} must be a [start, end] pair of integers, got {value!r}")
    start, end = value
    if start < 0 or end <= start:
        raise SpecError(f"{where} must satisfy 0 <= start < end, got {value!r}")
    return start, end


def _check_tiling(blocks: Sequence[Block], width: int, origin: str) -> None:
    """Blocks must cover every slot exactly once -- no gaps, no overlaps.

    A gap would mean a slot nobody can explain; an overlap means two claims about the
    same number. Both are the kind of error that is invisible once training starts.
    """
    covered = [0] * width
    for block in blocks:
        if block.end > width:
            raise SpecError(
                f"{origin}: block {block.name!r} ends at {block.end}, past width {width}"
            )
        for slot in range(block.start, block.end):
            covered[slot] += 1

    gaps = [i for i, n in enumerate(covered) if n == 0]
    overlaps = [i for i, n in enumerate(covered) if n > 1]
    if gaps:
        raise SpecError(f"{origin}: slots not covered by any block: {_ranges(gaps)}")
    if overlaps:
        raise SpecError(f"{origin}: slots claimed by more than one block: {_ranges(overlaps)}")


def _ranges(values: Sequence[int]) -> str:
    out, start, prev = [], None, None
    for value in values:
        if start is None:
            start = prev = value
        elif value == prev + 1:
            prev = value
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = value
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


def _reject(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{where} must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise SpecError(
            f"{where}: unknown key(s) {', '.join(unknown)}. "
            f"allowed: {', '.join(sorted(allowed))}"
        )
