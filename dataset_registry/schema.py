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
LAYOUTS_DIR = REGISTRY_DIR / "layouts"

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

_TOP_LEVEL = {
    "id", "name", "notes", "upstream", "mirrors", "delivered", "source", "lerobot",
}
_UPSTREAM = {"huggingface", "revision", "homepage", "license", "commercial_use"}
_MIRROR = {"kind", "uri", "layout", "objects", "bytes"}
MIRROR_KINDS = {"foundry", "naver", "other"}
_DELIVERED = {
    "path", "origin", "codebase_version", "episodes", "frames", "converted_by", "video",
}
_LEROBOT = {
    "robot_type", "fps", "embodiment_tag", "video", "state", "action", "features",
    "modality",
}
_VIDEO = {"cameras", "resize", "encoding", "keeps_original"}
_CAMERA = {"source", "shape"}
_STATE = {"width", "layout", "source_features", "blocks"}
_BLOCK = {"width", "pad", "source", "evidence", "note"}
_SOURCE = {"feature", "columns"}

# How the raw upstream files are arranged, and how to line their clocks up. Read by
# spec2lerobot; the format and clock names are the closed sets it implements.
_SOURCE_SPEC = {
    "builder", "args", "raw_dir", "format", "discover", "paths", "tasks", "clock",
    "features", "feature_widths", "layout", "note",
}

# Which program turns the raw source into LeRobot. `spec` is the data-driven path in
# spec2lerobot; the rest are the converters this repo already had, each of which
# carries its own dataset knowledge in code and emits observation.state itself --
# so a dataset built by one of those skips the state_layout step. `none` is a source
# that is already LeRobot and needs no conversion at all.
BUILDERS = {"spec", "openx", "agibot", "libero", "robocasa", "robomind", "none"}
_SOURCE_PATHS = {"episode", "video"}
_SOURCE_TASKS = {"file", "key", "prompt"}
_SOURCE_CLOCK = {"strategy", "data", "image", "image_format"}

_LAYOUT = {"order", "note"}


class SpecError(ValueError):
    """Raised for a malformed dataset spec."""


def available_layouts() -> list[str]:
    return sorted(path.stem for path in LAYOUTS_DIR.glob("*.yaml"))


def load_layout(name: str) -> tuple[str, ...]:
    """The block order a layout declares.

    Order lives here rather than in the dataset spec so that changing a convention
    is a one-file edit. A dataset declares which body parts it has and how wide they
    are; the layout decides where they sit. Slots are derived from the two, never
    written down, so they cannot drift.
    """
    import yaml

    path = LAYOUTS_DIR / f"{name}.yaml"
    if Path(name).name != name or not path.is_file():
        raise SpecError(
            f"unknown layout {name!r}. available: {', '.join(available_layouts())}"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    _reject(raw, _LAYOUT, str(path))

    order = raw.get("order")
    if (
        not isinstance(order, Sequence)
        or isinstance(order, str)
        or not order
        or not all(isinstance(item, str) for item in order)
    ):
        raise SpecError(f"{path}.order must be a non-empty list of block names")
    if len(set(order)) != len(order):
        raise SpecError(f"{path}.order repeats a block name")
    return tuple(order)


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int
    evidence: str
    feature: str | None = None
    src_start: int | None = None
    src_end: int | None = None
    # trailing slots of this block that no source column fills. A robot whose head
    # has 2 joints still occupies the skeleton's 3-wide `neck`, and the odd slot out
    # is a zero. Declared rather than inferred from the arithmetic, so that a source
    # range that is accidentally too short is an error instead of a silent pad.
    pad: int = 0
    note: str | None = None

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def sourced_width(self) -> int:
        """Slots this block actually copies; ``width`` minus any pad."""
        return 0 if self.src_start is None else self.src_end - self.src_start

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
            for offset in range(block.sourced_width):
                slots[block.start + offset] = (path, block.src_start + offset)
        return slots

    def evidence_counts(self) -> dict[str, int]:
        """Slots per evidence value. Pad slots are counted as ``pad``, not as the
        block's evidence: nothing was measured about them."""
        counts: dict[str, int] = {}
        for block in self.blocks:
            sourced = block.width - block.pad
            if sourced:
                counts[block.evidence] = counts.get(block.evidence, 0) + sourced
            if block.pad:
                counts["pad"] = counts.get("pad", 0) + block.pad
        return counts

    def unbuildable(self) -> list[Block]:
        """Blocks whose values cannot be produced, in spec order.

        A sourceless ``constant`` block is fine: it is a body part the robot does not
        have, so zeros are the answer rather than a missing one. Every other
        sourceless block is a hole -- ``declared`` and ``inferred`` mean we know the
        width and the name but never found the columns.

        Filling those with zeros would produce a dataset that trains without
        complaint on a quarter of a vector that is silently blank, so the loader
        refuses instead. See ``DatasetSpec.buildable``.
        """
        return [
            block
            for block in self.blocks
            if block.feature is None and block.evidence != "constant"
        ]


@dataclass(frozen=True)
class SourceSpec:
    """Where the raw upstream files are and how to read them.

    Everything here is a path template, a key name or the name of a mechanism --
    never code. ``format`` and ``clock.strategy`` select from the closed sets
    ``spec2lerobot`` implements, which grow with the number of file formats, not
    with the number of datasets.

    Templates take ``{id}`` (the episode id) and ``{camera}`` (the source-side
    directory from ``lerobot.video.cameras``).
    """

    builder: str = "spec"
    # flags the builder needs that only the dataset knows, e.g. which end-effector
    # subset of AgiBot World this is. Merged into the run config's source.args.
    args: Mapping[str, Any] = field(default_factory=dict)
    # the directory this dataset occupies inside a collection of raw sources, when
    # that is not just its own id. Two datasets can share one: the AgiBot dexhand and
    # gripper subsets are both read out of AgiBotWorld-Beta and differ only by
    # --eef-type.
    raw_dir: str | None = None
    format: str | None = None
    discover: str | None = None
    paths: Mapping[str, str] = field(default_factory=dict)
    tasks: Mapping[str, str] = field(default_factory=dict)
    clock: Mapping[str, Any] = field(default_factory=dict)
    # logical feature name -> where to read it inside the raw file, per side. This is
    # the counterpart of ``lerobot.state.source_features``, which names the *emitted*
    # LeRobot columns. The two are different namespaces and conflating them silently
    # turns verification into a no-op, so they are stated separately.
    features: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # true width of each source feature, by logical name. A fact about the source,
    # not derivable from the layout: a layout that reads columns 0..31 says nothing
    # about whether the array has 32 columns or 44. Stating it turns "this file is
    # from a different robot" into a skipped episode instead of silently wrong data.
    feature_widths: Mapping[str, int] = field(default_factory=dict)

    @property
    def strategy(self) -> str:
        return self.clock["strategy"]

    @property
    def builds_its_own_vectors(self) -> bool:
        """True when the converter emits observation.state itself.

        The pre-existing converters do; the spec-driven path deliberately does not,
        leaving assembly to the layout step. A dataset built by one of the others is
        buildable without any block sources, because nothing here assembles it.
        """
        return self.builder not in ("spec", "none")


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    raw: Mapping[str, Any] = field(repr=False)
    source: SourceSpec | None = None
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

    def buildable(self) -> list[str]:
        """Why this dataset cannot be rebuilt from its source; empty means it can.

        Checked before a run rather than during one. A dataset that is missing the
        columns for a quarter of its action vector should stop the pipeline at
        planning time, not emit zeros for two days and look like it worked.
        """
        problems: list[str] = []
        if self.source is None:
            problems.append("no source: section, so the raw files cannot be read")
        elif self.source.builds_its_own_vectors:
            # the converter writes observation.state itself, so there is no layout
            # here to satisfy and the holes below do not block a rebuild
            return problems
        for side in ("state", "action"):
            vector = self.vector(side)
            if vector is None:
                problems.append(f"no {side} layout")
                continue
            for block in vector.unbuildable():
                problems.append(
                    f"{side}.{block.name} ({block.width} slots at "
                    f"{block.start}-{block.end - 1}) is {block.evidence} with no "
                    "source column"
                )
        return problems

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
    def delivered_episodes(self) -> int:
        """How many episodes the delivered copy holds.

        Stands in for how much parallel work a rebuild carries, because the
        transform parallelises per file and files are roughly episodes times
        cameras. 0 when the delivered copy did not record it.
        """
        return (self.raw.get("delivered") or {}).get("episodes") or 0

    @property
    def delivered_video(self) -> Mapping[str, Mapping[str, Any]]:
        """Per camera, what the delivered copy actually is: geometry and codec.

        Kept apart from ``lerobot.video.cameras``, which describes the *source*.
        A rebuild is checked by comparing the two, so conflating them would make the
        check compare a value with itself.
        """
        return (self.raw.get("delivered") or {}).get("video") or {}

    @property
    def embodiment_tag(self) -> str | None:
        return (self.raw.get("lerobot") or {}).get("embodiment_tag")

    @property
    def cameras(self) -> Mapping[str, Mapping[str, Any]]:
        """LeRobot camera key -> ``{"source": <dir in the raw source>, "shape": [h, w, c]}``.

        Cameras differ in size within one dataset -- humanoid_everyday carries a
        640x480 original beside a 256x192 resize -- so the shape belongs to the
        camera, not to the dataset.
        """
        return self._video.get("cameras") or {}

    def camera_source(self, key: str) -> str:
        """The directory this camera has in the raw source; defaults to its own name."""
        return (self.cameras.get(key) or {}).get("source") or key

    def camera_shape(self, key: str) -> tuple[int, int, int] | None:
        shape = (self.cameras.get(key) or {}).get("shape")
        return tuple(shape) if shape else None

    @property
    def _video(self) -> Mapping[str, Any]:
        return ((self.raw.get("lerobot") or {}).get("video") or {}) or {}

    @property
    def is_resized(self) -> bool:
        """Whether the delivered copy was resized, and so whether a rebuild should be.

        A fact about the dataset rather than a choice: it was read back from the
        delivered encoding, where AV1 with a two-frame GOP means LeRobot's own writer
        output survived untouched. Running a resize over a dataset that never had one
        re-encodes video that was meant to pass straight through, and what comes out
        is not the dataset being reproduced. The profile decides *how* to resize;
        this decides *whether*.
        """
        return bool(self._video.get("resize"))

    @property
    def fps(self) -> int | None:
        return (self.raw.get("lerobot") or {}).get("fps")

    @property
    def robot_type(self) -> str | None:
        return (self.raw.get("lerobot") or {}).get("robot_type")

    @property
    def modality(self) -> Mapping[str, Any] | None:
        """How the training stack slices the flat vectors, if the delivered dataset
        declares it.

        Distinct from ``state.blocks``, which records where each slot's *value* comes
        from. The two disagree often: action_net's provenance is eight body-part
        blocks but its modality.json is one flat 0..44 block, because the training
        config asks it for ``modality_keys=["state"]``. Provenance is how to build
        the vector; modality is how the model reads it.
        """
        return (self.raw.get("lerobot") or {}).get("modality")


def available() -> list[str]:
    return sorted(path.stem for path in DATASETS_DIR.glob("*.yaml"))


def load(name: str, layouts: Mapping[str, str] | None = None) -> DatasetSpec:
    """Load a dataset spec, optionally building it under different layouts.

    ``layouts`` renames layouts as the spec is parsed: ``{"gr1_body_parts":
    "gr1_canonical"}`` lays every dataset that declares the first out under the
    second. This is how a processing profile switches the whole collection to a new
    slot order without touching a single dataset spec.
    """
    import yaml

    path = DATASETS_DIR / f"{name}.yaml"
    if Path(name).name != name or not path.is_file():
        raise SpecError(f"unknown dataset {name!r}. available: {', '.join(available())}")
    return parse(yaml.safe_load(path.read_text()) or {}, origin=str(path), layouts=layouts)


def load_all() -> list[DatasetSpec]:
    return [load(name) for name in available()]


def parse(
    raw: Mapping[str, Any],
    origin: str = "<spec>",
    layouts: Mapping[str, str] | None = None,
) -> DatasetSpec:
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
    video = lerobot.get("video") or {}
    _reject(video, _VIDEO, f"{origin}.lerobot.video")
    for key, camera in (video.get("cameras") or {}).items():
        where = f"{origin}.lerobot.video.cameras.{key}"
        _reject(camera, _CAMERA, where)
        shape = camera.get("shape")
        if shape is not None and (
            not isinstance(shape, Sequence) or isinstance(shape, str) or len(shape) != 3
        ):
            raise SpecError(f"{where}.shape must be [height, width, channels]")

    state_raw = lerobot.get("state")
    action_raw = lerobot.get("action")
    source_raw = raw.get("source")
    return DatasetSpec(
        id=raw["id"],
        name=raw["name"],
        raw=raw,
        source=_parse_source(source_raw, f"{origin}.source") if source_raw else None,
        state=(
            _parse_state(state_raw, f"{origin}.lerobot.state", layouts)
            if state_raw
            else None
        ),
        action=(
            _parse_state(action_raw, f"{origin}.lerobot.action", layouts)
            if action_raw
            else None
        ),
    )


def _parse_source(raw: Any, origin: str) -> SourceSpec:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{origin} must be a mapping")
    _reject(raw, _SOURCE_SPEC, origin)

    def text(mapping: Mapping[str, Any], key: str, where: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value:
            raise SpecError(f"{where}.{key} must be a non-empty string, got {value!r}")
        return value

    builder = raw.get("builder", "spec")
    if builder not in BUILDERS:
        raise SpecError(
            f"{origin}.builder must be one of {', '.join(sorted(BUILDERS))}, "
            f"got {builder!r}"
        )
    builder_args = raw.get("args") or {}
    if not isinstance(builder_args, Mapping):
        raise SpecError(f"{origin}.args must be a mapping of flag -> value")

    raw_dir = raw.get("raw_dir")
    if raw_dir is not None and (not isinstance(raw_dir, str) or not raw_dir):
        raise SpecError(f"{origin}.raw_dir must be a non-empty string")

    if builder != "spec":
        # only the spec-driven path reads the file description; the others carry
        # their own, so demanding one here would be asking for fiction
        return SourceSpec(builder=builder, args=dict(builder_args), raw_dir=raw_dir)

    paths = raw.get("paths") or {}
    _reject(paths, _SOURCE_PATHS, f"{origin}.paths")
    tasks = raw.get("tasks") or {}
    _reject(tasks, _SOURCE_TASKS, f"{origin}.tasks")
    clock = raw.get("clock") or {}
    _reject(clock, _SOURCE_CLOCK, f"{origin}.clock")

    for key in _SOURCE_PATHS:
        text(paths, key, f"{origin}.paths")
    for key in _SOURCE_TASKS:
        text(tasks, key, f"{origin}.tasks")
    text(clock, "strategy", f"{origin}.clock")

    features = raw.get("features") or {}
    if not isinstance(features, Mapping):
        raise SpecError(f"{origin}.features must be a mapping of name -> {{state, action}}")
    for name, sides in features.items():
        if not isinstance(sides, Mapping) or set(sides) != {"state", "action"}:
            raise SpecError(
                f"{origin}.features.{name} must map both 'state' and 'action' to a "
                f"path inside the raw file, got {sides!r}"
            )

    widths = raw.get("feature_widths") or {}
    if not isinstance(widths, Mapping):
        raise SpecError(f"{origin}.feature_widths must be a mapping of name -> width")
    for name, value in widths.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SpecError(
                f"{origin}.feature_widths.{name} must be a positive integer, "
                f"got {value!r}"
            )

    return SourceSpec(
        builder=builder,
        args=dict(builder_args),
        raw_dir=raw_dir,
        format=text(raw, "format", origin),
        discover=text(raw, "discover", origin),
        paths=dict(paths),
        tasks=dict(tasks),
        clock=dict(clock),
        features={name: dict(sides) for name, sides in features.items()},
        feature_widths=dict(widths),
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


def _parse_state(
    raw: Any, origin: str, layouts: Mapping[str, str] | None = None
) -> StateSpec:
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

    declared_layout = raw.get("layout")
    if not isinstance(declared_layout, str) or not declared_layout:
        raise SpecError(f"{origin}.layout must name a layout in layouts/")
    layout_name = (layouts or {}).get(declared_layout, declared_layout)
    order = load_layout(layout_name)

    declared = raw.get("blocks")
    if not isinstance(declared, Mapping):
        raise SpecError(
            f"{origin}.blocks must be a mapping of block name -> {{width, source, "
            "evidence}}. It is deliberately not a list: a list would let the spec "
            f"imply an order that disagrees with layouts/{layout_name}.yaml"
        )
    missing = [name for name in order if name not in declared]
    extra = sorted(set(declared) - set(order))
    if missing or extra:
        raise SpecError(
            f"{origin}.blocks must declare exactly the blocks in "
            f"layouts/{layout_name}.yaml"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unexpected {', '.join(extra)}" if extra else "")
        )

    blocks: list[Block] = []
    start = 0
    for name in order:
        entry = declared[name]
        where = f"{origin}.blocks.{name}"
        _reject(entry, _BLOCK, where)

        block_width = entry.get("width")
        if not isinstance(block_width, int) or isinstance(block_width, bool) or block_width < 1:
            raise SpecError(
                f"{where}.width must be a positive integer, got {block_width!r}"
            )
        end = start + block_width
        evidence = entry.get("evidence")
        if evidence not in EVIDENCE:
            raise SpecError(
                f"{where}.evidence must be one of {', '.join(sorted(EVIDENCE))}, "
                f"got {evidence!r}"
            )

        pad = entry.get("pad", 0)
        if not isinstance(pad, int) or isinstance(pad, bool) or pad < 0 or pad >= block_width:
            raise SpecError(
                f"{where}.pad must be an integer in 0..{block_width - 1}, got {pad!r}"
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
            if (src_end - src_start) + pad != block_width:
                raise SpecError(
                    f"{where}: {block_width} slots and pad {pad}, but "
                    f"{src_end - src_start} source columns"
                )
        elif pad:
            raise SpecError(
                f"{where}: pad is only meaningful alongside a source; a block with "
                "no source is already entirely constant"
            )
        elif evidence not in SOURCELESS_EVIDENCE:
            raise SpecError(
                f"{where}: a block with no source must be one of "
                f"{', '.join(sorted(SOURCELESS_EVIDENCE))}, got {evidence!r}"
            )

        blocks.append(
            Block(
                name=name,
                start=start,
                end=end,
                evidence=evidence,
                feature=feature,
                src_start=src_start,
                src_end=src_end,
                pad=pad,
                note=entry.get("note"),
            )
        )
        start = end

    if start != width:
        raise SpecError(
            f"{origin}: block widths sum to {start}, but width says {width}"
        )
    _check_tiling(blocks, width, origin)
    return StateSpec(
        width=width,
        layout=layout_name,
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
