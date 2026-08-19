# Dataset registry

One YAML per dataset, saying where it came from, how to read it, and how its vectors
are laid out. Adding a dataset whose source format is already implemented is this
file and nothing else — no Python.

```bash
uv run python -m dataset_registry.verify --upstream --all      # re-check the pins online
uv run python -m dataset_registry.verify action_net /path/to/dataset
uv run python -m dataset_registry.compare action_net --rebuilt /out --delivered /ref
uv run python -m dataset_registry.compare cmu_stretch --rebuilt /out --report records/
```

`verify` checks a spec against a dataset: does each slot really hold the column the
spec says.

`compare` asks how closely a rebuild matches the delivered copy, in **four steps**
rather than one verdict — because one verdict was the wrong shape. The rebuilds do not
control the order they write episodes in, some episodes land a row or two short, and
the two copies do not carry the same set of metadata files, so "does it reproduce"
collapses several answers into one and loses the useful ones. The steps run cheapest
and widest first, and each one explains the next.

| step | question | how it is judged |
|---|---|---|
| 1 · declaration | do the two even claim to be the same dataset | every `meta/info.json` field except the counts |
| 2 · episodes | how much of the delivered copy is there | **counted, never failed** |
| 3 · sample | is what is there the same | state and action exactly; video loosely on bytes and exactly on everything else |
| 4 · distributions | do the two describe the same data | over every episode **and** over the episodes they share |

Every step prints a verdict word next to its numbers, and the report ends with one line
saying which step decided the run. A reader should not have to know that `1.5e-15` is a
pass and `5.3e-02` is not.

**Step 1** diffs the two `info.json` files as dotted paths rather than against a schema
of the file — it has gained fields across LeRobot versions, and a comparison that listed
the ones it knew would go quiet on the rest. fps, robot type, feature shapes and dtypes,
codec and pixel format are failures; `total_episodes` and its neighbours are set aside,
because they follow from step 2 and failing here would report one finding twice.

**Step 2** pairs episodes on their own state and action bytes, since they carry no
source id and the rebuild does not write them in the delivered order. It pairs twice —
exactly, then on the first eight rows — so an episode that is *present but different*
is counted apart from one that is genuinely absent. The second pass catches two things
and only one of them is a trimmed tail, so the report splits them: a row or two short,
against the same length with values that go wrong past row 8.

**Step 3** holds the two halves to different standards: **state and action must be
identical**, since every slot is a copied float32, while **video bytes must only be
close**, since two ffmpeg builds given the same flags do not emit the same file.
Geometry, frame count, codec, **keyframe interval** and the **pictures** must match
exactly: those are decided by our settings, not by the encoder build. The keyframe
interval is read off the frames rather than the stream header, which does not carry it,
and the pictures are checked because a size ratio cannot see them — a rebuild with red
and blue exchanged came within 1% of the delivered size. Channel order is *not* a check
of its own; it is consulted only after the frames have already disagreed, to name the
likely cause. A time offset or a crop position is the next thing it could have been,
and one dedicated test per cause never ends.

Half the sample comes from each **end** of the paired index range. Taking the first N
looked thorough and was not: `openx2lerobot` converts in chunks of twenty-five
episodes, one worker to a chunk, so the front of the range is a single worker's output.
Two things also span *every* episode, from file metadata alone: the **task prompt** of
every paired episode, and each camera's **file count and mean size**. Without the
second, a rebuild that wrote a tenth of its videos passed on the sampled ones.

**Step 4** is the pair that makes the arrangement worth having. If the two rows
disagree, the difference is the episodes outside the shared set from step 2 and nothing
more; if the shared-episode row is *also* off, the values themselves are wrong. It reads
the parquet rather than `meta`, because the delivered copies carry quantiles in
`episodes_stats.jsonl` and a rebuild does not — the v3.0 → v2.1 downgrade keeps only the
five legacy keys — so comparing what each *says* about itself would compare the writers,
not the data.

The exit status follows steps 1, 3 and 4, the last over the shared episodes. It ignores
step 2: a run that lost episodes should still be able to say whether the ones it has are
right.

### The record a run leaves

`--report DIR` writes `DIR/<dataset>.json` and `DIR/<dataset>.txt`. The runs happen on
throwaway nodes, one dataset to a node, so the records are collected into one directory
afterwards and committed — which is why the JSON carries the **thresholds that decided
the verdict** rather than only the verdict, the **index map** of which rebuilt episode
was compared against which delivered one, and the **full per-dimension statistics** for
both sides rather than only the gap they were reduced to. The datasets are far too large
to keep beside the record, and a number nobody can recompute is a number nobody can
argue with. Re-running a dataset overwrites its own two files and nothing else.

Two things read these specs: [`spec2lerobot`](../spec2lerobot) converts a raw source
using the `source:` section, and `lerobot_pipeline`'s `state_layout` step assembles
the flat vectors using the `state:` / `action:` sections.

## Why this exists

The RLDX-1 pre-training datasets were built ad-hoc and **the conversion code was not
kept** — confirmed by ALIN Lab on 2026-07-29
([Notion](https://app.notion.com/p/3ac6cbdff6f680efaa98d85739443184), Q2: "변환용
전처리 코드는 없음… 즉석해서 처리해서 코드가 없음"). Rebuilding any of them means
reproducing a layout nobody wrote down.

So the layouts here were **recovered from the delivered data**: every column of
`observation.state` was matched, across several real episodes, against the columns of
the sub-feature vectors each dataset also carries. That makes them claims rather than
facts, which drives two design choices — the `evidence` field, and `verify.py`.

## Identity is the upstream release, not a mirror

A dataset is identified by its HuggingFace repo plus the revision that pins its bytes.
Mirrors are copies of that release and can come and go.

```yaml
upstream:
  huggingface: FourierIntelligence/ActionNet
  revision: eb3b5bb16f3a5a7052c1498c1072e78ac99042d0

mirrors:
  - kind: foundry
    uri: s3://rlwrld-foundry-data/external/action_net/eb3b5bb16f3a5a7052c1498c1072e78ac99042d0/
    layout: tar_archives
```

Foundry stores external sources under `external/<name>/<upstream revision>/`, so the
mirror path is a function of the upstream pin. Parsing enforces that a `foundry`
mirror's URI contains the declared revision — a mismatch means the mirror holds a
different release than the spec claims. `--upstream` re-checks the pins against the
HuggingFace API, so the whole mapping can be validated online without reading a byte
of any mirror.

Only three sources are mirrored into foundry today; `mirrors` is optional and grows.

## `evidence` — how much to trust each block

| value | meaning |
| --- | --- |
| `measured` | matched column-by-column against delivered episodes |
| `declared` | named outright by the dataset's own `meta/modality.json` |
| `inferred` | follows from conversion-code convention, not from the delivered bytes |
| `constant` | the slots never vary — indistinguishable from each other, and for the same reason harmless to get wrong |
| `unknown` | width is known, composition is not |

Only `measured` and `constant` are checkable, and `verify.py` checks exactly those.
A block with no `source` may not claim to be `measured` — there would have been
nothing to match it against.

## Layout shape — slots are derived, never written

A dataset says which body parts it has and how wide they are. `layouts/*.yaml` says
what order they go in. Slot numbers fall out of the two.

```yaml
lerobot:
  state:
    width: 44                                      # a checksum on the widths below
    layout: gr1_body_parts
    source_features:                               # emitted LeRobot column names
      robot_joints: {state: observation.robot_joints, action: action.robot_joints}
    blocks:                                        # a mapping, deliberately
      left_arm:
        width: 7
        source: {feature: robot_joints, columns: [18, 25]}
        evidence: measured
      neck:
        width: 3
        pad: 1                                     # a 2-DoF head in a 3-wide block
        source: {feature: head, columns: [0, 2]}
        evidence: measured
```

```yaml
# layouts/gr1_body_parts.yaml
order: [left_arm, left_hand, left_leg, neck, right_arm, right_hand, right_leg, waist]
```

`blocks` is a mapping and not a list on purpose: a list would let a spec imply an
order of its own, which could disagree with its layout. With order in one file,
switching the whole collection to a different slot order is one edit, and no dataset
spec has to change — that matters because the delivered order is load-bearing while
the current checkpoint is in use, and will be replaced later.

Validation: the block names must match the layout's exactly, the widths must sum to
`width`, and each block's source range must equal its width minus its `pad`.

`state.slot_map(side)` returns, per slot, the `(column, index)` it takes. That is
what the layout step consumes; it is more general than a permutation, which only
works when source and target widths match. Where `action` differs from `state`
(Galaxea: 18 vs 26) a separate `action:` section overrides.

## Reading the raw source

`source:` describes the files, in templates and key names rather than code.

```yaml
source:
  format: hdf5_episodes                            # a reader in spec2lerobot/formats
  discover: "*.hdf5"
  paths:
    episode: "{id}.hdf5"
    video: "{id}/{camera}/rgb.mp4"
  tasks: {file: metadata.json, key: id, prompt: prompt}
  clock:
    strategy: nearest_timestamp_dedup              # a strategy in spec2lerobot/clocks
    data: timestamp
    image: "{id}/{camera}/timestamps.json"
    image_format: "%Y-%m-%dT%H-%M-%S_%f"
  features:                                        # where to read, inside the file
    robot_joints: {state: state/robot, action: action/robot}
  feature_widths: {robot_joints: 32}               # what the source really is
```

`source.features` and `state.source_features` are different namespaces: the first is
where a vector is *read from*, the second is what it is *emitted as*. Conflating them
makes `verify.py` silently match nothing.

`feature_widths` is a fact the layout cannot supply — reading columns 0..31 says
nothing about whether the array has 32 columns or 44. Stating it turns "this file is
from a different robot" into a skipped episode instead of quietly wrong data.

## Who builds it, and is the video touched

```yaml
source:
  builder: openx          # spec | openx | agibot | libero | robocasa | robomind | none
  args: {eef_type: gripper}   # flags only the dataset knows

lerobot:
  video:
    resize: null          # this dataset was never resized; do not run a video step
    encoding: lerobot_av1_default
```

`builder` is which program turns the raw source into LeRobot. `spec` is the
data-driven path in `spec2lerobot`; the others are the converters this repo already
had, which write `observation.state` themselves and so skip `state_layout`; `none`
means the source is already LeRobot.

`resize` is a **fact, not a choice** — it was read back from the delivered encoding.
AV1 with a two-frame GOP is LeRobot's own writer default and survives only if
nothing re-encoded the file, so those datasets get no video step at all: running one
would re-encode video that was meant to pass straight through. The profile decides
*how* to resize; the spec decides *whether*.

`source.args` carries flags only the dataset knows. `agibot2lerobot` converts one
end-effector type per run, so without `eef_type` the dexhand and gripper subsets
would come out identical. Args are checked against the converter's own `argparse`
at config-load time, so a typo fails immediately rather than hours into a run.

## Can it be rebuilt?

`spec.buildable()` returns the reasons it cannot, and both the converter and the
layout step refuse to run when it returns anything. A sourceless `constant` block is
fine — it is a body part the robot does not have, so zeros are the answer. Every
other sourceless block is a hole, and filling it with zeros would produce a dataset
that trains without complaint on a blank stretch of vector.

## Status

| dataset | state | evidence | foundry |
| --- | --: | --- | --- |
| `action_net` | 44 | 32 measured, 12 constant | ✅ |
| `agibot_dexhand` | 44 | 30 measured, 14 constant | — |
| `agibot_gripper` | 34 | 20 measured, 14 constant | — |
| `galaxea` | 18 | 18 measured (action 26 unknown) | ✅ |
| `humanoid_everyday_g1` | 28 | 28 declared | ✅ |
| `neural_robocurate` | 44 | 44 declared | — |

Buildable today: `action_net`. The rest are missing either a `source:` section or the
columns for part of a vector; `python -m lerobot_pipeline.plan` reports which.

Not yet written: the 27 OpenX datasets. They carry no sub-feature vectors and a flat
`modality.json`, so their 8 slots cannot be recovered from the delivered data at all;
the semantics would have to come from `openx2lerobot/openx_rlds.py`'s `state_encoding`
convention and be marked `inferred`.

Background on how the layouts were recovered, and on what the model does with them:
[`docs/vla-pretrain-state-action-layout.md`](../docs/vla-pretrain-state-action-layout.md).
