# Dataset registry

One YAML per dataset, saying where it came from, how to read it, and how its vectors
are laid out. Adding a dataset whose source format is already implemented is this
file and nothing else — no Python.

```bash
uv run python -m dataset_registry.verify --upstream --all      # re-check the pins online
uv run python -m dataset_registry.verify action_net /path/to/dataset
```

`verify` checks a spec against a dataset: does each slot really hold the column the
spec says.

Checking a rebuild against the delivered copy is **not** here. That is
[`verification`](../verification), which reads these specs and is read by nothing in
this directory — what a rebuild is measured against must not be able to move when the
thing being measured does.

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

```yaml
lerobot:
  video:
    cameras:
      image: {shape: [128, 128, 3], modality: primary}   # a third name
      image_2: {shape: [256, 256, 3], modality: null}    # on disk, not exposed
```

A camera has **three** names and all three are in use at once. cmu_stretch's is the
feature `observation.images.image`, its video directory is named after that in full,
and `meta/modality.json` calls it `primary` — the alias `openx2lerobot`'s
`image_obs_keys` maps the source key to. The training stack reads the alias, so getting
it wrong renames the camera as far as the model is concerned. `modality` is read back
from the delivered `modality.json` files, for the same reason `resize` is: it is a fact
about what was shipped. Only the 19 OpenX sets whose alias differs from their camera key
declare it; for the other 8 the two are already the same word.

`modality: null` means **on disk and not exposed**. bridge_orig keeps two spare views of
four and berkeley_cable_routing a fourth wrist angle; the delivered modality files list
neither, and a camera nothing reads is not a camera as far as the training stack is
concerned. All 36 delivered `modality.json` files' `video` blocks are reproduced from
these declarations, which is checked in `tests/test_registry.py`.

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
