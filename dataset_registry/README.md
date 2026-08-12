# Dataset registry

One YAML per dataset, saying where it came from and how its vectors are laid out.

```bash
uv run python -m dataset_registry.verify --upstream --all      # re-check the pins online
uv run python -m dataset_registry.verify action_net /path/to/dataset
```

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

## Layout shape

```yaml
lerobot:
  state:
    width: 44
    layout: gr1_body_parts
    source_features:
      robot: {state: state/robot, action: action/robot}
    blocks:
      - name: left_arm
        slots: [0, 7]                              # half-open, into the emitted vector
        source: {feature: robot, columns: [18, 25]}
        evidence: measured
```

Blocks must tile the vector exactly — a gap is a slot nobody can explain, an overlap
is two claims about the same number, and both are invisible once training starts.

`state.slot_map(side)` returns, per slot, the `(source path, column)` it takes. That
is what a converter consumes; it is more general than a permutation, which only works
when source and target widths match. Where `action` differs from `state` (Galaxea:
18 vs 26) a separate `action:` section overrides.

## Status

| dataset | state | evidence | foundry |
| --- | --: | --- | --- |
| `action_net` | 44 | 32 measured, 12 constant | ✅ |
| `agibot_dexhand` | 44 | 30 measured, 14 constant | — |
| `agibot_gripper` | 34 | 20 measured, 14 constant | — |
| `galaxea` | 18 | 18 measured (action 26 unknown) | ✅ |
| `humanoid_everyday_g1` | 28 | 28 declared | ✅ |
| `neural_robocurate` | 44 | 44 declared | — |

Not yet written: the 27 OpenX datasets. They carry no sub-feature vectors and a flat
`modality.json`, so their 8 slots cannot be recovered from the delivered data at all;
the semantics would have to come from `openx2lerobot/openx_rlds.py`'s `state_encoding`
convention and be marked `inferred`.

Background on how the layouts were recovered, and on what the model does with them:
[`docs/vla-pretrain-state-action-layout.md`](../docs/vla-pretrain-state-action-layout.md).
