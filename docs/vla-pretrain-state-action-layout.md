# VLA pre-train dataset — what each state/action slot holds

Reference for the datasets under `/data/taeyoung/data/vla_pretrain_dataset`
(= `/storage1/sjw_dataset/dataset`, the RLDX-1 pre-training collection).

The code that produced these datasets does not exist. ALIN Lab confirmed on
2026-07-29 that preprocessing was done ad-hoc at Gaia/SKT and was not kept
([Notion](https://app.notion.com/p/3ac6cbdff6f680efaa98d85739443184), Q2). So the
layouts below were **recovered from the delivered data**, not read from a spec.

## Method

Every column of the flat `observation.state` / `action` vector was compared, across
several real episodes, against every column of the sub-feature vectors each dataset
also carries (`observation.robot_joints`, `observation.joint_position`, …). A column
matching exactly one source column is resolved; ties are narrowed by elimination
(the mapping is injective). Three traps had to be handled:

* `annotation.*` columns are constant scalars that spuriously equal any constant
  slot — excluded as sources;
* `observation.x` and `action.x` are identical whenever the robot tracks its command
  exactly, so `observation.state` only considers `observation.*` sources;
* all-zero columns are indistinguishable from each other. Those are reported as
  unresolved rather than guessed.

Script: `scratchpad/recover2.py`. Confidence per dataset is stated below.

## What the model actually consumes

Datasets do **not** need a common width. From `rldx/model/core/rldx.py`:

```python
self.state_encoder  = CategorySpecificMLP(num_categories=max_num_embodiments,   # 36
                                          input_dim=max_state_dim)              # 64
self.action_encoder = MultiEmbodimentActionEncoder(action_dim=64, num_embodiments=36)
self.action_decoder = CategorySpecificMLP(num_categories=36, output_dim=64)
```

and `processing_rldx.py` zero-pads each embodiment's state up to `max_state_dim`:

```python
normalized_states = torch.cat(
    [normalized_states, torch.zeros(n, self.max_state_dim - normalized_states.shape[1])], dim=-1)
```

So the common input is **64 zero-padded floats + an `embodiment_id`**, and each
embodiment has its own encoder/decoder weights. That is why 8-dim Franka and 44-dim
GR1 coexist in one mix.

> `max_state_dim` defaults to **32** in `processing_rldx.py` but **64** in
> `configs/model/rldx.py`. The real runs used 64 (`experiment_cfg/conf.yaml`).
> Reproducing with the wrong default silently truncates anything wider than 32.

### No cross-dataset alignment is needed

Each embodiment has its *own* weights, not a shared projection
(`model/modules/embodiment_conditioned_mlp.py`):

```python
self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))  # [36, 64, H]
selected_W = self.W[cat_ids]
return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

Slot 0 of `action_net` and slot 0 of `droid` multiply different matrices, so the two
orderings never have to agree. Normalization is per-embodiment too — `statistics.json`
is keyed by dataset (35 entries), with `action_net` holding one flat `state` block and
`droid` holding `end_effector_position` / `end_effector_rotation` / `gripper_position`.

What *is* required is consistency **within** an embodiment: the order used at training
must be the order used at inference. That is the real reason a rebuilt `action_net`
has to reproduce the 44-slot layout exactly — not to match other datasets, but to
match the projector the existing checkpoint already trained. Rebuild everything from
scratch and any self-consistent order would do.

The shared skeleton of family A is therefore a convention, not a constraint. It is not
even applied consistently: family B reorders the same blocks, family C has no skeleton
at all, and all of them trained together in one mix without trouble.

### Where the ordering *does* bite: `embodiment_id`

`EmbodimentTag`'s enum order is the projector index. Deleting a source from the middle
of the enum shifts every later embodiment onto the wrong weights. This came up when
planning the non-commercial exclusion (ActionNet is one of the three):

> Q4 … 소스를 mix에서 삭제하면 id 매핑이 밀려서 기존 PT 체크포인트와 호환이 깨지지 않습니까?
> **답변: data config에서 아예 삭제를 하면 된다.**

That answer is safe only when the checkpoint is being retrained. To keep an existing
checkpoint usable, drop a source with `mix_ratio: 0` and append new embodiments at the
end of the enum rather than renumbering.

## Layout family A — humanoid skeleton

Eight body-part blocks, left side before right:

```txt
left_arm | left_hand | left_leg | neck | right_arm | right_hand | right_leg | waist
```

Block widths vary per robot; absent body parts keep their slots as zeros.

| Dataset | l_arm | l_hand | l_leg | neck | r_arm | r_hand | r_leg | waist | total |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `action_net` | 7 | 6 | 6 | 3 | 7 | 6 | 6 | 3 | **44** |
| `agibot_dexhand` | 7 | 6 | 6 | 3 | 7 | 6 | 6 | 3 | **44** |
| `agibot_gripper0_part1‥4` | 7 | 1 | 6 | 3 | 7 | 1 | 6 | 3 | **34** |
| `neural_robocurate_v1‥3` | 7 | 6 | 6 | 3 | 7 | 6 | 6 | 3 | **44** |
| `march_robocurate_v1‥4` | 7 | 6 | 6 | 3 | 7 | 6 | 6 | 3 | **44** |
| `gen_all_lerobot_merged_clean` | 7 | 6 | 6 | 3 | 7 | 6 | 6 | 3 | **44** |

### `action_net` (44) — recovered from data

```txt
[ 0: 7] left_arm   <- robot_joints[18:25]   shoulder pitch/roll/yaw, elbow, wrist yaw/roll/pitch
[ 7:13] left_hand  <- hand_joints[0:6]
[13:19] left_leg   <- robot_joints[0:6]     hip roll/yaw/pitch, knee, ankle pitch/roll
[19:22] neck       <- robot_joints[15:18]   yaw, roll, pitch
[22:29] right_arm  <- robot_joints[25:32]
[29:35] right_hand <- hand_joints[6:12]
[35:41] right_leg  <- robot_joints[6:12]
[41:44] waist      <- robot_joints[12:15]   yaw, pitch, roll
```

**Confidence: 32/44 slots resolved directly**; the 12 leg slots are constant (hips
and knees identically zero, ankles a fixed ~1e-6) and are assigned by block
structure. `action` uses the same slots with `action.*` sources.

### `agibot_dexhand` (44) / `agibot_gripper` (34) — recovered from data

```txt
[ 0: 7] left_arm   <- joint_position[0:7]
[ 7:13] left_hand  <- effector_dexhand_qpos[0:6]     (gripper variant: [0:1], 1 wide)
[..:..] left_leg   <- zeros                          AgiBot A2D has no legs
[..:..] neck       <- head_position[0:2] + 1 zero    2-DoF head in a 3-wide block
[..:..] right_arm  <- joint_position[7:14]
[..:..] right_hand <- effector_dexhand_qpos[6:12]    (gripper variant: [1:2])
[..:..] right_leg  <- zeros
[..:..] waist      <- waist_position[0:2] + 1 zero   2-DoF waist in a 3-wide block
```

**Confidence: 30/44 (dexhand), 20/34 (gripper) resolved directly**; the rest are the
zero leg blocks and the pad slot at the end of `neck` and `waist`. The widths sum
exactly to 44 and 34, which is what confirms the skeleton.

### `neural_robocurate`, `march_robocurate`, `gen_all_lerobot_merged_clean`

**Not recovered — declared.** These carry no sub-feature vectors, but their
`meta/modality.json` names the blocks outright:

```json
{"left_arm": {"start": 0, "end": 7}, "left_hand": {"start": 7, "end": 13},
 "left_leg": {"start": 13, "end": 19}, "neck": {"start": 19, "end": 22},
 "right_arm": {"start": 22, "end": 29}, "right_hand": {"start": 29, "end": 35},
 "right_leg": {"start": 35, "end": 41}, "waist": {"start": 41, "end": 44}}
```

This is the independent confirmation that family A is a real convention and not a
coincidence of `action_net` and `agibot`.

## Layout family B — arms then hands

`humanoid_everyday` uses the same block *vocabulary* in a different *order*: both
arms first, then both hands. Declared in its `modality.json` and confirmed against
the data.

```txt
humanoid_everyday_g1 (28):  [0:7] left_arm | [7:14] right_arm | [14:21] left_hand | [21:28] right_hand
humanoid_everyday_h1 (26):  [0:7] left_arm | [7:14] right_arm | [14:20] left_hand | [20:26] right_hand
```

Recovered runs: `[0:14] <- arm_joints[0:14]`, `[14:28] <- hand_joints[0:14]`.
**Confidence: 28/28 and 26/26 resolved directly.**

Note these datasets carry far more sub-features than they use — IMU quaternion /
accelerometer / gyroscope / rpy, odometry, leg joints, tactile — none of which enter
`observation.state`.

## Layout family C — per-robot, no skeleton

### `galaxea_part1‥5` — state 18, action 26

```txt
[ 0: 6] <- joint_position_arm_left[0:6]
[ 6: 7] <- gripper_state_left[0:1]
[ 7:13] <- joint_position_arm_right[0:6]
[13:14] <- gripper_state_right[0:1]
[14:18] <- joint_position_torso[0:4]
```

**Confidence: 18/18 resolved directly**, identically across all five parts.
`modality.json` is a flat `{"state": {"start": 0, "end": 18}}`. Note state and
action differ in width (18 vs 26), and this family also carries `original_action`
alongside `absolute_action`.

### OXE (27 datasets) — state 8, action 7

**Not recoverable from the delivered data**: no sub-feature vectors, flat
`modality.json`, anonymous `m0..m7` names. The semantics come from the conversion
convention in this repo's `openx2lerobot/openx_rlds.py`, keyed on the dataset's
`state_encoding`:

| encoding | slots 0‥7 |
| --- | --- |
| `POS_EULER` | `x, y, z, roll, pitch, yaw, pad, gripper` |
| `POS_QUAT` | `x, y, z, rx, ry, rz, rw, gripper` |
| `JOINT` | `motor_0‥motor_6, gripper` (unused joints become `pad`) |

Action is 7 wide: `x, y, z, roll/axis_angle …, gripper`.

**This is inference, not measurement** — the delivered files do not record which
encoding each dataset used. Confirm per dataset against `OXE_DATASET_CONFIGS` before
relying on it.

## Summary of confidence

| Family | Datasets | Basis | Slots resolved |
| --- | --- | --- | --- |
| A — humanoid skeleton | action_net, agibot ×5 | measured | 32/44, 30/44, 20/34 |
| A — humanoid skeleton | neural_traj ×3, robocurate ×4, oneuniverse | declared in `modality.json` | 44/44 |
| B — arms then hands | humanoid_everyday ×2 | measured + declared | 28/28, 26/26 |
| C — per-robot | galaxea ×5 | measured | 18/18 |
| C — per-robot | OXE ×27 | inferred from `openx2lerobot` | 0/8 measured |

Unresolved slots are, without exception, blocks of identical constants (usually
zeros for a body part the robot does not have). They cannot be disambiguated from
the data because they are literally interchangeable — which also means getting their
internal order wrong has no numerical effect.
