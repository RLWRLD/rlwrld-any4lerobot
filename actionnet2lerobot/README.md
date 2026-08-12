# Fourier ActionNet to LeRobot

Converts the [Fourier ActionNet](https://action-net.org/) dataset (GR1-T1 humanoid,
30k teleoperated manipulation episodes) into the schema used by the RLDX-1
pre-training collection.

## Which schema this targets

Not the reference converter published with the dataset (`FFTAI/fourier-lerobot`).
That one emits a 32-dim state with the legs dropped, keeps end-effector pose
features, and names joints explicitly — none of which the training stack here
consumes. This converter targets the schema of the delivered dataset at
`/data/taeyoung/data/vla_pretrain_dataset/action_net`.

The code that produced that dataset does not exist: ALIN Lab confirmed on
2026-07-29 that preprocessing was done ad-hoc and was not kept. The schema below
was therefore recovered from the delivered artifacts and cross-checked against
RLDX-1's training config.

## Source layout

After untarring the dataset:

```txt
metadata.json                    # [{id, prompt, task, machine_id}, ...] for every episode
01JH00FCRH6EIBDXTA.hdf5          # robot side: state/, action/, timestamp
01JH00FCRH6EIBDXTA/
└── top/
    ├── rgb.mp4                  # h264, 800x1280
    ├── depth.mkv                # z16 depth (not converted, see below)
    └── timestamps.json          # wall-clock time of every rgb frame
```

## Features

| LeRobot feature | Width | Source |
| --- | --- | --- |
| `observation.images.primary` | 800x1280 | `<id>/top/rgb.mp4` |
| `observation.state` | 44 | `state/robot` + `state/hand`, reordered |
| `action` | 44 | `action/robot` + `action/hand`, reordered |
| `absolute_action` | 44 | copy of `action` |
| `observation.robot_joints` | 32 | `state/robot` unchanged |
| `observation.hand_joints` | 12 | `state/hand` unchanged |
| `action.robot_joints` | 32 | `action/robot` unchanged |
| `action.hand_joints` | 12 | `action/hand` unchanged |

`robot_type` is `ActionNet`, fps is 30, and motors are named positionally
(`m0`…`m43`) as every dataset in the collection does.

### The 44-dim layout

`observation.state` and `action` are the GR1's whole body, grouped by body part:

```txt
[ 0: 7] left_arm     <- robot[18:25]    shoulder pitch/roll/yaw, elbow, wrist yaw/roll/pitch
[ 7:13] left_hand    <- hand[0:6]
[13:19] left_leg     <- robot[0:6]      hip roll/yaw/pitch, knee, ankle pitch/roll
[19:22] neck         <- robot[15:18]    yaw, roll, pitch
[22:29] right_arm    <- robot[25:32]
[29:35] right_hand   <- hand[6:12]
[35:41] right_leg    <- robot[6:12]
[41:44] waist        <- robot[12:15]    yaw, pitch, roll
```

Recovered by comparing `observation.state` column-by-column against
`observation.robot_joints` / `observation.hand_joints` in a delivered parquet, and
independently confirmed by `neural_gr1`'s `modality.json` in RLDX-1, which names
the same eight blocks at the same offsets. `tests/test_layout.py` pins it.

The legs do not move during teleoperated manipulation — hips and knees are
identically zero, only the ankles carry values — but they keep their slots so the
vector is the GR1's whole body.

`meta/modality.json` exposes state and action as **one flat block**, matching the
delivered dataset. Splitting them into the eight named blocks (as `neural_gr1`
does) would not match `rldx/configs/data/pt_data_config.py`, which asks ActionNet
for `modality_keys=["state"]`.

## Timeline alignment

The robot streams at ~60 Hz and the camera at ~30 Hz, on separate clocks, so the
two are matched by timestamp rather than by index. This is a port of the reference
converter's matcher, including both of its filters: robot samples whose timestamp
does not advance are dropped, and so are video frames recorded at or after the last
robot sample. The second usually costs the final frame, so an episode is typically
one row shorter than its mp4 — the reference converter copies the mp4 unchanged and
leaves that trailing frame unreferenced, and so does this one.

Copying rather than re-encoding is why conversion is cheap: the mp4 is already
h264 and nothing about it needs to change at this stage. Resizing happens later, as
a pipeline step (see below).

## Not converted

* **`depth.mkv`.** The frames are z16, so the file cannot be reused as-is; LeRobot
  would need the depth decoded to arrays and re-encoded through
  `DepthEncoderConfig`. The delivered dataset has no depth either.
* **Point clouds.** The reference converter derives a 4096-point cloud per frame
  from the depth video. It is a Fourier-training-specific feature, not a LeRobot
  one, and is computed in a per-frame Python loop.

## Usage

Conversion alone, producing LeRobot v3.0 at full resolution:

```bash
uv sync --extra actionnet
uv run python actionnet2lerobot/actionnet_h5.py \
    --src-path /path/to/action_net \
    --output-path /path/to/output \
    --executor local \
    --episodes-per-task 100
```

`--episodes-per-task` sets how many episodes go into each temporary dataset; one
task is scheduled per chunk, so it controls the granularity of the parallelism.
Episodes are independent, so any chunking is correct — smaller chunks balance a
large machine better, larger chunks mean less aggregation work at the end.

Episodes missing an hdf5, an mp4 or a `timestamps.json`, and episodes whose joint
widths do not match GR1-T1, are skipped with a logged reason rather than failing
the chunk.

To get the delivered dataset's actual shape — resized to 192x288 and downgraded to
v2.1 — drive it from the pipeline instead:

```bash
uv run python -m lerobot_pipeline.run --config lerobot_pipeline/configs/actionnet_v21.yaml
```

That config pins the `rldx1_reference` encoding profile, which reproduces how the
delivered videos were encoded (libx264 High / yuv420p / GOP 250 / 3 B-frames).
