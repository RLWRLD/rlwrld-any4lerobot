# Fourier ActionNet to LeRobot

Converts the [Fourier ActionNet](https://action-net.org/) dataset (GR1-T1 humanoid,
30k teleoperated manipulation episodes) to LeRobot v3.0.

## Source layout

After untarring the dataset:

```txt
metadata.json                    # [{id, prompt, task, machine_id}, ...] for every episode
01JH00FCRH6EIBDXTA.hdf5          # robot side
01JH00FCRH6EIBDXTA/
└── top/
    ├── rgb.mp4                  # h264, 800x1280
    ├── depth.mkv                # z16 depth
    └── timestamps.json          # wall-clock time of every rgb frame
```

## Feature mapping

Follows the reference converter published with the dataset
(`FFTAI/fourier-lerobot`, `scripts/convert_hdf5_to_lerobot.py`), so the result lines
up with what Fourier's own training pipeline expects.

| LeRobot feature | Source | Width |
| --- | --- | --- |
| `observation.images.top` | `<id>/top/rgb.mp4` | 800x1280 |
| `observation.state` | `state/robot[:, 12:]` + `state/hand` | 20 + 12 |
| `action` | `action/robot[:, 12:]` + `action/hand` | 20 + 12 |
| `observation.state.pose` | `state/pose` minus head xyz + `state/hand` | 24 + 12 |
| `action.pose` | `action/pose` + `action/hand` | 24 + 12 |

The first 12 of the 32 joints are the legs, which do not move during teleoperated
manipulation, so both state and action drop them. `state/pose` carries a position
and an ortho6d rotation for the left hand, right hand and head; `action/pose` has no
head position, so the head xyz is dropped from the state side to match.

`--no-pose` emits only `observation.state` / `action`.

## Timeline alignment

The robot streams at ~60 Hz and the camera at ~30 Hz, on separate clocks, so the two
have to be matched by timestamp rather than by index. Each video frame takes the
robot sample nearest to it in time, and no robot sample is claimed twice.

The matching is a port of the reference converter's, including both of its filters:
robot samples whose timestamp does not advance are dropped, and so are video frames
recorded at or after the last robot sample. The second one usually costs the final
frame, so an episode is typically one row shorter than its mp4. The reference
converter copies the mp4 unchanged and leaves that trailing frame unreferenced, and
so does this one — see `generic_converter/prerendered_video.py`.

Copying rather than re-encoding is the reason the conversion is cheap: the mp4 is
already h264 and nothing about it needs to change.

## Not converted

* **`depth.mkv`.** The frames are z16, so reusing the file as-is is not possible;
  LeRobot would need the depth decoded to arrays and re-encoded through
  `DepthEncoderConfig`. Nothing here reads it yet.
* **Point clouds.** The reference converter derives a 4096-point cloud per frame
  from the depth video. It is a Fourier-training-specific feature, not a LeRobot
  one, and it is computed in a per-frame Python loop.

## Usage

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

Episodes missing an hdf5, an mp4 or a `timestamps.json`, and episodes whose state
or action widths do not match GR1-T1, are skipped with a logged reason rather than
failing the chunk.

To go on to v2.1, drive it from the pipeline instead:

```bash
uv run python -m lerobot_pipeline.run lerobot_pipeline/configs/actionnet_v21.yaml
```
