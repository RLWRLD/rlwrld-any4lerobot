# LeRobot Pipeline

Config-driven preprocessing for LeRobot datasets.

One YAML file describes a whole run — where the data comes from, which
preprocessing steps to apply in order, and what the final output should be. The
CLI reads that config, runs each stage in sequence, and keeps only the result.

```
source  →  [steps]  →  dest
```

## Why

RLDX-family models take images at their **original aspect ratio**, not square
resizes, and resizing during training is disabled — so the dataset itself has to
be preprocessed. That used to mean a one-off script per job. This package makes
the flow declarative and reusable.

## Quick start

```bash
python -m lerobot_pipeline.run --config lerobot_pipeline/configs/actionnet_rldx1.yaml
```

See what a config resolves to before running it — stages, the full slot map, the
video filters and encoder settings, and whether the dataset can be rebuilt at all:

```bash
python -m lerobot_pipeline.plan --config lerobot_pipeline/configs/actionnet_rldx1.yaml
```

Measure throughput on a sample before committing to a long run:

```bash
python -m lerobot_pipeline.bench --config lerobot_pipeline/configs/humanoid_everyday_g1_rldx.yaml --sample 20
```

## Three layers

A run is described by three files, each of which changes on a different schedule.

| layer | file | holds | changes |
| --- | --- | --- | --- |
| **fact** | `dataset_registry/datasets/*.yaml` | upstream pin, file layout, state layout | when a dataset is added |
| **convention** | `configs/profiles/*.yaml`, `dataset_registry/layouts/*.yaml` | block order, resize, encoding, output version | when the convention changes |
| **run** | `configs/*.yaml` | paths, workers, overrides | every run |

The profile is where the processing convention lives, state and video side by side:

```yaml
# configs/profiles/rldx1.yaml
state:
  layouts: {gr1_body_parts: gr1_body_parts}   # change a right-hand side to re-lay-out
video:
  resize: {type: resize_preserve_aspect_area, max_area: 65536, multiple: 32}
  encoding: rldx1_reference
dest:
  version: lerobot_v21
```

so a run config carries only what is specific to the run:

```yaml
name: actionnet_rldx1
dataset: action_net        # -> the registry: source layout, state layout, identity
profile: rldx1             # -> the convention above
source: {path: /scratch/data, args: {executor: local}}
dest: {path: /scratch/out/actionnet_rldx1}
```

Everything a profile supplies is a default; anything written in the run config wins,
so a one-off run does not need a second profile file.

## Config

```yaml
name: humanoid_everyday_g1_rldx      # used for logs and the work directory

dataset: action_net                  # optional; a registry entry
profile: rldx1                       # optional; a named convention

source:
  type: lerobot_v21                  # lerobot_v21 | lerobot_v30 | spec | agibot
                                     # | libero | openx | robocasa | robomind
  path: ~/data/humanoid_everyday/humanoid_everyday_g1
  args: {}                           # extra converter CLI flags; converter sources only

steps:
  - type: resize_preserve_aspect_area
    max_area: 65536                  # 256 ** 2
    multiple: 32                     # image patch size
    keys: null                       # null applies to every video key

dest:
  type: lerobot_v21                  # lerobot_v21 | lerobot_v30
  path: ~/data/humanoid_everyday/humanoid_everyday_g1_rldx

runtime:                             # optional; everything is auto-tuned by default
  workers: auto
  threads_per_ffmpeg: auto
  preset: null                       # null mirrors the source encoding
  crf: null
```

Validation is strict — an unknown key is an error rather than being ignored,
because a silently dropped key means a run that looks successful but did the
wrong thing.

## Steps

Steps are referenced by **registered name only**. Arbitrary import paths are not
accepted, so a typo fails config validation instead of importing something
unexpected.

| Step | Kind | What it does |
|---|---|---|
| `resize_preserve_aspect_area` | video | Downscale so `H*W <= max_area` with the aspect ratio preserved, then centre-crop both sides to multiples of `multiple`. |
| `state_layout` | table | Assemble `observation.state` and `action` from the source's own feature vectors, following the dataset's layout, and write `meta/modality.json`. |

Three kinds of step:

- **`video`** — operates directly on the mp4 files. One decode/encode pass.
- **`table`** — rewrites `parquet` columns and leaves the frame count alone, so
  every video file is hard-linked straight through.
- **`frame`** — changes how many frames there are, so the video has to be re-cut
  too. **Not implemented**; the interface is reserved.

Table steps run before video steps: they are cheap, and a failure there costs
seconds rather than a re-encode.

### Why the layout is a step and not part of each converter

Datasets that arrive as raw HDF5 and datasets that arrive already in LeRobot form go
through the same code, so the convention has one implementation instead of one per
source. And changing the convention later rewrites `parquet` only — re-laying-out
ActionNet takes minutes, against the days a full reconversion of 2.49 TiB would.

Several video steps compose into a *single* ffmpeg filter chain, so adding steps
does not add passes over the data.

## How a run is assembled

```
source type          steps          dest type        stages
-------------------  -------------  ---------------  -----------------------------------
lerobot_v21          video          lerobot_v21      transform
lerobot_v21          video          lerobot_v30      transform → version_convert
lerobot_v30          video          lerobot_v21      transform → version_convert
openx                video          lerobot_v30      convert → transform
openx                video          lerobot_v21      convert → transform → version_convert
```

Every stage is a directory → directory step, so the existing scripts in this repo
(`ds_version_convert/*`, the per-dataset converters) are reused untouched rather
than reimplemented.

Two ordering rules matter:

- the converter runs first, because converters only ever emit v3.0;
- the transform runs **before** any version conversion, so the version converter
  handles already-shrunk video — and for v2.1 sources it runs while the dataset
  still has many small files to parallelise over.

**When only `video` steps are configured and the version does not change, no
version conversion happens at all.** v2.1 in, v2.1 out, one pass over the data.

## What it guarantees

- **The source is never modified.** Not even by the version conversion scripts,
  one of which rewrites its input in place — such a stage reads from a staged
  hard-linked copy instead.
- **No wasted disk.** Files no step changes are hard-linked, not copied. That
  includes video keys the steps do not touch, and videos already at the target
  size.
- **Metadata matches the data.** `info.json` feature shapes and the nested
  `video.height` / `video.width` are updated together with the pixels.
- **The keyframe interval is preserved.** LeRobot samples random frames during
  training; re-encoding with ffmpeg's default 250-frame GOP would silently slow
  the dataloader down. The GOP and codec are mirrored from the source.
- **All or nothing.** Any failure removes the destination rather than leaving
  something that looks converted.

## Performance

The workload is one decode + encode per frame; the goal is to reach that floor
with every core busy.

- **Threads are auto-tuned from the file count.** v2.1 datasets (many small
  files) get one thread per process; v3.0 datasets (few large chunk files) get
  several threads each so cores are not left idle. Pinning `-threads` is for
  predictable, reproducible runs — see the measurements below for what it does
  *not* buy you.
- **Largest files are scheduled first**, so the run does not end waiting on one
  straggler.
- **Nothing is re-encoded twice.** Running the pipeline again over an already
  processed dataset hard-links instead of transcoding.

Use `bench` rather than guessing — the right settings depend on source
resolution, storage bandwidth and core count.

### Measured (2026-08-12)

24 files / 10,597 frames, `800x1280` → `192x288`, identical ffmpeg build
(6.1.1-3ubuntu5), both boxes 8 physical cores.

| Instance | workers × threads | wall | fps | fps / physical core |
|---|---|---|---|---|
| c7gd.2xlarge (Neoverse-V1, 8 vCPU) | 8 × 1 | 21.3s | 498 | 62.3 |
| | 8 × 8 | 19.8s | 535 | 66.9 |
| | 1 × 8 | 26.6s | 399 | 49.8 |
| c6id.4xlarge (Xeon 8375C, 16 vCPU) | 8 × 1 | 19.1s | 556 | 69.5 |
| | 16 × 1 | 12.3s | 862 | 107.7 |
| | 16 × 16 | 11.5s | 918 | 114.8 |
| | 1 × 8 | 20.3s | 522 | 65.2 |

What this changes:

- **File-level parallelism beats intra-file threading.** `1 × 8` is last on both
  machines. More workers is the right lever.
- **Thread oversubscription was _not_ the bottleneck** for this workload, which
  contradicts the original hypothesis behind the auto-tuner. At a 192×288 output
  x264 cannot saturate the threads it is given, so the spare ones sit idle rather
  than contending — `8 × 8` and `16 × 16` were marginally *faster*. Pinning
  threads remains worth keeping for predictability, but it is not a speed-up.
- **The auto planner lands near-best in production.** Planning with
  `os.cpu_count()` (16) picks `16 × 1` — within 6% of the best measured setting.
- **Per-core difference between architectures is ~12%** (556 vs 498); x86's real
  advantage here is SMT (`8 × 1` → `16 × 1` is +55%).
- **Transfer is ~30% of end-to-end time.** Output shrank 719MB → 77MB (9.3×), so
  upload is negligible and download dominates. In an S3 → EC2 → S3 flow,
  optimising the transfer matters about as much as optimising the resize.

Because work is parallelised per file, workers beyond the file count sit idle;
budget **4–8 files per worker** to absorb straggler variance when sizing an
instance.

`lerobot_pipeline/scripts/bench_raw_videos.py` produced these numbers and can be
pointed at any directory of raw videos.

## Development

```bash
uv run --no-project --with pytest --with pyyaml --with imageio-ffmpeg --python 3.10 \
  pytest lerobot_pipeline/tests -q
```

`imageio-ffmpeg` is only needed to supply an ffmpeg binary for the integration
tests on machines without a system ffmpeg; tests that need it skip when neither is
available.

## Limitations

- `frame` steps are not implemented.
- The converter and version-conversion stages shell out to this repo's existing
  scripts. Their command construction is unit-tested, but running them requires
  those scripts' own dependencies (`tfds`, `robosuite`, `lerobot`, …), so those
  paths are not covered by the automated tests.
- Image statistics in `meta/` are not recomputed after a resize. Channel means
  barely move under downscaling, and recomputing exactly would require decoding
  the whole dataset.
