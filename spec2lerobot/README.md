# spec2lerobot

Convert any registry dataset to LeRobot. Which dataset is a flag, not a module.

```bash
uv run python -m spec2lerobot --dataset action_net \
    --src-path /data/action_net --output-path /out
```

Everything the conversion needs — file layout, HDF5 keys, clock strategy, robot type,
frame rate, joint counts — comes from `dataset_registry/datasets/<name>.yaml`. No
dataset is named anywhere in this package.

Normally you do not run this directly; `lerobot_pipeline` runs it as the first stage
and then assembles the state vectors and resizes the video.

## What it emits

The source's own feature vectors and its video, and deliberately nothing else — no
`observation.state`, no `action`, no `modality.json`. Those are assembled afterwards
by `lerobot_pipeline`'s `state_layout` step from the same spec, so that a dataset
arriving as raw HDF5 and one arriving already converted go through **one**
implementation of the layout convention rather than two.

The mp4 is carried over rather than decoded and re-encoded.

## Two closed sets

Everything that varies between datasets is data. Two things cannot be:

**Formats** (`formats/`) — one reader per *file format*, not per dataset.

| name | shape | status |
| --- | --- | --- |
| `hdf5_episodes` | one HDF5 + one video directory per episode | implemented |

**Clock strategies** (`clocks.py`) — how a robot stream lines up with a camera
stream. This is an algorithm, so it is the one part of reading a dataset that cannot
be written in YAML; it is named instead.

| name | what it does |
| --- | --- |
| `nearest_timestamp_dedup` | nearest robot sample per video frame, each sample claimed at most once, plus the two upstream filters |
| `index` | one robot sample per frame, in order — for single-clock sources |

`nearest_timestamp_dedup` is ported verbatim from `FFTAI/fourier-lerobot`'s
`convert_hdf5_to_lerobot.py`, so a rebuilt dataset lines up with what the upstream
training pipeline was built against.

Adding a dataset that uses an existing format and strategy is a YAML file. Code grows
with the number of ways robot data is stored — a handful — not with the number of
datasets.

## Refusals

A dataset whose spec has blocks with no recovered source columns will not convert:
`SpecAdapter` raises before any work starts. Emitting zeros there would produce a
dataset that trains without complaint on a blank stretch of vector. Run
`python -m lerobot_pipeline.plan` to see the verdict for a config.

An episode is skipped, not reinterpreted, when its arrays are not the shape the spec
describes — a GR2 reports 29 joints where a GR1 reports 32, and its columns mean
other things.
