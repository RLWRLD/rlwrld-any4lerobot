# LeRobot Dataset v30 to v21

## Get started

1. Downgrade datasets:

   ```bash
   pip install "datasets<4.0.0"
   ```

   > Need to downgrade datasets first since `4.0.0` introduces `List` and `Column`.

2. Install v3.0 lerobot

   ```bash
   git clone https://github.com/huggingface/lerobot.git
   pip install -e .
   ```

2. Run the converter:
   ```bash
   python convert_dataset_v30_to_v21.py \
       --repo-id=your_id \
       --root=your_local_dir
   ```

## `--workers`

The two expensive steps are cutting the consolidated MP4s back into one file per
episode, and cutting the consolidated parquet files back into one file per
episode. Measured on a 135-episode, 2-camera dataset they are 77% and 23% of the
run; everything else -- the three JSON/JSONL files -- is under half a percent.

Both are lists of independent jobs: each reads a shared source read-only and
writes a path no other job writes. So they run on a pool of threads, and the
output does not depend on how many. `--workers` sets the size:

| value | meaning |
| --- | --- |
| `-1` (default) | one per core |
| `1` | serial, as this script used to be |
| `n` | `n` at a time |

Threads, not processes, because every job is either an ffmpeg subprocess or a
pyarrow read/write and both release the GIL for their whole duration.

The video pass has one job per episode per camera, so it saturates any core
count. The data pass has one job per *file*, so it is capped by however many
consolidated data files the dataset has -- which also caps its memory, since a
job holds one table while it slices it.

## What it writes about statistics

Both files, not one. `episodes_stats.jsonl` carries every key the v3.0 source had --
including the five quantiles, which v2.1 does not require but the delivered RLDX-1
copies all have. They used to be filtered out here, and that was a one-way door:
whole-dataset quantiles can be computed from per-episode ones, but per-episode ones
cannot be computed from anything except the data, so dropping them meant a second pass
over every frame to get them back. droid has 27 million of them.

`stats.json` is aggregated from those entries as they are written, so it costs no extra
pass. Deliberately not through lerobot's `aggregate_stats`: that one takes a
conservative envelope for the quantiles -- the minimum of the lower ones and the maximum
of the upper -- and says so in its own comment. The delivered copies used a
count-weighted mean. Measured against cmu_stretch's own `stats.json`, every feature and
every statistic: **4.2e-09** this way against **1.2e+04** for the envelope.

`std` is the one that cannot be averaged. How far a group spreads about its own mean
says nothing about how far that mean sits from everyone else's, so the pooled variance
needs both terms; averaging the per-episode values put `index` out by 7,168 and
`observation.state` by 0.08. Each feature is weighted by its **own** count, because
image statistics come from a hundred sampled frames while the vectors cover every row.
