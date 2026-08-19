# Bringing a machine up for a verification run

What has to be true on a fresh instance before `python -m orchestrator run --env ec2`
can be left alone. Every check in [`preflight.sh`](preflight.sh) is here because a run
was lost to it, so the order below is: build the machine, stage the data, run
preflight, then start.

## The machine

| | | why |
|---|---|---|
| type | `m7i.12xlarge` (48 vCPU, 185 GB) | see [memory](#memory-is-per-episode-not-per-core) |
| disk | ~1 TB root | sources are not deleted until a dataset is done; taco_play alone is 48 GB |
| region | `us-east-1` | where the mirrors are — a cross-region sync of 27 datasets is not worth paying for |
| IAM | an instance profile with SSM and read on `s3://rlwrld-foundry-data` | SSM is how the run is driven; there is no key pair |
| packages | `ffmpeg`, `uv` | ffmpeg does the video work; nothing else is needed at the system level |

### Memory is per episode, not per core

A worker holds one whole episode, so what it costs follows the episode size, not the
core count:

| dataset | per episode | workers | peak | per worker |
|---|---:|---:|---:|---:|
| bc_z | ~20 MB | 64 | 73.9 GB | **1,154 MB** |
| toto | ~301 MB | 64 | over 185 GB — killed | **>2.9 GB** |

`workers: -1` in [`env/ec2.yaml`](../../lerobot_pipeline/configs/env/ec2.yaml) asks the
machine for its core count rather than writing one down. That is safe at roughly
**4 GB per core**, which m7i gives. On a machine with less — c9gd is 2 GB per core —
the worker count has to be capped below the core count or the largest datasets will
be killed.

## Order of operations

```
1. launch                → user-data.sh runs at boot
2. stage the sources     → aws s3 sync, per dataset
3. stage the delivered   → must come from outside the VPC, see below
4. preflight.sh          → refuses to start on any known trap
5. orchestrator run
```

### Getting the repo onto the machine

`uv.lock` **is not in git** (`.gitignore:68`), so a `git clone` alone leaves a machine
that cannot resolve the environment — and a zero-length `uv.lock` fails with
`missing field 'version'`, which reads like a corrupt file rather than a missing one.
Ship the lock file alongside the source, or `uv lock` on the machine and accept that
the resolution is not the one that was tested.

An instance profile scoped to `rlwrld-foundry-data` cannot read a build bucket, so a
tarball staged elsewhere needs a presigned URL. **Presigned URLs signed with SSO
credentials die when the SSO session does**, not after `--expires-in` — an overnight
run that pulls one on wake-up will find it invalid. Pull everything up front.

### The delivered copies

They come from foundry artifact `c213aa21e25849dbb3dfa07742f92288`:

```bash
foundry pull --include '<dataset>/**' --artifact c213aa21e25849dbb3dfa07742f92288
```

**The foundry API is not reachable from inside the VPC** — its EIP does not hairpin —
so this cannot run on the instance. Pull the delivered copies somewhere else and
stage them to S3, or bake them into the image.

## Do not reuse an image made with `--no-reboot`

The AMI used for the last round (`ami-0d5900b553c125cc2`) was captured that way, which
snapshots a filesystem that was never quiesced. What came back:

- every `.py` in `generic_converter/`, and `uv.lock`, **truncated to zero bytes**
- **3,945 files inside `.venv`** truncated to zero bytes
- `/scratch/raw/taco_play` holding **472 of 511 shards**, with `dataset_info.json` and
  `features.json` absent

None of it announces itself. The truncated venv imports fine until it does not, and
the short source directory produces `KeyError: splits['train']` — which reads like a
dataset problem, not a disk problem. Stop the instance before creating the image.

## Known blockers, as of the last survey

All 27 openx mirrors carry complete tfds metadata (`dataset_info.json` and
`features.json`); the taco_play files that appeared missing were an incomplete local
copy, not a gap in S3. What is genuinely blocked:

| dataset | blocker |
|---|---|
| `bc_z` | mirror is **`array_record`**, not tfrecord — the only one of the 27. `as_dataset()` cannot read it; it needs `as_data_source()`. |
| `furniture_bench_...` | transform imports `tensorflow_graphics`, which is in no extra in `pyproject.toml` |
| `iamlab_cmu_pickup_insert_...` | same |

`cmu_playing_with_food` also needs `tensorflow_graphics` but is not in the registry.

## Traps

| symptom | cause | fix |
|---|---|---|
| `Invalid version ''` from tfds | `--raw-dir` points at the dataset, not the version directory | point at `<name>/<version>/`, e.g. `/scratch/raw/taco_play/0.1.0` |
| `KeyError: splits['train']` | `dataset_info.json` missing from the staged copy | re-sync; check the shard count against the `-of-NNNNN` suffix |
| `NotFoundError` on a tfrecord path | shards missing from an interrupted sync | same — count them |
| conversion holds the machine for hours at 0% CPU | datatrove's default forkserver inherits TensorFlow thread state | already fixed: the converter passes `start_method="spawn"` |
| OOM with workers well under the core count | `--image-writer-process` defaults to 5, so each worker spawns six processes each holding the imports | already set to 0 in `env/ec2.yaml` |
| `dataset_registry.available()` lists `._name` entries | a `tar` made on macOS wrote AppleDouble files | `find /opt/oxe -name '._*' -delete`, or use `--disable-copyfile` |
| a step is "done" but its output is gone | a record survived the thing it made | already fixed: `Steps.done()` checks `record.created` still exists |

## Files here

- [`user-data.sh`](user-data.sh) — runs at boot: packages, directories, environment
- [`preflight.sh`](preflight.sh) — refuses to start a run on any of the above
