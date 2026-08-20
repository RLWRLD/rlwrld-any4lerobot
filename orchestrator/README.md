# Orchestrator

`download → preprocess → upload`, unattended, over every dataset in the registry.

```bash
python -m orchestrator run     --env ec2                  # all of it
python -m orchestrator fetch   --env ec2                  # one stage
python -m orchestrator build   --env ec2 --dataset droid  # one dataset
python -m orchestrator status  --env ec2                  # how far it got
python -m orchestrator timings --env ec2                  # how long it took
```

`timings` measures nothing. Every step already records `started`, `finished` and, for
`fetch`, `bytes` -- a record that cannot say when it ran cannot answer whether a re-run
got slower -- so the report only reads them, and works on a run nobody was watching.
`--json` dumps the records instead, for aggregating several nodes into one table.

Only `fetch` gets a rate. It is the S3-to-instance leg and the one stage whose work is a
known number of bytes; `build` is reported per episode, because how many frames the
converter was handed is a property of the source and not of the record.

Every step records what it did, so a re-run picks up where the last one stopped and
skips what is already done. That is also what lets a stage be run by hand without
confusing the others.

## The stages do not overlap

Overlapping them was measured against the whole collection first. The registry
declares 154,685,462 frames, and the three datasets with a mirror size on record
average 179 KB per frame, so the source is about 27.6 TB. On a 100 Gbps, 96-core
machine:

| | preprocess | download | upload | total |
|---|---|---|---|---|
| sequential | 3.9 h | 1.2 h | 0.1 h | **5.2 h** |
| overlapped | 3.9 h | (hidden) | (hidden) | **3.9 h** |

Overlapping buys **1.3 hours on a full pass**. Assume a slower encoder — 8,000 fps
instead of 11,000 — and it is 6.7 h against 5.4 h: still 1.3 hours, because the
saving is fixed and only the ratio moves. The conclusion gets stronger, not weaker,
as the estimate gets worse.

What it costs is capacity prediction, backpressure, reservation accounting and
reclaim ordering. Those decide when to delete things. 1.3 hours is not the price of
that risk. Keeping the stages apart also keeps an encoder measurement free of
network I/O, which is half of why the machine exists.

## Within a stage, datasets are batched

One dataset does not keep a large machine busy. Work is parallelised per file, so
[workers beyond the file count sit idle](../lerobot_pipeline/README.md), and **14 of
the 36 datasets have fewer than a thousand episodes** — austin_buds has 50, viola
135. Those leave most of the machine doing nothing.

So a batch is filled by work, not by count:

```
accumulate datasets until the episodes reach target_episodes
a dataset that reaches it alone is a batch of one
no batch exceeds max_datasets
```

Datasets are ordered smallest-first before this runs. Without that the grouping does
almost nothing: the registry is alphabetical, so the small datasets are scattered the
length of it and each ends up in its own batch anyway. Ordering by size takes the
shipped collection from 7 undersized datasets running alone to 1.

The rule also keeps disk flat. The only datasets that get multiplied are the small
ones; anything large enough to matter already runs by itself, so the peak is one
dataset's footprint either way. Note that a build's peak is roughly
`source + output x 2`, because `generic_converter` aggregates the per-task temporary
datasets into the final one *before* deleting the temporaries.

## What gets deleted, and what cannot be

A source is removed once it has been built, and an output once it has been
published. Three things must hold first, each ruling out a different accident:

1. **a step recorded creating the path.** A source staged by hand is never created
   by `fetch`, so it has no recorded path and is unreachable from here. The
   protection is the absence of a claim, not a check that could be got wrong.
2. **the step that consumes it succeeded.** A failed build never costs the source it
   failed on.
3. **no other dataset still needs it.** `agibot_dexhand` and `agibot_gripper` are one
   tree read twice with different flags; deleting it after the first build would
   strand the second.

`--keep` turns deletion off entirely. `--dry-run` reports what would happen and
touches nothing.

## Where the settings live

The existing three layers, unchanged: what a dataset is lives in
`dataset_registry`, how it is processed in its profile, where it lives in the
environment.

```yaml
# lerobot_pipeline/configs/profiles/rldx1.yaml -- the collection's own address
dest:
  version: lerobot_v21
  uri: s3://rlwrld-foundry-data/lerobot/{profile}/{dataset}/
```

```yaml
# lerobot_pipeline/configs/env/ec2.yaml -- this machine
state_root: /scratch/state
batch:
  max_datasets: 3
  target_episodes: 768   # workers x 4
```

No instance type is detected and no bandwidth is named. Declaring more bandwidth
than the NIC has measured *slower*, so that number is a property of the machine and
belongs in `aws configure` — see [Downloading source data](../README.md).

## Not all 36 can be fetched yet

Only 3 datasets declare a `mirrors:` entry, so the rest have nowhere to be fetched
from. `fetch` reports those as **skipped**, not failed: that is data not yet
delivered, not a broken run. Stage such a source by hand and point `env.paths` at
it, and `build` proceeds — and, per rule 1 above, that source will never be deleted.

## Development

```bash
uv run --no-project --with pytest --with pyyaml --python 3.10 pytest orchestrator/tests -q
```

Batching and deletion are pure functions, so every awkward case — a shared source, a
failed build, a hand-staged tree — is reachable without a network, an encoder or a
terabyte of disk.
