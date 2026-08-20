# Verification — does a rebuild reproduce the delivered copy

The one directory here that is not conversion work. Everything else in this repository
exists to **produce** a dataset — the converters, [`lerobot_pipeline`](../lerobot_pipeline),
[`orchestrator`](../orchestrator), [`dataset_registry`](../dataset_registry) — and this
exists to ask whether what came out is the same as what was delivered.

Which is the whole reason it is separate. `compare` reads `dataset_registry` to know what
a dataset is and imports nothing else here: **what is being measured must not be able to
change what measures it.** A comparison that shared a resizer, an encoder profile or a
stats routine with the converter would agree with it by construction. So it shells out to
ffmpeg and ffprobe for video and reads the parquet itself, rather than reusing the code
that wrote either.

Conversion work stays where it belongs and is *not* here: the RLDS sharding and
ArrayRecord reader in [`openx2lerobot`](../openx2lerobot), the channel-order rules in
`openx2lerobot/video_rules.py`, the parallel v3.0 → v2.1 downgrade in
[`ds_version_convert`](../ds_version_convert). Those are improvements to the converters,
and a verification directory that absorbed them would be verifying itself.

```bash
uv run python -m verification.compare cmu_stretch --rebuilt /out --delivered /ref
uv run python -m verification.compare cmu_stretch --rebuilt /out --report records/
```

| | |
|---|---|
| `compare.py` | the funnel below |
| `tests/` | its tests |
| `modality/<dataset>.json` | each delivered copy's own `meta/modality.json`, verbatim |
| `addresses.tsv` | where each dataset's source and delivered copy live |
| `records/<dataset>.{json,txt}` | what a run leaves behind, collected from the nodes and committed |

The reference material — `modality/` and `addresses.tsv` — is a **snapshot**, taken on
2026-08-19 from `jy-park-data-pod` in namespace `p-rlwrld`, which mounts the training
storage at `/data/taeyoung/data/vla_pretrain_dataset`. The delivered copies themselves
are the authority; these are kept so that a comparison can be set up and argued about
without a mount, a cluster or a running pod. Where a question can be settled by reading
the real thing, read the real thing.

## The funnel — four steps, not one verdict

One pass/fail was the wrong shape. The rebuilds do not control the order they write
episodes in, some episodes land a row or two short, and the two copies do not carry the
same set of metadata files, so "does it reproduce" collapses several answers into one
and loses the useful ones. The steps run cheapest and widest first, and each explains
the next.

| step | question | how it is judged |
|---|---|---|
| 1 · declaration | do the two even claim to be the same dataset | every file under `meta/` that needs no pairing |
| 2 · episodes | how much of the delivered copy is there | **counted, never failed** |
| 3 · sample | is what is there the same | state and action exactly; video loosely on bytes and exactly on everything else |
| 4 · distributions | do the two describe the same data | over every episode **and** over the episodes they share |

Every step prints a verdict word next to its numbers, and the report ends with one line
saying which step decided the run. A reader should not have to know that `1.5e-15` is a
pass and `5.3e-02` is not.

**Step 1** diffs the two `info.json` files as dotted paths rather than against a schema
of the file — it has gained fields across LeRobot versions, and a comparison that listed
the ones it knew would go quiet on the rest.

Only the fields **both** copies declare are judged. A field one side does not have is
absence of evidence, not disagreement, and usually not the rebuild's doing: the delivered
copies were written by an older LeRobot that recorded fewer encoder settings and kept
`is_depth_map` one level further in. On cmu_stretch that is eight one-sided fields —
`crf`, `preset`, `g`, `fast_decode`, `video_backend`, `extra_options` and `is_depth_map`
twice over — against a rebuild whose 135 episodes are byte-identical. Failing on them
would mark a faithful rebuild wrong, and would go on doing it for every dataset in the
collection.

Nothing is lost by that. What the one-sided fields describe is the encoding, and step 3
opens the actual files: codec, pixel format, geometry and keyframe interval come off the
video rather than out of a writer's opinion of it. A camera the rebuild renamed appears
here as one feature key on each side, and step 3 names it too, off the video directories.

`total_episodes` and its neighbours are set aside for a different reason — they follow
from step 2, and failing here would report one finding twice.

### All of `meta/`, and the one file left out

Every file under `meta/` is compared except **`relative_stats.json`**, which is left out
by decision: it is two bytes — `{}` — in every delivered copy examined, so there is
nothing in it to reproduce. A file under `meta/` that nothing here compares is still
named in the report, because the collection has grown metadata before and a silent new
file is how a schema drifts without anyone deciding to.

| file | compared as | what is set aside, and why |
|---|---|---|
| `info.json` | field by field | the counts; fields only one copy declares |
| `tasks.jsonl` | the **set** of prompts | which index each is given — the rebuild numbers them in the order it first meets them, the delivered copy numbered them alphabetically |
| `modality.json` | field by field | nothing; every entry is a claim about the data |
| `stats.json` | per feature and statistic | the ordering features; quantiles the rebuild does not carry |
| `episodes.jsonl` | per pair: prompt and length | — |
| `episodes_stats.jsonl` | per pair, per feature and statistic | as `stats.json` |

The last two are compared in **step 3**, not here, because they are keyed by episode
index and need the pairing: 134 of cmu_stretch's 135 episodes sit somewhere else, so a
line-by-line diff calls every line different and says nothing. Pair by pair they are
exact. Reported per statistic rather than per episode — a statistic that is wrong is
usually wrong in every episode, and 405 lines saying so is one finding printed 405
times.

Three kinds of difference are set aside there, each with the reason printed beside it:

- **ordering features** — `index`, `episode_index`, `task_index`. These are bookkeeping,
  not data, and they move whenever the rebuild writes episodes or numbers tasks in a
  different order. That those are the *only* three was measured: over all 135
  cmu_stretch pairs `index` differs by 2.4e4, `episode_index` by 128 and `task_index`
  by 3, while `observation.state`, `action`, `timestamp` and `frame_index` are exact to
  the bit.
- **image statistics**, within a tolerance — a rebuild is a second lossy generation of
  the same picture. `min` and `max` are allowed further than `mean` and `std`: one pixel
  decides an extreme while a mean averages a hundred frames of them, and 8 of 135 pairs
  put `max` between 0.05 and 0.0667 while every `mean` stayed under 0.0059.
- **absent quantiles** — the v3.0 → v2.1 downgrade keeps five keys and the delivered
  copies have ten.

What is *not* set aside is a statistic the delivered copy never computed. Every
delivered cmu_stretch episode records image `std` as exactly zero, where the pixels have
a spread of 0.24 — so the rebuild and the delivered copy differ because the rebuild
computed it and the delivered copy did not. That is reported as the difference it is,
with the cause named, rather than tolerated away.

**Step 2** pairs episodes on their own state and action bytes, since they carry no
source id and the rebuild does not write them in the delivered order. It pairs twice —
exactly, then on the first eight rows — so an episode that is *present but different*
is counted apart from one that is genuinely absent. The second pass catches two things
and only one of them is a trimmed tail, so the report splits them: a row or two short,
against the same length with values that go wrong past row 8.

**Step 3** holds the two halves to different standards: **state and action must be
identical**, since every slot is a copied float32, while **video bytes must only be
close**, since two ffmpeg builds given the same flags do not emit the same file.
Geometry, frame count, codec, **keyframe interval** and the **pictures** must match
exactly: those are decided by our settings, not by the encoder build. The keyframe
interval is read off the frames rather than the stream header, which does not carry it,
and the pictures are checked because a size ratio cannot see them — a rebuild with red
and blue exchanged came within 1% of the delivered size. Channel order is *not* a check
of its own; it is consulted only after the frames have already disagreed, to name the
likely cause. A time offset or a crop position is the next thing it could have been,
and one dedicated test per cause never ends.

Half the sample comes from each **end** of the paired index range. Taking the first N
looked thorough and was not: `openx2lerobot` converts in chunks of twenty-five
episodes, one worker to a chunk, so the front of the range is a single worker's output.
Two things also span *every* episode, from file metadata alone: the **task prompt** of
every paired episode, and each camera's **file count and mean size**. Without the
second, a rebuild that wrote a tenth of its videos passed on the sampled ones.

**Step 4** is the pair that makes the arrangement worth having. If the two rows
disagree, the difference is the episodes outside the shared set from step 2 and nothing
more; if the shared-episode row is *also* off, the values themselves are wrong. It reads
the parquet rather than `meta`, because the delivered copies carry quantiles in
`episodes_stats.jsonl` and a rebuild does not — the v3.0 → v2.1 downgrade keeps only the
five legacy keys — so comparing what each *says* about itself would compare the writers,
not the data.

The exit status follows steps 1, 3 and 4, the last over the shared episodes. It ignores
step 2: a run that lost episodes should still be able to say whether the ones it has are
right.

## `records/` — what a run leaves behind

`--report DIR` writes `DIR/<dataset>.json` and `DIR/<dataset>.txt`. The runs happen on
throwaway nodes, one dataset to a node, so the records are pulled off each node and
collected here before being committed — which is why the JSON carries the **thresholds
that decided the verdict** rather than only the verdict, the **index map** of which
rebuilt episode was compared against which delivered one, and the **full per-dimension
statistics** for both sides rather than only the gap they were reduced to. The datasets
are far too large to keep beside the record, and a number nobody can recompute is a
number nobody can argue with. Re-running a dataset overwrites its own two files and
nothing else.

## Reference material

## What this round covers

29 of the 36 datasets in the registry: the 27 Open X-Embodiment sets, plus
humanoid_everyday's g1 and h1.

Out of scope, and not for any reason to do with the data: `action_net`,
`agibot_dexhand`, `agibot_gripper`, `galaxea`, `march_robocurate`,
`neural_robocurate`, `oneuniverse_simul`. Their rows stay in `addresses.tsv` and
their modality files stay in `modality/`, because leaving them out of a table is how
a dataset gets forgotten rather than deferred.

### humanoid_everyday is the shape nothing else in scope has

Three ways, each of which the rest of the collection does not need:

**Its upstream is already LeRobot.** Not RLDS. The mirror is an `hf_repo` snapshot
with `data/`, `meta/` and `videos/` and `codebase_version: 2.1`, so a rebuild is a
resize and a re-encode rather than a conversion. `builder: none` in those two specs
means "needs no converter", not "upstream unknown".

**One upstream, two datasets.** The snapshot's prefix says `humanoid_everyday_g1`
but it holds both robots -- its `meta/info.json` counts 8949 episodes and 3436171
frames, exactly g1's 4064 + h1's 4885 and 1779287 + 1656884. Every episode in
`meta/episodes.jsonl` carries a `robot_type`, so the split is recoverable, and both
specs now point at the same snapshot. h1 had no mirror recorded before this; it was
not missing, it was in here.

The split itself is not a build step. It decides which episodes a dataset contains,
which is settled before a build rather than inside one: the two trees are staged
separately, each from the robot_type its episodes carry, and each build then sees
only its own.

**One upstream camera, two delivered ones.** The upstream has
`observation.images.egocentric` at 480x640 and nothing else. The delivered copies
carry that plus `egocentric_resized` at 192x256, and the two were encoded
differently -- medium crf 23 for the original, fast crf 18 for the resized. Only the
resized one is in `modality.json`, so only that one is read, rebuilt and checked.

## `modality/<dataset>.json`

Each delivered dataset's own `meta/modality.json`, verbatim, for all 36.

This is the file that says **which cameras the training stack reads**, and it does
not always list everything on disk. Four datasets keep a camera they do not expose:

| dataset | on disk | exposed | kept but unexposed |
| --- | --- | --- | --- |
| `bridge_orig` | 4 | 2 | `image_2`, `image_3` |
| `berkeley_cable_routing` | 4 | 3 | `wrist225_image` |
| `humanoid_everyday_g1` | 2 | 1 | `egocentric` |
| `humanoid_everyday_h1` | 2 | 1 | `egocentric` |

`compare.py` reads the delivered copy's modality file to decide which
cameras a comparison is about, so these are the files that decide it. humanoid_everyday
is the case that matters most: its two entries are the same camera twice, 640x480 and
256x192, and they were not encoded alike -- the exposed one at preset fast crf 18 and
the unexposed one at preset medium crf 23. Holding a rebuild to both would hold it to
something no single encoding can produce.

Note the two spellings. Most datasets name a camera's directory after the modality
entry's `original_key` in full -- `observation.images.rgb_static` -- while
humanoid_everyday names it `egocentric_resized`, the last segment alone. Both have to
be recognised; matching one leaves the other with no cameras selected, which reads as
a pass.

## `addresses.tsv`

Where each dataset's source and delivered copy live, as one table: dataset, source
mirror, delivered path. Generated from the specs in `dataset_registry/datasets`,
which stay authoritative -- this exists so a run can be set up from one file instead
of thirty-six.

Six rows have no source mirror: `agibot_dexhand`, `agibot_gripper`,
`humanoid_everyday_h1`, `march_robocurate`, `neural_robocurate` and
`oneuniverse_simul`. That is not a gap in this table. Their upstream was never
traced, so their specs declare `builder: none` and there is nothing to mirror; a
delivered copy is all that exists of them.

## The delivered copies away from the pod

foundry artifact `c213aa21e25849dbb3dfa07742f92288`, fetched per dataset:

```bash
foundry pull c213aa21e25849dbb3dfa07742f92288 --include '<dataset>/**'
```

A node pulls its own, as it goes. That needs the **internal** ALB, because
`foundry.internal.rlwrld.ai` resolves to the public ALB's addresses and those do not
hairpin from inside the VPC:

```bash
export FOUNDRY_URL=http://internal-rlwrld-foundry-api-425985869.us-east-1.elb.amazonaws.com/api
```

Which group a node launches with decides whether that works — the internal ALB admits
named source groups only. See
[reaching Foundry from a node](../orchestrator/bootstrap/README.md#reaching-foundry-from-a-node).

## Refreshing

```bash
while read -r name dir; do
  kubectl cp -n p-rlwrld "jy-park-data-pod:$dir/meta/modality.json" \
    "verification/modality/$name.json" </dev/null
done < <(awk -F'\t' 'NR>1 {print $1, $3}' verification/addresses.tsv)
```

`</dev/null` is not decoration: `kubectl cp` reads standard input, and without it the
first copy swallows the rest of the loop's input. Under zsh, name the loop variable
anything but `path` -- `$path` is tied to `$PATH`, and assigning to it empties it.

## A verification run needs `--keep`

The orchestrator deletes an output once it has been published, which is correct for a
production run and fatal for a verification one — the first attempt on austin_buds ran
`fetch build publish` and then had nothing left to compare:

```
rebuilt dataset not found: /scratch/out/austin_buds_dataset_converted_externally_to_rlds
```

So a node that is going to compare passes `--keep`, or compares between `build` and
`publish`. This is not a bug to fix in the orchestrator: reclaiming disk is what lets a
27 TB pass run at all, and verification is deliberately not one of its steps.

## What the records say so far

Rebuilt on image `7bcf55b`, which resizes with **sinc**.

| dataset | episodes | downscale | video | verdict |
|---|--:|---|--:|---|
| `cmu_stretch` | 135/135 | none | 0.98x | FAIL — delivered image `std` is zero |
| `austin_buds_…_rlds` | 50/50 | none | 0.99x / 0.97x | FAIL — same, both cameras |
| `austin_sirius_…_rlds` | 559/559 | 1.31x | 1.10x / 1.09x | FAIL — same |
| `dlr_edan_…_rlds` | 104/104 | 1.94x | 0.96x | FAIL — same |
| `ucsd_kitchen_…_rlds` | 150/150 | 2.50x | 0.99x | FAIL — same |

**Every video clears the 15% band now, at every downscale factor measured.** The two
datasets that are not resized did not move, which is the check that the filter was the
whole of it.

All five reproduce their data exactly — every episode byte-identical on state and
action, every sampled episode identical on state and action, prompts all agreeing,
distributions to 1e-14 or better. dlr_edan was the one failure that was **ours**, and
the `downscale` column is why: it was the first dataset in the funnel whose source has
to be downscaled at all. It is fixed — the profile now names `sinc`, and dlr_edan's
video went from 0.79x with 63 of 64 sampled episodes failing to **0.96x with none**,
while ucsd went 0.885x to 0.99x. Every remaining difference in the collection so far is
the delivered copy's zero `std`.

### Then 15 more, four nodes at once

A parallel run over 25 datasets, stopped part way. 15 finished the full cycle; timings
and the stall that ended it are in [`records/speed.md`](records/speed.md). Every one of
them reproduces its data and fails on the same zero `std`, with four exceptions worth
naming:

| dataset | also differs in |
|---|---|
| `iamlab_cmu_pickup_insert_…` | `robot_type` `'franka'` vs `'Franka'`; state slot names — ours `motor_0…motor_6, gripper`, delivered `x, y, z, roll, pitch, yaw, pad, gripper` |
| `bc_z` | `robot_type` `'google_robot'` vs `'Google Robot'` |
| `berkeley_autolab_ur5` | one prompt differs by a **trailing space** |
| `stanford_hydra`, `taco_play` | video size — the gentle end of the resize |

The first three are spec data this repository can correct. Two are the capitalisation of
a string; the third is real layout information the delivered copy carries and our spec
does not — `x, y, z, roll, pitch, yaw` says the block is a pose, where `motor_0…6` says
only that it is seven of something.

`absolute_action` turned up in **7 of the 15**, not the 1 of 3 first recorded. It is set
aside every time, which is the undeclared-column rule working: no delivered `info.json`
declares it.

### One filter is not enough, and this is the evidence

`sinc` fixed the strong downscales and broke the gentle ones. Of the 15, exactly two fail
on video size — `stanford_hydra` at 1.25x down and `taco_play` at 1.21x — which is
precisely where the sweep said sinc would overshoot: 1.09-1.10x measured at 1.31x against
a 15% bound, and gentler than that is worse. Everything else lands in 0.96-1.15x.

The change is still a net gain, five or more datasets fixed against two broken. But the
conclusion the sweep hedged on is settled: **the resampler has to follow the scale
factor, not the collection.**

So the profile now names `by_scale` rather than a filter, and the threshold is 1.3x:

| downscale | filter | measured |
|---|---|--:|
| 1.21x `taco_play`, 1.25x `stanford_hydra` | bicubic | 0.985-1.019x |
| 1.31x `austin_sirius` | sinc | 1.088-1.100x |
| 1.94x `dlr_edan` | sinc | 0.959x |
| 2.50x `ucsd_kitchen` + 4 more cameras | sinc | 0.988x |

The step decides per camera rather than per dataset, because a dataset can hold a
480x640 view and an 84x84 one and those are different downscales. Both resize paths
resolve the rule through the same step object, which a test asserts directly — a rule
that answered differently on the two would build half a collection one way and half the
other, which is the failure the declared value was introduced to prevent.

Still to check on a node: that this puts `stanford_hydra` and `taco_play` back inside the
band. Everything above is measured, but the threshold itself has not yet been run
end-to-end.

### The delivered image `std` is zero

Every delivered episode of both datasets records image `std` as exactly `[0,0,0]` where
the pixels have a spread of 0.22 to 0.29. Two datasets and three cameras in, this looks
like a property of how the collection was made rather than one bad run. It is reported
as a difference with its cause named, not tolerated: the rebuild computed the statistic
and the delivered copy did not.

### The resize loses detail, and dlr_edan is where it shows

63 of dlr_edan's 64 sampled episodes fail on video size and nothing else: state and
action identical, geometry, frame count, codec and keyframe interval all matching,
pixels agreeing at 0.997 — and every file about 0.80x the delivered one. A ratio that
steady across 63 episodes is a setting, not noise.

It is the downscale. cmu_stretch and austin_buds are 128x128 at the source and are not
resized; their video lands at 0.98x and 0.99x, which is the encoder build difference
alone. dlr_edan is 360x640 down to 192x320, and `resize_frame` low-passes more than
whatever rldx1 used: measured on episode 6, our frames carry **78% of the delivered
high-frequency detail** — almost exactly the 0.79x size ratio, because a softer picture
costs the encoder less.

`resize_frame`'s docstring already measured this and chose swscale BICUBIC as the one
that "stays inside tolerance everywhere". dlr_edan is the counterexample: ucsd at 2.5x
down came out 0.86x and cleared the size tolerance; this does not.

**Lanczos was never in that table.** Resizing the real source frames every available
way and measuring the detail each keeps, with the delivered file as the target:

| filter | detail kept, vs delivered |
|---|--:|
| swscale BILINEAR | 0.68x |
| swscale AREA | 0.80x |
| cv2 INTER_AREA | 0.84x |
| swscale BICUBIC — *what ships* | 0.85x |
| **swscale LANCZOS** | **0.93x** |
| swscale SINC | 1.12x |
| cv2 INTER_CUBIC | 1.19x |
| cv2 INTER_LANCZOS4 | 1.22x |

Two cautions before anyone changes the filter. This measures *detail*, not file size,
and the two are related without being the same number. And it is one dataset at one
scale factor — the original table's own point was that the offset moves with how gentle
the downscale is, so Lanczos has to be re-measured on ucsd and taco_play, on the size
metric, before it replaces bicubic. What is settled is narrower and still worth having:
bicubic is not the best available on a 2x downscale, and the option that beats it was
never tried.

### `absolute_action`, and why the rebuild has none

austin_buds' delivered parquet carries an eighth column its own `meta/info.json` does
not declare. It is real derived data — `state + action`, the delta action resolved to an
absolute pose, matching to the last digit on every column but the gripper — and it is in
`meta/stats.json` too. The RLDS source has no such field, so nothing in the source says
to build it.

It is set aside with its reason printed rather than failed. Undeclared is the
load-bearing word. A LeRobot loader builds its feature set from
`info.json`, so a column absent from there is not read by anything downstream, which is
also why this is a difference to decide about rather than a hole to fill. Of the three
delivered copies examined it appears in one, so it is not a collection-wide convention
either.
