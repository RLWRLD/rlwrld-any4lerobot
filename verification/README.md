# Verification — does a rebuild reproduce the delivered copy

One question none of the converters can answer, kept apart from all of them. `compare`
reads [`dataset_registry`](../dataset_registry) to know what a dataset is, and nothing
under any4lerobot's own directories imports anything here — what a rebuild is measured
against must not move when the thing being measured does.

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
| 1 · declaration | do the two even claim to be the same dataset | every `meta/info.json` field except the counts |
| 2 · episodes | how much of the delivered copy is there | **counted, never failed** |
| 3 · sample | is what is there the same | state and action exactly; video loosely on bytes and exactly on everything else |
| 4 · distributions | do the two describe the same data | over every episode **and** over the episodes they share |

Every step prints a verdict word next to its numbers, and the report ends with one line
saying which step decided the run. A reader should not have to know that `1.5e-15` is a
pass and `5.3e-02` is not.

**Step 1** diffs the two `info.json` files as dotted paths rather than against a schema
of the file — it has gained fields across LeRobot versions, and a comparison that listed
the ones it knew would go quiet on the rest. fps, robot type, feature shapes and dtypes,
codec and pixel format are failures; `total_episodes` and its neighbours are set aside,
because they follow from step 2 and failing here would report one finding twice.

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

The API is not reachable from inside the VPC -- its address resolves to a public EIP
that does not hairpin -- so a run on EC2 has to have the copies staged onto the image
beforehand rather than pulling them as it goes.

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
