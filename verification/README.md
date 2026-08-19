# Reference material for checking a rebuild

Not part of any converter. Nothing here is imported, and removing the directory
would not change what any4lerobot produces — these are copies of things the
delivered datasets say about themselves, kept so that a comparison can be set up
and argued about without a mount, a cluster or a running pod.

The delivered copies themselves are the authority. This is a snapshot of them,
taken on 2026-08-19 from `jy-park-data-pod` in namespace `p-rlwrld`, which mounts
the training storage at `/data/taeyoung/data/vla_pretrain_dataset`. Where a
question can be settled by reading the real thing, read the real thing.

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

`dataset_registry.compare` reads the delivered copy's modality file to decide which
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
