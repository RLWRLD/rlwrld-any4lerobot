# Bringing a machine up for a verification run

What has to be true on a fresh node before `python -m orchestrator run --env ec2` can
be left alone. Every check in [`preflight.sh`](preflight.sh) is here because a run was
lost to it, so the order is always: build the node, stage the data, run preflight,
then start.

## Use the image

The [`Dockerfile`](../../Dockerfile) at the repo root is the supported way to build a
node. One node does this:

```bash
docker run --rm \
  -e DATASETS="taco_play toto" \
  -e NIC_RATE=100Gb/s \
  -v /scratch:/scratch \
  487592470682.dkr.ecr.us-east-1.amazonaws.com/rlwrld/inhouse-services/any4lerobot/node:<tag>
```

`DATASETS` is that node's share; [`node.sh`](node.sh) runs preflight and then hands
the names to the orchestrator, which fetches, builds and publishes each in turn.

### Why a container and not an AMI

Nothing about a node's *data* is baked in — each node downloads its own share and
uploads its own output — so an image only ever held the environment, which is the
part an AMI held badly. The AMI this replaces was captured with `--no-reboot`,
snapshotting a filesystem that was never quiesced, and came back with **3,945
zero-length files in `.venv`** and `generic_converter/` gone, without a word. Nothing
noticed until a run failed hours later with `KeyError: splits['train']`, which reads
like a dataset problem.

Layers are content-addressed. A damaged one fails to pull; it does not start and
misbehave. And two nodes pulling the same digest are running the same thing, which is
the property that matters once there is more than one node.

That is also why `uv.lock` is tracked and why the build passes `--locked`: if the
lock and `pyproject.toml` disagree the build stops, rather than quietly resolving
something new.

### Building and pushing

Nodes are `m7i`, so the image has to be **`linux/amd64`** whatever it is built on —
on an arm64 machine that means an emulated build, which is slow but correct:

```bash
REPO=487592470682.dkr.ecr.us-east-1.amazonaws.com/rlwrld/inhouse-services/any4lerobot/node
FOUNDRY=../rlwrld-foundry            # a checkout at the revision in foundry-cli.pin

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${REPO%%/*}"

git -C "$FOUNDRY" fetch -q origin && git -C "$FOUNDRY" checkout -q "$(cat foundry-cli.pin)"

docker buildx build --platform linux/amd64 --provenance=false \
  --build-context foundry="$FOUNDRY" \
  --build-arg FOUNDRY_REVISION="$(git -C "$FOUNDRY" rev-parse HEAD)" \
  -t "$REPO:$(git rev-parse --short HEAD)" --push .
```

The **foundry build context** is what puts the `foundry` CLI and its SDK in the image,
for the delivered copies a comparison is measured against and for publishing a
preprocessed dataset back. Neither package is on PyPI or in an internal index, so the
source has to come from a checkout, and the build refuses to proceed unless that
checkout's revision matches [`foundry-cli.pin`](../../foundry-cli.pin) — which CLI an
image carries should be a reviewed fact, not whatever the builder had lying around.
The image records it at `/opt/foundry/REVISION`. Bumping the CLI is an edit to the pin.

**On a build node**, which has no GitHub credentials, both trees arrive through
`s3://rlwrld-foundry-data/tmp/any4lerobot-build/` — the one prefix
`rlwrld-any4lerobot-build` is allowed to read, and what it is for. Upload this working
tree and the pinned foundry packages there, extract them on the node, and point
`--build-context` at the second. Clear the prefix afterwards; it is a transit, not a
store.

Tag with the commit, not `latest`. A run should be able to say which image produced
its output, and `latest` cannot answer that. Pull by digest in anything scheduled:
`$REPO@sha256:…` is the only form that guarantees every node in a fan-out got the
same bytes.

`--provenance=false` keeps the push to a plain image manifest. Without it buildx
writes an OCI index with an attestation beside the image, which some runtimes will
not pull.

## Without the image

[`user-data.sh`](user-data.sh) prepares a bare instance the same way, for when a
container is inconvenient — it installs the packages and makes the directories, and
leaves the repo to be delivered separately. Everything below applies either way.

## The machine

| | | why |
|---|---|---|
| type | `m7i.12xlarge` (48 vCPU, 185 GB) | see [memory](#memory-is-per-episode-not-per-core) |
| disk | ~1 TB root | sources are not deleted until a dataset is done; taco_play alone is 48 GB |
| region | `us-east-1` | where the mirrors are — a cross-region sync of 27 datasets is not worth paying for |
| IAM | instance profile `rlwrld-any4lerobot-ec2` | SSM, read on `external/`, write on `lerobot/`, ECR pull |
| network | security group `rlwrld-any4lerobot-ec2-sg` | outbound-only, and the one group the Foundry API admits — see below |
| packages | `ffmpeg`, `uv` | ffmpeg does the video work; nothing else is needed at the system level |

Both are declared in `rlwrld-terraform` — the profile in `envs/iam/any4lerobot_ec2.tf`,
the group in `envs/prod/any4lerobot.tf` — and they share a name because they are the
two halves of the same thing: the profile says what a node may call, the group says
what it may reach. Launching with a group made at the console is what the second one
replaces, and a node launched that way cannot reach Foundry.

### Reaching Foundry from a node

The image sets this already; it is here because the value is the surprising part:

```bash
FOUNDRY_URL=http://internal-rlwrld-foundry-api-425985869.us-east-1.elb.amazonaws.com/api
FOUNDRY_HOME_LOCATION=aws-ssot
```

`http`, not `https` — the internal ALB has one listener and it is plain HTTP on 80.
Both values are the ones `rlwrld-foundry`'s own `deploy/agents/targets.yaml` declares
for this cluster, under `skt`. `FOUNDRY_HOME_LOCATION` decides where an *upload* is
stored, which matters once preprocessed datasets are published back.

`foundry.internal.rlwrld.ai` does not work from a node despite the name: it resolves
to the *public* ALB's addresses, which do not hairpin from inside the VPC. Measured on
a node: the internal ALB answers `HTTP 200` in 27 ms, the public name times out after
15 s. The bytes
do not cross either ALB in any case — Foundry answers with presigned S3 URLs, and the
VPC has an S3 gateway endpoint, so the download is S3 to instance and never leaves the
network.


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

**This has now been hit, and on a node picked for verification rather than a full run.**
c7i.2xlarge is 8 vCPU and 16 GB, so 2 GB per core, and stanford_hydra's build died
immediately: 570 episodes at 240x320, two cameras, eight workers. Capping to 4 fixed it.
The three datasets verified before it — 50, 104 and 150 episodes — were small enough not
to reach it, which is exactly how this stays hidden until a real dataset arrives.

So a verification node is an `m7i` too, even though it handles one dataset at a time.
Picking a c-family instance because the work looks small trades 2 GB per core for a
failure that reads like a dataset problem.

**And m7i was not enough either.** On 2026-08-20 three m7i.12xlarge nodes -- 48 cores,
185 GB, 4 GB per core exactly as recommended -- were killed anyway: furniture_bench's
workers peaked at **4.74 GB each**, so 48 of them asked for 226 GB. The rule was wrong,
not the instance. The constraint is memory per *worker*, and `workers: -1` now resolves
through [`worker_budget`](../../generic_converter/pipeline.py), which divides the machine's
memory by a measured 6 GB and takes the smaller of that and the core count: 30 on this
instance rather than 48. `ANY4LEROBOT_WORKER_MEMORY_GB` raises it for a dataset whose
episodes are larger still -- toto's are ~301 MB against bc_z's ~20 MB, and no single
number serves both.

What made that expensive was not the kill. It was that the parent then waited in
`do_wait` for children the kernel had removed, at load average 0.00, with nothing in any
log to say why -- 66 to 107 minutes per node before anyone looked, and the cause visible
only in `dmesg` and `/proc/<pid>/wchan`. A stall watchdog now bounds it: if nothing is
written under the output or logging directory for 20 minutes the converter prints what to
check and exits 75, so an OOM costs a re-run rather than a node.

### Transfer settings, and the trap of setting them in the wrong place

The CRT client and the 64 MB chunk size are **in the image**, not in this script. They
were here once, and that failed quietly: a driver that starts a stage with
`--entrypoint python` never runs this script, and one that runs `aws configure` on the
host does not reach the container, which has its own `$HOME`. Either way the aws CLI
that does the work gets stock defaults, and a 3.9 TB pass moved at 324 MB/s instead of
677 without anything reporting a problem.

Measured on 22.3 GB with a cold page cache:

| | MB/s |
| --- | --: |
| stock defaults | 323.6 |
| classic client, 100 concurrent requests | 348.8 |
| **CRT client, 64 MB chunks** | **676.6** |
| CRT + `target_bandwidth 18Gb/s` | 697.7 |
| CRT, written to tmpfs instead of gp3 | 1594.7 |

So `NIC_RATE` is worth about 3% and stays here because it is the one genuinely
machine-specific value. And past ~700 MB/s the volume is the limit rather than the
link: gp3 at 16000 IOPS against ~40 KB shard writes works out to 625 MB/s, which is
where it lands, so provisioning more MB/s alone would not move it. A node that needs
to go faster than that wants more IOPS, striped volumes, or local NVMe.

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
copy, not a gap in S3. Nothing is blocked on code any more, but one of the three former
blockers has not been run since it was addressed:

| dataset | was | now |
|---|---|---|
| `bc_z` | mirror is **`array_record`**, not tfrecord — the only one of the 27, and `as_dataset()` cannot read it | the adapter reads the format from `dataset_info.json` and switches to `as_data_source()`. **Never run against the real mirror** — what the reader assumes about tfds is tested (`openx2lerobot/tests/test_rlds_decoding.py`), the 1090-shard source is not. Expect to debug this one on first contact. |
| `furniture_bench_...` | transform imports `tensorflow_graphics`, which was in no extra | `tensorflow_graphics` is in the `openx` extra |
| `iamlab_cmu_pickup_insert_...` | same | same |

`cmu_playing_with_food` also needs `tensorflow_graphics` but is not in the registry.

`toto`'s OOM was addressed by `workers: -1` and has also never been re-run.

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
