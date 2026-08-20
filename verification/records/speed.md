# How fast a rebuild actually is

Measured 2026-08-20 on four `m7i.12xlarge` nodes (48 vCPU, 185 GB, gp3 at 1000 MB/s)
running 25 datasets in parallel shares. The run was stopped part way; 15 datasets
completed a full fetch-build-compare-publish cycle and four more finished only their
download. Every number here comes from the step records, which are committed alongside
under `timings/`.

## S3 to instance: 448 MB/s, and that is 19% of the NIC

| dataset | source | fetch | MB/s |
| --- | --: | --: | --: |
| `droid` | 1834.8 GB | 67.8m | 451.1 |
| `fmb_dataset` | 1285.7 GB | 46.4m | 462.3 |
| `furniture_bench` | 123.5 GB | 4.8m | 428.7 |
| `fractal20220817_data` | 119.3 GB | 4.4m | 450.0 |
| `dobbe` | 92.0 GB | 3.5m | 438.1 |
| `berkeley_autolab_ur5` | 82.0 GB | 3.1m | 443.4 |
| … 13 smaller | | | 320-427 |
| **total** | **3916.1 GB** | **2.43h** | **447.9** |

The rate rises with dataset size and then stops dead at about **450 MB/s**, which is
3.6 Gbps. m7i.12xlarge has 18.75 Gbps of baseline network, so **four fifths of the link
is idle** — and it is not the disk either, since gp3 was provisioned at 1000 MB/s and
this is under half of it.

That 450 MB/s ceiling holds from 82 GB up to 1.8 TB, across datasets with very different
file counts. A limit that flat is a configured limit, not a physical one.

### It was: the transfer settings never reached the process doing the transfer

`node.sh` sets `preferred_transfer_client crt`, a 64 MB chunk size and
`target_bandwidth` — and this run never executed `node.sh`, because its driver started
each stage with `--entrypoint python`. The launcher set the same three on the *host*
instead, in user-data, and the transfer runs **inside the container**, which has its own
`$HOME`. So the aws CLI that moved 3.9 TB ran on stock defaults.

Measured on a fresh m7i.12xlarge, 22.3 GB of `utaustin_mutex`, cold page cache, config
varied first and destination second:

| config | destination | MB/s | Gbps |
| --- | --- | --: | --: |
| stock defaults | gp3 | 323.6 | 2.59 |
| classic, 100 concurrent requests | gp3 | 348.8 | 2.79 |
| **CRT, 64 MB chunks** | gp3 | **676.6** | 5.41 |
| CRT + `target_bandwidth 18Gb/s` | gp3 | 697.7 | 5.58 |
| CRT + `target_bandwidth 30Gb/s` | gp3 | 697.7 | 5.58 |
| **CRT, 64 MB chunks** | **tmpfs** | **1594.7** | **12.76** |
| stock defaults | tmpfs | 446.5 | 3.57 |

Three things fall out, and the first correction is to an earlier draft of this file,
which said `target_bandwidth` was the thing to check.

**The CRT client is the lever, worth 2.1x** — 324 to 677 MB/s. Nothing else in the
config comes close.

**`target_bandwidth` is worth 3%,** and 18 versus 30 Gbps is worth nothing at all. The
over-declaration trap the bootstrap README documents is real for a 100 Gbps link; at
18.75 Gbps it does not measure.

**Past ~700 MB/s the disk is the limit, not the network.** The same CRT transfer into
tmpfs is 2.4x faster, so gp3 is what holds it — and not by throughput: the volume was
provisioned for 1000 MB/s, and 16000 IOPS against RLDS shard writes of about 40 KB
works out to 625 MB/s, which is where it landed. It is IOPS-bound on small writes, so
raising provisioned MB/s alone would not move it.

Stock-into-tmpfs at 446.5 MB/s is worth noticing: it reproduces the 447.9 MB/s of the
whole 3.9 TB run almost exactly. The production run was config-bound the entire time,
and never came close to the disk ceiling it would have hit next.

## Instance-side: 1,343 frames/s, six to eight times slower than planned

`orchestrator/README.md` sizes the whole collection at 11,000 fps, and calls 8,000 fps
the pessimistic case. Measured over the 15 datasets that finished:

| frames/s | frames | build | source per frame | dataset |
| --: | --: | --: | --: | --- |
| 283 | 38,240 | 2.2m | 128.0 KB | `berkeley_cable_routing` |
| 307 | 146,241 | 8.0m | 360.6 KB | `iamlab_cmu_pickup_insert` |
| 370 | 87,783 | 4.0m | 912.5 KB | `berkeley_autolab_ur5` |
| 394 | 70,127 | 3.0m | 138.2 KB | `jaco_play` |
| 518 | 68,913 | 2.2m | 158.2 KB | `viola` |
| 742 | 168,423 | 3.8m | 282.6 KB | `roboturk` |
| 839 | 358,234 | 7.1m | 212.2 KB | `stanford_hydra` |
| 1,059 | 213,972 | 3.4m | 234.1 KB | `taco_play` |
| 1,652 | 1,139,911 | 11.5m | 78.8 KB | `dobbe` |
| 1,717 | 5,471,693 | 53.1m | 10.0 KB | `bc_z` |
| 2,434 | 279,939 | 1.9m | 24.5 KB | `austin_sirius` |
| **1,343** | **8,855,514** | **109.9m** | | **aggregate** |

**Throughput follows pixels, not episodes.** Source bytes per frame is the column that
predicts the rate: 912 KB/frame gives 370 fps, 10 KB/frame gives 1,717. Episodes per
second spans 1.0 to 16.1 across the same runs and predicts nothing, which is why
`timings` reports it only as a secondary figure.

At 1,343 fps the 60.1M frames still unbuilt in this round are 12.4 node-hours, or about
3 hours across four nodes -- if nothing stalls. Something did.

### It is not the CPU, and not the three things that looked likeliest

`jaco_play`, 70,127 frames, one m7i.12xlarge, everything else held still:

| configuration | wall | frames/s | CPU idle |
| --- | --: | --: | --: |
| 8 workers | 271s | 259 | 66% |
| 16 workers | 182s | 385 | 50% |
| **30 workers** | **156s** | **450** | **36%** |
| 48 workers | 182s | 385 | 39% |
| 30 workers, TF threads capped to 1 | 153s | 458 | 36% |
| 30 workers, 5 episodes per task | 160s | 438 | 35% |
| 30 workers, both | 157s | 447 | 34% |
| 48 workers, both | 164s | 428 | 33% |

Four hypotheses, three of them dead:

- **More cores would help.** No: 48 workers is *slower* than 30, and a third of the CPU
  is idle in every configuration including the fastest.
- **Thread oversubscription.** One worker process carries 226 threads, all TensorFlow's,
  so 30 workers ask 48 cores for thousands. Capping them changes nothing: 153s against
  156s.
- **Task granularity.** 976 episodes at 25 per task is 40 tasks, so past 30 workers most
  finish early and the run waits on stragglers -- the work tree confirms it, showing 0 to
  4 live converters late in a 48-worker build. Cutting tasks to 5 episodes, 196 of them,
  changes nothing either: 160s.
- **The two-frame keyframe interval.** Also no, and backwards: GOP 250 encodes *slower*
  than GOP 2 (1.20 against 0.94 ms/frame), because at 224x224 the intra frames are cheap
  and the long-GOP motion search is not.

### Where a frame's time actually goes

Same node, 300 frames of jaco_play at 224x224, each stage timed on its own:

| stage | ms/frame | share of the measured pipeline |
| --- | --: | --: |
| **read + decode from the tfrecord** | **8.18** | **89%** |
| resize | 0.10 | 1% |
| encode AV1, GOP 2, default threads | 0.94 | 10% |
| encode AV1, GOP 2, one thread | 0.54 | — |

**The decode dominates, at eight times the encode.** That is TensorFlow parsing the
protobuf and decoding the JPEG, and its cost scales with pixels -- which is why source
bytes per frame predicts the rate and episodes per second predicts nothing. CPU *type*
is not the issue in either direction: the encoder is fine, and 0.54 ms/frame with one
thread against 0.94 with svt-av1's default says the encoder's own threading costs more
than it buys at this frame size.

**What this does not explain is most of the wall clock.** 9.2 ms of pipeline per frame
is 109 fps a worker, so 30 workers should give about 3,300 fps and give 450. Six sevenths
of the per-worker time is outside the frame loop -- LeRobot's writer, the per-episode
bookkeeping, per-task process startup, the aggregation -- and none of it is measured yet.
That is the next thing to instrument, and until it is, "the decode is the bottleneck" is
true of the frame loop and not of the run.

Two caveats on the numbers above. The 8.18 ms includes building the tf.data pipeline,
amortised over only 300 frames, so steady-state decode is somewhat cheaper. And every
figure is one dataset at one frame size; the 912 KB/frame datasets will not have the same
split.

## The largest cost was not slowness. It was a stall.

Three of the four nodes were doing nothing when the run was stopped:

| node | dataset | build started | idle at 11:04 | OOM kills |
| --- | --- | --- | --: | --: |
| 1 | `droid` | 09:40 | 84m | 15 |
| 2 | `fmb_dataset` | 09:58 | 66m | 12 |
| 3 | `furniture_bench` | 09:05 | 107m | 2 |

Load average 0.00, one surviving `openx_rlds` process, `wchan` = `do_wait`. The kernel
had killed the workers and the parent was waiting for children that no longer existed:

```
Out of memory: Killed process 1618330 (python) anon-rss:4740220kB
State: S (sleeping)          wchan: do_wait
```

**A worker peaked at 4.7 GB.** `workers: -1` asks for one per core, so 48 of them wanted
226 GB on a 185 GB machine. Two things follow, and the second matters more:

1. **The worker count has to come from memory, not cores.** `4 GB per core, which m7i
   gives` is the wrong rule -- the constraint is GB per *worker*, measured at 4.7 GB
   here, so the cap is `min(cores, RAM / ~5 GB)`: 36 on this instance, not 48. Capping
   to 16 by hand unblocked all three nodes.
2. **A killed worker must fail the run, not hang it.** ~4.3 node-hours were spent at
   load 0.00, and nothing in any log said why -- the failure surfaced only through
   `dmesg` and `/proc/<pid>/wchan`. An OOM that stops a run in two minutes costs a
   retry; one that stops it silently costs the whole node until somebody looks.

This is the same hang seen earlier on `stanford_hydra` and written off then as
instance-specific. It is not: it is any dataset whose episodes are large enough that one
worker per core overcommits, and it will recur on every dataset bigger than the ones
that have finished.

## Where the time went

Over the four nodes, `fetch` was 57% of recorded time and `build` 43% -- but that is an
artefact of the stall, since the four largest builds never ran. On the 15 datasets that
completed the split is 14% fetch and 86% build, which is the figure to plan with. The
orchestrator's decision not to overlap the stages is unaffected either way: it was taken
on a 1.3-hour saving over a full pass, and nothing here moves that.


# 2026-08-21: m8i.16xlarge, and the dataset that never finished

`toto` is the dataset `orchestrator/bootstrap/README.md` records as OOM-killed. It hung
twice more at the start of this session. Everything below is one node --
**m8i.16xlarge**, 64 vCPU, 247 GB, 30 Gb/s NIC, EBS good for 20 Gb/s and 80,000 IOPS --
with `/scratch` on **four gp3 volumes striped**, because one gp3 volume tops out at
16,000 IOPS and that was the download ceiling on the previous node.

## Download: 447.9 to 2,916.4 MB/s, six and a half times

| change | MB/s | Gb/s |
| --- | --: | --: |
| the 3.9 TB production pass, as it ran | 447.9 | 3.58 |
| CRT client, now carried by the image | 676.6 | 5.41 |
| + four-way gp3 stripe | 1,370.7 | 10.97 |
| + `target_bandwidth` declared at the NIC rate | 2,687.7 | 21.50 |
| **end to end from the image, `orchestrator fetch`** | **2,916.4** | **23.33** |

**`target_bandwidth` is worth 2x, and this file previously said 3%.** That earlier
figure was taken on a node pinned at 677 MB/s by a single gp3 volume -- a measurement
made where the answer could not vary. Same node, same 130.7 GB, nothing else changed:
1,330.8 MB/s undeclared, 2,687.7 declared.

Chunk size matters less and not monotonically: 16 MB gave 2,797.4 and 128 MB gave
1,877.7, against 64 MB's 2,687.7. 64 MB stays, since 16 MB's 4% is inside the run-to-run
spread and 128 MB is clearly worse.

**What is left.** 2,916.4 MB/s is 78% of this instance's 30 Gb/s NIC and 117% of its
nominal 20 Gb/s EBS budget, so the remaining headroom is small and the next constraint
is the link. A faster download now needs a bigger instance, not a better setting.

## The build: two hangs, both with names

Both found with `py-spy`, which is the tool this needed all along -- neither failure
produces a log line, a return code, or CPU.

**One: `start_method` reached the pool and not the manager.** datatrove's
`LocalPipelineExecutor.run` calls `multiprocess.Manager()` with the default context and
`multiprocess.get_context(self.start_method)` only for the pool. The manager is forked
from a parent holding TensorFlow, inherits a lock no thread in it owns, and wedges; the
pool workers block on the queue it serves. Parent in `pool.py:861 next`, child in
`managers.py:176 serve_forever` via `popen_fork`. Fixed by setting the process default.

**Two: an OOM-killed worker hangs the run instead of failing it.** With the manager
fixed it hung again, differently: 42 workers all in `synchronize.py:101` inside
`Queue.get`, parent in `pool.next`, 128 GB resident, output frozen at 1,472 MB, load
average 0.00. `dmesg`: `Killed process ... anon-rss:9278632kB`. A worker killed while
holding the pool's task-queue lock never releases it.

That one is the more important, because no worker-count estimate will be right always.
`watch_for_oom` reads `/proc/vmstat`'s `oom_kill` counter -- whole-machine, no root, and
it exits 76 within ten seconds naming the mechanism and how to retry.

And the estimate that allowed it: 6 GB a worker, a constant, where toto's workers reach
9.3 GB and jaco_play's 4.74. Now fitted on both against source bytes per episode, and
toto gets 22 workers on this node rather than 41.

## Parallelism is not the constraint any more

toto, worker count forced by lowering the per-worker estimate:

| workers | s | frames/s | peak resident | CPU idle |
| --: | --: | --: | --: | --: |
| **22** (what the model picks) | 249 | 1,181 | 169 GB | 34% |
| 35 | 260 | 1,131 | 244 GB | 9% |
| 49 | 248 | 1,186 | 253 GB | 5% |
| 61 | 247 | 1,191 | 260 GB | 5% |

**Flat from 22 to 61.** Thirty-nine more workers buy 0.8% and cost 91 GB and all the
CPU headroom. The model's 22 is the right number and it is not leaving throughput on the
table.

Task size does help, on both axes at once, and is now 8 rather than 25:

| episodes per task | tasks | frames/s | peak resident |
| --: | --: | --: | --: |
| 25 | 37 | 1,177 | 160.7 GB |
| **8** | 113 | **1,262** | **132.5 GB** |
| 3 | 301 | 1,158 | 103.6 GB |

## Where a worker's time actually goes -- and the earlier answer was wrong

Sixty stacks sampled from one worker mid-build:

| | share |
| --- | --: |
| **`resize_frame`** | **55%** |
| PIL JPEG decode | 20% |
| numpy reductions, the episode statistics | 12% |
| `encode_video_frames` | 5% |
| the TensorFlow iterator | 2% |

The earlier entry in this file put decode at 89% and resize at 1%. That measurement
resized 224x224 frames to 224x224 -- `resize_frame` returns early when the shape already
matches, so it timed a no-op and called it cheap. On a dataset that actually downscales,
the resize is the largest single cost.

### The fidelity choice costs 24% of build throughput

`by_scale` picks sinc above 1.3x down, and toto is 2.5x, so toto pays for the widest
kernel libswscale has:

Speed against the fidelity each filter was chosen or rejected on. The last column is
the video-size ratio from `resize-filter-sweep.md`, on the two strong-downscale datasets:

| filter | s | frames/s | video written | fidelity (dlr_edan / ucsd) |
| --- | --: | --: | --: | --: |
| **sinc -- what ships** | 231 | **1,273** | 1,429 MB | **0.959 / 0.988** |
| lanczos | 194 | 1,516 | 1,303 MB | 0.857 / 0.913 |
| area | 192 | 1,532 | 1,257 MB | 0.778 / 0.875 |
| bicubic | 186 | 1,581 | 1,254 MB | 0.811 / 0.885 |
| bilinear | 186 | 1,581 | 1,116 MB | 0.698 / 0.809 |

Bicubic and bilinear tie exactly, so below sinc the resize stops being the bottleneck --
this is sinc's own cost, not the resize step's.

**Sinc stays, and now the price is known.** It is 19% slower than lanczos and 24% slower
than bicubic, and it is the only filter that put both strong-downscale datasets inside
5%: lanczos clears the 15% tolerance on dlr_edan by 0.7 points, which is not a margin
when four more cameras sit at that scale factor. The ordering is also a warning about
reading this table the other way -- speed and fidelity are almost perfectly inverted
here, so any future move down this list is buying throughput with reproduction.

`resize_frame` also converts RGB24 to RGB24 through libswscale, which cannot take the
fast paths it has for planar formats, and the frame is encoded as yuv420p afterwards
anyway. Resizing straight into the encoder's format would remove one conversion. Not
attempted: LeRobot's writer takes RGB arrays, so it is a change to the boundary rather
than to this function.
