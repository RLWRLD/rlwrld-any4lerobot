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
file counts. A limit that flat is a configured limit, not a physical one. The first thing
to check is `target_bandwidth`, which this run set to `18Gb/s` — the README already
records that declaring more than the link has measured *slower* (41.4 to 37.4 Gbps),
and 18 of 18.75 is declaring essentially all of it.

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
