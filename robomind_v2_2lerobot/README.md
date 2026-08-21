# robomind_v2_2lerobot

Convert RoboMIND 2.0 (12 embodiments, 16 source repos, 261,961 usable episodes) to
LeRobot v3.0. Converting zero episodes is treated as a failure, not a silent no-op —
see [Why not `robomind2lerobot` (v1)](#why-not-robomind2lerobot-v1) for the regression
this closes.

```bash
uv sync --extra robomind_v2
```

Edit the paths in `convert.sh`, then run it:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0

python robomind_v2_2lerobot/robomind_v2_h5.py \
    --src-paths /path/to/robomind_2_0/franka-part-1 \
                /path/to/robomind_2_0/franka-part-2 \
                /path/to/robomind_2_0/franka-part-3 \
                /path/to/robomind_2_0/franka-part-4 \
                /path/to/robomind_2_0/franka-part-5 \
    --output-path /path/to/local \
    --cpus-per-task 2
```

Which embodiment gets read is never a flag. The source layout is
`data/<embodiment>/<task>/success_episodes/<stamp>/data/<name>.hdf5`, so the directory
name under `data/` selects the config in `configs/`. `--src-paths` accepts more than one
root because this release ships one robot (Franka) as five separate repos that all
start at the same `data/franka` path — they need to land in one embodiment, one config,
one set of LeRobot datasets.

Other flags: `--debug` runs serially with no Ray (this is also what the test suite
uses); `--cpus-per-task` (default 2) is how many CPUs Ray reserves for each task when
not running with `--debug`; `--save-depth` additionally writes each depth camera as an
`image` feature; `--min-frames` (default 50) is the frame count below which an episode
is treated as a broken recording rather than a short task, carried over from v1's floor.

A run also writes `summary.json` beside `--output-path`: written/skipped counts and skip
reasons per embodiment, plus which tasks (if any) failed outright. The same tally is
logged at the end of a run, but under Ray a worker's per-episode log lines never reach
the driver's console — this file is what actually survives, and how to reconcile what
landed against what was expected (4,502 files in the release are already known-broken).

### Memory and scratch disk per task

`read_episode` holds every camera's fully decoded frames in memory at once before
building the frame list, so peak memory scales with an episode's frame count, camera
count, and resolution. Measured directly, not assumed: a 2,653-frame, 3-camera,
480×640 episode holds 6.83 GiB of decoded colour alone, +3.80 GiB more with
`--save-depth`; a 310-frame, 6-camera, 720×1280 episode is 4.79 GiB colour, +3.19 GiB
with depth. `--cpus-per-task` is a CPU reservation only, so it never accounted for
this — every task this converter submits to Ray now also carries a `memory`
reservation, sized off the worse of those two per-camera figures (~2.3 GiB/camera
without depth, ~3.6 GiB/camera with it, plus 50% headroom for `decode_color`'s
transient extra copy while it builds a list before copying it into one contiguous
array). Concretely: a 6-camera task reserves ~21 GiB (~32 GiB with depth); a 1-camera
task ~3.5 GiB (~5.4 GiB with depth).

Ray admits fewer concurrent tasks on a memory-constrained node rather than
oversubscribing it, which is the point — but it does mean `--cpus-per-task 2` on a
64-vCPU node no longer implies 32 concurrent tasks unless the node also has the RAM
for 32 of whatever the actual embodiment mix needs. `--cpus-per-task`'s default stays
2 here: there is no single node size this project always runs on to size it against.
Pick the node's RAM for the concurrency you want, or raise `--cpus-per-task` yourself
to deliberately cap concurrency lower on a smaller one.

`add_frame` separately writes one file per frame per camera to scratch disk, ahead of
`save_episode`'s own video-encoding step — roughly 3 GB per episode for the widest
robot in the release. Size scratch disk for that too, not just for the final dataset.

Run the tests from the repo root with `uv run pytest robomind_v2_2lerobot/tests`.

## Sample data

The full release is 114.28 TB — too large to pull to a laptop, and it should be
converted near the S3 mirror rather than from one. `fetch_sample.sh` pulls a
handful of real episodes instead, for local development and manual smoke-testing:

```bash
export AWS_PROFILE=rlwrld   # `default` may be expired; `rlwrld` and `company-sso` both work
robomind_v2_2lerobot/fetch_sample.sh tienyi 2      # -> ./sample/tienyi/, ~0.2-0.5 GB
uv run python robomind_v2_2lerobot/robomind_v2_h5.py \
    --src-paths robomind_v2_2lerobot/sample/tienyi --output-path ./out --debug
```

`fetch_sample.sh <embodiment> [count]` (default `count` 2) lists the embodiment's
objects on S3, keeps only files that look like a real episode (`.hdf5` over 20 KB),
downloads the first `count` of them into `./sample/<embodiment>/` in the same
`data/<embodiment>/<task>/success_episodes/<stamp>/data/*.hdf5` shape the converter
discovers, and also pulls every `zh_description.txt` under that embodiment so
`instruction.source: zh_file` has something to read. `DEST` overrides the `./sample`
destination; the embodiment-to-repo-slug mapping it needs (one robot ships as five
repos, `tienyi` the directory is `tianyi` the repo, ...) lives in the script itself.

Episode size varies enormously by embodiment — from about 24 MB (the smallest
`tienkung_sim` episodes) to about 738 MB (the largest `agilex_mobile` ones) — so
`count` controls how many episodes land on disk, not a fixed download size.

## Why not `robomind2lerobot` (v1)

`robomind2lerobot` reads RoboMIND 1.x. Every key it opens moved in 2.0:

| | v1.x (what the converter reads) | RoboMIND 2.0 (measured) |
|---|---|---|
| RGB | `observations/rgb_images/<cam>` | `camera_observations/color_images/<cam>` |
| depth | `observations/depth_images/<cam>` | `camera_observations/depth_images/<cam>` |
| state | `puppet/<key>` → `(T, D)` | `puppet/<stream>_align/{data, is_intervene, timestamp}` |
| action | `master/<key>` → `(T, D)` | `master/<stream>_align/{data, is_intervene, timestamp}` |
| camera metadata | none | `camera_{color_resolution,depth_resolution,color_channel,model}/<cam>` |

A stream went from one dataset to a three-member group, and `joint_effort_*` /
`joint_velocity_*`, which v1 reads, do not exist in 2.0 at all. Beyond keys: there is no
`train`/`val` split or `benchmark` tier to select with v1's `--benchmark` flag; v1's
embodiment config keys (`agilex_3rgb`, `franka_3rgb`, `ur_1rgb`, ...) don't match 2.0's
directory names and camera counts (`franka` grew from 3 cameras to 6); there is no
`RoboMIND_v1_2_instr.csv` for instructions; and v1 hardcodes `fps=30` where 2.0
actually runs from about 7 Hz to about 101 Hz, varying by episode.

The part that motivates this task specifically: **v1 fails silently.** Its
`get_all_tasks()` finds no `h5_<embodiment>/` directories in a 2.0 tree and yields
nothing, no error. Even if the directories are hand-massaged into v1's shape,
`load_local_dataset()` `KeyError`s on the first read; that's caught by a broad
`except (FileNotFoundError, OSError, KeyError)`, so every episode is skipped;
`total_episodes == 0` at the end makes v1 `rmtree` its own output directory and exit 0.
"0 episodes converted" is indistinguishable from success. `NothingConverted` in
`robomind_v2_h5.py` exists to make that outcome impossible here: an empty result raises
rather than returning, at every one of the ways it can happen (no episodes found, an
embodiment with no config, or every found episode skipped).

A task that fails outright — partway through, after writing some real episodes — is a
different outcome again, and is not swallowed either: `convert_task` re-raises rather
than returning as if the task were merely empty, `main` raises `TasksFailed` (a non-zero
exit) even when every other task succeeded, and the failed task's own output directory
is renamed to `<task>.failed` rather than left finalized and looking complete. See
`convert_task`'s docstring for exactly which failures are treated as recoverable at the
single-episode level versus fatal to the whole task.

## The 12 embodiments

One repo can hold several tasks and, for Franka, several source repos share one
embodiment. Arm/eef widths and camera counts are measured, not assumed, per
embodiment — sampled from 1-2 episodes each, not a full scan; see
[Design](#design) below for that caveat.

| embodiment | usable episodes | TB | tasks | cameras | arm DoF | eef pos DoF | extra streams | fps |
|---|---|---|---|---|---|---|---|---|
| `franka` | 112,624 | 50.87 | 213 | 6 | 8 | 1 | — | 10–14 |
| `ur` | 49,940 | 15.58 | 150 | 6 | 6 | 1 | — | ~7 |
| `agilex` | 22,206 | 13.28 | 66 | 3 | 6 | 1 | — | ~101 |
| `ark` | 14,857 | 10.45 | 54 | 3 | 6 | 1 | — | 56–61 |
| `agilex_mobile` | 13,873 | 10.23 | 77 | 3 | 6 | 1 | chassis pose/twist, tactile `(T,2,6)` | 63–66 |
| `tienkung` | 14,740 | 5.85 | 47 | 1 | 7 | 1 | — | 20–29 |
| `ark_mobile` | 4,295 | 3.09 | 20 | 3 | 6 | 1 | chassis pose/twist | ~27 |
| `franka_sim` | 10,692 | 2.14 | 34 | 6 | 7 | 1 | `_raw` duplicate, metadata, intrinsics/extrinsics | ~30 |
| `tienyi` | 7,145 | 1.90 | 36 | 1 | 7 | 1 | — | ~48 |
| `ur_dex` | 1,795 | 0.41 | 6 | 6 | 6 | **12** | — | ~8 |
| `tienyi_mobile` | 1,779 | 0.29 | 10 | 1 | 7 | 1 | chassis twist, head_position | ~19 |
| `tienkung_sim` | 8,015 | 0.19 | 25 | 1 (`camera_head`) | 7 | **12 puppet / 6 master — dropped** | `_raw` duplicate, metadata, intrinsics/extrinsics | ~30 |

fps for the two `_sim` embodiments looked unmeasurable at first glance — their
`camera_observations/timestamp` was assumed frozen, by analogy with a design that never
actually checked. Measuring real files found it advancing on every frame like any other
embodiment, just in milliseconds rather than `real`'s seconds; see
[Why fps and resolution are not config fields](#why-fps-and-resolution-are-not-config-fields).

Camera names, by count:

| cameras | embodiments | names |
|---|---|---|
| 6 | `franka` `franka_sim` `ur` `ur_dex` | `camera_{front,left,right,top,wrist_left,wrist_right}` |
| 3 | `agilex` `agilex_mobile` | `camera_{front,left,right}` |
| 3 | `ark` `ark_mobile` | `camera_{left,right,top}` |
| 1 | `tienkung` `tienyi` `tienyi_mobile` | `camera_top` |
| 1 | `tienkung_sim` | `camera_head` |

## Config fields

`configs/<embodiment>.yaml` is pure data — dtype, shape, and names, nothing that reads
like Python. Adding an embodiment this converter already knows how to handle is one
YAML file. `robomind_v2_utils/configs.py` validates every field below; an unknown key
or a malformed value raises `ConfigError` rather than being ignored.

- **`cameras`** — mapping of camera name to `{depth: bool}`. `depth: true` marks a
  camera whose depth stream exists to read; only `--save-depth` actually reads and
  writes it as an `image` feature. Without `--save-depth`, `depth: true` requires
  nothing extra from the file — an episode missing its depth data is still usable, and
  every real conversion so far has run without `--save-depth`.
- **`streams`** — mapping of stream name to `{width: int}`. Each entry reads
  `puppet/<name>_align` as `observation.states.<name>` and `master/<name>_align` as
  `actions.<name>`. The width is checked against the file, not trusted: a mismatch
  skips the episode rather than reinterpreting it (this is what closes trap ① below).
- **`extra`** — mapping of group name to member name to `{shape: [...]}`, for a
  per-frame array that is neither a camera nor a puppet/master stream. Only
  `agilex_mobile` has one: two tactile pads at `tactile_observations/tactile_{left,right}_align`,
  shape `(T, 2, 6)`.
- **`instruction.source`** — one of `zh_file` (read `<task>/zh_description.txt`, the
  ten real embodiments), `h5_metadata` (read `metadata/language_instruction` from the
  file, the two sim embodiments), or `dirname` (no source on disk; `ur_dex` has
  neither of the other two). This names where the instruction text lives, not the
  text itself — instructions are data, not config.
- **`layout`** — `real` or `sim`. Marks the two embodiments whose files carry a numeric
  task-directory prefix, duplicate `_raw` streams, and extra sim-only keys (`metadata`,
  `camera_intrinsics`, `camera_extrinsics`, `base_to_robot_transformation`) — and tells
  `reader.episode_fps` which unit `camera_observations/timestamp` is in (seconds for
  `real`, milliseconds for `sim`). There is no `fps` field: every embodiment's rate is
  measured, not stated. See below for why.

### Why fps and resolution are not config fields

The rule this project uses: config states only what is constant within an embodiment
*and* cannot be computed from the file. fps and resolution both fail that test:

- **fps** varies from about 7 Hz to about 101 Hz across embodiments, and moves between
  episodes of the same embodiment, so it is computed per episode from
  `camera_observations/timestamp` (`reader.episode_fps`) rather than stated once. An
  earlier version of this design believed the two simulated embodiments were the one
  exception — that their clock never advances, so a config would have to state their
  rate instead. Measuring real files disproved that: `camera_observations/timestamp`
  advances on `sim` too, on every episode checked, just in milliseconds rather than
  `real`'s seconds. `episode_fps` converts by `layout` and measures both; one simulated
  episode's clock jumps backward once, mid-episode, so `sim` averages the *positive*
  frame-to-frame steps rather than using the first-to-last span, which that jump would
  otherwise corrupt. The mean, not the median: the timestamps are whole milliseconds,
  so a true 33.333 ms tick still has to land on 33 or 34 every frame, and the median
  just picks whichever is more common — reading a systematically fast 30.3030 Hz
  instead of 30.0000. There was never a case where the rate genuinely couldn't be
  measured, so there is no `fps` config field for either layout.
- **one task's rate is a median, not any single episode's measurement.** One LeRobot
  dataset is opened per `(embodiment, task)` at one fps, but the per-episode rate above
  does move between episodes of the same task — real evidence from this release: two
  episodes of one task measured 26.94 Hz and 31.33 Hz, 16% apart. `convert_task` takes
  the *median* of every episode's own rate as the task's fps (`reader.task_fps`, a
  cheap pre-pass over just the tiny timestamp array — no frame decoding, no cost
  anywhere near `read_episode`'s), and skips, rather than writes onto the wrong time
  base, any episode whose own rate drifts from that median by more than 10%.
- **resolution** is measured by decoding an episode's first frame
  (`images.frame_shape`), never read from the file's own `camera_color_resolution`
  field. That field lies about axis order (see trap ④ below), so trusting it would
  silently transpose height and width for half the corpus.

## Six traps this converter guards against

These came out of measuring the actual 2.0 corpus (114.28 TB, 269,569 objects) rather
than assuming it matches 1.x or matches itself across embodiments. Full detail on each
was in the design doc named in [Design](#design) below; this is the short version and
where the guard lives.

| # | trap | guarded in |
|---|---|---|
| ① | `ur_dex` uses the same stream names as `ur`; only the width differs (`end_effector_*_position` is 1-wide gripper on `ur`, 12-wide dexterous hand on `ur_dex`). Mapping by name alone reads a dexterous hand as a gripper with no error. | `configs.py` (`Stream.width` is required and validated); `reader._stream_data` (checks the file's width against it, raises `EpisodeSkipped` on mismatch) |
| ② | Two different kinds of broken file exist upstream: 4,500 files (all `ur`) carry the full group skeleton but no dataset anywhere inside it, at any depth; 2 files were truncated mid-write and hold only their first stream. Filtering on size alone catches only the first kind — and `handle.keys()` alone can't even tell that one from a whole file, since the skeleton keeps the top-level group names non-empty either way. | `reader.check_usable` (a whole-tree walk via `handle.visititems(...)` for the first kind, since top-level keys stay non-empty regardless; a required-key count for truncated, told apart from a wrong-embodiment config by whether *any* of the config's own keys are present at all) |
| ③ | The two `sim` embodiments are a different format: episode filename is `<episode_id>.hdf5` instead of `trajectory.hdf5`, task directories carry a numeric prefix, every stream exists as both `_align` and `_raw`, sim-only keys (`metadata`, intrinsics/extrinsics) appear, and `camera_observations/timestamp` advances in milliseconds rather than `real`'s seconds. | `reader.discover` globs any `*.hdf5` regardless of name; `reader._from_dirname` strips the numeric prefix; `reader.read_streams` only ever builds `_align` paths, so `_raw` is never read regardless of layout; `reader.episode_fps` picks the unit from `layout` and measures both, rather than a config stating a `sim` rate it was once assumed couldn't be measured |
| ④ | `camera_color_resolution` stores axes as (H, W) for a real episode and (W, H) for a simulated one, while the decoded pixels are (H, W) in both. | `images.frame_shape` (and `decode_color`) never read that field — resolution comes from decoding the first frame |
| ⑤ | Instructions come from three different places and none of them is complete: `h5_metadata` only exists for sim, `ur_dex` has neither of the file-based sources, and even `zh_file` has holes — `ur/assemble_lego_letters` has 521 episodes but no `zh_description.txt`, and `ark_mobile/grab_beaker_from_left_and_place_on_right` has the file but zero episodes. | `reader.instruction` tries the config's named source first, `reader._from_dirname` is the universal fallback that guarantees a non-empty prompt even when the named source is missing, unreadable, or empty |
| ⑥ | The good one: `_align` streams are already time-aligned to the cameras — `camera`/`puppet`/`master` timestamps are byte-identical across all 12 embodiments in the sample. | Nowhere, deliberately — no clock-alignment algorithm exists in this converter because none was needed; `reader.read_streams` simply never reads anything but `_align` |

## Design

Full measurements (S3 inventory, per-embodiment schema, the repo→embodiment mapping,
sample strategy, and the 1-2-episode sampling caveat referenced above) and the
rationale behind decisions summarized in this file — why not `spec2lerobot`, why the
config folders stay separate, what counts as config versus data — lived in a design
doc at `docs/superpowers/specs/2026-08-21-robomind-v2-converter-design.md` while this
converter was being built. That directory is gitignored by convention (design docs are
working notes, not shipped artifacts), so the file will not exist in a fresh clone —
this paragraph is a pointer for whoever still has it locally, not a dependency. Nothing
above assumes it exists: everything a maintainer needs — the traps, the config schema,
why fps and resolution are measured rather than configured — is in this file already.
