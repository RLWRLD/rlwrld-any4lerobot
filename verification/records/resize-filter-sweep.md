# Which resampler the collection was resized with

`video.resize.filter` decides how much detail survives a downscale, and the value it
shipped with -- bicubic -- was measured on two datasets and chosen with one candidate
missing. This is that measurement redone: every filter, on the size metric, against
the delivered copies.

## Method

Rebuild one dataset per filter through the real pipeline (nothing reimplemented), then
compare **total video bytes per camera** against the delivered copy.

The totals are deliberate. A first attempt paired videos by filename and produced
ratios from 0.28 to 2.59, which measured nothing: the rebuild does not write episodes
in the delivered order -- 100 of dlr_edan's 104 sit at a different index -- so
`episode_000006.mp4` on the two sides is two different episodes and the ratio was
mostly their length difference. Both sides hold the same episodes, so the sum over a
camera compares like for like without needing to know which is which.

## Results

Ratio of our total video bytes to the delivered copy's. 1.000 is exact; datasets that
are not resized at all land at 0.98-0.99, which is the encoder build difference by
itself and the best any filter could do.

| filter | ucsd_kitchen 2.5x down | dlr_edan 2.0x down |
| --- | --: | --: |
| bilinear | 0.809 | 0.698 |
| area | 0.875 | 0.778 |
| bicubic -- *what shipped* | 0.885 | 0.811 |
| lanczos | 0.913 | 0.857 |
| **sinc** | **0.988** | **0.959** |

Sinc is the only filter within 5% on either, and it is within 5% on both. Bicubic is
11-19% short.

The ucsd column cross-checks the old table in `resize_frame`'s docstring, which had
bicubic at 0.86 against 0.885 here and sinc at 0.97 against 0.988. Close enough to
say that table's ucsd row was sound and that this method agrees with it.

### A prediction that did not survive

Detail retained was measured first, on dlr_edan episode 6, and it put **lanczos**
closest at 0.93x of the delivered high-frequency content against bicubic's 0.85x. On
the size metric lanczos is 0.857 and sinc is 0.959, so the detail proxy ranked the
wrong filter first. Detail and encoded size move together but not tightly enough to
choose on; the size metric is the one that decides the comparison, so it is the one
that decides the filter.

## The gentle end

`austin_sirius` at 1.31x down, which is **taco_play's `rgb_gripper` geometry exactly**
(84x84 to 64x64) at a sixth of taco_play's episodes:

| filter | image | wrist_image |
| --- | --: | --: |
| **bicubic** | **0.985** | **1.019** |
| lanczos | 1.021 | 1.053 |
| sinc | 1.100 | 1.088 |

The ranking inverts. Bicubic is nearly exact here and sinc is 9-10% over -- which
matches the old table's taco_play sinc figures (1.10/1.15) closely enough to say that
row was sound too.

`stanford_hydra` at 1.25x was tried first and abandoned: 2.1 GB of delivered video made
each build a two-hour job on the original node, and after moving to a larger one the
build **hung** -- three live processes at load 0.00, output stalled at 2178 MB with the
aggregation never starting. Worth a look on its own; not worth blocking a filter
decision on when austin_sirius answers the same question in a tenth of the pixels.

## The decision: sinc

No filter is flat across scale factors. Every one comes out larger the gentler the
downscale, which is the offset `resize_frame`'s docstring already described, and
picking a filter does not remove it -- it only slides the whole curve. So the question
is not which filter is *right* but which one keeps every dataset inside
`SIZE_TOLERANCE`, which is 15%.

| filter | dlr_edan 2.0x | ucsd 2.5x | sirius 1.31x | worst deviation |
| --- | --: | --: | --: | --: |
| bicubic | 0.811 | 0.885 | 0.985 / 1.019 | **18.9% -- fails dlr_edan** |
| lanczos | 0.857 | 0.913 | 1.021 / 1.053 | 14.3% -- passes by 0.7 points |
| **sinc** | **0.959** | **0.988** | 1.100 / 1.088 | **10.0%** |

Sinc, and not because it wins the most columns -- bicubic wins the gentle ones outright.
Because its curve is the flattest of the three, so it is the only one with room at both
ends. Lanczos clears dlr_edan by 0.7 points, which is not a margin; one dataset at a
slightly stronger downscale would take it out, and the collection has four more cameras
at 2.5x.

That the residual offset survives every filter says the remaining difference is not the
resampler -- most likely the encoder build, which is the same thing that puts unresized
datasets at 0.98-0.99 rather than 1.00. Nothing here can close that, and a filter chosen
to hide it would be fitting noise.

### Scope of the change

Eight cameras across five datasets resize at all; the rest of the collection is already
at its target size and is untouched by this. The five: `dlr_edan` (2.0x), `ucsd_kitchen`,
`berkeley_autolab_ur5` (2 cameras), `toto`, `roboturk` (all 2.5x), plus the gentle three
`taco_play` (2), `stanford_hydra` (2) and `austin_sirius` (2). Datasets already verified
under bicubic that do **not** resize -- cmu_stretch, austin_buds -- produce identical
output and need no rebuild.

## Incidental: `workers: -1` overcommits on a 2 GB/core node

stanford_hydra's build died immediately on the verification node until the worker count
was capped. c7i.2xlarge is 8 vCPU and 16 GB -- 2 GB per core -- and `workers: -1` asks
for all eight. The bootstrap README predicts exactly this ("on a machine with less --
c9gd is 2 GB per core -- the worker count has to be capped below the core count"), and
m7i at 4 GB per core is what it recommends. The three datasets measured before this
were small enough not to reach it: 50, 104 and 150 episodes.

Worker count does not affect what is being measured here -- it changes which worker
encodes which episode, not the bytes any episode encodes to -- so capping it to 4 for
the sweep is safe.

## Verified on the built image

Rebuilt and re-compared on an image built from `7bcf55b` -- the image carrying
`filter: sinc`, nothing mounted over it, so what was measured is what a run would use.

| dataset | video ratio, before | after | sampled identical, before | after |
| --- | --: | --: | --: | --: |
| `dlr_edan` 2.0x | 0.79x, **63 of 64 SIZE failures** | **0.96x, none** | 1 of 64 | **64 of 64** |
| `ucsd_kitchen` 2.5x | 0.885x by totals | **0.99x** | -- | **64 of 64** |
| `austin_buds` not resized | 0.99x / 0.97x | 0.99x / 0.97x | 50 of 50 | 50 of 50 |

austin_buds is the control: it is 128x128 at the source, the resize composes to
nothing, and its output is byte-identical to the bicubic build. A filter change that
moved it would have meant the resize was running where it should not.

dlr_edan's own record file went from 31,845 bytes to 6,694 -- the 63 failing episodes
are simply gone from it.

**All three now fail on one thing and it is not ours**: the delivered copy records image
`std` as exactly zero in every episode. Nothing about video, geometry, encoding, state,
action, prompts or distributions differs any more.


## Re-verified on the shipped image

The image built from `7bcf55b`, no mounted patches -- whatever the image carries is
what was measured. Four datasets rebuilt and compared end to end on the node.

| dataset | camera | scale | bicubic | **sinc** |
| --- | --- | --- | --: | --: |
| `dlr_edan` | image | 1.94x | 0.79x **FAIL** | **0.96x** |
| `ucsd_kitchen` | image | 2.50x | 0.885x | **0.99x** |
| `austin_sirius` | image | 1.31x | 0.985x | 1.10x |
| `austin_sirius` | wrist_image | 1.31x | 1.019x | 1.09x |
| `austin_buds` | image | none | 0.99x | 0.99x |
| `austin_buds` | wrist_image | none | 0.97x | 0.97x |

**Episodes flagged SIZE: 0, on all four.** dlr_edan alone was 63 of 64 before.
austin_buds is unchanged because it is 128x128 at the source and never resized, which
is the control: the change touches only what it should.

Every remaining failure is the delivered copy's image `std` of exactly zero, now seen
on four datasets and six cameras. One new marginal difference appeared on
austin_sirius -- `image.max`, 1 pair of 559, 0.1059 against an allowance of 0.1 -- which
is the expected direction: sinc keeps more detail, so a single-pixel extreme lands
slightly further out.

## Still open: cv2, which Slack points at and no measurement has covered

Searching Slack turned up the house convention for downscaling, from the RLDX data
recipe channel on 2026-08-02:

> Sejune Joo: when I resize datasets it is with Inter-area, while rrc resizes with
> Inter-linear at inference.

and, on the rrc side, "resize interpolation changed to `INTER_AREA`; training was
already AREA and only rrc was LINEAR". So the house answer for a downscale is
`cv2.INTER_AREA`.

Two reasons that is not the conclusion here. That thread is about the tactile/teleop
datasets, not the OXE pre-training collection -- whose conversion code ALIN Lab
confirmed was never kept. And nothing in Notion records the resampler; the top hit for
the question is this repo's own table.

But it does name the one candidate the deciding metric has never covered. `cv2
INTER_AREA` has been measured for detail (0.84x) and pixel error (4.077) and never for
file size; what *was* measured for size is **swscale** area, a different
implementation. That the two families diverge sharply is already on record -- on ucsd,
swscale bicubic is 0.885 and cv2 INTER_CUBIC is 1.03, same name, and cv2's kernel width
scales with the downscale ratio in a way swscale's does not. That property is exactly
what would explain the residue no swscale filter removes: every one of them comes out
larger the gentler the downscale, which is why sinc lands at 1.10 on austin_sirius while
reaching 0.96 on dlr_edan.

Adopting cv2 would cost something the current design deliberately bought. The
transform stage expresses its resize as an ffmpeg `scale=W:H:flags=` chain, which can
only name libswscale filters, so a cv2 choice splits the two paths that
`video.resize.filter` was introduced to keep together. The split is at least
lopsided rather than even: the datasets that resize before the writer take the Python
path, where cv2 is available, and the transform stage covers only the nine
h264/GOP250 datasets.
