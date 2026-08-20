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
