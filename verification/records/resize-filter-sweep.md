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

## What is still open

The gentle end. The old table's other dataset, taco_play at 1.2x down, is where bicubic
looked best (1.01/1.04) and sinc overshot (1.10/1.15) -- and those numbers may carry the
same filename-pairing flaw as my first attempt, since taco_play is reordered too. If
they hold, no single filter fits both ends and the choice becomes a per-scale rule
rather than one value.

`stanford_hydra` at 1.25x down is the cheap probe for that regime: 570 episodes against
taco_play's 3,242, near the same scale factor, and two cameras rather than one.

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
