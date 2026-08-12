# Download

First stage of `download → preprocess → upload`. Give it an S3 prefix and a
directory; it pulls everything as fast as the machine allows.

```bash
uv sync --extra s3
python3 -m download s3://bucket/prefix/ /scratch/data
```

```python
from download import download

download("s3://bucket/prefix/", "/scratch/data")
```

Re-running skips objects already present at the right size, so an interrupted run
resumes for free.

## Defaults

Everything is derived from the instance's NIC rate, read from the link speed or the
instance type — no IAM permissions needed. Override with `--target-gbps` /
`--concurrency` only if you have a reason.

| | value | why |
| --- | --- | --- |
| bandwidth target | detected NIC rate | asking for more than the NIC has makes transfers **slower**, measured 41.4 → 37.4 Gbps |
| concurrency | `target ÷ 3 Gbps`, capped at 32 | one transfer sustains ~3 Gbps in aggregate; past 32 concurrent transfers throughput drops |
| part size | 64 MB | 8 MB parts run out of request budget before bandwidth (~9,400 GET/s needed at 600 Gbps vs ~5,500/s per prefix) |
| ordering | largest first | keeps the run from ending on one straggler |

A single client drives every transfer rather than one process per object: it owns
the connection pool and the bandwidth budget, which separate processes cannot
share.

## Measured throughput

| machine | NIC | achieved | utilisation |
| --- | --- | --- | --- |
| c6id.16xlarge | 25 Gbps | 23.7 | 95% |
| c6id.32xlarge | 50 Gbps | 41.4 | 83% |
| c9gd.48xlarge | 100 Gbps | 66.7 (8.3 GB/s) | 67% |
| c8gn.48xlarge | 600 Gbps | 89.3 | 15% |

A bigger NIC stops paying off well before 600 Gbps. **For large jobs add
instances, not a bigger NIC** — per-second billing keeps the total cost flat while
wall clock divides.

## Disk is often the limit

On c9gd.48xlarge, writing to disk cost 6% versus streaming, and provisioned gp3
was *slower* than the local NVMe it was meant to beat:

| path | GB/s |
| --- | --- |
| local NVMe RAID0, write only | 14.9 |
| 9 × gp3 RAID0 (9,000 MB/s provisioned), write only | 9.0 |
| S3 → nowhere | 8.3 |
| S3 → local NVMe RAID0 | 7.8 |
| S3 → gp3 RAID0 | 6.9 |

Stripe the local NVMe and skip EBS unless local capacity is genuinely too small.
Per-device NVMe throughput varies a lot between instance generations, so measure
rather than extrapolate.
