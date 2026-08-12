# Download

The first stage of `download → preprocess → upload`. Pulls source data from S3 as
fast as the machine allows, so the preprocess stage
([`lerobot_pipeline/`](../lerobot_pipeline/)) is never waiting on bytes.

```bash
# fetch
python3 -m download s3://bucket/prefix/ --dest /scratch/data --suffix .tar

# why is my transfer slow? runs the sample twice -- to disk and to /dev/null --
# and attributes the shortfall to the disk or to the client
python3 -m download s3://bucket/prefix/ --dest /scratch/data --diagnose --sample 8

# network only, disk taken out of the path
python3 -m download s3://bucket/prefix/ --discard --sample 8
```

Needs the `s3` extra: `uv sync --extra s3` (brings in `awscrt` and `boto3`).

## Files

- `plan.py` — what to fetch, in what order, how many at once. Pure decisions.
- `nic.py` — NIC rate detection. Isolated because getting it wrong is silent.
- `s3.py` — listing and execution (one CRT client, boto3 fallback).
- `cli.py` — the command line, including `--diagnose`.

## Why the defaults are what they are

Measured on this project's own data
(`s3://rlwrld-foundry-data/external/action_net/`, 322 tars, 2.735 TB, us-east-1).

### One process, many transfers

A single transfer client peaks near **14 Gbps** regardless of instance size —
14.90 on c6id.16xlarge, 14.15 on c6id.32xlarge, 13.65 on c8gn.48xlarge. So
concurrency is mandatory.

But it has to live *inside one process*. Forking `aws s3 cp` per object reached
only 89 Gbps on a 600 Gbps NIC, and past 64 processes it started losing objects:

| concurrent `aws s3 cp` processes | ok / total | Gbps |
| --- | --- | --- |
| 32 | 604 / 604 | 89.3 |
| 64 | 593 / 611 | 73.1 |
| 96 | 400 / 604 | 26.7 |
| 128 | 171 / 604 | 7.0 |

Hence `MAX_CONCURRENCY = 32`, and hence a single CRT client owning the connection
pool rather than N independent ones fighting over the NIC.

### Never declare more bandwidth than the NIC has

On a 50 Gbps instance, telling the client it had 100 Gbps made it **slower** —
41.41 → 37.42 Gbps, over three alternating runs each (σ 0.36 / 0.46, ranges not
overlapping). `resolve_target_gbps` clamps to the detected NIC rate.

### 64MB parts

At 8MB parts, 600 Gbps would need ~9,400 GET/s, above the ~5,500/s per-prefix
guidance — and these datasets keep every object under one prefix. 8MB parts held
~17 Gbps where 64MB reached 89.

### Detection must not fail quietly

On c9gd.48xlarge the instance role lacked `ec2:DescribeInstanceTypes`. Detection
fell back to 25 Gbps, derived concurrency 8 instead of 32, and reported the
resulting **2.9 GB/s** as if it were the machine's limit. The same machine did
**7.8 GB/s** once configured correctly.

So detection tries, in order: sysfs link speed → a static instance-type table →
the EC2 API. All but the last need no IAM. If nothing answers it **warns** and
tells you to pass `--target-gbps`.

### The discard baseline stays in C

`--discard` writes to `os.devnull` through CRT's `recv_filepath`, not an `on_body`
callback. The callback crossed into Python per body chunk and measured *slower*
than the disk path it was meant to be a baseline for (−11%).

## Measured ceilings, for sizing

| machine | NIC | achieved | utilisation |
| --- | --- | --- | --- |
| c6id.16xlarge | 25 Gbps | 23.7 | 95% |
| c6id.32xlarge | 50 Gbps | 41.4 | 83% |
| c9gd.48xlarge | 100 Gbps | 66.7 (8.34 GB/s) | 67% |
| c8gn.48xlarge | 600 Gbps | 89.3 | 15% |

A bigger NIC stops paying off well before 600 Gbps. **For large jobs, add
instances rather than a bigger NIC** — per-second billing keeps the total flat
while wall clock divides.

## Disk is often the real limit — but check, do not assume

On c9gd.48xlarge (3 local NVMe):

| path | GB/s |
| --- | --- |
| fio, local NVMe RAID0 write | **14.87** |
| fio, 9× gp3 RAID0 write (9,000 MB/s provisioned) | 9.00 |
| S3 → `/dev/null` | 8.34 |
| S3 → local NVMe RAID0 | 7.84 |
| S3 → gp3 RAID0 | 6.85 |

Two things worth carrying forward. **Local NVMe throughput varies enormously by
instance generation** — extrapolating c6id's ~1.2 GB/s per device predicted
3.3–4 GB/s for c9gd and was wrong by 4x. And **provisioned gp3 was both slower
and not free** here, so reach for it only when local capacity is genuinely
insufficient.

`--diagnose` exists so this gets measured instead of estimated.
