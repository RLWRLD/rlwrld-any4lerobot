"""`fetch_sample.sh` used to exit 141 on a perfectly good fetch.

It pipes a listing into `head -n "$COUNT"`, which exits (successfully) the
moment it has its lines. Against the real bucket, the listing always has far
more matches than `$COUNT`, so `head` always cuts the still-writing
`aws s3 ls`/`awk` off early -- SIGPIPEing them. With `set -o pipefail`, that
made the whole line, and thus the script, exit 141 even though every
download inside the loop had already completed correctly. A caller scripting
`fetch_sample.sh ... && convert...` would see the fetch as failed.

These tests pin the fix against a fake `aws` -- no network, no S3, no real
credentials -- so they can run as part of the regular synthetic suite. One
proves a fetch that genuinely succeeds (and, incidentally, hits the exact
early-pipe-closure this script always triggers against a real bucket) exits
0; the other proves a real `aws` failure is still reported as one.
"""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "fetch_sample.sh"

# A minimal `aws` stand-in. `s3 ls` emits a synthetic, oversized listing (or
# fails, if asked to) in the same column shape `--recursive` output has:
# <date> <time> <size> <key>. `s3 cp` either "downloads" one file (touches
# the destination given as its second argument) or, for the recursive
# zh_description.txt sweep the real script also issues, does nothing --
# that sweep isn't what these tests are about, and the script already
# tolerates it failing (`|| true`).
FAKE_AWS = """#!/usr/bin/env bash
set -euo pipefail
case "$1 $2" in
  "s3 ls")
    if [[ -n "${FAKE_AWS_LS_FAIL:-}" ]]; then
      echo "fake aws: simulated ListObjectsV2 failure" >&2
      exit 254
    fi
    for i in $(seq 1 "${FAKE_AWS_LS_COUNT:-500}"); do
      printf '2026-08-11 00:00:00 30000 external/fake/task/ep_%05d/data/trajectory.hdf5\\n' "$i"
    done
    ;;
  "s3 cp")
    # `aws s3 cp <source> <dest> [options...]` -- source/dest are always the
    # 3rd/4th words of the whole invocation, before any --flag appears.
    dst="$4"
    case " $* " in
      *" --recursive "*) : ;;  # the zh_description.txt sweep -- not under test here
      *) : > "$dst" ;;
    esac
    ;;
  *)
    echo "fake aws: unsupported invocation: $*" >&2
    exit 99
    ;;
esac
"""


def _fake_aws_env(tmp_path: Path, **extra: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_aws = bin_dir / "aws"
    fake_aws.write_text(FAKE_AWS)
    fake_aws.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AWS_PROFILE": "fake-profile",
        **extra,
    }


def test_a_listing_bigger_than_count_still_exits_zero(tmp_path):
    """The ordinary case against a real bucket: far more matches than
    `$COUNT`, so `head` always cuts `aws s3 ls`/`awk` off early. That must
    not be reported as a failure, and the right number of files must still
    land where the converter expects them.
    """
    dest = tmp_path / "sample"
    env = _fake_aws_env(tmp_path, FAKE_AWS_LS_COUNT="5000", DEST=str(dest))

    result = subprocess.run(
        ["bash", str(SCRIPT), "tienyi", "2"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    downloaded = sorted(dest.rglob("trajectory.hdf5"))
    assert len(downloaded) == 2, downloaded


def test_a_listing_that_never_completes_still_fails(tmp_path):
    """A real `aws` error (bad profile, expired token, ...) must still fail
    the script -- tolerating the SIGPIPE case must not turn into tolerating
    every exit status.
    """
    dest = tmp_path / "sample"
    env = _fake_aws_env(tmp_path, FAKE_AWS_LS_FAIL="1", DEST=str(dest))

    result = subprocess.run(
        ["bash", str(SCRIPT), "tienyi", "2"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert not list(dest.rglob("trajectory.hdf5"))


def test_an_unknown_embodiment_still_fails(tmp_path):
    """Unrelated to the SIGPIPE fix, but worth pinning alongside it: a typo'd
    embodiment name must still be rejected before any `aws` call is made.
    """
    env = _fake_aws_env(tmp_path, DEST=str(tmp_path / "sample"))

    result = subprocess.run(
        ["bash", str(SCRIPT), "not_a_real_embodiment", "2"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "unknown embodiment" in result.stderr
