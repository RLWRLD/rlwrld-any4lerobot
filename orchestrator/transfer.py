"""S3 in and S3 out, via the AWS CLI.

There is no transfer code here on purpose. A coordinated CRT client was built for
this once and measured *slower* than the CLI it was meant to replace (6.67 GB/s
against 7.39 for ``aws s3 cp`` with ``xargs -P16``, same 2.735 TB), so it was
removed. What was worth keeping is the configuration, and configuration is not
code: ``preferred_transfer_client``, ``target_bandwidth`` and
``multipart_chunksize`` are set once per machine with ``aws configure``, and the
repo README carries the numbers.

Nothing here names an instance type or a bandwidth. Declaring more bandwidth than
the NIC has measured slower, so the right value is a property of the machine and
belongs on the machine.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path


class TransferError(RuntimeError):
    """Raised when a transfer did not happen."""


def sync_command(
    source: str, dest: str | Path, *, dry_run: bool = False
) -> list[str]:
    """``aws s3 sync``, which resumes correctly: a re-run skips what is already
    there at the right size, so an interrupted transfer costs nothing to repeat."""
    command = ["aws", "s3", "sync", str(source), str(dest), "--only-show-errors"]
    if dry_run:
        command.append("--dryrun")
    return command


def declare_bandwidth(nic_rate: str | None, *, run=None) -> bool:
    """Tell the CRT client how much link it has, before anything is transferred.

    ``target_bandwidth`` has no command-line flag -- it is read from the aws config --
    so it has to be written before the sync rather than passed to it. That is why this
    used to live in ``node.sh``, and why it kept not applying: a stage started with
    ``--entrypoint python`` skips that script, and a value set on the host does not
    reach the container. Doing it here means the process that transfers is the process
    that sets it.

    Worth 2x, not the 3% an earlier measurement suggested. On m8i.16xlarge with a
    four-way gp3 stripe, 130.7 GB of toto: 1,330.8 MB/s with nothing declared,
    2,687.7 MB/s at ``30Gb/s``. The 3% came from a node pinned at 677 MB/s by a single
    volume, where no client setting could have shown up.

    Returns whether anything was set, so a caller can say so.
    """
    if not nic_rate:
        return False
    _execute(["aws", "configure", "set",
              "default.s3.target_bandwidth", nic_rate], run)
    return True


def sync(
    source: str,
    dest: str | Path,
    *,
    dry_run: bool = False,
    nic_rate: str | None = None,
    run=None,
) -> None:
    declare_bandwidth(nic_rate, run=run)
    _execute(sync_command(source, dest, dry_run=dry_run), run)


def remote_bytes(uri: str, *, run=None) -> int | None:
    """Total size behind an S3 prefix, or ``None`` if it cannot be read.

    Read before transferring rather than estimated: the answer is exact and costs
    one listing.
    """
    command = ["aws", "s3", "ls", "--summarize", "--recursive", uri]
    try:
        completed = _execute(command, run, capture=True)
    except TransferError:
        return None
    return parse_total_bytes(completed.stdout or "")


def parse_total_bytes(listing: str) -> int | None:
    for line in reversed(listing.splitlines()):
        label, separator, value = line.partition(":")
        if separator and label.strip().lower() == "total size":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def tree_bytes(path: str | Path) -> int:
    """How much of the disk a directory occupies, by the files in it.

    Symlinks are not followed and hard links are counted once per name, which is
    what the caller wants: this reports what a transfer moved, not what a
    deduplicating filesystem would free.
    """
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for entry in root.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def _execute(command: Sequence[str], run, *, capture: bool = False):
    runner = run or subprocess.run
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise TransferError(
            f"could not run {command[0]!r}: the AWS CLI is not on PATH. "
            "It is what moves data in and out of this pipeline; install it and "
            "configure it as the repo README describes."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or f"exit status {completed.returncode}"
        raise TransferError(f"{' '.join(command[:4])} failed: {detail}")
    return completed


__all__ = [
    "TransferError",
    "parse_total_bytes",
    "remote_bytes",
    "sync",
    "sync_command",
    "tree_bytes",
]
