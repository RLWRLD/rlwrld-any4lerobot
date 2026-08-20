import subprocess

import pytest

from orchestrator.transfer import (
    TransferError,
    parse_total_bytes,
    sync,
    sync_command,
    tree_bytes,
)


class FakeRun:
    """Stands in for subprocess.run, recording what it was asked to do."""

    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        return subprocess.CompletedProcess(command, self.returncode, "", self.stderr)


# -- the command --------------------------------------------------------------


def test_sync_copies_a_prefix_into_a_directory():
    command = sync_command("s3://bucket/prefix/", "/scratch/raw/a")

    assert command[:3] == ["aws", "s3", "sync"]
    assert command[3:5] == ["s3://bucket/prefix/", "/scratch/raw/a"]


def test_a_dry_run_asks_aws_not_to_transfer_anything():
    assert "--dryrun" in sync_command("s3://b/p/", "/tmp/x", dry_run=True)
    assert "--dryrun" not in sync_command("s3://b/p/", "/tmp/x")


def test_no_bandwidth_settings_are_baked_into_the_command():
    """Those belong to `aws configure` on the machine, not to this repo -- a number
    right for one instance is wrong for the next, and declaring more bandwidth than
    the NIC has measured slower."""
    command = sync_command("s3://b/p/", "/tmp/x")

    assert not [arg for arg in command if "bandwidth" in arg or "Gb/s" in arg]


# -- running it ---------------------------------------------------------------


def test_a_successful_sync_runs_exactly_one_command(tmp_path):
    run = FakeRun()

    sync("s3://bucket/prefix/", tmp_path / "a", run=run)

    assert len(run.calls) == 1


def test_a_failing_sync_reports_what_aws_said(tmp_path):
    run = FakeRun(returncode=1, stderr="fatal error: Access Denied")

    with pytest.raises(TransferError, match="Access Denied"):
        sync("s3://bucket/prefix/", tmp_path / "a", run=run)


def test_a_missing_aws_cli_says_so_rather_than_failing_obscurely(tmp_path):
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    with pytest.raises(TransferError, match="aws"):
        sync("s3://bucket/prefix/", tmp_path / "a", run=missing)


# -- sizes --------------------------------------------------------------------


def test_the_summarised_listing_yields_a_total():
    listing = (
        "2026-08-14 09:00:00  1024 prefix/a.mp4\n"
        "2026-08-14 09:00:01  2048 prefix/b.mp4\n"
        "\n"
        "Total Objects: 2\n"
        "   Total Size: 3072\n"
    )

    assert parse_total_bytes(listing) == 3072


def test_a_listing_without_a_total_is_unknown():
    assert parse_total_bytes("prefix/ is empty\n") is None


def test_a_tree_is_measured_by_the_files_in_it(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "b.bin").write_bytes(b"y" * 7)

    assert tree_bytes(tmp_path) == 17


def test_a_missing_tree_measures_zero(tmp_path):
    assert tree_bytes(tmp_path / "nope") == 0


class TestBandwidthIsDeclaredBeforeTransfer:
    """target_bandwidth has no CLI flag, so it must be written before the sync.

    It lived in node.sh and kept not applying: a stage started with
    `--entrypoint python` skips that script, and setting it on the host does not reach
    the container that runs the transfer. A 3.9 TB pass moved at stock speed as a
    result. Now the process that transfers is the process that sets it.
    """

    def test_it_sets_target_bandwidth(self):
        from orchestrator.transfer import declare_bandwidth

        seen = []

        def run(command, **kwargs):
            seen.append(command)

            class Ok:
                returncode = 0
                stderr = ""

            return Ok()

        assert declare_bandwidth("30Gb/s", run=run) is True
        assert seen == [["aws", "configure", "set",
                         "default.s3.target_bandwidth", "30Gb/s"]]

    def test_no_rate_declares_nothing(self):
        """A machine that has not said what its link is must not get a guess."""
        from orchestrator.transfer import declare_bandwidth

        seen = []
        assert declare_bandwidth(None, run=lambda c, **k: seen.append(c)) is False
        assert seen == []

    def test_sync_declares_before_it_copies(self):
        """Order matters: the CLI reads the config when it starts, not while running."""
        from orchestrator.transfer import sync

        seen = []

        def run(command, **kwargs):
            seen.append(command)

            class Ok:
                returncode = 0
                stderr = ""

            return Ok()

        sync("s3://bucket/prefix/", "/scratch/raw/x", nic_rate="30Gb/s", run=run)
        assert seen[0][:3] == ["aws", "configure", "set"]
        assert seen[1][:3] == ["aws", "s3", "sync"]
