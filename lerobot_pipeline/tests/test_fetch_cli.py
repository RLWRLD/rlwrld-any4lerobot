import pytest

from lerobot_pipeline.scripts.fetch_s3 import parse_args, split_s3_uri


def test_parses_an_s3_uri_into_bucket_and_prefix():
    assert split_s3_uri("s3://my-bucket/a/b/c/") == ("my-bucket", "a/b/c/")


def test_bucket_only_uri_has_an_empty_prefix():
    assert split_s3_uri("s3://my-bucket") == ("my-bucket", "")


def test_a_non_s3_uri_is_rejected():
    with pytest.raises(ValueError, match="s3://"):
        split_s3_uri("/local/path")


def test_source_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_dest_and_discard_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["s3://b/p", "--dest", "/tmp/x", "--discard"])


def test_one_of_dest_or_discard_is_required():
    with pytest.raises(SystemExit):
        parse_args(["s3://b/p"])


def test_discard_mode_needs_no_destination():
    args = parse_args(["s3://b/p", "--discard"])
    assert args.discard is True and args.dest is None


def test_tuning_flags_default_to_automatic():
    args = parse_args(["s3://b/p", "--dest", "/tmp/x"])
    assert args.concurrency is None
    assert args.target_gbps is None


def test_resume_is_on_by_default_and_can_be_turned_off():
    assert parse_args(["s3://b/p", "--dest", "/tmp/x"]).resume is True
    assert parse_args(["s3://b/p", "--dest", "/tmp/x", "--no-resume"]).resume is False


def test_diagnose_mode_compares_disk_against_streaming(tmp_path):
    """The measured cost of landing bytes on disk was 30-43%, so a slow transfer
    is often a disk problem. --diagnose runs both and reports the gap."""
    args = parse_args(["s3://b/p", "--dest", str(tmp_path), "--diagnose"])
    assert args.diagnose is True


def test_sample_limits_how_many_objects_are_touched():
    assert parse_args(["s3://b/p", "--discard", "--sample", "8"]).sample == 8
