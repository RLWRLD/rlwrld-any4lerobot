import pytest

from download import Result, plan, split_uri
from download.__main__ import main


def sizes(*values):
    return [(f"p/o{i}.tar", v) for i, v in enumerate(values)]


# --- URI ---------------------------------------------------------------------


def test_splits_bucket_and_prefix():
    assert split_uri("s3://b/a/c/") == ("b", "a/c/")


def test_bucket_only_uri_has_an_empty_prefix():
    assert split_uri("s3://b") == ("b", "")


@pytest.mark.parametrize("bad", ["/local/path", "s3://", "https://x/y"])
def test_rejects_anything_that_is_not_an_s3_uri(bad):
    with pytest.raises(ValueError):
        split_uri(bad)


# --- ordering and resume -----------------------------------------------------


def test_largest_objects_go_first(tmp_path):
    todo, _, _, _ = plan(sizes(10, 300, 100), tmp_path, target_gbps=25)
    assert [size for _, _, size in todo] == [300, 100, 10]


def test_destination_mirrors_the_key(tmp_path):
    todo, _, _, _ = plan(sizes(10), tmp_path, target_gbps=25)
    assert todo[0][1] == tmp_path / "p" / "o0.tar"


def test_objects_already_present_are_skipped(tmp_path):
    done = tmp_path / "p" / "o0.tar"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"x" * 10)

    todo, skipped, _, _ = plan(sizes(10, 20), tmp_path, target_gbps=25)
    assert skipped == ["p/o0.tar"]
    assert [key for key, _, _ in todo] == ["p/o1.tar"]


def test_a_truncated_file_is_refetched(tmp_path):
    partial = tmp_path / "p" / "o0.tar"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 40)

    todo, skipped, _, _ = plan(sizes(100), tmp_path, target_gbps=25)
    assert skipped == [] and len(todo) == 1


# --- concurrency and target --------------------------------------------------


def test_concurrency_scales_with_the_target(tmp_path):
    _, _, _, slow = plan(sizes(*([1] * 100)), tmp_path, target_gbps=25)
    _, _, _, fast = plan(sizes(*([1] * 100)), tmp_path, target_gbps=100)
    assert fast > slow


def test_concurrency_is_capped(tmp_path):
    _, _, _, workers = plan(sizes(*([1] * 100)), tmp_path, target_gbps=600)
    assert workers == 32


def test_never_more_workers_than_objects(tmp_path):
    _, _, _, workers = plan(sizes(1, 1, 1), tmp_path, target_gbps=600)
    assert workers == 3


def test_explicit_concurrency_wins(tmp_path):
    _, _, _, workers = plan(sizes(1), tmp_path, target_gbps=25, concurrency=8)
    assert workers == 8


def test_a_target_above_the_nic_is_clamped(tmp_path, monkeypatch):
    import download

    monkeypatch.setattr(download, "nic_gbps", lambda: 50.0)
    _, _, target, _ = plan(sizes(1), tmp_path, target_gbps=600)
    assert target == 50.0


def test_an_undetectable_nic_warns_rather_than_guessing_high(tmp_path, monkeypatch):
    import download

    monkeypatch.setattr(download, "nic_gbps", lambda: None)
    with pytest.warns(UserWarning, match="could not determine the NIC rate"):
        _, _, target, _ = plan(sizes(1), tmp_path)
    assert target == 25.0


# --- reporting ---------------------------------------------------------------


def test_reports_gbps_and_gigabytes_per_second():
    r = Result(1, 0, 50_000_000_000, 10.0, 16, 100.0)
    assert r.gbps == pytest.approx(40.0)
    assert r.gigabytes_per_s == pytest.approx(5.0)


def test_zero_duration_does_not_divide_by_zero():
    assert Result(1, 0, 10, 0.0, 1, 25.0).gbps == 0.0


# --- CLI ---------------------------------------------------------------------


def test_source_and_dest_are_both_required():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["s3://b/p"])


def test_a_bad_uri_is_reported_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["/local/path", "/tmp/x"])
    assert "s3://" in str(exc.value)
