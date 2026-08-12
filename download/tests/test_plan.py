import pytest

from download.plan import (
    MAX_CONCURRENCY,
    FetchTask,
    RemoteObject,
    plan_fetch,
    resolve_concurrency,
    resolve_target_gbps,
    summarize_fetch,
)


def _objs(*sizes):
    return [RemoteObject(key=f"p/o{i}.tar", size=s) for i, s in enumerate(sizes)]


# --- concurrency ------------------------------------------------------------
# Measured: one transfer process peaks near 14 Gbps, so concurrency is required;
# but past ~32 concurrent transfers the client degrades and then dies outright
# (128 processes lost 433 of 604 objects on c8gn.48xlarge).


def test_concurrency_scales_with_the_bandwidth_target():
    assert resolve_concurrency(n_objects=1000, target_gbps=100) > resolve_concurrency(
        n_objects=1000, target_gbps=25
    )


def test_concurrency_is_capped_because_the_client_collapses_past_it():
    assert resolve_concurrency(n_objects=1000, target_gbps=600) == MAX_CONCURRENCY


def test_never_more_concurrency_than_objects():
    assert resolve_concurrency(n_objects=3, target_gbps=600) == 3


def test_at_least_one():
    assert resolve_concurrency(n_objects=0, target_gbps=100) == 1


def test_explicit_request_wins():
    assert resolve_concurrency(n_objects=1000, target_gbps=25, requested=24) == 24


def test_explicit_request_above_the_cap_warns_but_is_honoured():
    with pytest.warns(UserWarning, match="beyond the measured safe range"):
        assert resolve_concurrency(n_objects=1000, target_gbps=25, requested=200) == 200


# --- bandwidth target -------------------------------------------------------
# Measured: setting target_bandwidth above the real NIC made throughput *worse*
# (41.41 -> 37.42 Gbps when 50 Gbps was declared as 100).


def test_target_defaults_to_the_detected_nic_rate():
    assert resolve_target_gbps(nic_gbps=100) == 100


def test_target_is_clamped_to_the_nic_because_overstating_it_slows_transfers():
    with pytest.warns(UserWarning, match="above the NIC"):
        assert resolve_target_gbps(nic_gbps=50, requested=600) == 50


def test_a_target_below_the_nic_is_left_alone():
    assert resolve_target_gbps(nic_gbps=100, requested=25) == 25


def test_unknown_nic_falls_back_conservatively_rather_than_guessing_high():
    assert resolve_target_gbps(nic_gbps=None) == 25


# --- plan -------------------------------------------------------------------


def test_largest_objects_are_fetched_first(tmp_path):
    plan = plan_fetch(_objs(10, 300, 100), dest_dir=tmp_path, target_gbps=25)
    assert [t.obj.size for t in plan.tasks] == [300, 100, 10]


def test_destination_mirrors_the_key(tmp_path):
    plan = plan_fetch(_objs(10), dest_dir=tmp_path, target_gbps=25)
    assert plan.tasks[0].dest == tmp_path / "p" / "o0.tar"


def test_already_complete_files_are_skipped_on_resume(tmp_path):
    objs = _objs(10, 20)
    done = tmp_path / "p" / "o0.tar"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"x" * 10)

    plan = plan_fetch(objs, dest_dir=tmp_path, target_gbps=25)
    assert [o.key for o in plan.skipped] == ["p/o0.tar"]
    assert [t.obj.key for t in plan.tasks] == ["p/o1.tar"]


def test_a_truncated_file_is_refetched_not_trusted(tmp_path):
    objs = _objs(100)
    partial = tmp_path / "p" / "o0.tar"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 40)

    plan = plan_fetch(objs, dest_dir=tmp_path, target_gbps=25)
    assert plan.skipped == ()
    assert len(plan.tasks) == 1


def test_resume_can_be_disabled(tmp_path):
    objs = _objs(10)
    done = tmp_path / "p" / "o0.tar"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"x" * 10)

    plan = plan_fetch(objs, dest_dir=tmp_path, target_gbps=25, resume=False)
    assert plan.skipped == ()


def test_total_bytes_counts_only_what_will_be_fetched(tmp_path):
    objs = _objs(10, 20)
    done = tmp_path / "p" / "o0.tar"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"x" * 10)

    assert plan_fetch(objs, dest_dir=tmp_path, target_gbps=25).total_bytes == 20


def test_part_size_defaults_large_because_small_parts_cannot_reach_high_rates(tmp_path):
    """At 8MB parts, 600 Gbps would need ~9,400 GET/s -- above the ~5,500/s
    per-prefix guidance, and every object here shares one prefix. 64MB parts need
    ~1,170/s. Measured: 8MB parts held ~17 Gbps where 64MB reached 89."""
    from download.plan import DEFAULT_PART_SIZE

    assert DEFAULT_PART_SIZE >= 64 * 1024 * 1024
    assert plan_fetch(_objs(10), dest_dir=tmp_path, target_gbps=25).part_size == DEFAULT_PART_SIZE


def test_part_size_can_be_overridden(tmp_path):
    plan = plan_fetch(
        _objs(10), dest_dir=tmp_path, target_gbps=25, part_size=16 * 1024 * 1024
    )
    assert plan.part_size == 16 * 1024 * 1024


def test_measurement_mode_writes_nowhere_so_disk_cannot_cap_the_result(tmp_path):
    """Streaming to nowhere isolates the network from the disk -- measured cost of
    landing bytes on NVMe was 30-43%."""
    plan = plan_fetch(_objs(10), dest_dir=None, target_gbps=25)
    assert plan.tasks[0].dest is None
    assert plan.discard is True


# --- reporting --------------------------------------------------------------


def test_reports_both_gbps_and_gigabytes_per_second():
    s = summarize_fetch(total_bytes=50_000_000_000, seconds=10.0, concurrency=16)
    assert s.gbps == pytest.approx(40.0)
    assert s.gigabytes_per_s == pytest.approx(5.0)


def test_reports_nic_utilisation_when_the_nic_rate_is_known():
    s = summarize_fetch(
        total_bytes=50_000_000_000, seconds=10.0, concurrency=16, nic_gbps=100
    )
    assert s.nic_utilisation_pct == pytest.approx(40.0)


def test_zero_duration_does_not_divide_by_zero():
    s = summarize_fetch(total_bytes=10, seconds=0.0, concurrency=1)
    assert s.gbps == 0.0 and s.gigabytes_per_s == 0.0
