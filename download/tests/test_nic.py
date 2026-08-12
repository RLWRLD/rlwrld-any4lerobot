import pytest


# --- NIC rate detection must not depend on IAM, and must not fail quietly ----
# A run on c9gd.48xlarge silently fell back to 25 Gbps (concurrency 8 instead of
# 32) because the instance role lacked ec2:DescribeInstanceTypes, and reported a
# plausible-looking 2.9 GB/s. Detection now avoids the API, and says so when it
# cannot determine the rate.


def test_reads_the_link_rate_from_sysfs_without_needing_any_iam():
    from download.nic import nic_gbps_from_sysfs

    assert nic_gbps_from_sysfs(read=lambda: "100000") == 100.0


def test_sysfs_sentinel_values_are_not_believed():
    from download.nic import nic_gbps_from_sysfs

    for value in ("-1", "0", "", "Unknown!", "4294967295"):
        assert nic_gbps_from_sysfs(read=lambda v=value: v) is None


def test_instance_type_maps_to_a_nic_rate_without_calling_the_api():
    from download.nic import nic_gbps_from_table

    assert nic_gbps_from_table("c9gd.48xlarge") == 100.0
    assert nic_gbps_from_table("c8gn.48xlarge") == 600.0
    assert nic_gbps_from_table("c6id.32xlarge") == 50.0


def test_an_unknown_instance_type_returns_none_rather_than_a_guess():
    from download.nic import nic_gbps_from_table

    assert nic_gbps_from_table("zz9zza.99xlarge") is None


def test_detection_prefers_sources_that_need_no_permissions():
    from download import nic as f

    calls = []

    def api(_):
        calls.append("api")
        return 999.0

    assert f.detect_nic_gbps(
        instance_type="c9gd.48xlarge", sysfs=lambda: None, api=api
    ) == 100.0
    assert calls == []  # the table answered; the API was never consulted


def test_the_api_is_still_used_as_a_last_resort():
    from download import nic as f

    assert f.detect_nic_gbps(
        instance_type="zz9zza.99xlarge", sysfs=lambda: None, api=lambda _: 42.0
    ) == 42.0


def test_undetectable_rate_warns_instead_of_silently_using_a_wrong_target():
    from download import nic as f

    with pytest.warns(UserWarning, match="could not determine the NIC rate"):
        assert f.detect_nic_gbps(instance_type=None, sysfs=lambda: None, api=lambda _: None) is None
