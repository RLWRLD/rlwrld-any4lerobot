from download.plan import FetchTask, RemoteObject


# --- the discard path must stay in C -----------------------------------------
# Measured on c9gd: --discard came out *slower* than writing to disk (-11%),
# because every body chunk crossed into a Python on_body callback while the disk
# path let CRT write the file natively. A discard baseline that is slower than
# the thing it is a baseline for is worse than no baseline.


def test_discard_writes_to_devnull_natively_instead_of_a_python_callback(tmp_path):
    import os

    from download.s3 import crt_request_kwargs

    kwargs = crt_request_kwargs(FetchTask(obj=RemoteObject("k", 1), dest=None))
    assert kwargs["recv_filepath"] == os.devnull
    assert "on_body" not in kwargs


def test_a_real_destination_is_passed_straight_through(tmp_path):
    from download.s3 import crt_request_kwargs

    dest = tmp_path / "a" / "b.tar"
    kwargs = crt_request_kwargs(FetchTask(obj=RemoteObject("k", 1), dest=dest))
    assert kwargs["recv_filepath"] == str(dest)
    assert "on_body" not in kwargs
