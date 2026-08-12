import json

import pytest

from lerobot_pipeline.video_ops import (
    MAX_THREADS_PER_FFMPEG,
    EncodingParams,
    build_ffmpeg_command,
    order_by_size_desc,
    parse_ffprobe_video_stream,
    plan_parallelism,
)

ENCODING = EncodingParams(codec="libx264", preset="fast", crf=18, gop=2, pix_fmt="yuv420p")


# --- parallelism auto-tuning -------------------------------------------------
# The reference script ran `cores` processes each with ffmpeg's default thread
# count (= cores), oversubscribing by cores^2. These tests pin the fix.


def test_many_small_files_use_one_thread_each_and_fill_every_core():
    plan = plan_parallelism(file_count=1000, cores=64)
    assert plan.threads == 1
    assert plan.workers == 64


def test_few_large_files_use_multiple_threads_to_avoid_idle_cores():
    plan = plan_parallelism(file_count=4, cores=64)
    assert plan.workers == 4
    assert plan.threads == 16
    assert plan.workers * plan.threads == 64


def test_never_spawns_more_workers_than_files():
    plan = plan_parallelism(file_count=3, cores=64)
    assert plan.workers == 3


def test_threads_are_capped_because_x264_threading_scales_poorly():
    plan = plan_parallelism(file_count=1, cores=256)
    assert plan.threads == MAX_THREADS_PER_FFMPEG


def test_single_core_still_produces_a_usable_plan():
    plan = plan_parallelism(file_count=100, cores=1)
    assert plan.workers == 1
    assert plan.threads == 1


def test_no_files_does_not_produce_a_zero_plan():
    plan = plan_parallelism(file_count=0, cores=8)
    assert plan.workers >= 1
    assert plan.threads >= 1


def test_explicit_overrides_win_over_auto_tuning():
    plan = plan_parallelism(file_count=1000, cores=64, workers=2, threads_per_ffmpeg=3)
    assert (plan.workers, plan.threads) == (2, 3)


def test_partial_override_still_derives_the_other_side():
    plan = plan_parallelism(file_count=1000, cores=64, threads_per_ffmpeg=4)
    assert plan.threads == 4
    assert plan.workers == 16


# --- straggler avoidance -----------------------------------------------------


def test_largest_files_are_scheduled_first(tmp_path):
    small, medium, large = (tmp_path / n for n in ("s.mp4", "m.mp4", "l.mp4"))
    small.write_bytes(b"x" * 10)
    medium.write_bytes(b"x" * 100)
    large.write_bytes(b"x" * 1000)

    assert order_by_size_desc([small, large, medium]) == [large, medium, small]


def test_ordering_is_stable_for_equal_sizes(tmp_path):
    a, b = (tmp_path / n for n in ("a.mp4", "b.mp4"))
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"x" * 10)
    assert order_by_size_desc([a, b]) == [a, b]


# --- ffmpeg command construction --------------------------------------------


def test_filters_are_chained_into_a_single_pass():
    cmd = build_ffmpeg_command(
        "in.mp4", "out.mp4", ("scale=256:192", "crop=256:192"), ENCODING, threads=1
    )
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == "scale=256:192,crop=256:192"


def test_audio_is_dropped_rather_than_copied():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=256:192",), ENCODING, threads=1)
    assert "-an" in cmd
    assert "copy" not in cmd


def test_gop_is_pinned_so_random_access_training_reads_stay_fast():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=256:192",), ENCODING, threads=1)
    assert cmd[cmd.index("-g") + 1] == "2"
    # scene-cut detection would make the interval nondeterministic
    assert cmd[cmd.index("-sc_threshold") + 1] == "0"


def test_thread_count_is_set_for_both_decode_and_encode():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=256:192",), ENCODING, threads=4)
    assert [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-threads"] == ["4", "4"]
    # the decode-side -threads must precede -i to apply to the input
    assert cmd.index("-threads") < cmd.index("-i")


def test_encoding_parameters_are_passed_through():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=256:192",), ENCODING, threads=1)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_runs_unattended_and_overwrites():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", ("scale=256:192",), ENCODING, threads=1)
    assert cmd[0] == "ffmpeg"
    for flag in ("-y", "-nostdin"):
        assert flag in cmd
    assert cmd[-1] == "out.mp4"
    assert cmd[cmd.index("-i") + 1] == "in.mp4"


def test_empty_filter_chain_is_rejected_because_it_would_be_a_pointless_reencode():
    with pytest.raises(ValueError):
        build_ffmpeg_command("in.mp4", "out.mp4", (), ENCODING, threads=1)


# --- ffprobe output parsing --------------------------------------------------


def test_parses_shape_and_frame_count_from_ffprobe_json():
    payload = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "height": 480,
                    "width": 640,
                    "nb_frames": "300",
                    "avg_frame_rate": "30/1",
                }
            ]
        }
    )
    info = parse_ffprobe_video_stream(payload)
    assert (info.height, info.width) == (480, 640)
    assert info.frames == 300
    assert info.fps == pytest.approx(30.0)


def test_skips_non_video_streams():
    payload = json.dumps(
        {
            "streams": [
                {"codec_type": "audio", "height": 0, "width": 0},
                {
                    "codec_type": "video",
                    "height": 192,
                    "width": 256,
                    "nb_frames": "10",
                    "avg_frame_rate": "30/1",
                },
            ]
        }
    )
    info = parse_ffprobe_video_stream(payload)
    assert (info.height, info.width) == (192, 256)


def test_missing_frame_count_is_reported_as_unknown_not_zero():
    payload = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "height": 192,
                    "width": 256,
                    "nb_frames": "N/A",
                    "avg_frame_rate": "30/1",
                }
            ]
        }
    )
    assert parse_ffprobe_video_stream(payload).frames is None


def test_no_video_stream_raises():
    payload = json.dumps({"streams": [{"codec_type": "audio"}]})
    with pytest.raises(ValueError):
        parse_ffprobe_video_stream(payload)
