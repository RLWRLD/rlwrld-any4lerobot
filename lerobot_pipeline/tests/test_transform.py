import json

import pytest

from lerobot_pipeline.meta import load_info
from lerobot_pipeline.registry import build_step
from lerobot_pipeline.transform import link_or_copy, materialize, plan_transform

CAM = "observation.images.egocentric"
WRIST = "observation.images.wrist"

VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _info():
    def camera(height, width):
        return {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
            },
        }

    return {
        "codebase_version": "v2.1",
        "fps": 30,
        "video_path": VIDEO_TEMPLATE,
        "features": {
            CAM: camera(480, 640),
            WRIST: camera(192, 256),
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
        },
    }


def _dataset(tmp_path, episodes=2, sizes=None):
    """A miniature v2.1 layout: videos plus the parquet/meta files around them."""
    root = tmp_path / "src"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(_info()))
    (root / "meta" / "episodes.jsonl").write_text('{"episode_index": 0}\n')

    for episode in range(episodes):
        data = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(b"parquet-" + bytes([episode]))

        for key in (CAM, WRIST):
            video = root / "videos" / "chunk-000" / key / f"episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            size = (sizes or {}).get((key, episode), 100)
            video.write_bytes(b"v" * size)

    return root


def _resize(**kwargs):
    return build_step({"type": "resize_preserve_aspect_area", **kwargs})


def _rel(root, paths):
    return sorted(str(p.relative_to(root)) for p in paths)


# --- planning ----------------------------------------------------------------


def test_a_resized_key_produces_one_transcode_per_video_file(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])

    sources = _rel(root, [job.src for job in plan.transcodes])
    assert sources == [
        f"videos/chunk-000/{CAM}/episode_000000.mp4",
        f"videos/chunk-000/{CAM}/episode_000001.mp4",
    ]


def test_a_key_already_at_the_target_shape_is_linked_not_re_encoded(tmp_path):
    """WRIST is already 192x256; re-encoding it would only lose quality."""
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])

    assert all(WRIST not in str(job.src) for job in plan.transcodes)
    assert any(WRIST in str(src) for src, _ in plan.links)


def test_transcode_carries_the_composed_filter_chain(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    assert plan.transcodes[0].filters == ("scale=256:192:flags=bicubic",)


def test_transcode_mirrors_the_source_codec(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    assert plan.transcodes[0].encoding.codec == "libx264"


def test_destination_mirrors_the_source_layout(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    plan = plan_transform(root, dest, [_resize()])
    job = plan.transcodes[0]
    assert job.dst.relative_to(dest) == job.src.relative_to(root)


def test_non_video_files_are_linked(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    linked = _rel(root, [src for src, _ in plan.links])
    assert "data/chunk-000/episode_000000.parquet" in linked
    assert "meta/episodes.jsonl" in linked


def test_info_json_is_rewritten_rather_than_linked(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    assert all("info.json" not in str(src) for src, _ in plan.links)


def test_plan_patches_the_shape_of_the_resized_key_only(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    assert plan.info["features"][CAM]["shape"] == [192, 256, 3]
    assert plan.info["features"][WRIST]["shape"] == [192, 256, 3]
    assert plan.info["features"][CAM]["info"]["video.height"] == 192


def test_no_steps_means_nothing_is_re_encoded(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [])
    assert plan.transcodes == ()
    assert len(plan.links) == 7  # 4 videos + 2 parquet + 1 episodes.jsonl


def test_biggest_files_are_scheduled_first(tmp_path):
    root = _dataset(tmp_path, episodes=3, sizes={(CAM, 0): 10, (CAM, 1): 900, (CAM, 2): 50})
    plan = plan_transform(root, tmp_path / "out", [_resize()])
    assert [job.src.stem for job in plan.transcodes] == [
        "episode_000001",
        "episode_000002",
        "episode_000000",
    ]


def test_a_key_restricted_step_leaves_other_keys_alone(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize(keys=[WRIST])])
    assert plan.transcodes == ()


# --- materialising -----------------------------------------------------------


def test_materialise_without_transcodes_reproduces_every_file(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    materialize(plan_transform(root, dest, []))

    assert (dest / "data" / "chunk-000" / "episode_000000.parquet").read_bytes() == b"parquet-\x00"
    assert (dest / "videos" / "chunk-000" / CAM / "episode_000000.mp4").exists()
    assert (dest / "meta" / "episodes.jsonl").exists()


def test_materialise_writes_the_patched_info(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    materialize(plan_transform(root, dest, []))
    assert load_info(dest)["codebase_version"] == "v2.1"


def test_unchanged_files_are_hardlinked_so_no_disk_is_wasted(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    materialize(plan_transform(root, dest, []))

    source = root / "data" / "chunk-000" / "episode_000000.parquet"
    assert (dest / "data" / "chunk-000" / "episode_000000.parquet").stat().st_ino == source.stat().st_ino


def test_link_or_copy_creates_missing_parents(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "deep" / "nested" / "a.bin"

    link_or_copy(src, dst)
    assert dst.read_bytes() == b"payload"


def test_link_or_copy_falls_back_to_a_copy_when_linking_is_impossible(tmp_path, monkeypatch):
    src = tmp_path / "a.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "b.bin"

    def refuse(*args, **kwargs):
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr("os.link", refuse)
    link_or_copy(src, dst)

    assert dst.read_bytes() == b"payload"
    assert dst.stat().st_ino != src.stat().st_ino


def test_materialise_refuses_to_write_into_an_existing_destination(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(FileExistsError):
        materialize(plan_transform(root, dest, []))


def test_materialise_can_be_told_to_overwrite(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.txt").write_text("old")

    materialize(plan_transform(root, dest, []), overwrite=True)
    assert not (dest / "stale.txt").exists()


def test_encoding_overrides_from_runtime_config_reach_the_transcode(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(
        root, tmp_path / "out", [_resize()], preset="veryfast", crf=23
    )
    assert plan.transcodes[0].encoding.preset == "veryfast"
    assert plan.transcodes[0].encoding.crf == 23


def test_encoding_overrides_do_not_change_the_mirrored_codec(tmp_path):
    root = _dataset(tmp_path)
    plan = plan_transform(root, tmp_path / "out", [_resize()], preset="veryfast")
    assert plan.transcodes[0].encoding.codec == "libx264"
    assert plan.transcodes[0].encoding.gop == 2
