"""LeRobot 쪽 접합부 — feature 사전, 에피소드 통계, v3 writer.

``compute_episode_stats`` 와 아래 writer 서브클래스는 ``robomind2lerobot`` 에서
가져왔다. 원본은 손대지 않았다 (v1 산출물의 재현성을 지킨다). 가져오면서 V2.0 에
존재하지 않는 두 가지를 뺐다:

* ``save_episode`` 의 ``split`` 인자 — V2.0 에는 train/val 층이 없다
* ``save_episode`` 의 ``action_config`` 인자 — V2.0 에는
  ``language_description_annotation_json`` 이 없다

feature 이름은 v1 의 관례를 지킨다: RoboMIND 의 원본 필드 이름을 그대로 쓰고,
평탄한 ``observation.state`` 를 조립하지 않는다.
"""

import concurrent.futures
import inspect
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.dataset_writer import DatasetWriter, _encode_video_worker
from lerobot.datasets.feature_utils import validate_episode_buffer
from lerobot.datasets.io_utils import load_image_as_numpy
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH


def build_features(config, shapes: dict[str, tuple[int, ...]]) -> dict:
    """The LeRobot feature dict for one embodiment.

    Video shapes come from ``shapes`` -- measured per episode by decoding the
    first frame -- because the file's own ``camera_color_resolution`` stores its
    axes in one order for real episodes and the other for simulated ones.

    Depth is an ``image`` feature rather than ``video``: encoding it would put a
    lossy codec across distance values.
    """
    features: dict[str, dict] = {}

    for camera in config.cameras:
        key = f"observation.images.{camera.name}"
        if key in shapes:
            features[key] = {
                "dtype": "video",
                "shape": tuple(shapes[key]),
                "names": ["height", "width", "rgb"],
            }
        depth_key = f"{key}_depth"
        if camera.depth and depth_key in shapes:
            features[depth_key] = {
                "dtype": "image",
                "shape": tuple(shapes[depth_key]),
                "names": ["height", "width", "channel"],
                # Flags this key in the pinned lerobot's `depth_keys` (see
                # `DatasetMetadata.depth_keys`, which looks for exactly this
                # `info.is_depth_map` nesting). That property gates two things
                # this feature needs: `DatasetWriter._get_image_file_path` writes
                # it as `.tiff` instead of `.png`, and `load_image_as_numpy`
                # (called back on that path by `sample_images` below) branches on
                # the `.tiff` extension to read the native dtype instead of
                # downcasting through a forced 3-channel RGB convert. Without
                # this, both the storage format and the statistics silently
                # treat single-channel depth as truncated 8-bit colour.
                "info": {"is_depth_map": True},
            }

    for stream in config.streams:
        entry = {
            "dtype": "float32",
            "shape": (stream.width,),
            "names": {"motors": [f"m{index}" for index in range(stream.width)]},
        }
        features[f"observation.states.{stream.name}"] = dict(entry)
        features[f"actions.{stream.name}"] = dict(entry)

    for extra in config.extras:
        features[f"observation.{extra.name}"] = {
            "dtype": "float32",
            "shape": tuple(extra.shape),
            "names": None,
        }

    return features


def sample_images(input):
    """Copied from ``robomind2lerobot/robomind_uitls/lerobot_uitls.py``.

    Both branches below are reachable. The pinned lerobot's ``DatasetWriter.add_frame``
    writes every ``image``/``video``-dtype value out to a per-frame file on disk and
    leaves the episode buffer holding that file's path instead of the decoded array;
    every other feature's values stay in memory. ``compute_episode_stats`` calls this
    for every ``image``/``video`` feature, so dropping either branch here makes
    ``save_episode`` raise for any dataset that has a camera -- which is every
    embodiment in this project.

    Changed from v1: the ``ndarray`` branch also transposes a channel-last frame
    (``(H, W, 3)`` or ``(H, W, 1)``, what this project's own decoded frames look like)
    to channel-first before downsampling -- v1's ndarray branch never needed this,
    since it never received a raw decoded frame there. And a value that is neither a
    list nor an ndarray now raises ``TypeError`` instead of falling through: v1 left
    ``images`` unassigned in that case, so its own ``return images`` raised an obscure
    ``UnboundLocalError`` there (verified by calling it directly), not the silent
    ``None`` previously assumed here.

    Two more divergences: the ``list`` branch's accumulating buffer is now allocated
    with ``dtype=img.dtype``, preserving whatever ``load_image_as_numpy`` returned
    (native dtype for depth, ``uint8`` for colour), where v1 hardcoded ``dtype=np.uint8``
    for that buffer regardless of what was loaded. And the ``ndarray`` branch now guards
    on the input's number of dimensions -- ``input if input.ndim == 4 else
    input[:, None, :, :]`` -- where v1 unconditionally inserted the axis
    (``input[:, None, :, :]``), assuming a 3-D input every time.
    """
    if type(input) is list:
        image_paths = input

        sampled_indices = sample_indices(len(image_paths))
        images = None
        for i, idx in enumerate(sampled_indices):
            path = image_paths[idx]

            # we load RGB images as uint8 to reduce memory usage; depth keeps its native dtype
            img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=img.dtype)

            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input if input.ndim == 4 else input[:, None, :, :]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for position, index in enumerate(sampled_indices):
            image = frames_array[index]
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = np.transpose(image, (2, 0, 1))
            image = auto_downsample_height_width(image)
            if images is None:
                images = np.empty((len(sampled_indices), *image.shape), dtype=image.dtype)
            images[position] = image
    else:
        raise TypeError(
            f"sample_images expected a list of image paths or a decoded ndarray, "
            f"got {type(input).__name__}"
        )
    return images


def compute_episode_stats(episode_data, features: dict, quantile_list=None) -> dict:
    """Copied from ``robomind2lerobot``. Depth normalises by 1.0, colour by 255."""
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        if features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            ep_ft_array,
            axis=axes_to_reduce,
            keepdims=keepdims,
            quantile_list=quantile_list,
        )

        if features[key]["dtype"] in ["image", "video"]:
            value_norm = 1.0 if "depth" in key else 255.0
            ep_stats[key] = {
                name: value if name == "count" else np.squeeze(value / value_norm, axis=0)
                for name, value in ep_stats[key].items()
            }

    return ep_stats


# ---------------------------------------------------------------------------
# Below: copied from ``robomind2lerobot/robomind_h5.py`` (``RoboMINDDatasetMetadata``,
# ``RoboMINDDatasetWriter``, ``RoboMINDDataset``), renamed with a ``v2`` infix.
# v1 itself is untouched -- this is a copy, not an import, so v1's output stays
# reproducible. What got removed while porting (and why) is described in the
# module docstring above; ``compute_episode_stats`` below resolves to the
# function defined earlier in this module, so -- unlike v1, which imported it
# from ``robomind_uitls.lerobot_uitls`` -- no import of it is needed here.
#
# One more adaptation was required that the module docstring doesn't cover:
# two call sites pass a video codec through to the writer. v1 wrote this as a
# single ``vcodec`` string, which the pinned lerobot here no longer accepts --
# ``DatasetWriter`` now takes ``rgb_encoder``/``depth_encoder`` instead (see
# ``RoboMINDv2Dataset.create`` and the parallel-encoding branch of
# ``RoboMINDv2DatasetWriter.save_episode`` below for exactly what changed and
# why). Nothing about how much encoding configuration is threaded through
# changes -- v1 never set the old parameter either, so both sides still
# default to "let the library choose."
#
# ``RoboMINDv2DatasetMetadata.save_episode`` no longer exists as an override at all --
# it used to access the ``DatasetInfo`` dataclass by its fields instead of the deprecated
# dict-style shim, but once that was its only remaining difference from the base
# ``LeRobotDatasetMetadata.save_episode``, the override was deleted outright rather than
# kept as a no-op copy. See that class's own docstring for the reasoning and for what is
# still overridden.
#
# ``RoboMINDv2Dataset.create`` also carries forward a v1 behaviour worth flagging rather
# than silently repeating: right after the ``super().create(*args, **kwargs)`` call above
# it has already written a dataset to ``params["root"]``, it calls
# ``shutil.rmtree(params["root"], ignore_errors=True)`` and rebuilds ``obj.meta``/
# ``obj.writer`` from scratch. This deletes whatever the base ``create()`` just wrote --
# and anything else already sitting at that path -- and makes resuming an existing dataset
# there impossible. v1 did this too, so the behaviour stays unchanged, but it is a real
# hazard once a path with something worth keeping can land in ``root``: a later task wires
# a user-supplied path into this driver, at which point this line deletes whatever that
# user points it at, not just this call's own just-written output.
#
# ``RoboMINDv2DatasetWriter.save_episode``'s stacking loop is the one place control flow
# itself diverges from v1: it skips ``ft["dtype"] in ["image", "video"]``, where v1's
# equivalent loop skipped only ``["video"]``. v1's narrower filter left an ``image``-dtype
# key -- a depth feature, here -- as its raw list of on-disk path strings when it reached
# this point; ``np.stack``-ing that list, as v1's filter would have let happen, collapses
# it into a single ndarray of path *strings* before ``compute_episode_stats`` ever calls
# ``sample_images`` on it, and ``sample_images`` has no branch that can sample strings.
# Skipping ``image`` here keeps the path list intact for ``sample_images``' own list
# branch to consume, the same way it already left ``video`` keys alone for that reason.
#
# Everything else -- control flow (apart from that one stacking-loop condition), comments,
# the broad ``except Exception`` in the parallel video-encoding block -- is unchanged from v1.
# ---------------------------------------------------------------------------


class RoboMINDv2DatasetMetadata(LeRobotDatasetMetadata):
    """Copied from ``robomind_h5.py::RoboMINDDatasetMetadata``. See module docstring.

    ``save_episode`` is no longer overridden here: once the ``split``/``action_config``
    removal left it a verbatim, statement-for-statement copy of the base
    ``LeRobotDatasetMetadata.save_episode`` (the only difference was
    ``self.info.splits["train"] = ...`` vs. the base's ``self.info.splits = {"train": ...}``,
    equivalent since ``splits`` starts empty and only ``"train"`` is ever written), the
    override was deleted and the base method is now inherited unchanged.

    ``_flush_metadata_buffer`` is the one method that still genuinely differs: this port
    passes ``schema=`` into ``pa.Table.from_pydict(...)``, while the base instead calls
    ``table.select(schema.names)`` to realign a differently-ordered batch onto the writer's
    established schema. The base has therefore already solved, in its own way, the same
    schema-alignment problem this override exists for -- so a later reader should treat
    this override, too, as a candidate for removal, once someone verifies the base's
    approach behaves the same way here.
    """

    def _flush_metadata_buffer(self) -> None:
        """Write all buffered episode metadata to parquet file."""
        if not hasattr(self, "_metadata_buffer") or len(self._metadata_buffer) == 0:
            return

        combined_dict = {}
        for episode_dict in self._metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                # Extract value and serialize numpy arrays
                # because PyArrow's from_pydict function doesn't support numpy arrays
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        schema = None if not self._pq_writer else self._pq_writer.schema
        table = pa.Table.from_pydict(combined_dict, schema=schema)

        if not self._pq_writer:
            path = Path(self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()


class RoboMINDv2DatasetWriter(DatasetWriter):
    """Copied from ``robomind_h5.py::RoboMINDDatasetWriter``. See module docstring."""

    def save_episode(
        self,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        """Save the current episode in self.episode_buffer to disk."""
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self._meta.total_episodes, self._meta.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self._meta.total_frames, self._meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self._meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self._meta.get_task_index(task) for task in tasks])

        for key, ft in self._meta.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()
        has_video_keys = len(self._meta.video_keys) > 0
        use_streaming = self._streaming_encoder is not None and has_video_keys
        use_batched_encoding = self._batch_encoding_size > 1

        if use_streaming:
            non_video_buffer = {
                k: v for k, v in episode_buffer.items() if self._meta.features.get(k, {}).get("dtype") not in ("video",)
            }
            non_video_features = {k: v for k, v in self._meta.features.items() if v["dtype"] != "video"}
            ep_stats = compute_episode_stats(non_video_buffer, non_video_features)
        else:
            ep_stats = compute_episode_stats(episode_buffer, self._meta.features)

        ep_metadata = self._save_episode_data(episode_buffer)

        if use_streaming:
            streaming_results = self._streaming_encoder.finish_episode()
            for video_key in self._meta.video_keys:
                temp_path, video_stats = streaming_results[video_key]
                if video_stats is not None:
                    ep_stats[video_key] = {
                        k: v if k == "count" else np.squeeze(v.reshape(1, -1, 1, 1) / 255.0, axis=0)
                        for k, v in video_stats.items()
                    }
                ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=temp_path))
        elif has_video_keys and not use_batched_encoding:
            num_cameras = len(self._meta.video_keys)
            if parallel_encoding and num_cameras > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
                    future_to_key = {
                        executor.submit(
                            _encode_video_worker,
                            video_key,
                            episode_index,
                            self._root,
                            self._meta.fps,
                            # v1 passed a single `self._vcodec` string here. This
                            # pinned lerobot's DatasetWriter has no such attribute
                            # any more -- it holds separate `_rgb_encoder` and
                            # `_depth_encoder` configs instead -- so the choice is
                            # made the same way `DatasetWriter._encode_temporary_
                            # episode_video` (the non-parallel path, unmodified)
                            # already makes it.
                            self._depth_encoder if video_key in self._meta.depth_keys else self._rgb_encoder,
                            self._encoder_threads,
                        ): video_key
                        for video_key in self._meta.video_keys
                    }

                    results = {}
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        try:
                            temp_path = future.result()
                            results[video_key] = temp_path
                        except Exception as exc:
                            logging.error(f"Video encoding failed for {video_key}: {exc}")
                            raise exc

                for video_key in self._meta.video_keys:
                    temp_path = results[video_key]
                    ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=temp_path))
            else:
                for video_key in self._meta.video_keys:
                    ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` be executed after encoding the videos
        self._meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            self._episodes_since_last_encoding += 1
            if self._episodes_since_last_encoding == self._batch_encoding_size:
                start_ep = self._meta.total_episodes - self._batch_encoding_size
                end_ep = self._meta.total_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self._episodes_since_last_encoding = 0

        if not episode_data:
            self.clear_episode_buffer(delete_images=len(self._meta.image_keys) > 0)


class RoboMINDv2Dataset(LeRobotDataset):
    """Copied from ``robomind_h5.py::RoboMINDDataset``. See module docstring."""

    @classmethod
    def create(cls, *args, **kwargs) -> "RoboMINDv2Dataset":
        sig = inspect.signature(super().create)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        obj = super().create(*args, **kwargs)

        shutil.rmtree(params["root"], ignore_errors=True)
        obj.meta: RoboMINDv2DatasetMetadata = RoboMINDv2DatasetMetadata.create(
            repo_id=params["repo_id"],
            fps=params["fps"],
            robot_type=params["robot_type"],
            features=params["features"],
            root=params["root"],
            use_videos=params["use_videos"],
            metadata_buffer_size=params["metadata_buffer_size"],
        )
        obj.writer: RoboMINDv2DatasetWriter = RoboMINDv2DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            # v1 constructed this with a single `vcodec=obj._vcodec` -- an
            # attribute `LeRobotDataset.create` no longer sets at all in this
            # pinned lerobot. `rgb_encoder`/`depth_encoder` are its modern
            # replacement, and both are already sitting in `params` (with the
            # same `None` default v1's unset `vcodec` effectively had), pulled
            # from the exact same bound-arguments dict used for `fps`/
            # `robot_type`/etc. above.
            rgb_encoder=params["rgb_encoder"],
            depth_encoder=params["depth_encoder"],
            encoder_threads=obj._encoder_threads,
            batch_encoding_size=obj._batch_encoding_size,
        )
        return obj

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        self._require_writer("save_episode")
        self.writer.save_episode(episode_data, parallel_encoding)
