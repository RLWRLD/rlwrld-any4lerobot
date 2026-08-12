"""LeRobot dataset classes for sources that already ship encoded video.

Most raw robot datasets store one mp4 per camera per episode. LeRobot's default
writer expects individual frames and re-encodes them, which for these sources
means decoding a video only to encode it again. The classes here keep the source
file: ``save_episode`` takes a ``videos`` mapping of ``video_key -> Path`` and the
encode hook copies that file instead of rendering frames.

Adapters that need to *change* the video (resize, trim, re-time) should hand in a
path to an already-transformed file rather than subclassing further.
"""

import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.feature_utils import (
    get_hf_features_from_features,
    validate_episode_buffer,
    validate_frame,
)
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH


class PrerenderedMetadata(LeRobotDatasetMetadata):
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
                combined_dict[key].append(
                    val.tolist() if isinstance(val, np.ndarray) else val
                )

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        schema = None if not self._pq_writer else self._pq_writer.schema
        table = pa.Table.from_pydict(combined_dict, schema=schema)

        if not self._pq_writer:
            path = Path(
                self.root
                / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()


class PrerenderedWriter(DatasetWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_features = get_hf_features_from_features(self._meta.features)
        self.current_videos: dict[str, Path] = {}

    def add_frame(self, frame: dict) -> None:
        """Buffer one frame. Video keys are supplied per episode, not per frame."""
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {
            key: value
            for key, value in self._meta.features.items()
            if key in self.hf_features
        }  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self._create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(frame_index / self._meta.fps)
        self.episode_buffer["task"].append(frame.pop("task"))

        for key, value in frame.items():
            if key not in self._meta.features:
                raise ValueError(
                    f"An element of the frame is not in the features. "
                    f"'{key}' not in '{self._meta.features.keys()}'."
                )
            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(
        self,
        videos: dict[str, Path],
        episode_data: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        """Save the buffered episode, taking its videos from ``videos``."""
        episode_buffer = (
            episode_data if episode_data is not None else self.episode_buffer
        )

        validate_episode_buffer(
            episode_buffer, self._meta.total_episodes, self._meta.features
        )

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self._meta.total_frames, self._meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self._meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array(
            [self._meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self._meta.features.items():
            # index, episode_index, task_index are already processed above, and
            # video keys are handled by copying the source file below
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in [
                "video"
            ]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self._meta.video_keys:
            episode_buffer[key] = str(videos[key])

        ep_stats = compute_episode_stats(episode_buffer, self._meta.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self._meta.video_keys) > 0
        use_batched_encoding = self._batch_encoding_size > 1

        self.current_videos = videos
        if has_video_keys and not use_batched_encoding:
            for video_key in self._meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        if extra_metadata:
            ep_metadata.update(extra_metadata)
        self._meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, ep_metadata
        )

        if has_video_keys and use_batched_encoding:
            self._episodes_since_last_encoding += 1
            if self._episodes_since_last_encoding == self._batch_encoding_size:
                start_ep = self._meta.total_episodes - self._batch_encoding_size
                end_ep = self._meta.total_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self._episodes_since_last_encoding = 0

        if not episode_data:
            self.clear_episode_buffer(delete_images=len(self._meta.image_keys) > 0)

    def _encode_temporary_episode_video(
        self, video_key: str, episode_index: int
    ) -> Path:
        """Take the source file as-is instead of encoding frames into a new one."""
        temp_path = (
            Path(tempfile.mkdtemp(dir=self._root))
            / f"{video_key}_{episode_index:03d}.mp4"
        )
        shutil.copy(self.current_videos[video_key], temp_path)
        return temp_path


class PrerenderedDataset(LeRobotDataset):
    @classmethod
    def create(cls, *args, **kwargs) -> "PrerenderedDataset":
        sig = inspect.signature(super().create)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        obj = super().create(*args, **kwargs)

        shutil.rmtree(params["root"], ignore_errors=True)
        obj.meta = PrerenderedMetadata.create(
            repo_id=params["repo_id"],
            fps=params["fps"],
            robot_type=params["robot_type"],
            features=params["features"],
            root=params["root"],
            use_videos=params["use_videos"],
            metadata_buffer_size=params["metadata_buffer_size"],
            video_files_size_in_mb=params["video_files_size_in_mb"],
            data_files_size_in_mb=params["data_files_size_in_mb"],
        )
        obj.writer = PrerenderedWriter(
            meta=obj.meta,
            root=obj.root,
            camera_encoder=obj.writer._camera_encoder,
            encoder_threads=obj.writer._encoder_threads,
            batch_encoding_size=obj.writer._batch_encoding_size,
            streaming_encoder=obj.writer._streaming_encoder,
        )
        return obj

    def save_episode(
        self,
        videos: dict[str, Path],
        episode_data: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        self._require_writer("save_episode")
        self.writer.save_episode(videos, episode_data, extra_metadata)


def sample_images(video_path: str | np.ndarray) -> np.ndarray:
    """Decode a spread of frames from a video for per-episode statistics.

    Only the sampled frames are decoded. Pulling the whole video into an array
    first costs a gigabyte per 800x1280 episode and, multiplied by one worker per
    core, is what runs a large box out of memory.
    """
    if isinstance(video_path, np.ndarray):
        frames_array = video_path[:, None, :, :]  # Shape: [T, C, H, W]
        sampled = sample_indices(len(frames_array))
        decoded = frames_array[list(sampled)]
    else:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path)
        sampled = sample_indices(len(decoder))
        decoded = decoder.get_frames_at(list(sampled)).data.numpy()  # [T, C, H, W]

    images = None
    for i, frame in enumerate(decoded):
        img = auto_downsample_height_width(frame)
        if images is None:
            images = np.empty((len(decoded), *img.shape), dtype=np.uint8)
        images[i] = img
    return images


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
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
                k: v if k == "count" else np.squeeze(v / value_norm, axis=0)
                for k, v in ep_stats[key].items()
            }

    return ep_stats
