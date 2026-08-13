"""One HDF5 file plus one already-encoded video directory per episode.

The shape ActionNet ships, and the shape several other HDF5 sources ship with
different key names -- which is exactly what the spec supplies. Nothing here names
a dataset.

Layout, as described by ``source:`` in a dataset spec::

    <episode_id>.hdf5                 robot side: the datasets under source_features
    <episode_id>/<camera>/rgb.mp4     camera side, already encoded
    <episode_id>/<camera>/timestamps.json
    metadata.json                     episode id -> prompt

The mp4 is carried over rather than decoded and re-encoded. Any resizing is a later
pipeline step, so a source whose video is already the right size costs one hard link.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..clocks import ClockError, align, parse_times
from . import EpisodeSkipped, FormatError, register_format


@dataclass(frozen=True)
class Episode:
    episode_id: str
    frames: list[dict[str, Any]]
    videos: dict[str, Path]


@register_format("hdf5_episodes")
class Hdf5EpisodeReader:
    def __init__(self, spec, root: Path):
        self.spec = spec
        self.root = Path(root).expanduser().resolve()
        self.source = spec.source
        # lerobot video key -> source-side directory, e.g. observation.images.primary
        # -> "top". One camera carries the clock; the rest follow it.
        self.cameras = {
            f"observation.images.{key}": value for key, value in spec.cameras.items()
        }
        if not self.cameras:
            raise FormatError(f"{spec.id}: lerobot.video.cameras is empty")
        self.primary_camera = next(iter(spec.cameras.values()))

    # -- discovery ------------------------------------------------------------

    def episode_ids(self) -> list[str]:
        """Episode ids on disk, sorted. ActionNet ids are ULIDs, so this is also
        chronological; for other sources it is merely stable."""
        return sorted(path.stem for path in self.root.glob(self.source.discover))

    def prompts(self) -> dict[str, str]:
        tasks = self.source.tasks
        path = self.root / tasks["file"]
        if not path.exists():
            raise FormatError(f"{path} not found; it holds the prompt for every episode")
        entries = json.loads(path.read_text())
        return {entry[tasks["key"]]: entry.get(tasks["prompt"], "") for entry in entries}

    # -- one episode ----------------------------------------------------------

    def _path(self, template: str, episode_id: str, camera: str | None = None) -> Path:
        return self.root / template.format(id=episode_id, camera=camera or "")

    def read_episode(self, episode_id: str, prompt: str) -> Episode:
        import h5py

        paths = self.source.paths
        hdf5_path = self._path(paths["episode"], episode_id)
        videos = {
            key: self._path(paths["video"], episode_id, camera)
            for key, camera in self.cameras.items()
        }

        missing = [
            str(path.relative_to(self.root))
            for path in (hdf5_path, *videos.values())
            if not path.exists()
        ]
        if missing:
            raise EpisodeSkipped(f"missing {', '.join(missing)}")

        image_times = self._image_times(episode_id)
        columns, data_times = self._read_hdf5(h5py, hdf5_path)

        try:
            matched = align(self.source.strategy, data_times, image_times)
        except ClockError as exc:
            raise EpisodeSkipped(str(exc)) from exc
        if matched.size == 0:
            raise EpisodeSkipped("no frames survived clock alignment")

        frames = [
            {name: values[index] for name, values in columns.items()} | {"task": prompt}
            for index in matched
        ]
        return Episode(episode_id=episode_id, frames=frames, videos=videos)

    def _image_times(self, episode_id: str) -> np.ndarray:
        clock = self.source.clock
        path = self._path(clock["image"], episode_id, self.primary_camera)
        if not path.exists():
            raise EpisodeSkipped(f"missing {path.name}")
        try:
            return parse_times(
                json.loads(path.read_text()), clock.get("image_format")
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise EpisodeSkipped(f"unreadable {path.name}: {exc}") from exc

    def _read_hdf5(self, h5py, path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Every source feature the layout refers to, as a LeRobot column name.

        A feature the spec maps to ``state/robot`` on the state side and
        ``action/robot`` on the action side becomes two columns,
        ``observation.robot`` and ``action.robot``. Those names are what the layout
        step reads later, and what the delivered datasets carry alongside the
        assembled vectors.
        """
        wanted = self._wanted_columns()
        columns: dict[str, np.ndarray] = {}
        features: dict[str, str] = {}
        with h5py.File(path, "r") as handle:
            try:
                data_times = np.asarray(handle[self.source.clock["data"]], np.float64)
                for (column_name, feature), dataset_path in wanted.items():
                    columns[column_name] = np.asarray(handle[dataset_path], np.float32)
                    features[column_name] = feature
            except KeyError as exc:
                raise EpisodeSkipped(f"hdf5 missing dataset {exc}") from exc

        self._check_widths(columns, features)
        return columns, data_times

    def _wanted_columns(self) -> dict[tuple[str, str], str]:
        """``(emitted LeRobot column, logical feature name)`` -> path in the hdf5.

        Two namespaces meet here. ``lerobot.state.source_features`` names the columns
        to *emit*; ``source.features`` says where to *read* them. Keeping them apart
        is what lets the same layout describe a dataset arriving as raw hdf5 and one
        arriving already converted.
        """
        raw_paths = self.source.features
        wanted: dict[tuple[str, str], str] = {}
        for side in ("state", "action"):
            vector = self.spec.vector(side)
            if vector is None:
                continue
            for name, sides in vector.source_features.items():
                if name not in raw_paths:
                    raise FormatError(
                        f"{self.spec.id}: source.features has no entry for {name!r}, "
                        "so there is nowhere to read it from"
                    )
                wanted[(sides[side], name)] = raw_paths[name][side]
        return wanted

    def _check_widths(
        self, columns: dict[str, np.ndarray], features: dict[str, str]
    ) -> None:
        """Reject an episode whose arrays are not the shape the spec describes.

        Two checks, and they catch different mistakes. ``feature_widths`` is a fact
        about the source, so a mismatch means this file is from a different robot --
        a GR2 reports 29 joints where a GR1 reports 32, and its columns mean other
        things. The derived minimum catches a spec that reads past the end of an
        array it did describe correctly.
        """
        declared = self.source.feature_widths
        needed = self._minimum_widths()
        for column_name, values in columns.items():
            if values.ndim != 2:
                raise EpisodeSkipped(f"{column_name} is {values.shape}, expected 2-D")
            got = values.shape[1]
            feature = features[column_name]
            want = declared.get(feature)
            if want is not None and got != want:
                raise EpisodeSkipped(
                    f"{column_name} is {values.shape}, but the spec says {feature} is "
                    f"{want} wide. A robot variant with a different joint count needs "
                    "its own spec"
                )
            if got < needed.get(column_name, 0):
                raise EpisodeSkipped(
                    f"{column_name} is {values.shape}, but the layout reads up to "
                    f"column {needed[column_name] - 1}"
                )

    def _minimum_widths(self) -> dict[str, int]:
        """The highest source column any block reads, per emitted column name."""
        widths: dict[str, int] = {}
        for side in ("state", "action"):
            vector = self.spec.vector(side)
            if vector is None:
                continue
            for block in vector.blocks:
                if block.feature is None:
                    continue
                column_name = vector.source_features[block.feature][side]
                widths[column_name] = max(widths.get(column_name, 0), block.src_end)
        return widths
