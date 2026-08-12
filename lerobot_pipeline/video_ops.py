"""ffmpeg command construction, probing and parallelism planning.

Everything that can be a pure function is one, so the parts that decide *what*
work happens are testable without ffmpeg installed. Only :func:`probe_video` and
:func:`run_ffmpeg` actually shell out.
"""

import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# x264's frame-level threading gains little past this point, and every extra
# thread costs synchronisation. Beyond it we would rather run another file.
MAX_THREADS_PER_FFMPEG = 16

FFMPEG_TIMEOUT_S = 60 * 60


@dataclass(frozen=True)
class EncodingParams:
    """Output encoding settings. Mirrored from the source so re-encoding does not
    silently change properties the training data loader depends on."""

    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 18
    gop: int = 2
    pix_fmt: str = "yuv420p"


@dataclass(frozen=True)
class Parallelism:
    workers: int
    threads: int


@dataclass(frozen=True)
class VideoInfo:
    height: int
    width: int
    frames: int | None
    fps: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def plan_parallelism(
    file_count: int,
    cores: int,
    workers: int | None = None,
    threads_per_ffmpeg: int | None = None,
) -> Parallelism:
    """Decide how to split ``cores`` between concurrent ffmpeg processes.

    Many small files (v2.1, one video per episode) want one thread each so that
    every core runs an independent file. Few large files (v3.0, videos
    concatenated into chunks) would leave cores idle that way, so each ffmpeg
    gets several threads instead.

    Measured on 800x1280 -> 192x288 (see the package README): file-level
    parallelism is what matters. Oversubscribing threads did *not* hurt at this
    output size -- x264 cannot saturate them, so the spare threads idle rather
    than contend. Pinning ``-threads`` is therefore about reproducible runs, not
    throughput, and the plan this returns lands within ~6% of the best measured
    setting.
    """
    cores = max(1, cores)

    threads = threads_per_ffmpeg
    if threads is None:
        if file_count <= 0:
            threads = 1
        else:
            threads = min(max(cores // file_count, 1), MAX_THREADS_PER_FFMPEG)

    if workers is None:
        workers = max(cores // threads, 1)
        if file_count > 0:
            workers = min(workers, file_count)

    return Parallelism(workers=max(1, workers), threads=max(1, threads))


def order_by_size_desc(paths: Iterable[Path]) -> list[Path]:
    """Longest-processing-time-first: hand out the biggest files while every
    worker is still free, so the run does not end waiting on one straggler."""
    return sorted(paths, key=lambda p: -Path(p).stat().st_size)


def build_ffmpeg_command(
    src: str | Path,
    dst: str | Path,
    filters: Sequence[str],
    encoding: EncodingParams,
    threads: int,
) -> list[str]:
    """Build a single-pass decode -> filter -> encode command."""
    if not filters:
        raise ValueError(
            "refusing to build an ffmpeg command with an empty filter chain; "
            "a no-op should be hard-linked instead of re-encoded"
        )

    return [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        # before -i: decode-side threads
        "-threads",
        str(threads),
        "-i",
        str(src),
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        encoding.codec,
        "-preset",
        encoding.preset,
        "-crf",
        str(encoding.crf),
        "-g",
        str(encoding.gop),
        # keep the keyframe interval deterministic; scene-cut detection would
        # otherwise insert extra keyframes and make -g meaningless
        "-sc_threshold",
        "0",
        "-pix_fmt",
        encoding.pix_fmt,
        # after the input: encode-side threads
        "-threads",
        str(threads),
        str(dst),
    ]


def parse_ffprobe_video_stream(payload: str) -> VideoInfo:
    """Parse ``ffprobe -show_streams -of json`` output into a :class:`VideoInfo`."""
    try:
        streams = json.loads(payload).get("streams", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse ffprobe output as JSON: {exc}") from exc

    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        return VideoInfo(
            height=int(stream["height"]),
            width=int(stream["width"]),
            frames=_optional_int(stream.get("nb_frames")),
            fps=_parse_rate(stream.get("avg_frame_rate")),
        )

    raise ValueError("ffprobe reported no video stream")


def probe_video(path: str | Path) -> VideoInfo:
    _require_binary("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=FFMPEG_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return parse_ffprobe_video_stream(result.stdout)


def run_ffmpeg(command: Sequence[str]) -> None:
    _require_binary("ffmpeg")
    result = subprocess.run(
        list(command), capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}"
        )


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} not found on PATH; it is required for video preprocessing"
        )


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _optional_float(value)
    numerator, _, denominator = value.partition("/")
    try:
        den = float(denominator)
        return float(numerator) / den if den else None
    except ValueError:
        return None


def _optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
