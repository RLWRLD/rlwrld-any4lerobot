"""Real ffmpeg, and a v3.0 video file to cut up, for the tests that need both.

Same fallback as ``lerobot_pipeline/tests/conftest.py``: the system binary when
there is one, otherwise the one bundled with ``imageio-ffmpeg``, otherwise the
tests that need it are skipped. Not a ``conftest.py``, because none of this is a
fixture and two test packages called ``tests`` cannot both have one.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _ensure_ffmpeg_on_path() -> bool:
    if shutil.which("ffmpeg"):
        return True

    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bindir = Path(tempfile.mkdtemp(prefix="v30-to-v21-ffmpeg-"))
    link = bindir / "ffmpeg"
    try:
        link.symlink_to(executable)
    except OSError:
        shutil.copy2(executable, link)
        link.chmod(0o755)

    os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    return shutil.which("ffmpeg") is not None


HAVE_FFMPEG = _ensure_ffmpeg_on_path()

requires_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG,
    reason="ffmpeg not available (install it, or `pip install imageio-ffmpeg`)",
)

FPS = 10


def write_concatenated_video(path: Path, seconds: float) -> None:
    """One camera's v3.0 file: every episode's frames, end to end.

    Keyframed every frame, because the downgrade cuts it with ``-c copy`` and a
    stream copy can only start on a keyframe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size=64x64:rate={FPS}:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
