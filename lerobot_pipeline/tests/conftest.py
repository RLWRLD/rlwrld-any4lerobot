"""Make an ffmpeg binary available to the integration tests.

Uses the system ffmpeg when present. Otherwise falls back to the binary bundled
with ``imageio-ffmpeg`` if that package is installed, so the video path can be
verified for real on machines without a system ffmpeg. Tests that need it are
skipped when neither is available.
"""

import os
import shutil
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
    bindir = Path(tempfile.mkdtemp(prefix="lerobot-pipeline-ffmpeg-"))
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
