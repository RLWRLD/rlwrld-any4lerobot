"""Check a config's ``source.args`` against the converter that will receive them.

``args`` is passed straight through to a converter's command line, so a key that
converter does not accept is a run that dies at argument parsing -- after the
machine is provisioned and the source is staged, which is the expensive moment to
find a typo.

The flags are read out of each converter's own ``argparse`` calls rather than being
listed here, because a list here would be a second copy that drifts. Reading is done
with ``ast`` and not by importing: several converters import tensorflow or h5py at
module scope, and validating a config should not require the whole conversion
environment to be installed.
"""

import ast
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Flags the pipeline supplies itself; a config must not also set them
RESERVED = {"src_path", "output_path", "raw_dir", "local_dir"}


class ArgumentError(ValueError):
    """Raised when a config passes something a converter will reject."""


@lru_cache(maxsize=None)
def converter_flags(script: str) -> frozenset[str] | None:
    """Every ``--flag`` the converter accepts, as python identifiers.

    ``None`` means the flags could not be read, in which case validation is skipped
    rather than guessed at.
    """
    path = REPO_ROOT / script
    if not path.suffix:  # a package, e.g. spec2lerobot
        path = REPO_ROOT / script / "__main__.py"
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None

    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                flags.add(str(arg.value)[2:].replace("-", "_"))
    return frozenset(flags) or None


def check(script: str, args, source_type: str) -> list[str]:
    """Problems with these args, as messages; empty means they will be accepted."""
    problems = []
    for key in args:
        if key in RESERVED:
            problems.append(
                f"source.args.{key} is supplied by the pipeline itself; remove it"
            )

    flags = converter_flags(script)
    if flags is None:
        return problems

    for key in args:
        if key in RESERVED or key in flags:
            continue
        suggestion = _closest(key, flags)
        problems.append(
            f"source.args.{key} is not a flag of the {source_type} converter"
            + (f"; did you mean {suggestion}?" if suggestion else "")
        )
    return problems


def _closest(key: str, flags) -> str | None:
    import difflib

    matches = difflib.get_close_matches(key, sorted(flags), n=1, cutoff=0.7)
    return matches[0] if matches else None
