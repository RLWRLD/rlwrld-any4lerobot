"""What verification is allowed to import.

The point of the directory is that it does not share code with what it measures. A
comparison that used the converter's resizer, its encoder profile or its stats routine
would agree with it by construction, and would go on agreeing after both went wrong
together. That property is invisible in a passing test run, so it is asserted here
rather than left to a reviewer noticing an import.
"""

import ast
import sys
from pathlib import Path

VERIFICATION = Path(__file__).resolve().parents[1]
REPO = VERIFICATION.parent

sys.path.insert(0, str(REPO))

# The registry says what a dataset is -- which columns exist, how wide the vectors
# are, where the delivered copy lives. Reading it is not sharing conversion code with
# the converter: it is data, and the same data both sides are held to.
ALLOWED = {"dataset_registry", "verification"}


def repo_packages() -> set[str]:
    """Every importable top-level directory in the repository."""
    return {
        path.name
        for path in REPO.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__")) and (
            (path / "__init__.py").exists() or any(path.glob("*.py"))
        )
    }


def imported_packages(source: Path) -> set[str]:
    """Top-level names ``source`` imports, including inside functions."""
    tree = ast.parse(source.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_verification_imports_no_conversion_code():
    siblings = repo_packages() - ALLOWED
    assert siblings, "the repository should have other packages, or this proves nothing"
    for source in sorted(VERIFICATION.rglob("*.py")):
        if "tests" in source.parts:
            continue
        overlap = imported_packages(source) & siblings
        assert not overlap, (
            f"{source.relative_to(REPO)} imports {sorted(overlap)}; verification must "
            "not share code with what it measures"
        )


def test_no_conversion_code_imports_verification():
    """The other direction, which would be the worse one: a converter that reached
    into this directory would make the comparison part of the thing compared."""
    for package in sorted(repo_packages() - {"verification"}):
        for source in sorted((REPO / package).rglob("*.py")):
            assert "verification" not in imported_packages(source), (
                f"{source.relative_to(REPO)} imports verification"
            )
