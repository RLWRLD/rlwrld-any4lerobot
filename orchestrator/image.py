"""Building the node image, and refusing to build one nobody can trace back.

The rule was already written down -- "tag with the commit, not ``latest``" in
``bootstrap/README.md`` -- and it was broken five times in one day. The images
``parallel``, ``crt``, ``by-scale``, ``resize-sinc`` and ``foundry-cli`` were pushed
under names because a name was quicker to type than a check, and two of them ended up
cited in committed records as the provenance of measurements. Recovering which commit
those had been built from took bracketing their push times against the git log.

So the rule is a function now. A tag is derived, never chosen:

  * from the commit, short-form, because that is what a later reader has to be able
    to look up;
  * only from a commit that is on ``origin/main``, because a hash that exists on one
    laptop answers nothing;
  * only from a clean tree, because a dirty tree is not any commit at all.

The image also records its own source revision, at ``/opt/any4lerobot/REVISION``. It
did not before -- it recorded the foundry CLI's revision and nothing about itself --
which is why an image could not be asked what it was and the whole weight of that
answer sat on the tag. With the revision inside, a mistagged image is still
identifiable.
"""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = ("487592470682.dkr.ecr.us-east-1.amazonaws.com"
        "/rlwrld/inhouse-services/any4lerobot/node")

# Nodes are m7i/m8i, so the image is amd64 whatever it is built on. On an arm64
# machine that means an emulated build: correct, and slow enough that the practical
# answer is to build on a node instead.
PLATFORM = "linux/amd64"

# Without this, buildx writes an OCI index with an attestation beside the image, and
# some runtimes will not pull that.
PROVENANCE = "--provenance=false"


class ImageError(RuntimeError):
    """Raised when the image that would be built could not be traced to main."""


def _run(command: Sequence[str], run=None, cwd: Path | None = None):
    runner = run or subprocess.run
    return runner(list(command), capture_output=True, text=True, cwd=cwd)


def _git(args: Sequence[str], *, run=None, cwd: Path | None = None) -> str:
    completed = _run(["git", *args], run, cwd)
    if completed.returncode != 0:
        raise ImageError(
            f"git {' '.join(args)} failed: "
            f"{(completed.stderr or '').strip() or completed.returncode}"
        )
    return (completed.stdout or "").strip()


def head_revision(*, run=None, cwd: Path | None = None) -> str:
    return _git(["rev-parse", "HEAD"], run=run, cwd=cwd)


def is_clean(*, run=None, cwd: Path | None = None) -> bool:
    """Whether the tree holds nothing the commit does not.

    Untracked files count. ``COPY . .`` takes the working tree, so an untracked file
    is in the image and is not in the commit the tag names -- which is the failure
    this is for, not a tidiness preference.
    """
    return _git(["status", "--porcelain"], run=run, cwd=cwd) == ""


def on_main(revision: str, *, run=None, cwd: Path | None = None) -> bool:
    """Whether ``revision`` is reachable from ``origin/main``.

    Ancestry rather than equality, so an image can be built from a commit that main
    has already moved past -- rebuilding an older commit is a legitimate thing to
    want, and it is still a commit anyone can find.
    """
    completed = _run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/main"], run, cwd
    )
    return completed.returncode == 0


def foundry_revision(source: Path, *, run=None) -> str:
    """The revision of the foundry checkout that would go into the image."""
    return head_revision(run=run, cwd=source)


def pinned_foundry(root: Path) -> str:
    return (root / "foundry-cli.pin").read_text().strip()


def tag_for(root: Path, *, run=None, fetch: bool = True) -> str:
    """The tag this tree is allowed to push under, or a refusal saying why not.

    Every branch here is an image that was pushed at some point and should not have
    been.
    """
    if fetch:
        # A stale origin/main would refuse a commit that is on main, so the check
        # reads a fetched ref rather than whatever was last seen.
        _run(["git", "fetch", "--quiet", "origin", "main"], run, root)

    if not is_clean(run=run, cwd=root):
        raise ImageError(
            "the working tree has changes, so no commit describes what would be "
            "built. `COPY . .` copies the tree, not the commit, so the tag would "
            "name something the image does not contain. Commit first."
        )

    revision = head_revision(run=run, cwd=root)
    if not on_main(revision, run=run, cwd=root):
        raise ImageError(
            f"{revision[:7]} is not on origin/main. A tag is only an answer if a "
            "later reader can look it up, and a hash that exists on one machine "
            "cannot be looked up. Merge it first, or build the commit you merged."
        )
    return revision[:7]


def build_command(
    root: Path, foundry: Path, tag: str, revision: str, foundry_rev: str,
    *, repo: str = REPO,
) -> list[str]:
    """``docker buildx build``, with both revisions passed in as build arguments.

    The foundry one is checked against ``foundry-cli.pin`` inside the build, which is
    the right place for it: which CLI an image carries should be a reviewed fact.
    """
    return [
        "docker", "buildx", "build",
        "--platform", PLATFORM,
        PROVENANCE,
        "--build-context", f"foundry={foundry}",
        "--build-arg", f"FOUNDRY_REVISION={foundry_rev}",
        "--build-arg", f"SOURCE_REVISION={revision}",
        "-t", f"{repo}:{tag}",
        "--push", str(root),
    ]


def build(
    root: Path, foundry: Path, *, repo: str = REPO, run=None, fetch: bool = True
) -> str:
    """Build and push the image for this tree, returning the tag it went out under."""
    tag = tag_for(root, run=run, fetch=fetch)
    revision = head_revision(run=run, cwd=root)

    pinned, present = pinned_foundry(root), foundry_revision(foundry, run=run)
    if pinned != present:
        raise ImageError(
            f"the foundry checkout is at {present[:7]} and foundry-cli.pin asks for "
            f"{pinned[:7]}. The build would refuse this too, five minutes later."
        )

    completed = _run(
        build_command(root, foundry, tag, revision, pinned, repo=repo), run
    )
    if completed.returncode != 0:
        raise ImageError(
            f"the build failed: {(completed.stderr or '').strip()[-2000:]}"
        )
    return tag


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="the any4lerobot tree to build (default: cwd)")
    parser.add_argument("--foundry", type=Path, required=True,
                        help="a rlwrld-foundry checkout at the revision in "
                             "foundry-cli.pin")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--no-fetch", action="store_true",
                        help="trust the origin/main already on disk")
    parser.add_argument("--tag-only", action="store_true",
                        help="print the tag this tree would push under, and stop")
    args = parser.parse_args(argv)

    try:
        if args.tag_only:
            print(tag_for(args.root, fetch=not args.no_fetch))
            return 0
        tag = build(args.root, args.foundry, repo=args.repo,
                    fetch=not args.no_fetch)
    except ImageError as exc:
        print(f"image: {exc}", file=sys.stderr)
        return 2
    print(f"{args.repo}:{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
