"""What the image build refuses to do.

Each refusal here is an image that was actually pushed. `parallel`, `crt`,
`by-scale`, `resize-sinc` and `foundry-cli` all went out under names in one day,
against a rule the README already stated, and two of them ended up cited in
committed records as the provenance of measurements -- so recovering which commit
they came from meant bracketing push times against the git log. A rule that is only
written down is a rule that gets skipped when a name is quicker to type.
"""

import subprocess
from pathlib import Path

import pytest

from orchestrator import image


class _Git:
    """A git that answers from a script instead of a repository."""

    def __init__(self, *, head="a" * 40, porcelain="", ancestor=True, build=0,
                 heads=None):
        self.head, self.porcelain = head, porcelain
        self.ancestor, self.build_code = ancestor, build
        # per-directory answers, so the foundry checkout is not the same tree
        self.heads = {str(k): v for k, v in (heads or {}).items()}
        self.commands: list[list[str]] = []

    def __call__(self, command, capture_output=True, text=True, cwd=None):
        self.commands.append(command)
        if command[:2] == ["git", "rev-parse"]:
            return self._done(0, self.heads.get(str(cwd), self.head))
        if command[:2] == ["git", "status"]:
            return self._done(0, self.porcelain)
        if command[:2] == ["git", "merge-base"]:
            return self._done(0 if self.ancestor else 1, "")
        if command[:2] == ["git", "fetch"]:
            return self._done(0, "")
        if command[0] == "docker":
            return self._done(self.build_code, "", "no space left on device")
        raise AssertionError(f"unexpected command: {command}")

    @staticmethod
    def _done(code, out, err=""):
        return subprocess.CompletedProcess([], code, stdout=out, stderr=err)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "foundry-cli.pin").write_text("c" * 40 + "\n")
    return tmp_path


class TestTheTagComesFromTheCommit:
    def test_it_is_the_short_revision(self, tree):
        git = _Git(head="21ecf8c668aefe4b88ef22ec957e9412c897472e")
        assert image.tag_for(tree, run=git) == "21ecf8c"

    def test_origin_main_is_fetched_before_it_is_trusted(self, tree):
        """A stale origin/main refuses a commit that is on main, which teaches the
        person building to pass --no-fetch and lose the check entirely."""
        git = _Git()
        image.tag_for(tree, run=git)
        assert ["git", "fetch", "--quiet", "origin", "main"] in git.commands

    def test_a_dirty_tree_is_refused(self, tree):
        """`COPY . .` copies the tree, so an uncommitted change is in the image and
        is not in the commit the tag names."""
        git = _Git(porcelain=" M orchestrator/image.py\n")
        with pytest.raises(image.ImageError, match="no commit describes"):
            image.tag_for(tree, run=git)

    def test_an_untracked_file_is_refused_too(self, tree):
        """It reaches the image the same way a modified one does."""
        git = _Git(porcelain="?? scratch.py\n")
        with pytest.raises(image.ImageError, match="no commit describes"):
            image.tag_for(tree, run=git)

    def test_a_commit_that_is_not_on_main_is_refused(self, tree):
        """The point of the hash is that a later reader can look it up."""
        git = _Git(ancestor=False)
        with pytest.raises(image.ImageError, match="not on origin/main"):
            image.tag_for(tree, run=git)

    def test_a_commit_main_has_moved_past_is_allowed(self, tree):
        """Ancestry, not equality: rebuilding an older commit is legitimate, and it
        is still a commit anybody can find."""
        git = _Git(head="b" * 40, ancestor=True)
        assert image.tag_for(tree, run=git) == "bbbbbbb"


@pytest.fixture
def foundry(tmp_path):
    """A checkout sitting at the revision foundry-cli.pin asks for."""
    at_the_pin = tmp_path / "foundry"
    at_the_pin.mkdir()
    return at_the_pin


class TestWhatReachesTheBuild:
    def _git(self, foundry, *, head="a" * 40, **kwargs):
        return _Git(head=head, heads={foundry: "c" * 40}, **kwargs)

    def _built(self, tree, foundry, git):
        image.build(tree, foundry, run=git)
        return next(c for c in git.commands if c[0] == "docker")

    def test_both_revisions_are_passed_in(self, tree, foundry):
        """The image records its own commit as well as the foundry CLI's. It used to
        record only the CLI's, so it could not be asked what it was."""
        built = self._built(tree, foundry, self._git(foundry, head="d" * 40))
        assert f"SOURCE_REVISION={'d' * 40}" in built
        assert f"FOUNDRY_REVISION={'c' * 40}" in built

    def test_the_tag_is_the_commit_and_not_a_name(self, tree, foundry):
        git = self._git(foundry, head="21ecf8c668aefe4b88ef22ec957e9412c897472e")
        assert f"{image.REPO}:21ecf8c" in self._built(tree, foundry, git)

    def test_it_builds_for_amd64_whatever_it_runs_on(self, tree, foundry):
        """Nodes are m7i/m8i. An arm64 build of this image cannot run on one."""
        built = self._built(tree, foundry, self._git(foundry))
        assert built[built.index("--platform") + 1] == "linux/amd64"

    def test_provenance_is_off(self, tree, foundry):
        """buildx otherwise writes an OCI index with an attestation beside the
        image, which some runtimes will not pull."""
        assert "--provenance=false" in self._built(tree, foundry, self._git(foundry))

    def test_a_foundry_checkout_off_the_pin_is_refused_before_the_build(
        self, tree, foundry
    ):
        """The build checks this too, five minutes in. Failing here costs nothing."""
        (tree / "foundry-cli.pin").write_text("e" * 40 + "\n")
        git = self._git(foundry)
        with pytest.raises(image.ImageError, match="foundry-cli.pin asks for"):
            image.build(tree, foundry, run=git)
        assert not [c for c in git.commands if c[0] == "docker"]

    def test_a_failed_build_is_an_error_and_says_why(self, tree, foundry):
        git = self._git(foundry, build=1)
        with pytest.raises(image.ImageError, match="no space left on device"):
            image.build(tree, foundry, run=git)


class TestTheDockerfileRecordsIt:
    def test_the_build_argument_is_declared_and_written(self):
        """A build arg no stage declares is silently dropped, which would leave the
        image quietly saying `unrecorded`."""
        text = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
        assert "ARG SOURCE_REVISION" in text
        assert "/opt/any4lerobot/REVISION" in text
