"""How far each dataset has got, and what may be deleted as a result.

One file per (dataset, step), not one per dataset. That is the whole reason no lock
is needed anywhere: ``build`` is the only writer of ``build.json``, ``publish`` the
only writer of ``publish.json``, and a reader that arrives mid-write sees either the
old file or the new one because the write is a rename.

Deletion is the reason this module exists at all. A step records the paths it
created, and those are the only paths anything will ever delete -- so a source
directory staged by hand, which no step created, cannot be deleted by any rule we
later get wrong. Every uncertainty resolves the same way: a record that cannot be
read counts as "not done", which redoes work (idempotent) rather than deleting it
(not idempotent).
"""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEPS = ("fetch", "build", "publish")

# what has to succeed before a step's output is no longer needed
_NEXT = {"fetch": "build", "build": "publish"}

_STATUSES = ("ok", "failed")


class StepError(ValueError):
    """Raised for a record that could never be written back out."""


def now() -> str:
    # timezone.utc rather than datetime.UTC: the repo's documented test command
    # runs on 3.10, where the shorter spelling does not exist
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Record:
    step: str
    dataset: str
    status: str
    started: str | None = None
    finished: str | None = None
    # what the step depended on, so a later run can tell whether it still holds
    spec_sha: str | None = None
    profile: str | None = None
    source_uri: str | None = None
    dest_uri: str | None = None
    # the only paths anything will ever delete
    created: tuple[str, ...] = ()
    bytes: int | None = None
    error: str | None = None


class Steps:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def path(self, dataset: str, step: str) -> Path:
        return self.root / dataset / f"{step}.json"

    # -- reading --------------------------------------------------------------

    def read(self, dataset: str, step: str) -> Record | None:
        """The record, or ``None`` if there is not a readable one.

        A malformed file is ``None`` rather than an exception: the callers use this
        to decide whether to redo work and whether to delete, and "redo, delete
        nothing" is the safe answer to "I cannot tell".
        """
        path = self.path(dataset, step)
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, Mapping):
            return None
        known = {f.name for f in fields(Record)}
        try:
            return Record(
                **{
                    key: tuple(value or ()) if key == "created" else value
                    for key, value in raw.items()
                    if key in known
                }
            )
        except TypeError:
            return None

    def done(self, dataset: str, step: str, **expected: Any) -> bool:
        """Whether ``step`` succeeded, what it made is still there, and its inputs hold.

        The caller names the inputs that matter: ``fetch`` cares that the mirror is
        the same one, ``build`` that the spec and profile are unchanged. A dataset
        spec that has been edited since makes its build stale without anyone having
        to remember to clear anything.

        The output is checked too, because a record outliving what it describes is
        the normal case here rather than an odd one: ``reclaim`` deletes a source as
        soon as its build succeeds and leaves the fetch record saying ok. A later run
        that needs that source again would be told it was already there and would
        convert nothing. A step that claims no paths -- a hand-staged source, which
        the pipeline did not create -- has nothing to have gone missing and stays
        done.
        """
        record = self.read(dataset, step)
        if record is None or record.status != "ok":
            return False
        if not all(Path(path).exists() for path in record.created):
            return False
        return all(getattr(record, key) == value for key, value in expected.items())

    def reclaimable(self, dataset: str, step: str) -> tuple[str, ...]:
        """Paths ``step`` created that are no longer needed.

        Empty unless the step that consumes them has succeeded. Empty for the last
        step, which nothing consumes. Empty when the step created nothing, which is
        what a hand-staged source looks like.
        """
        following = _NEXT.get(step)
        if following is None or not self.done(dataset, following):
            return ()
        record = self.read(dataset, step)
        if record is None or record.status != "ok":
            return ()
        return tuple(record.created)

    def datasets(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(path.name for path in self.root.iterdir() if path.is_dir())

    # -- writing --------------------------------------------------------------

    def write(self, record: Record) -> None:
        """Replace the record for one (dataset, step), atomically."""
        if record.step not in STEPS:
            raise StepError(
                f"unknown step {record.step!r}; expected one of {', '.join(STEPS)}"
            )
        if record.status not in _STATUSES:
            raise StepError(
                f"unknown status {record.status!r}; expected one of "
                f"{', '.join(_STATUSES)}"
            )
        path = self.path(record.dataset, record.step)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(asdict(record))
        payload["created"] = list(record.created)

        # written beside the target so the rename stays on one filesystem, and
        # removed on failure so a reader never sees a half-written file
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
