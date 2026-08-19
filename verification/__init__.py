"""Checking a rebuilt dataset against the delivered copy it reproduces.

The one thing here that is not conversion work. Everything else in this repository
exists to *produce* a dataset -- the converters, `lerobot_pipeline`, the orchestrator,
`dataset_registry` -- and this exists to ask whether what came out is the same as what
was delivered.

Which is why it is its own directory rather than a module inside one of theirs. It
reads `dataset_registry` to know what a dataset is and imports nothing else from this
repository: **what is being measured must not be able to change what measures it.** A
comparison that shared a resizer, an encoder profile or a stats routine with the
converter would agree with it by construction.
"""
