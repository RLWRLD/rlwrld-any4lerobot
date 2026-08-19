"""Checking a rebuilt dataset against the delivered copy it reproduces.

Separate from the converters, and from ``dataset_registry`` which describes the
datasets: this asks one question those cannot, which is whether a rebuild came out
the same. It reads a spec to know what a dataset is and then never touches the
conversion path -- what is being verified must not be able to change what verifies it.

Nothing under any4lerobot's own directories imports this.
"""
