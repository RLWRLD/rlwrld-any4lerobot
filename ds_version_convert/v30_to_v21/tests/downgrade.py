"""Import the downgrade script, which lives next to this package as a script.

It is run by path in production -- ``lerobot_pipeline`` builds a command line for
it -- so there is no package to import it from. The tests load it the same way
the pipeline runs it: by file.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "convert_dataset_v30_to_v21.py"

_spec = importlib.util.spec_from_file_location("convert_dataset_v30_to_v21", _SCRIPT)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)
