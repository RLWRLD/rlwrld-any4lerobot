import sys
from pathlib import Path

# robomind_v2_h5.py and its helpers are run as a script from this directory, so
# that is where their imports resolve from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
