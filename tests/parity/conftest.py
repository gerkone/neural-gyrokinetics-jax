"""Parity tests against the torch reference; skipped unless torch and ``neugk`` import.

``NEUGK_TORCH_REPO`` points at the torch repository (default: the parent of this repo).
"""

import importlib.util
import os
import sys
from pathlib import Path

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
_REPO = os.environ.get("NEUGK_TORCH_REPO", str(Path(__file__).resolve().parents[3]))
if _REPO not in sys.path:
    sys.path.append(_REPO)
