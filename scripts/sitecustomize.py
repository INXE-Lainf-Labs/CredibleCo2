"""Ensure repository-root imports work for scripts executed by file path.

Python imports ``sitecustomize`` automatically during interpreter startup when
it is available on ``sys.path``. Because GitHub Actions executes
``python scripts/run_lstm_tuning.py``, the ``scripts`` directory is on
``sys.path`` but the repository root is not. This module adds that root before
the tuning runner imports ``src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
root_text = str(REPOSITORY_ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)
