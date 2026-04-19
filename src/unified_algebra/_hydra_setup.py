"""Ensure Hydra's two source roots are on sys.path before any hydra imports."""

import sys
from pathlib import Path

_initialized = False


def setup():
    global _initialized
    if _initialized:
        return
    root = Path(__file__).resolve().parent.parent.parent.parent / "hydra"
    for sub in [
        root / "heads" / "python" / "src" / "main" / "python",
        root / "dist" / "python" / "hydra-kernel" / "src" / "main" / "python",
    ]:
        p = str(sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    _initialized = True


setup()
