"""Hydra bootstrap and term-navigation helpers.

Ensures Hydra's two source roots are on sys.path before any hydra imports,
and provides helpers for extracting values from Hydra record terms.
"""

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


# ---------------------------------------------------------------------------
# Hydra term-navigation helpers
# ---------------------------------------------------------------------------

def record_fields(term) -> dict[str, object]:
    """Extract a Hydra record's fields as a {name_str: Term} dict."""
    return {f.name.value: f.term for f in term.value.fields}


def string_value(term) -> str:
    """Extract a plain string from a Hydra TermLiteral(LiteralString(...))."""
    return term.value.value


def float_value(term) -> float:
    """Extract a float from a Hydra TermLiteral(LiteralFloat(...))."""
    return term.value.value
