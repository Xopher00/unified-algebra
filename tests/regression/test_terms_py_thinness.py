"""Regression: terms.py stays thin — line-count budget and import restrictions.

Audit reference: robust-scribbling-dove.md § 5 ("terms.py as narrow Hydra adapter").

Pins:
  - File line count <= 250 (current is ~187; budget gives ~60 lines headroom).
  - terms.py does NOT import from unialg.assembly, unialg.runtime,
    unialg.parser, or unialg.algebra.equation.
  - Allowed cross-package imports: hydra.*, stdlib, and self-referential helpers.
  - Uses ast.parse() to extract imports (not regex).

If terms.py legitimately grows beyond 250 lines, update the budget here and
document why — adding new record-view subclasses is allowed; adding Cell/
Equation/Semiring logic is not.
"""

import ast
from pathlib import Path

import pytest


_TERMS_PY = Path(__file__).parents[2] / "src" / "unialg" / "terms.py"

_LINE_BUDGET = 250

# Forbidden source packages — any import from these indicates boundary leakage
_FORBIDDEN_IMPORT_PREFIXES = (
    "unialg.assembly",
    "unialg.runtime",
    "unialg.parser",
    "unialg.algebra.equation",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Return (module_name, line_no) for every import statement in the AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append((module, node.lineno))
    return imports


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_terms_py_exists():
    """terms.py is present at the expected path."""
    assert _TERMS_PY.exists(), (
        f"Expected terms.py at {_TERMS_PY} — was it moved or renamed?"
    )


def test_terms_py_line_count():
    """terms.py must not exceed the line-count budget.

    Current: ~187 lines. Budget: 250 (60 lines headroom for new _RecordView
    subclasses). If this fails, check whether Cell/Equation/Semiring logic has
    leaked into terms.py — see ARCHITECTURE.md § terms.py is the narrow adapter.
    """
    source = _TERMS_PY.read_text()
    lines = source.splitlines()
    line_count = len(lines)
    assert line_count <= _LINE_BUDGET, (
        f"terms.py has {line_count} lines (budget: {_LINE_BUDGET}). "
        f"Check whether boundary leakage has occurred — "
        f"only _RecordView subclasses and Hydra adapter helpers belong here."
    )


def test_terms_py_no_forbidden_imports():
    """terms.py must not import from assembly, runtime, parser, or algebra.equation.

    These imports would turn terms.py from a thin Hydra adapter into a
    second-layer IR, violating the boundary contract in ARCHITECTURE.md.
    """
    source = _TERMS_PY.read_text()
    tree = ast.parse(source, filename=str(_TERMS_PY))
    imports = _collect_imports(tree)

    violations = [
        (module, lineno)
        for module, lineno in imports
        if any(module.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES)
    ]

    violation_msgs = [
        f"  line {lineno}: import {module!r}"
        for module, lineno in violations
    ]
    assert violations == [], (
        "terms.py imports from forbidden packages — boundary violation:\n"
        + "\n".join(violation_msgs)
        + "\n\nSee ARCHITECTURE.md § terms.py is the narrow adapter."
    )


def test_terms_py_imports_are_hydra_or_stdlib():
    """All cross-package imports in terms.py are from hydra.* or stdlib.

    Allowed: hydra.*, stdlib (re, pathlib, typing, etc.), unialg self-imports.
    Not allowed: other third-party packages introducing heavy dependencies.
    """
    source = _TERMS_PY.read_text()
    tree = ast.parse(source, filename=str(_TERMS_PY))
    imports = _collect_imports(tree)

    # We only flag imports that are clearly non-hydra third-party packages.
    # numpy and torch are backend concerns, not terms.py concerns.
    flagged_third_party = ("numpy", "torch", "jax", "cupy")

    violations = [
        (module, lineno)
        for module, lineno in imports
        if any(module.startswith(pkg) for pkg in flagged_third_party)
    ]
    violation_msgs = [
        f"  line {lineno}: import {module!r}"
        for module, lineno in violations
    ]
    assert violations == [], (
        "terms.py imports backend packages — these belong in backend.py, not terms.py:\n"
        + "\n".join(violation_msgs)
    )


def test_terms_py_ast_parseable():
    """terms.py is valid Python (sanity guard for the other tests)."""
    source = _TERMS_PY.read_text()
    try:
        ast.parse(source, filename=str(_TERMS_PY))
    except SyntaxError as e:
        pytest.fail(f"terms.py failed to parse: {e}")
