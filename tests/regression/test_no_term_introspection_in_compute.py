"""Regression: no isinstance(result, core.Term*) calls inside compute functions.

Audit reference: robust-scribbling-dove.md § "2. Boundary assessment" item 1,
and Phase 3.1.

The Hydra primitive contract:
  A compute function registered with prim1/prim2/prim3 must receive and return
  native Python/numpy values — NOT Hydra terms. The coders (encode/decode)
  handle the boundary before and after the compute function is called.

Current violation:
  ``_primitives.py:83-84`` — ``_lens_fwd_compute`` contains:
      elif isinstance(result, core.TermPair):
  This is inside a Hydra primitive's compute closure, where ``result`` should
  already be a plain Python value.

This test is marked xfail until Phase 3.1 ships and removes the branch.

After Phase 3.1, remove the xfail marker and the violation should be gone.
"""

import ast
import re
import textwrap
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ASSEMBLY_DIR = Path(__file__).parents[2] / "src" / "unialg" / "assembly"

# Regex that matches isinstance(..., core.Term...) including TermPair, TermList, etc.
_TERM_ISINSTANCE_PATTERN = re.compile(
    r"isinstance\s*\(\s*[^,]+,\s*core\.Term",
    re.MULTILINE,
)


def _find_compute_functions_with_term_introspection(path: Path) -> list[tuple[str, int, str]]:
    """Return (func_name, line_no, matched_text) for each violation found."""
    violations = []
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    # Collect line numbers of all function/method definitions whose names
    # end in "_compute" or contain "compute".
    compute_fn_ranges: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "compute" in node.name.lower():
                # end_lineno available in Python 3.8+
                end = getattr(node, "end_lineno", node.lineno + 200)
                compute_fn_ranges.append((node.lineno, end, node.name))

    if not compute_fn_ranges:
        return violations

    lines = source.splitlines()
    for fn_start, fn_end, fn_name in compute_fn_ranges:
        for lineno in range(fn_start - 1, min(fn_end, len(lines))):
            line = lines[lineno]
            m = _TERM_ISINSTANCE_PATTERN.search(line)
            if m:
                violations.append((fn_name, lineno + 1, m.group(0)))

    return violations


# ---------------------------------------------------------------------------
# The regression test — xfail until Phase 3.1
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Audit Phase 3.1 will remove isinstance(result, core.TermPair) from "
        "_lens_fwd_compute in _primitives.py. "
        "See robust-scribbling-dove.md § Phase 3.1."
    ),
    strict=True,  # must FAIL (not error) until fixed; becomes XPASS after fix
)
def test_no_term_introspection_in_primitives_compute():
    """_primitives.py _*_compute functions must not contain isinstance(*, core.Term*).

    This XFAILS because _lens_fwd_compute at line 83-84 currently contains:
        elif isinstance(result, core.TermPair):
    When Phase 3.1 removes the branch, this test will XPASS and the xfail
    marker should be removed.
    """
    target = _ASSEMBLY_DIR / "_primitives.py"
    assert target.exists(), f"Expected {target} to exist"

    violations = _find_compute_functions_with_term_introspection(target)
    violation_msgs = [
        f"  {fn}:{lineno}: {text}"
        for fn, lineno, text in violations
    ]
    assert violations == [], (
        "Found isinstance(*, core.Term*) inside compute functions in "
        f"{target.name}:\n" + "\n".join(violation_msgs)
    )


# ---------------------------------------------------------------------------
# Non-xfail: verify the scanner itself works on a clean file
# ---------------------------------------------------------------------------

def test_scanner_finds_no_violations_in_equation_resolution():
    """_equation_resolution.py has no isinstance(*, core.Term*) in compute fns."""
    target = _ASSEMBLY_DIR / "_equation_resolution.py"
    assert target.exists(), f"Expected {target} to exist"
    violations = _find_compute_functions_with_term_introspection(target)
    violation_msgs = [
        f"  {fn}:{lineno}: {text}"
        for fn, lineno, text in violations
    ]
    assert violations == [], (
        "Unexpected core.Term introspection in _equation_resolution.py:\n"
        + "\n".join(violation_msgs)
    )


def test_scanner_detects_synthetic_violation(tmp_path):
    """Verify the scanner actually catches a synthetic violation."""
    synthetic = tmp_path / "synthetic_prims.py"
    synthetic.write_text(textwrap.dedent("""\
        import hydra.core as core

        def my_compute(result, other):
            if isinstance(result, core.TermPair):
                return result.value
            return result
    """))

    # Reuse the scanner on the synthetic file
    source = synthetic.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        pytest.fail("Synthetic source failed to parse")

    compute_fn_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "compute" in node.name.lower():
                end = getattr(node, "end_lineno", node.lineno + 200)
                compute_fn_ranges.append((node.lineno, end, node.name))

    violations = []
    lines = source.splitlines()
    for fn_start, fn_end, fn_name in compute_fn_ranges:
        for lineno in range(fn_start - 1, min(fn_end, len(lines))):
            line = lines[lineno]
            m = _TERM_ISINSTANCE_PATTERN.search(line)
            if m:
                violations.append((fn_name, lineno + 1, m.group(0)))

    assert len(violations) == 1, (
        f"Scanner should have found exactly 1 violation, found: {violations}"
    )
    assert violations[0][0] == "my_compute"
