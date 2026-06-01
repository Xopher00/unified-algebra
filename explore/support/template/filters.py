"""
Pre-test filters for ArchSpec candidates.

Each filter returns (passed: bool, reason: str).
passed=True means the spec survives the filter (not rejected).

Usage (CLI):
    python explore/support/template/filters.py <spec.json>
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spec_utils import (
    render_poly_f, spec_equations, spec_activations, spec_semiring, catalog_rows,
)

FAMILY_TREE = Path("explore/support/family_tree.csv")


# ── 1. Tucker / einsum dedup ─────────────────────────────────────────────────

def _signature(poly_f_str: str, equations: list[str],
               activations: list[str], sr_add: str, sr_mul: str) -> tuple:
    return (poly_f_str, tuple(sorted(equations)),
            tuple(sorted(activations)), sr_add, sr_mul)


def check_dedup(spec: dict) -> tuple[bool, str]:
    """
    Reject if the catalog already contains a row with the same
    (poly_f, equations, activations, semiring_add, semiring_mul) signature.
    A name change does not make a structurally identical spec novel.
    """
    poly_f_str = render_poly_f(spec["arch"]["poly_f"])
    equations  = spec_equations(spec)
    activations = spec_activations(spec)
    sr_add, sr_mul = spec_semiring(spec)
    new_sig = _signature(poly_f_str, equations, activations, sr_add, sr_mul)

    for row in catalog_rows(FAMILY_TREE):
        existing_eqs = [e for e in row["tensor_equations"].split(";") if e]
        existing_acts = [a for a in row["activation"].split(";") if a]
        existing_sig = _signature(
            row["poly_f"], existing_eqs, existing_acts,
            row["semiring_add"], row["semiring_mul"],
        )
        if new_sig == existing_sig:
            return False, f"duplicate of catalog entry '{row['label']}'"

    # Also reject if label already exists.
    existing_labels = {r["label"] for r in catalog_rows(FAMILY_TREE)}
    if spec["label"] in existing_labels:
        return False, f"label '{spec['label']}' already in catalog"

    return True, ""


# ── 2. Triviality ────────────────────────────────────────────────────────────

def check_triviality(spec: dict) -> tuple[bool, str]:
    """
    Reject specs whose cell body is computationally trivial:

    (a) Identity: for Ana/Hylo/Pure arches, the declared output is the
        state var or input var directly — no transformation applied.

    (b) No contraction: cell has no Contraction bindings and no Activation
        bindings — the cell performs no weighted computation at all.
        (Pure ElemOp chains on constants are not learnable.)
    """
    cell   = spec["cell"]
    result = cell["result"]
    tag    = result["tag"]
    bindings = cell["bindings"]

    # (a) identity output
    if tag == "Ana":
        if result["output"] == result["state_var"]:
            return False, "trivial: output is the state variable unchanged"
        if result["output"] == result["input_var"]:
            return False, "trivial: output is the input variable unchanged"
        if result["next_state"] == result["state_var"]:
            return False, "trivial: next_state is the state variable unchanged"
    elif tag == "Pure":
        if result["result"] == result["input_var"]:
            return False, "trivial: result is the input variable unchanged"

    # (b) no weighted computation
    has_contraction = any(b["expr"]["tag"] == "Contraction" for b in bindings)
    has_activation  = any(b["expr"]["tag"] == "Activation"  for b in bindings)
    if not has_contraction and not has_activation:
        return False, "trivial: no Contraction or Activation — cell is unweighted"

    return True, ""


# ── run all filters ───────────────────────────────────────────────────────────

def run_filters(spec: dict) -> list[tuple[str, bool, str]]:
    """Returns [(filter_name, passed, reason), ...]."""
    return [
        ("dedup",      *check_dedup(spec)),
        ("triviality", *check_triviality(spec)),
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: filters.py <spec.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        spec = json.load(f)

    results = run_filters(spec)
    all_pass = True
    for name, passed, reason in results:
        status = "PASS" if passed else "REJECT"
        msg = f"  [{status}] {name}"
        if reason:
            msg += f": {reason}"
        print(msg)
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
