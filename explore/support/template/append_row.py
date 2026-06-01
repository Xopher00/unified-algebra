"""
Derive a family_tree.csv row from an ArchSpec JSON file and append it.

All structured fields are computed mechanically from the spec.
Only `notes` requires agent input.

Usage:
    python explore/support/template/append_row.py <spec.json> [--notes "..."]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spec_utils import (
    render_poly_f, poly_depth,
    spec_equations, spec_activations, spec_semiring,
)

FAMILY_TREE = Path("explore/support/family_tree.csv")

FIELDNAMES = [
    "label", "arch_class", "poly_f", "tensor_equations", "activation",
    "semiring_add", "semiring_mul", "ref_strength", "arm", "depth", "notes",
]


# ── row derivation ────────────────────────────────────────────────────────────

def derive_row(spec: dict, notes: str) -> dict:
    arch = spec["arch"]
    ref  = spec["ref"]
    sr_add, sr_mul = spec_semiring(spec)
    return {
        "label":            spec["label"],
        "arch_class":       arch["class"],
        "poly_f":           render_poly_f(arch["poly_f"]),
        "tensor_equations": ";".join(spec_equations(spec)),
        "activation":       ";".join(spec_activations(spec)),
        "semiring_add":     sr_add,
        "semiring_mul":     sr_mul,
        "ref_strength":     ref.get("strength", "numpy-only"),
        "arm":              spec.get("arm", ""),
        "depth":            poly_depth(arch["poly_f"]),
        "notes":            notes,
    }


# ── append ────────────────────────────────────────────────────────────────────

def append_row(row: dict) -> None:
    existing = set()
    if FAMILY_TREE.exists():
        with FAMILY_TREE.open(newline="") as f:
            for r in csv.DictReader(f):
                existing.add(r["label"])

    if row["label"] in existing:
        print(f"[skip] {row['label']} already in {FAMILY_TREE}", file=sys.stderr)
        return

    with FAMILY_TREE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    print(f"[ok] appended {row['label']} to {FAMILY_TREE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec",    help="Path to ArchSpec JSON file")
    parser.add_argument("--notes", default="", help="Free-text notes for this row")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    row = derive_row(spec, args.notes)

    for k, v in row.items():
        print(f"  {k:20s} {v}")
    print()

    append_row(row)


if __name__ == "__main__":
    main()
