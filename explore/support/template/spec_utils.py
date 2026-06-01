"""Shared helpers for reading and normalizing ArchSpec JSON."""

from __future__ import annotations
import csv
from pathlib import Path


# ── poly_f → mathematical form ────────────────────────────────────────────────

def render_poly_f(node: dict, parent: str | None = None) -> str:
    tag = node["tag"]
    if tag == "KUnit":   return "1"
    if tag == "KConst":  return "C"
    if tag == "Hole":    return "X"
    if tag == "Sum":
        s = f"{render_poly_f(node['left'])} + {render_poly_f(node['right'])}"
        return f"({s})" if parent in ("Product", "Exp") else s
    if tag == "Product":
        return (f"{render_poly_f(node['left'], 'Product')} × "
                f"{render_poly_f(node['right'], 'Product')}")
    if tag == "Exp":
        return f"X^{render_poly_f(node['arg'], 'Exp')}"
    raise ValueError(f"Unknown PolyF tag: {tag!r}")


def poly_depth(node: dict) -> int:
    tag = node["tag"]
    if tag in ("KUnit", "KConst", "Hole"):
        return 0
    if tag in ("Sum", "Product"):
        return 1 + max(poly_depth(node["left"]), poly_depth(node["right"]))
    if tag == "Exp":
        return 1 + poly_depth(node["arg"])
    raise ValueError(f"Unknown PolyF tag: {tag!r}")


# ── einsum normalization ──────────────────────────────────────────────────────

def normalize_equation(eq: str) -> str:
    """
    Canonical form: rename indices to a, b, c, … with output indices first,
    then remaining input-only indices.

    Examples:
      oi,i->o   →  ab,b->a
      hi,i->h   →  ab,b->a  (same — both are matvec)
      ij,jk->ik →  ab,bc->ac
    """
    inputs_str, output_str = eq.split("->")
    mapping: dict[str, str] = {}
    counter = [0]

    def assign(c: str) -> None:
        if c.isalpha() and c not in mapping:
            mapping[c] = chr(ord("a") + counter[0])
            counter[0] += 1

    for c in output_str:
        assign(c)
    for c in inputs_str:
        assign(c)

    def remap(s: str) -> str:
        return "".join(mapping[c] if c.isalpha() else c for c in s)

    return remap(inputs_str) + "->" + remap(output_str)


# ── extract structured fields from a spec ────────────────────────────────────

def spec_equations(spec: dict) -> list[str]:
    """Unique normalized einsum equations from all Contraction bindings."""
    seen: list[str] = []
    for b in spec["cell"]["bindings"]:
        if b["expr"]["tag"] == "Contraction":
            n = normalize_equation(b["expr"]["equation"])
            if n not in seen:
                seen.append(n)
    return seen


def spec_activations(spec: dict) -> list[str]:
    """Unique activation kinds from all Activation bindings."""
    seen: list[str] = []
    for b in spec["cell"]["bindings"]:
        if b["expr"]["tag"] == "Activation":
            k = b["expr"]["kind"]
            if k not in seen:
                seen.append(k)
    return seen


def spec_semiring(spec: dict) -> tuple[str, str]:
    """(add, mul) from the first Contraction binding; ('', '') if none."""
    for b in spec["cell"]["bindings"]:
        if b["expr"]["tag"] == "Contraction":
            sr = b["expr"]["semiring"]
            return sr["add"], sr["multiply"]
    return "", ""


def catalog_rows(family_tree: Path) -> list[dict]:
    if not family_tree.exists():
        return []
    with family_tree.open(newline="") as f:
        return list(csv.DictReader(f))
