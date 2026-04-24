"""Composition of equations as Hydra lambda terms.

    path()      → sequential: f_n . ... . f_1
    fan()       → parallel: merge(b1(x), b2(x), ...)
    fold()      → catamorphism: foldl(step, init, seq)
    unfold()    → anamorphism: unfold_n(step, n, state)
    fixpoint()  → iterate until convergence
    lens_path() → bidirectional sequential
    lens_fan()  → bidirectional parallel
"""

from __future__ import annotations

import hydra.core as core
from hydra.dsl.meta.phantoms import var, int32, list_, TTerm

from unialg.terms import bind_composition


# ---------------------------------------------------------------------------
# Path (sequential composition)
# ---------------------------------------------------------------------------

def path(
    name: str,
    eq_names: list[str],
    params: dict[str, list[core.Term]] | None = None,
    residual: bool = False,
    residual_semiring: str | None = None,
) -> tuple[core.Name, core.Term]:
    """Build a sequential composition as a Hydra lambda term."""
    if not eq_names:
        raise ValueError(f"Path '{name}' must have at least one equation")

    body: TTerm = var("x")
    for eq_name in eq_names:
        fn: TTerm = var(f"ua.equation.{eq_name}")
        if params and eq_name in params:
            for p in params[eq_name]:
                fn = fn @ TTerm(p)
        body = fn @ body

    if residual:
        sr_name = residual_semiring or "default"
        body = var(f"ua.prim.residual_add.{sr_name}") @ body @ var("x")

    return bind_composition("path", name, "x", body)


# ---------------------------------------------------------------------------
# Fan (parallel composition)
# ---------------------------------------------------------------------------

def fan(
    name: str,
    branch_names: list[str],
    merge_name: str,
) -> tuple[core.Name, core.Term]:
    """Build a parallel composition as a Hydra lambda term."""
    if not branch_names:
        raise ValueError(f"Fan '{name}' must have at least one branch")

    branch_results = [
        var(f"ua.equation.{bname}") @ var("x")
        for bname in branch_names
    ]
    body = var(f"ua.equation.{merge_name}") @ list_(branch_results)

    return bind_composition("fan", name, "x", body)


# ---------------------------------------------------------------------------
# Fold (catamorphism)
# ---------------------------------------------------------------------------

def fold(
    name: str,
    step_name: str,
    init_term: core.Term,
) -> tuple[core.Name, core.Term]:
    """Build a fold as a Hydra lambda term."""
    step_fn = var(f"ua.equation.{step_name}")
    body = var("hydra.lib.lists.foldl") @ step_fn @ TTerm(init_term) @ var("seq")
    return bind_composition("fold", name, "seq", body)


def unfold(
    name: str,
    step_name: str,
    n_steps: int,
) -> tuple[core.Name, core.Term]:
    """Build an unfold as a Hydra lambda term."""
    step_fn = var(f"ua.equation.{step_name}")
    body = var("ua.prim.unfold_n") @ step_fn @ int32(n_steps) @ var("state")
    return bind_composition("unfold", name, "state", body)


def fixpoint(
    name: str,
    step_name: str,
    predicate_name: str,
    epsilon: float,
    max_iter: int,
) -> tuple[core.Name, core.Term]:
    """Build a fixpoint iteration as a Hydra lambda term."""
    step_fn = var(f"ua.equation.{step_name}")
    pred_fn = var(f"ua.equation.{predicate_name}")
    body = var(f"ua.prim.fixpoint.{epsilon}.{max_iter}") @ step_fn @ pred_fn @ var("state")
    return bind_composition("fixpoint", name, "state", body)


# ---------------------------------------------------------------------------
# Lens compositions
# ---------------------------------------------------------------------------

def lens_path(
    name: str,
    fwd_eq_names: list[str],
    bwd_eq_names: list[str],
    params: dict[str, list[core.Term]] | None = None,
    has_residual: bool = False,
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses sequentially into a bidirectional path."""
    if not fwd_eq_names:
        raise ValueError(f"lens_path '{name}' must have at least one lens")

    if has_residual and len(fwd_eq_names) > 1:
        fwd_pair = bind_composition("path", f"{name}.fwd", "x",
                                   var("ua.prim.lens_fwd") @ list_([var(f"ua.equation.{n}") for n in fwd_eq_names]) @ var("x"))
        bwd_pair = bind_composition("path", f"{name}.bwd", "p",
                                   var("ua.prim.lens_bwd") @ list_([var(f"ua.equation.{n}") for n in bwd_eq_names]) @ var("p"))
        return fwd_pair, bwd_pair

    fwd_name_obj, fwd_term = path(f"{name}.fwd", fwd_eq_names, params)
    bwd_name_obj, bwd_term = path(f"{name}.bwd", list(reversed(bwd_eq_names)))
    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)


def lens_fan(
    name: str,
    fwd_branch_names: list[str],
    bwd_branch_names: list[str],
    merge_fwd_name: str,
    merge_bwd_name: str,
) -> tuple[tuple[core.Name, core.Term], tuple[core.Name, core.Term]]:
    """Compose lenses in parallel into a bidirectional fan."""
    if not fwd_branch_names:
        raise ValueError(f"lens_fan '{name}' must have at least one branch lens")

    fwd_name_obj, fwd_term = fan(f"{name}.fwd", fwd_branch_names, merge_fwd_name)
    bwd_name_obj, bwd_term = fan(f"{name}.bwd", bwd_branch_names, merge_bwd_name)
    return (fwd_name_obj, fwd_term), (bwd_name_obj, bwd_term)
