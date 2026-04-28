"""Algebra-hom walker — F-driven inductive and coinductive (co)algebra homs.

Given a polynomial Functor F and a list of cells (one per summand for
algebra direction), synthesise the unique morphism induced by initiality
or terminality. List-cata, tree-cata, n-ary tree-cata, and single-cell
coalgebras are all instances of the same code, parameterised by F.

This module is imported by ``_para_runtime`` for the algebraHom dispatch
case. It does not import from ``_para_runtime`` at module load time —
sub-cell compilation goes through a callable passed in by the runtime
(``compile_subcell``) to break the circular dependency.
"""
from __future__ import annotations

from collections.abc import Callable

from unialg.assembly._para import Cell


CompileSubcell = Callable[[Cell], Callable | None]


def compile_algebra_hom(cell: Cell, backend, matchers: dict[str, Callable],
                        compile_subcell: CompileSubcell) -> Callable | None:
    """Dispatch on direction; build the F-driven walker."""
    functor = cell.functor
    cells = cell.cells
    summands = functor.summands()
    if len(cells) != len(summands):
        raise ValueError(
            f"algebra_hom: cells length {len(cells)} != functor "
            f"{functor.name!r} summand count {len(summands)}"
        )
    if cell.direction == "algebra":
        return _compile_inductive(functor, cells, backend, matchers, compile_subcell)
    if cell.direction == "coalgebra":
        return _compile_coinductive(functor, cells, backend, compile_subcell)
    raise AssertionError(f"unreachable direction: {cell.direction}")


def _compile_inductive(functor, cells, backend, matchers, compile_subcell):
    matcher = matchers.get(functor.name)
    if matcher is None:
        raise ValueError(
            f"algebra_hom over {functor.name!r}: no matcher registered. "
            f"The inductive walker needs a matcher to decompose input values "
            f"into (summand_idx, [recs], [consts])."
        )
    case_handlers: list[Callable] = []
    for c in cells:
        handler = compile_subcell(c)
        if handler is None:
            return None
        case_handlers.append(handler)

    def walker(value):
        idx, recs, consts = matcher(value)
        rec_results = [walker(r) for r in recs]
        return case_handlers[idx](*consts, *rec_results)

    return backend.compile(walker)


def _compile_coinductive(functor, cells, backend, compile_subcell):
    """F = X or F = prod-with-id: a single-step coalgebra producing the next
    state (and optional output). Truncation / iteration is a separate
    primitive that operates on the returned closure.
    """
    if len(cells) != 1:
        raise ValueError(
            f"coalgebra_hom over {functor.name!r}: expected 1 cell, got {len(cells)}"
        )
    cell_fn = compile_subcell(cells[0])
    if cell_fn is None:
        return None
    body = functor.body
    if body.kind == "id":
        return backend.compile(cell_fn)
    if body.kind == "prod":
        left, right = body.left, body.right
        if (left.kind == "id" and right.kind != "id") or \
           (right.kind == "id" and left.kind != "id"):
            return backend.compile(cell_fn)
    raise NotImplementedError(
        f"coalgebra_hom: body shape {body.kind!r} not supported by the "
        f"coinductive walker (current support: id, prod-with-id-position)."
    )
