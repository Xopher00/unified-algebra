"""Cell-expression resolution — extracted from _resolver.py."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hydra.dsl.terms as Terms
from hydra.dsl.python import Just
from unialg.terms import _literal_value
from unialg.morphism import TypedMorphism as T
from unialg.algebra import Equation
from ._cell_ast import CellExpr
from ._decl_ast import CellDecl


@dataclass(frozen=True)
class NamedCell:
    """A named morphism expression destined for graph registration."""
    name: str
    cell: object


@dataclass
class _CellResolverContext:
    equations_by_name: dict   # live dict — mutated by _ensure_adjoint_eq
    equations_list: list      # live list — appended to by _resolve_modified_eq
    sorts: dict
    functors_by_name: dict


def _get_sort(ctx: _CellResolverContext, name: str):
    if name not in ctx.sorts:
        raise ValueError(f"Unknown spec {name!r} — declared specs: {list(ctx.sorts)}")
    return ctx.sorts[name]


def _ensure_adjoint_eq(ctx: _CellResolverContext, base_name: str) -> tuple[str, "Equation | None"]:
    """Return ``(adjoint_name, new_eq)`` where ``new_eq`` is the freshly created
    Equation if it did not already exist, or ``None`` if it was already registered.

    The caller is responsible for appending ``new_eq`` to the equations list when
    it is not ``None`` — this function no longer mutates ``ctx.equations_list``.
    It still registers the equation in ``ctx.equations_by_name`` so that
    subsequent modifier chains resolve correctly within the same call.
    """
    if base_name not in ctx.equations_by_name:
        raise ValueError(f"unknown equation {base_name!r}")
    base = ctx.equations_by_name[base_name]
    if not base.einsum:
        raise NotImplementedError(
            f"equation modifier \"'\" on {base_name!r}: adjoint "
            "references require an einsum-backed op"
        )
    adjoint_name = f"{base_name}__adjoint"
    if adjoint_name not in ctx.equations_by_name:
        eq = Equation(
            adjoint_name, base.einsum,
            base.domain_sort, base.codomain_sort, base.semiring,
            nonlinearity=base.nonlinearity,
            inputs=tuple(base.inputs),
            param_slots=tuple(base.param_slots),
            adjoint=True, skip=base.skip,
        )
        ctx.equations_by_name[adjoint_name] = eq
        return adjoint_name, eq
    return adjoint_name, None


def _resolve_modified_eq(ctx: _CellResolverContext, base_name: str, modifiers: str):
    if base_name not in ctx.equations_by_name:
        raise ValueError(f"unknown equation {base_name!r}")
    resolved_name = base_name
    for modifier in modifiers:
        if modifier == "'":
            resolved_name, new_eq = _ensure_adjoint_eq(ctx, resolved_name)
            if new_eq is not None:
                ctx.equations_list.append(new_eq)
        elif modifier == "?":
            raise NotImplementedError(
                f"equation modifier '?' on {resolved_name!r}: masked "
                "references are parsed but not implemented"
            )
        else:
            raise ValueError(f"unsupported equation modifier {modifier!r}")
    return resolved_name, ctx.equations_by_name[resolved_name]


def _build_typed(ctx: _CellResolverContext, cell_codomain: Any, node: CellExpr):
    from unialg.morphism import eq, lit, iden, copy, delete, seq, par, lens, algebra_hom, summand_domain
    k = node.kind
    if k == "eq":
        rec = node._payload_record_fields()
        base_name = _literal_value(rec["baseName"])
        modifiers = _literal_value(rec["modifiers"])
        resolved_name, equation = _resolve_modified_eq(ctx, base_name, modifiers)
        return eq(resolved_name, domain=equation.domain_sort, codomain=equation.codomain_sort)
    if k == "lit":
        v = _literal_value(node._payload)
        return lit(Terms.float32(v), cell_codomain)
    if k in ("copy", "delete", "iden"):
        sort_name = _literal_value(node._payload)
        sort = _get_sort(ctx, sort_name)
        if k == "copy":   return copy(sort)
        if k == "delete": return delete(sort)
        if k == "iden":   return iden(sort)
    if k in ("seq", "par"):
        payload = node._payload
        l_term, r_term = payload.value[0], payload.value[1]
        left  = _build_typed(ctx, cell_codomain, CellExpr(l_term))
        right = _build_typed(ctx, cell_codomain, CellExpr(r_term))
        if k == "seq": return seq(left, right)
        return par(left, right)
    if k == "lens":
        rec = node._payload_record_fields()
        fwd_node = CellExpr(rec["fwd"])
        bwd_node = CellExpr(rec["bwd"])
        res_maybe = rec["residual"]
        if isinstance(res_maybe.value, Just):
            residual_sort = ctx.sorts[_literal_value(res_maybe.value.value)]
        else:
            residual_sort = None
        return lens(
            _build_typed(ctx, cell_codomain, fwd_node),
            _build_typed(ctx, cell_codomain, bwd_node),
            residual_sort,
        )
    if k in ("cata", "ana"):
        rec = node._payload_record_fields()
        f_name = _literal_value(rec["functor"])
        if f_name not in ctx.functors_by_name:
            label = k
            raise ValueError(f"cell {label}: unknown functor {f_name!r} — declare via 'functor'")
        functor = ctx.functors_by_name[f_name]
        direction = "algebra" if k == "cata" else "coalgebra"
        args = [CellExpr(a) for a in rec["args"].value]
        if direction == "algebra":
            summands = functor.summands()
            morphisms = []
            for arg_node, summand in zip(args, summands):
                m = _build_typed(ctx, cell_codomain, arg_node)
                dom = summand_domain(summand, cell_codomain)
                morphisms.append(T(m.term, dom, cell_codomain))
        else:
            morphisms = [_build_typed(ctx, cell_codomain, a) for a in args]
        return algebra_hom(functor, direction, morphisms)
    raise ValueError(f"cell: unknown kind {k!r}")


def handle_cell(ctx: _CellResolverContext, decl: CellDecl, resolve_sort_ref) -> NamedCell:
    cell_codomain = resolve_sort_ref(decl.sig[1])
    return NamedCell(name=decl.name, cell=_build_typed(ctx, cell_codomain, decl.expr))
