"""Name resolution pass for parsed .ua declarations.

Walks the raw declaration list from the grammar and produces a UASpec.
Compositions are expressed as `cell` declarations using the operator
sub-grammar; `share` records 2-cell weight-tying groups; `functor`
declares polynomial endofunctors used by ``cata``/``ana`` cells.
"""
from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from unialg.algebra import Sort, ProductSort, Semiring, Equation
from ._resolve_cells import _CellResolverContext, handle_cell
from hydra.ast import ExprConst, ExprOp
from ._decl_ast import (
    ImportDecl, AlgebraDecl, SpecDecl, OpDecl,
    ShareDecl, DefineDecl, FunctorDecl, CellDecl,
)


@dataclass
class UASpec:
    """Parsed .ua program before compilation."""
    semirings: dict = field(default_factory=dict)
    sorts: dict = field(default_factory=dict)
    equations: list = field(default_factory=list)
    defines: list = field(default_factory=list)
    backend_name: str | None = None
    share_groups: dict = field(default_factory=dict)
    cells: list = field(default_factory=list)


def _build_poly(node, get_sort):
    """Recursively build a polynomial functor from a parsed expression node.

    ``get_sort`` is a callable that resolves a sort name to a Sort object;
    it is passed explicitly so this function does not close over resolver state.
    """
    from unialg.morphism import sum_, prod, one, zero, id_, const
    if isinstance(node, ExprConst):
        sym_val = node.value.value
        if sym_val == "0": return zero()
        if sym_val == "1": return one()
        if sym_val == "X": return id_()
        return const(get_sort(sym_val))
    if isinstance(node, ExprOp):
        op_expr = node.value
        sym_str = op_expr.op.symbol.value
        if sym_str == "@":
            raise NotImplementedError("functor composition (@) not yet supported")
        left = _build_poly(op_expr.lhs, get_sort)
        right = _build_poly(op_expr.rhs, get_sort)
        if sym_str == "+": return sum_(left, right)
        if sym_str == "&": return prod(left, right)
        raise ValueError(f"functor: unknown operator {sym_str!r}")
    raise ValueError(f"functor: invalid polynomial AST node {node!r}")


def _resolve_spec(raw_decls: list[tuple]) -> UASpec:
    """Resolve raw declarations into a UASpec.

    Processes declarations in source order. Lookups (sort, op, functor)
    work because the user is required to declare a name before referring
    to it; this is enforced by raising on missing names.
    """

    backend_name: str | None = None
    semirings: dict[str, Any] = {}
    sorts: dict[str, Any] = {}
    defines: list[tuple] = []
    equations_by_name: dict[str, Any] = {}
    equations_list: list[Any] = []
    share_groups: dict[str, list[str]] = {}
    cells: list[Any] = []
    functors_by_name: dict[str, Any] = {}

    # --- lookup helpers ---

    def _lookup(name: str, d: dict, label: str) -> Any:
        if name not in d:
            raise ValueError(
                f"Unknown {label} {name!r} — declared {label}s: {list(d)}"
            )
        return d[name]

    def _get_sr(name):   return _lookup(name, semirings, 'algebra')
    def _get_sort(name): return _lookup(name, sorts, 'spec')
    def _get_eq(name):   return _lookup(name, equations_by_name, 'op')

    def _resolve_sort_ref(ref):
        if isinstance(ref, str):
            return _get_sort(ref)
        if isinstance(ref, tuple) and ref[0] == '_product':
            return ProductSort([_get_sort(n) for n in ref[1]])
        raise ValueError(f"Invalid sort reference: {ref}")

    # --- per-kind handlers ---

    _cell_ctx = _CellResolverContext(
        equations_by_name=equations_by_name,
        equations_list=equations_list,
        sorts=sorts,
        functors_by_name=functors_by_name,
    )

    for decl in raw_decls:
        match decl:
            case DefineDecl(name=name, arity=arity, params=params, body=body):
                defines.append((arity, name, params, body))
            case ImportDecl(backend=name):
                backend_name = name
            case AlgebraDecl(name=name, kw_args=kw_args):
                sr_term = Semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                                   zero=kw_args['zero'], one=kw_args['one'],
                                   contraction=kw_args.get('strategy') or kw_args.get('contraction', ''),
                                   residual=kw_args.get('residual', ''),
                                   leq=kw_args.get('leq', ''))
                semirings[name] = sr_term
            case SpecDecl(name=name, sr_name=sr_name, batched=batched, axes=axes):
                sr_term = _get_sr(sr_name)
                sort_term = Sort(name, sr_term, batched=batched, axes=axes)
                sorts[name] = sort_term
            case OpDecl(name=name, sig=(dom_name, cod_name), attrs=attr_dict):
                dom_sort = _resolve_sort_ref(dom_name)
                cod_sort = _resolve_sort_ref(cod_name)
                einsum = attr_dict.get('einsum', None) or None
                nl = attr_dict.get('nonlinearity', None) or None
                sr_name = attr_dict.get('algebra', None)
                sr_term = _get_sr(sr_name) if sr_name else None
                is_adjoint = bool(attr_dict.get('adjoint', False))
                inputs_val = attr_dict.get('inputs', [])
                if isinstance(inputs_val, str):
                    inputs_val = [inputs_val]
                eq_term = Equation(name, einsum, dom_sort, cod_sort,
                                   sr_term, nonlinearity=nl,
                                   inputs=tuple(inputs_val),
                                   adjoint=is_adjoint)
                equations_by_name[name] = eq_term
                equations_list.append(eq_term)
            case FunctorDecl(name=name, body=body_node, attrs=attrs):
                from unialg.morphism import Functor

                category = attrs.get('category', 'set')
                if category not in ('set', 'poset'):
                    raise ValueError(
                        f"functor {name!r}: category must be 'set' or 'poset', got {category!r}"
                    )
                f = Functor(name, _build_poly(body_node, _get_sort), category=category)
                f.validate()
                functors_by_name[name] = f
            case ShareDecl(name=name, op_names=op_names):
                if name in share_groups:
                    raise ValueError(f"share '{name}': already declared")
                if len(op_names) < 2:
                    raise ValueError(f"share '{name}': must list at least two ops, got {op_names}")
                for op_name in op_names:
                    _get_eq(op_name)
                sorts_seen = [equations_by_name[n].domain_sort for n in op_names]
                first_eq = op_names[0]
                first_sr = sorts_seen[0].semiring_name
                for nm, s in zip(op_names[1:], sorts_seen[1:]):
                    if s.semiring_name != first_sr:
                        raise ValueError(
                            f"share '{name}': ops have incompatible domain algebras "
                            f"('{first_eq}' is {first_sr}, '{nm}' is {s.semiring_name})"
                        )
                share_groups[name] = list(op_names)
            case CellDecl():
                cell_obj = handle_cell(_cell_ctx, decl, _resolve_sort_ref)
                cells.append(cell_obj)
            case _:
                raise ValueError(f"unknown declaration: {decl!r}")

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        defines=defines,
        backend_name=backend_name,
        share_groups=share_groups,
        cells=cells,
    )
