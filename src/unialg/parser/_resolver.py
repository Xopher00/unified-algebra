"""Name resolution pass for parsed .ua declarations.

Walks the raw declaration list from the grammar and produces a UASpec.
Compositions are expressed as `cell` declarations using the operator
sub-grammar; `share` records 2-cell weight-tying groups; `functor`
declares polynomial endofunctors used by ``cata``/``ana`` cells.
"""
from __future__ import annotations

from typing import Any

import hydra.core as core

from unialg.algebra import Sort, ProductSort, Semiring, Equation, register_defines
from ._types import NamedCell, UASpec
from unialg.morphism import algebra_hom, summand_domain, lens
from unialg.morphism import TypedMorphism as T
from unialg.morphism import eq, lit, iden, copy, delete, seq, par

from . import UASpec


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

    def _handle_define(decl):
        _, name, arity, params, expr_ast = decl
        defines.append((arity, name, params, expr_ast))

    def _handle_import(decl):
        nonlocal backend_name
        backend_name = decl[1]

    def _handle_algebra(decl):
        _, name, kw_args = decl
        sr_term = Semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                               zero=kw_args['zero'], one=kw_args['one'],
                               contraction=kw_args.get('strategy') or kw_args.get('contraction', ''),
                               residual=kw_args.get('residual', ''),
                               leq=kw_args.get('leq', ''))
        semirings[name] = sr_term

    def _handle_spec(decl):
        _, name, sr_name, batched, axes = decl
        sr_term = _get_sr(sr_name)
        sort_term = Sort(name, sr_term, batched=batched, axes=axes)
        sorts[name] = sort_term

    def _handle_op(decl):
        _, name, (dom_name, cod_name), attr_dict = decl
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

    def _handle_functor(decl):
        from unialg.morphism.functor import (
            Functor, sum_, prod, one, zero, id_, const,
        )
        _, name, body_node, attrs = decl

        def _build_poly(node):
            tag = node[0]
            if tag == 'poly_zero':
                return zero()
            if tag == 'poly_one':
                return one()
            if tag == 'poly_id':
                return id_()
            if tag == 'poly_const':
                return const(_get_sort(node[1]))
            if tag == 'poly_sum':
                return sum_(_build_poly(node[1]), _build_poly(node[2]))
            if tag == 'poly_prod':
                return prod(_build_poly(node[1]), _build_poly(node[2]))
            if tag == 'poly_compose':
                raise NotImplementedError(
                    "functor composition (@) not yet supported"
                )
            raise ValueError(f"functor: unknown polynomial AST tag {tag!r}")

        category = attrs.get('category', 'set')
        if category not in ('set', 'poset'):
            raise ValueError(
                f"functor {name!r}: category must be 'set' or 'poset', got {category!r}"
            )
        f = Functor(name, _build_poly(body_node), category=category)
        f.validate()
        functors_by_name[name] = f

    def _handle_cell(decl):

        def _literal(node):
            return core.TermLiteral(value=core.LiteralFloat(value=float(node[1])))

        _, name, sig, expr_node = decl
        cell_codomain = _resolve_sort_ref(sig[1])

        def _split_modifiers(eq_name: str) -> tuple[str, str]:
            i = len(eq_name)
            while i > 0 and eq_name[i - 1] in "'?":
                i -= 1
            return eq_name[:i], eq_name[i:]

        def _ensure_adjoint_eq(base_name: str):
            base = _get_eq(base_name)
            if not base.einsum:
                raise NotImplementedError(
                    f"equation modifier \"'\" on {base_name!r}: adjoint "
                    "references require an einsum-backed op"
                )
            adjoint_name = f"{base_name}__adjoint"
            if adjoint_name not in equations_by_name:
                eq = Equation(
                    adjoint_name,
                    base.einsum,
                    base.domain_sort,
                    base.codomain_sort,
                    base.semiring,
                    nonlinearity=base.nonlinearity,
                    inputs=tuple(base.inputs),
                    param_slots=tuple(base.param_slots),
                    adjoint=True,
                    skip=base.skip,
                )
                equations_by_name[adjoint_name] = eq
                equations_list.append(eq)
            return adjoint_name

        def _resolve_modified_eq(eq_name: str):
            base_name, modifiers = _split_modifiers(eq_name)
            if not base_name:
                raise ValueError(f"invalid equation reference {eq_name!r}")
            if base_name not in equations_by_name:
                raise ValueError(f"unknown equation {base_name!r}")
            resolved_name = base_name
            for modifier in modifiers:
                if modifier == "'":
                    resolved_name = _ensure_adjoint_eq(resolved_name)
                elif modifier == "?":
                    raise NotImplementedError(
                        f"equation modifier '?' on {resolved_name!r}: masked "
                        "references are parsed but not implemented"
                    )
                else:
                    raise ValueError(f"unsupported equation modifier {modifier!r}")
            return resolved_name, _get_eq(resolved_name)

        def _build_typed(node):
            tag = node[0]
            if tag == 'cell_eq':
                resolved_name, equation = _resolve_modified_eq(node[1])
                return eq(
                    resolved_name,
                    domain=equation.domain_sort,
                    codomain=equation.codomain_sort,
                )
            if tag == 'cell_lit':
                return lit(_literal(node), cell_codomain)
            if tag == 'cell_copy':
                return copy(_get_sort(node[1]))
            if tag == 'cell_delete':
                return delete(_get_sort(node[1]))
            if tag == 'cell_iden':
                return iden(_get_sort(node[1]))
            if tag == 'cell_seq':
                return seq(_build_typed(node[1]), _build_typed(node[2]))
            if tag == 'cell_par':
                return par(_build_typed(node[1]), _build_typed(node[2]))
            if tag in {'cell_lens', 'cell_cata', 'cell_ana'}:
                if tag == 'cell_lens':
                    _, fwd_node, bwd_node, residual_name = node
                    if residual_name is not None:
                        _get_sort(residual_name)
                    return lens(_build_typed(fwd_node), _build_typed(bwd_node))
                _, f_name, arg_nodes = node
                if f_name not in functors_by_name:
                    label = "cata" if tag == "cell_cata" else "ana"
                    raise ValueError(
                        f"cell {label}: unknown functor {f_name!r} — declare via 'functor'"
                    )
                functor = functors_by_name[f_name]
                direction = "algebra" if tag == "cell_cata" else "coalgebra"
                if direction == "algebra":
                    summands = functor.summands()
                    morphisms = []
                    for arg_node, summand in zip(arg_nodes, summands):
                        m = _build_typed(arg_node)
                        dom = summand_domain(summand, cell_codomain)
                        morphisms.append(T(m.term, dom, cell_codomain))
                else:
                    morphisms = [_build_typed(a) for a in arg_nodes]
                return algebra_hom(functor, direction, morphisms)
            raise ValueError(f"cell: unknown AST tag {tag!r}")

        cell_obj = _build_typed(expr_node)
        cells.append(NamedCell(name=name, cell=cell_obj))

    def _handle_share(decl):
        _, name, op_names = decl
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

    _handlers = {
        'define':  _handle_define,
        'import':  _handle_import,
        'algebra': _handle_algebra,
        'spec':    _handle_spec,
        'op':      _handle_op,
        'functor': _handle_functor,
        'share':   _handle_share,
        'cell':    _handle_cell,
    }

    for decl in raw_decls:
        kind = decl[0]
        if kind in _handlers:
            _handlers[kind](decl)
        else:
            raise ValueError(f"Unknown declaration kind: {kind!r}")

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        defines=defines,
        backend_name=backend_name,
        share_groups=share_groups,
        cells=cells,
    )
