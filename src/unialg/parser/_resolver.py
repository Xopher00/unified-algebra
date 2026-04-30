"""Name resolution pass for parsed .ua declarations.

Walks the raw declaration list from the grammar and produces a UASpec.
Compositions are expressed as `cell` declarations using the operator
sub-grammar; `share` records 2-cell weight-tying groups; `functor`
declares polynomial endofunctors used by ``cata``/``ana`` cells.
"""
from __future__ import annotations

from typing import Any

import unialg.algebra as alg
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
            from unialg.algebra.sort import ProductSort
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
        sr_term = alg.Semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                               zero=kw_args['zero'], one=kw_args['one'],
                               contraction=kw_args.get('strategy') or kw_args.get('contraction', ''),
                               residual=kw_args.get('residual', ''),
                               leq=kw_args.get('leq', ''))
        semirings[name] = sr_term

    def _handle_spec(decl):
        _, name, sr_name, batched, axes = decl
        sr_term = _get_sr(sr_name)
        sort_term = alg.Sort(name, sr_term, batched=batched, axes=axes)
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
        eq_term = alg.Equation(name, einsum, dom_sort, cod_sort,
                               sr_term, nonlinearity=nl,
                               inputs=tuple(inputs_val),
                               adjoint=is_adjoint)
        equations_by_name[name] = eq_term
        equations_list.append(eq_term)

    def _handle_functor(decl):
        from unialg.assembly.functor import (
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
        from unialg.assembly import morphism
        from unialg.assembly.para._para_graph import NamedCell
        import hydra.core as core

        def _contains_legacy_only(node) -> bool:
            tag = node[0]
            if tag in {"cell_seq", "cell_par"}:
                return _contains_legacy_only(node[1]) or _contains_legacy_only(node[2])
            if tag in {"cell_cata", "cell_ana"}:
                return any(_contains_legacy_only(a) for a in node[2])
            return False

        def _literal(node):
            return core.TermLiteral(value=core.LiteralFloat(value=float(node[1])))

        def _build_cell_legacy(node):
            from unialg.assembly.para._para import (
                eq as cell_eq,
                lit as cell_lit,
                iden as cell_iden,
                seq as cell_seq,
                par as cell_par,
                copy as cell_copy,
                delete as cell_delete,
                lens as cell_lens,
                algebra_hom as cell_algebra_hom,
            )
            tag = node[0]
            if tag == 'cell_eq':
                return cell_eq(node[1])
            if tag == 'cell_lit':
                return cell_lit(_literal(node))
            if tag == 'cell_copy':
                return cell_copy(_get_sort(node[1]))
            if tag == 'cell_delete':
                return cell_delete(_get_sort(node[1]))
            if tag == 'cell_iden':
                return cell_iden(_get_sort(node[1]))
            if tag == 'cell_seq':
                return cell_seq(_build_cell_legacy(node[1]), _build_cell_legacy(node[2]))
            if tag == 'cell_par':
                return cell_par(_build_cell_legacy(node[1]), _build_cell_legacy(node[2]))
            if tag == 'cell_lens':
                _, fwd_node, bwd_node, residual_name = node
                fwd, bwd = _build_cell_legacy(fwd_node), _build_cell_legacy(bwd_node)
                residual = _get_sort(residual_name) if residual_name else None
                return cell_lens(fwd, bwd, residual=residual)
            if tag == 'cell_cata':
                _, f_name, arg_nodes = node
                if f_name not in functors_by_name:
                    raise ValueError(
                        f"cell cata: unknown functor {f_name!r} — declare via 'functor'"
                    )
                return cell_algebra_hom(functors_by_name[f_name], "algebra",
                                        [_build_cell_legacy(a) for a in arg_nodes])
            if tag == 'cell_ana':
                _, f_name, arg_nodes = node
                if f_name not in functors_by_name:
                    raise ValueError(
                        f"cell ana: unknown functor {f_name!r} — declare via 'functor'"
                    )
                return cell_algebra_hom(functors_by_name[f_name], "coalgebra",
                                        [_build_cell_legacy(a) for a in arg_nodes])
            raise ValueError(f"cell: unknown AST tag {tag!r}")

        _, name, sig, expr_node = decl
        cell_codomain = _resolve_sort_ref(sig[1])

        def _build_typed(node):
            tag = node[0]
            if tag == 'cell_eq':
                if node[1] not in equations_by_name:
                    raise ValueError(f"unknown equation {node[1]!r}")
                eq = _get_eq(node[1])
                return morphism.eq(
                    node[1],
                    domain=eq.domain_sort,
                    codomain=eq.codomain_sort,
                )
            if tag == 'cell_lit':
                return morphism.lit(_literal(node), cell_codomain)
            if tag == 'cell_copy':
                return morphism.copy(_get_sort(node[1]))
            if tag == 'cell_delete':
                return morphism.delete(_get_sort(node[1]))
            if tag == 'cell_iden':
                return morphism.iden(_get_sort(node[1]))
            if tag == 'cell_seq':
                if _contains_legacy_only(node):
                    return _build_cell_legacy(node)
                return morphism.seq(_build_typed(node[1]), _build_typed(node[2]))
            if tag == 'cell_par':
                if _contains_legacy_only(node):
                    return _build_cell_legacy(node)
                return morphism.par(_build_typed(node[1]), _build_typed(node[2]))
            if tag in {'cell_lens', 'cell_cata', 'cell_ana'}:
                if tag == 'cell_lens':
                    _, fwd_node, bwd_node, residual_name = node
                    if residual_name is not None:
                        _get_sort(residual_name)
                    return morphism.lens(_build_typed(fwd_node), _build_typed(bwd_node))
                _, f_name, arg_nodes = node
                if f_name not in functors_by_name:
                    label = "cata" if tag == "cell_cata" else "ana"
                    raise ValueError(
                        f"cell {label}: unknown functor {f_name!r} — declare via 'functor'"
                    )
                direction = "algebra" if tag == "cell_cata" else "coalgebra"
                return morphism.algebra_hom(
                    functors_by_name[f_name],
                    direction,
                    [_build_typed(a) for a in arg_nodes],
                )
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
