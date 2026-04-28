"""Name resolution pass for parsed .ua declarations.

Resolves algebra/spec/op references, expands template morphisms,
and builds spec objects.  Pure function: raw tuples in, UASpec out.
"""
from __future__ import annotations

from typing import Any

import unialg.algebra as alg
import unialg.assembly.specs as sp
from . import UASpec

_SPEC_CLASSES = {
    'seq': sp.PathSpec, 'branch': sp.FanSpec, 'scan': sp.FoldSpec,
    'unroll': sp.UnfoldSpec, 'fixpoint': sp.FixpointSpec,
    'lens_seq': sp.PathSpec, 'lens_branch': sp.FanSpec,
    'parallel': sp.ParallelSpec,
}


def _resolve_spec(raw_decls: list[tuple]) -> UASpec:
    """Second pass: resolve name references, build DSL terms.

    Processes declarations in dependency order:
    1. semirings (no deps)
    2. sorts (depend on semirings by name)
    3. equations (depend on sorts + semirings by name)
    4. compositions (depend on equations and lenses by name)
    """

    backend_name: str | None = None
    semirings: dict[str, Any] = {}
    sorts: dict[str, Any] = {}
    defines: list[tuple] = []
    equations_by_name: dict[str, Any] = {}
    equations_list: list[Any] = []
    templates_by_name: dict[str, tuple] = {}
    specs: list[Any] = []
    lenses: list[Any] = []
    lenses_by_name: dict[str, Any] = {}

    # --- lookup helpers (closures over shared dicts) ---

    def _lookup(name: str, d: dict, label: str) -> Any:
        if name not in d:
            raise ValueError(
                f"Unknown {label} {name!r} — declared {label}s: {list(d)}"
            )
        return d[name]

    def _get_sr(name):   return _lookup(name, semirings, 'algebra')
    def _get_sort(name): return _lookup(name, sorts, 'spec')
    def _get_eq(name):   return _lookup(name, equations_by_name, 'op')
    def _get_lens(name): return _lookup(name, lenses_by_name, 'lens')

    def _resolve_sort_ref(ref):
        if isinstance(ref, str):
            return _get_sort(ref)
        if isinstance(ref, tuple) and ref[0] == '_product':
            from unialg.algebra.sort import ProductSort
            return ProductSort([_get_sort(n) for n in ref[1]])
        raise ValueError(f"Invalid sort reference: {ref}")

    _fresh_counters: dict[str, int] = {}
    compositions_by_name: dict[str, object] = {}

    def _clone_eq(base_name, new_name, *, adjoint=None, skip=None):
        """Create a named copy of an equation, optionally overriding adjoint/skip."""
        if new_name not in equations_by_name:
            base_eq = _get_eq(base_name)
            copy_eq = alg.Equation(
                new_name, base_eq.einsum, base_eq.domain_sort, base_eq.codomain_sort,
                base_eq.semiring, nonlinearity=base_eq.nonlinearity,
                inputs=tuple(base_eq.inputs),
                adjoint=base_eq.adjoint if adjoint is None else adjoint,
                skip=base_eq.skip if skip is None else skip)
            equations_by_name[new_name] = copy_eq
            equations_list.append(copy_eq)

    def _clone_composition(comp, new_name, suffix, *, residual=False):
        """Shallow-clone a PathSpec or FanSpec: fresh equation copies, new composition name."""
        if isinstance(comp, sp.PathSpec):
            new_eq_names = []
            for eq_name in comp.eq_names:
                if eq_name in equations_by_name:
                    fresh = f"{eq_name}__{suffix}"
                    _clone_eq(eq_name, fresh)
                    new_eq_names.append(fresh)
                else:
                    new_eq_names.append(eq_name)
            infer_sr = None
            if residual:
                for n in new_eq_names:
                    eq = equations_by_name.get(n)
                    if eq and eq.semiring:
                        infer_sr = eq.semiring_name
                        break
            clone = sp.PathSpec(
                name=new_name, eq_names=new_eq_names,
                domain_sort=comp.domain_sort, codomain_sort=comp.codomain_sort,
                residual=residual or comp.residual,
                residual_semiring=infer_sr or comp.residual_semiring or "")
            specs.append(clone)
            compositions_by_name[new_name] = clone
            return new_name
        if isinstance(comp, sp.FanSpec):
            new_branches = []
            for br in comp.branches:
                if br in equations_by_name:
                    fresh = f"{br}__{suffix}"
                    _clone_eq(br, fresh)
                    new_branches.append(fresh)
                else:
                    new_branches.append(br)
            clone = sp.FanSpec(
                name=new_name, branches=new_branches,
                merge_names=list(comp.merge_names),
                domain_sort=comp.domain_sort, codomain_sort=comp.codomain_sort)
            specs.append(clone)
            compositions_by_name[new_name] = clone
            return new_name
        raise ValueError(
            f"~{comp.name}: fresh copies are supported for seq and branch compositions only")

    def _expand_template_ref(ref):
        if isinstance(ref, str):
            return ref
        tag = ref[0]
        if tag == '_fresh':
            _, inner = ref
            base_name = _expand_template_ref(inner)
            n = _fresh_counters.get(base_name, 0)
            _fresh_counters[base_name] = n + 1
            fresh_name = f"{base_name}__{n}"
            if base_name in compositions_by_name:
                if fresh_name not in compositions_by_name:
                    _clone_composition(compositions_by_name[base_name], fresh_name, n)
            else:
                _clone_eq(base_name, fresh_name)
            return fresh_name
        if tag == '_adj':
            _, inner = ref
            base_name = _expand_template_ref(inner)
            adj_name = f"{base_name}__adj"
            if adj_name not in equations_by_name:
                _clone_eq(base_name, adj_name, adjoint=True)
            return adj_name
        if tag == '_res':
            _, inner = ref
            base_name = _expand_template_ref(inner)
            res_name = f"{base_name}__res"
            if base_name in compositions_by_name:
                if res_name not in compositions_by_name:
                    _clone_composition(compositions_by_name[base_name], res_name, 'res',
                                       residual=True)
            else:
                if res_name not in equations_by_name:
                    _clone_eq(base_name, res_name, skip=True)
            return res_name
        _, tpl_name, prefix = ref
        concrete_name = f"{prefix}_{tpl_name}"
        if concrete_name not in equations_by_name:
            if tpl_name not in templates_by_name:
                raise ValueError(
                    f"Unknown template {tpl_name!r} in {tpl_name}[{prefix}] — "
                    f"declared templates: {list(templates_by_name)}"
                )
            einsum, dom_sort, cod_sort, sr_term, nl = templates_by_name[tpl_name]
            eq_term = alg.Equation(concrete_name, einsum, dom_sort, cod_sort,
                                  sr_term, nonlinearity=nl)
            equations_by_name[concrete_name] = eq_term
            equations_list.append(eq_term)
        return concrete_name

    # --- per-kind handlers (closures; mutate shared state) ---

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
        is_template = attr_dict.get('template', False)
        is_adjoint = bool(attr_dict.get('adjoint', False))
        inputs_val = attr_dict.get('inputs', [])
        if isinstance(inputs_val, str):
            inputs_val = [inputs_val]

        if is_template:
            templates_by_name[name] = (einsum, dom_sort, cod_sort, sr_term, nl)
        else:
            eq_term = alg.Equation(name, einsum, dom_sort, cod_sort,
                                  sr_term, nonlinearity=nl,
                                  inputs=tuple(inputs_val),
                                  adjoint=is_adjoint)
            equations_by_name[name] = eq_term
            equations_list.append(eq_term)

    def _handle_lens(decl):
        _, name, (_, _), attrs = decl
        res_name = attrs.get('residual')
        res_sort = _get_sort(res_name) if res_name else None
        lens_term = alg.Lens(name, attrs['fwd'], attrs['bwd'],
                             residual_sort=res_sort)
        lenses.append(lens_term)
        lenses_by_name[name] = lens_term

    def _handle_composition(kind, decl):
        kw = dict(expand_ref=_expand_template_ref)
        if kind.startswith('lens_'):
            kw['get_lens'] = _get_lens
        spec = _SPEC_CLASSES[kind].from_parsed(decl, _get_sort, **kw)
        specs.append(spec)
        compositions_by_name[spec.name] = spec

    # --- dispatch loop ---

    _handlers = {
        'define':  _handle_define,
        'import':  _handle_import,
        'algebra': _handle_algebra,
        'spec':    _handle_spec,
        'op':      _handle_op,
        'lens':    _handle_lens,
    }

    for decl in raw_decls:
        kind = decl[0]
        if kind in _handlers:
            _handlers[kind](decl)
        elif kind in _SPEC_CLASSES:
            _handle_composition(kind, decl)

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        specs=specs,
        lenses=lenses,
        defines=defines,
        backend_name=backend_name,
    )
