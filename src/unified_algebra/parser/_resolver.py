"""Name resolution pass for parsed .ua declarations.

Resolves semiring/sort/equation references, expands template morphisms,
and builds spec objects.  Pure function: raw tuples in, UASpec out.
"""
from __future__ import annotations

from typing import Any

from . import UASpec


def _resolve_spec(raw_decls: list[tuple]) -> UASpec:
    """Second pass: resolve name references, build DSL terms.

    Processes declarations in dependency order:
    1. semirings (no deps)
    2. sorts (depend on semirings by name)
    3. equations (depend on sorts + semirings by name)
    4. compositions (depend on equations and lenses by name)
    """
    from ..algebra.semiring import semiring as mk_semiring
    from ..algebra.sort import sort as mk_sort
    from ..algebra.morphism import equation as mk_equation
    from ..composition.lenses import lens as mk_lens
    from ..specs import PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec, LensPathSpec, LensFanSpec

    semirings: dict[str, Any] = {}
    sorts: dict[str, Any] = {}
    equations_by_name: dict[str, Any] = {}
    equations_list: list[Any] = []
    templates_by_name: dict[str, tuple] = {}
    specs: list[Any] = []
    lenses: list[Any] = []
    lenses_by_name: dict[str, Any] = {}

    def _lookup(name: str, d: dict, label: str) -> Any:
        if name not in d:
            raise ValueError(
                f"Unknown {label} {name!r} — declared {label}s: {list(d)}"
            )
        return d[name]

    def _get_sr(name):   return _lookup(name, semirings, 'semiring')
    def _get_sort(name): return _lookup(name, sorts, 'sort')
    def _get_eq(name):   return _lookup(name, equations_by_name, 'equation')

    def _expand_template_ref(ref):
        """Expand a template ref ('_tpl', name, prefix) into a concrete equation name."""
        if isinstance(ref, str):
            return ref
        _, tpl_name, prefix = ref
        concrete_name = f"{prefix}_{tpl_name}"
        if concrete_name not in equations_by_name:
            if tpl_name not in templates_by_name:
                raise ValueError(
                    f"Unknown template {tpl_name!r} in {tpl_name}[{prefix}] — "
                    f"declared templates: {list(templates_by_name)}"
                )
            einsum, dom_sort, cod_sort, sr_term, nl = templates_by_name[tpl_name]
            eq_term = mk_equation(concrete_name, einsum, dom_sort, cod_sort,
                                  sr_term, nonlinearity=nl)
            equations_by_name[concrete_name] = eq_term
            equations_list.append(eq_term)
        return concrete_name

    for decl in raw_decls:
        kind = decl[0]

        if kind == 'semiring':
            _, name, kw_args = decl
            sr_term = mk_semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                                  zero=kw_args['zero'], one=kw_args['one'])
            semirings[name] = sr_term

        elif kind == 'sort':
            _, name, sr_name, batched = decl
            sr_term = _get_sr(sr_name)
            sort_term = mk_sort(name, sr_term, batched=batched)
            sorts[name] = sort_term

        elif kind == 'equation':
            _, name, (dom_name, cod_name), attr_dict = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            einsum = attr_dict.get('einsum', None) or None
            nl = attr_dict.get('nonlinearity', None) or None
            sr_name = attr_dict.get('semiring', None)
            sr_term = _get_sr(sr_name) if sr_name else None
            is_template = attr_dict.get('template', False)

            if is_template:
                templates_by_name[name] = (einsum, dom_sort, cod_sort, sr_term, nl)
            else:
                eq_term = mk_equation(name, einsum, dom_sort, cod_sort,
                                      sr_term, nonlinearity=nl)
                equations_by_name[name] = eq_term
                equations_list.append(eq_term)

        elif kind == 'path':
            _, name, (dom_name, cod_name), eq_names_raw, attr_dict = decl
            eq_names = [_expand_template_ref(en) for en in eq_names_raw]
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for en in eq_names:
                _get_eq(en)
            residual = attr_dict.get('residual', False)
            residual_sr = attr_dict.get('semiring', None) if residual else None
            specs.append(PathSpec(
                name=name,
                eq_names=eq_names,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
                residual=residual,
                residual_semiring=residual_sr,
            ))

        elif kind == 'fan':
            _, name, (dom_name, cod_name), branches_raw, merge = decl
            branches = [_expand_template_ref(bn) for bn in branches_raw]
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for bn in branches:
                _get_eq(bn)
            _get_eq(merge)
            specs.append(FanSpec(
                name=name,
                branch_names=branches,
                merge_name=merge,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

        elif kind == 'fold':
            _, name, (dom_name, state_name), step = decl
            dom_sort = _get_sort(dom_name)
            state_sort = _get_sort(state_name)
            _get_eq(step)
            specs.append(FoldSpec(
                name=name,
                step_name=step,
                init_term=None,
                domain_sort=dom_sort,
                state_sort=state_sort,
            ))

        elif kind == 'unfold':
            _, name, (dom_name, state_name), step, n_steps = decl
            dom_sort = _get_sort(dom_name)
            state_sort = _get_sort(state_name)
            _get_eq(step)
            specs.append(UnfoldSpec(
                name=name,
                step_name=step,
                n_steps=n_steps,
                domain_sort=dom_sort,
                state_sort=state_sort,
            ))

        elif kind == 'fixpoint':
            _, name, sort_name, attr_dict = decl
            dom_sort = _get_sort(sort_name)
            step = attr_dict.get('step')
            predicate = attr_dict.get('predicate')
            epsilon = float(attr_dict.get('epsilon', 1e-6))
            max_iter = int(attr_dict.get('max_iter', 100))
            if step:
                _get_eq(step)
            if predicate:
                _get_eq(predicate)
            specs.append(FixpointSpec(
                name=name,
                step_name=step,
                predicate_name=predicate,
                epsilon=epsilon,
                max_iter=max_iter,
                domain_sort=dom_sort,
            ))

        elif kind == 'lens':
            _, name, (_, _), fwd, bwd = decl
            _get_eq(fwd)
            _get_eq(bwd)
            lens_term = mk_lens(name, fwd, bwd)
            lenses.append(lens_term)
            lenses_by_name[name] = lens_term

        elif kind == 'lens_path':
            _, name, (dom_name, cod_name), lens_names = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for ln in lens_names:
                if ln not in lenses_by_name:
                    raise ValueError(
                        f"Unknown lens {ln!r} — declared lenses: {list(lenses_by_name)}"
                    )
            specs.append(LensPathSpec(
                name=name,
                lens_names=lens_names,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

        elif kind == 'lens_fan':
            _, name, (dom_name, cod_name), lens_names, merge_lens = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for ln in lens_names:
                if ln not in lenses_by_name:
                    raise ValueError(
                        f"Unknown lens {ln!r} — declared lenses: {list(lenses_by_name)}"
                    )
            if merge_lens not in lenses_by_name:
                raise ValueError(
                    f"Unknown merge lens {merge_lens!r} — declared lenses: {list(lenses_by_name)}"
                )
            specs.append(LensFanSpec(
                name=name,
                lens_names=lens_names,
                merge_lens_name=merge_lens,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        specs=specs,
        lenses=lenses,
    )
