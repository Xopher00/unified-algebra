"""Name resolution pass for parsed .ua declarations.

Resolves algebra/spec/op references, expands template morphisms,
and builds spec objects.  Pure function: raw tuples in, UASpec out.
"""
from __future__ import annotations

from typing import Any

import unialg.algebra as alg
import unialg.assembly.specs as sp
from unialg.parser import UASpec


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

    def _get_sr(name):   return _lookup(name, semirings, 'algebra')
    def _get_sort(name): return _lookup(name, sorts, 'spec')
    def _get_eq(name):   return _lookup(name, equations_by_name, 'op')
    def _get_lens(name): return _lookup(name, lenses_by_name, 'lens')

    def _expand_template_ref(ref):
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
            eq_term = alg.Equation(concrete_name, einsum, dom_sort, cod_sort,
                                  sr_term, nonlinearity=nl)
            equations_by_name[concrete_name] = eq_term
            equations_list.append(eq_term)
        return concrete_name

    _SPEC_CLASSES = {
        'seq': sp.PathSpec, 'branch': sp.FanSpec, 'scan': sp.FoldSpec,
        'unroll': sp.UnfoldSpec, 'fixpoint': sp.FixpointSpec,
        'lens_seq': sp.LensPathSpec, 'lens_branch': sp.LensFanSpec,
    }

    for decl in raw_decls:
        kind = decl[0]

        if kind == 'import':
            backend_name = decl[1]

        elif kind == 'algebra':
            _, name, kw_args = decl
            sr_term = alg.Semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                                   zero=kw_args['zero'], one=kw_args['one'])
            semirings[name] = sr_term

        elif kind == 'spec':
            _, name, sr_name, batched = decl
            sr_term = _get_sr(sr_name)
            sort_term = alg.Sort(name, sr_term, batched=batched)
            sorts[name] = sort_term

        elif kind == 'op':
            _, name, (dom_name, cod_name), attr_dict = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            einsum = attr_dict.get('einsum', None) or None
            nl = attr_dict.get('nonlinearity', None) or None
            sr_name = attr_dict.get('algebra', None)
            sr_term = _get_sr(sr_name) if sr_name else None
            is_template = attr_dict.get('template', False)

            if is_template:
                templates_by_name[name] = (einsum, dom_sort, cod_sort, sr_term, nl)
            else:
                eq_term = alg.Equation(name, einsum, dom_sort, cod_sort,
                                      sr_term,
                                      nonlinearity=nl)
                equations_by_name[name] = eq_term
                equations_list.append(eq_term)

        elif kind == 'lens':
            _, name, (_, _), attrs = decl
            lens_term = alg.Lens(name, attrs['fwd'], attrs['bwd'])
            lenses.append(lens_term)
            lenses_by_name[name] = lens_term

        elif kind in _SPEC_CLASSES:
            specs.append(_SPEC_CLASSES[kind].from_parsed(
                decl, _get_sort, expand_ref=_expand_template_ref, get_lens=_get_lens))

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        specs=specs,
        lenses=lenses,
        backend_name=backend_name,
    )
