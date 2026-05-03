"""Graph assembly: equation resolution, validation, composition, and Hydra Graph construction."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import hydra.core as core
import hydra.graph
import hydra.typing
import hydra.substitution as subst
from hydra.dsl.python import FrozenDict, Nothing
from hydra.lexical import elements_to_graph, graph_with_primitives
from hydra.sources.libraries import standard_library

from unialg.terms import register_tensor_schema
from ._morphism_compile import register_cells
from ._equation_resolution import _resolve_equations

if TYPE_CHECKING:
    from unialg.backend import Backend


def build_graph(sort_terms, primitives=None, bound_terms=None):
    schema = {}
    register_tensor_schema(schema)
    for st in sort_terms:
        st.register_schema(schema)
    schema_types = FrozenDict(schema)
    parent = graph_with_primitives(
        tuple(standard_library().values()),
        tuple((primitives or {}).values()),
    )
    bindings = tuple(
        core.Binding(name, term, Nothing())
        for name, term in (bound_terms or {}).items()
    )
    return elements_to_graph(parent, schema_types, bindings)


def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
    params: dict[str, core.Term] | None = None,
    semirings: dict[str, core.Term] | None = None,
    cells: list | None = None,
) -> tuple[hydra.graph.Graph, dict, dict]:
    """Resolve equations, assemble a Hydra Graph, and register morphism cells.

    ``cells`` is a list of ``NamedCell`` entries whose morphism terms are
    compiled to Hydra primitives or bound_terms. Lenses produce two primitives
    (forward + backward).
    """
    eq_by_name, primitives, native_fns, coder, schema_types, list_packed_info = \
        _resolve_equations(eq_terms, backend, semirings, extra_sorts=extra_sorts)

    bound_terms: dict[core.Name, core.Term] = {}
    if params:
        for param_name, param_term in params.items():
            if param_term is not None:
                bound_terms[core.Name(f"ua.param.{param_name}")] = param_term

    seen_sorts: dict[str, core.Term] = {}
    for eq in eq_by_name.values():
        for attr in ("domain_sort", "codomain_sort", "state_sort"):
            st = getattr(eq, attr, None)
            if st is not None:
                seen_sorts.setdefault(str(st.type_), st)
    sort_list = list(seen_sorts.values()) + list(extra_sorts or [])

    if cells:
        preliminary = build_graph(sort_list, primitives=primitives, bound_terms=bound_terms)
        register_cells(cells, preliminary, bound_terms, primitives, native_fns, coder, backend)

    # Final graph captures the post-registration dict state into FrozenDicts.
    graph = build_graph(sort_list, primitives=primitives, bound_terms=bound_terms)
    return graph, native_fns, list_packed_info


def rebind_params(graph, updates):
    param_updates = {core.Name(f"ua.param.{k}"): v for k, v in updates.items()}
    ts = hydra.typing.TermSubst(FrozenDict(param_updates))
    new_terms = {name: subst.substitute_in_term(ts, term) for name, term in graph.bound_terms.items()}
    new_terms.update(param_updates)
    return dataclasses.replace(graph, bound_terms=FrozenDict(new_terms))
