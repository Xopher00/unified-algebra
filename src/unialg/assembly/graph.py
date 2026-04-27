"""Graph assembly: equation resolution, validation, composition, and Hydra Graph construction."""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import TYPE_CHECKING

import hydra.core as core
import hydra.graph
from hydra.dsl.python import FrozenDict, Nothing
from hydra.lexical import empty_graph
from hydra.sources.libraries import standard_library
from hydra.typing import TypeConstraint
import hydra.substitution as subst
import hydra.typing

from unialg.algebra.equation import Equation
from unialg.terms import unify_or_raise

if TYPE_CHECKING:
    from unialg.backend import Backend


def topo_edges(equations: list) -> list:
    """Return (upstream, downstream, slot) triples in topological order."""
    by_name = {eq.name: eq for eq in equations}
    edges, in_degree, children = [], {eq.name: 0 for eq in equations}, {eq.name: [] for eq in equations}
    for eq in equations:
        for slot, inp in enumerate(eq.inputs):
            if inp in by_name:
                edges.append((by_name[inp], eq, slot))
                children[inp].append(eq.name)
                in_degree[eq.name] += 1
    queue = deque(n for n, d in in_degree.items() if d == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != len(equations):
        raise ValueError("Cycle detected in equation DAG")
    return edges


def _build_schema(equations: list) -> FrozenDict:
    schema: dict = {}
    for eq in equations:
        eq.register_sorts(schema)
    return FrozenDict(schema)


def validate_pipeline(equations: list, schema_types=None) -> None:
    """Check sort and rank junctions across all equations."""
    if schema_types is None:
        schema_types = _build_schema(equations)
    cs = []
    for u, d, slot in topo_edges(equations):
        cs.append(TypeConstraint(
            u.codomain_sort.type_,
            d.domain_sort.type_,
            f"'{u.name}' codomain != '{d.name}' domain",
        ))
        out_rank = u.output_rank
        in_rank = d.input_rank(slot)
        if out_rank is not None and in_rank is not None and out_rank != in_rank:
            raise TypeError(
                f"Rank mismatch: '{u.name}' output rank {out_rank} != "
                f"'{d.name}' input rank {in_rank} at slot {slot}")
        cod_axes = u.codomain_sort.axes
        dom_axes = d.domain_sort.axes
        if cod_axes and dom_axes:
            cod_names = u.codomain_sort.axis_names
            dom_names = d.domain_sort.axis_names
            if tuple(cod_names) != tuple(dom_names):
                raise TypeError(
                    f"Axis mismatch: '{u.name}' codomain axes {cod_names} != "
                    f"'{d.name}' domain axes {dom_names}")
            for i, (cd, dd) in enumerate(zip(u.codomain_sort.axis_dims,
                                              d.domain_sort.axis_dims)):
                if cd is not None and dd is not None and cd != dd:
                    raise TypeError(
                        f"Dimension mismatch: '{u.name}' axis '{cod_names[i]}' "
                        f"size {cd} != '{d.name}' size {dd}")
    unify_or_raise(cs, schema_types)


def _resolve_equations(eq_terms, backend, merge_names, semirings):
    """Parse, dedupe, and resolve equations. Returns (eq_by_name, primitives, native_fns, resolved_semirings, coder, schema_types)."""
    eq_by_name: dict[str, Equation] = {}
    for i, t in enumerate(eq_terms):
        eq = Equation.from_term(t)
        if eq.name in eq_by_name:
            raise ValueError(f"Duplicate equation name '{eq.name}' (positions {list(eq_by_name).index(eq.name)} and {i})")
        eq_by_name[eq.name] = eq

    primitives: dict = dict(standard_library())
    native_fns: dict = {}
    coder = None
    resolved_semirings = Equation.resolve_semirings(semirings, backend) if semirings else {}

    schema: dict = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
        prim, native_fn, sr, eq_coder = eq.resolve(backend)
        sr_name = eq.semiring_name
        if sr_name and sr is not None:
            resolved_semirings.setdefault(sr_name, sr)
        if coder is None:
            coder = eq_coder
        primitives[prim.name] = prim
        native_fns[prim.name] = native_fn
        if eq.name in merge_names:
            merge_key = core.Name(f"ua.equation.{eq.name}.__merge__")
            merge_prim, merge_fn, _, _ = eq.resolve_as_merge(backend, prim_name_override=merge_key)
            primitives[merge_prim.name] = merge_prim
            native_fns[merge_prim.name] = merge_fn

    return eq_by_name, primitives, native_fns, resolved_semirings, coder, schema


def _build_compositions(specs, eq_by_name, primitives, native_fns, bound_terms, schema_types, coder, backend, **kwargs):
    from hydra.dsl.prims import prim1
    from unialg.assembly.compositions import Composition
    compiled_fns = {}
    for spec in specs:
        spec.validate(eq_by_name, schema_types)
        first_term = None
        for entry in spec.build(primitives, native_fns, coder=coder, **kwargs):
            if isinstance(entry, Composition):
                prim, fn = entry.resolve(native_fns, coder, backend, bound_terms)
                if prim is not None:
                    primitives[prim.name] = prim
                    compiled_fns[spec.name] = fn
                else:
                    name, term = entry.to_lambda()
                    bound_terms[name] = term
                    if fn is not None:
                        compiled_fns[spec.name] = fn
                    if first_term is None:
                        first_term = term
            else:
                name, term = entry
                bound_terms[name] = term
        eq_key = core.Name(f"ua.equation.{spec.name}")
        if spec.name in compiled_fns:
            alias_prim = prim1(eq_key, compiled_fns[spec.name], [], coder, coder)
            primitives[eq_key] = alias_prim
            native_fns[eq_key] = compiled_fns[spec.name]
        elif first_term is not None:
            bound_terms[eq_key] = first_term
        eq_by_name[spec.name] = spec
    return compiled_fns


def build_graph(sort_terms, primitives=None, bound_terms=None):
    tensor_name = core.Name("ua.tensor.NDArray")
    schema = {tensor_name: core.TypeScheme((), core.TypeVariable(tensor_name), Nothing())}
    for st in sort_terms:
        st.register_schema(schema)
    return dataclasses.replace(
        empty_graph(),
        bound_terms=FrozenDict(bound_terms or {}),
        primitives=FrozenDict(primitives or {}),
        schema_types=FrozenDict(schema),
    )


def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
    specs: list | None = None,
    hyperparams: dict[str, core.Term] | None = None,
    lenses: list[core.Term] | None = None,
    semirings: dict[str, core.Term] | None = None,
) -> tuple[hydra.graph.Graph, dict, dict]:
    """Resolve equations, assemble a Hydra Graph, and compile compositions."""
    all_specs = list(specs or [])
    merge_names = set()
    for spec in all_specs:
        if hasattr(spec, 'merge_names'):
            merge_names.update(spec.merge_names)

    eq_by_name, primitives, native_fns, resolved_semirings, coder, schema = \
        _resolve_equations(eq_terms, backend, merge_names, semirings)

    for st in (extra_sorts or []):
        if st is not None:
            st.register_schema(schema)
    schema_types = FrozenDict(schema)
    validate_pipeline(list(eq_by_name.values()), schema_types)

    bound_terms: dict[core.Name, core.Term] = {}
    if hyperparams:
        for param_name, param_term in hyperparams.items():
            bound_terms[core.Name(f"ua.param.{param_name}")] = param_term

    compiled_fns = _build_compositions(
        all_specs, eq_by_name, primitives, native_fns, bound_terms, schema_types,
        coder=coder, backend=backend, resolved_semirings=resolved_semirings)

    for eq_name in eq_by_name:
        fn = native_fns.get(core.Name(f"ua.equation.{eq_name}"))
        if fn is not None:
            compiled_fns[eq_name] = fn

    seen_sorts: dict[str, core.Term] = {}
    for eq in eq_by_name.values():
        for attr in ("domain_sort", "codomain_sort", "state_sort"):
            st = getattr(eq, attr, None)
            if st is not None:
                seen_sorts.setdefault(str(st.type_), st)
    graph = build_graph(list(seen_sorts.values()) + list(extra_sorts or []),
                        primitives=primitives, bound_terms=bound_terms)
    return graph, native_fns, compiled_fns


def rebind_hyperparams(graph, updates):
    param_updates = {core.Name(f"ua.param.{k}"): v for k, v in updates.items()}
    ts = hydra.typing.TermSubst(FrozenDict(param_updates))
    new_terms = {name: subst.substitute_in_term(ts, term) for name, term in graph.bound_terms.items()}
    new_terms.update(param_updates)
    return dataclasses.replace(graph, bound_terms=FrozenDict(new_terms))
