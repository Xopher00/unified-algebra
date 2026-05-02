"""Pipeline validation: topology, schema construction, type-constraint checking."""

from collections import deque

from hydra.dsl.python import FrozenDict, Left
from hydra.lexical import empty_context
from hydra.typing import TypeConstraint
from hydra.unification import unify_type_constraints

_EMPTY_CX = empty_context()


def unify_or_raise(constraints, schema):
    if constraints:
        result = unify_type_constraints(_EMPTY_CX, schema, tuple(constraints))
        if isinstance(result, Left):
            raise TypeError(result.value.message)


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
