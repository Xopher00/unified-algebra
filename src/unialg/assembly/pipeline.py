"""Unified assembly pipeline: schema registration, equation resolution, and validation.

Replaces the separate topology and resolution modules with a single pass over
eq_terms that builds schema_types, primitives, and eq_by_name together.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Left
from hydra.dsl.prims import prim1, prim2, prim3, float32 as float32_coder, list_ as list_coder
from hydra.sources.libraries import standard_library
from hydra.typing import TypeConstraint
from hydra.unification import unify_type_constraints

from unialg.algebra.equation import Equation
from unialg.algebra.sort import sort_wrap
from unialg.algebra.contraction import contract_and_apply, contract_merge

if TYPE_CHECKING:
    from unialg.backend import Backend

_CX = Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unify(constraints: list, schema: dict) -> None:
    if constraints:
        result = unify_type_constraints(_CX, FrozenDict(schema) if not isinstance(schema, FrozenDict) else schema, tuple(constraints))
        if isinstance(result, Left):
            raise TypeError(result.value.message)


def topo_edges(eq_terms: list) -> list:
    """Return (upstream_term, downstream_term, slot) triples in topological order."""
    parsed = [(Equation.from_term(t), t) for t in eq_terms]
    by_name = {e.name: t for e, t in parsed}
    edges, in_degree, children = [], {e.name: 0 for e, _ in parsed}, {e.name: [] for e, _ in parsed}
    for eq_obj, raw in parsed:
        for slot, inp in enumerate(eq_obj.inputs):
            if inp in by_name:
                edges.append((by_name[inp], raw, slot))
                children[inp].append(eq_obj.name)
                in_degree[eq_obj.name] += 1
    queue = deque(n for n, d in in_degree.items() if d == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != len(eq_terms):
        raise ValueError("Cycle detected in equation DAG")
    return edges


# ---------------------------------------------------------------------------
# Standalone validation functions (called directly from tests)
# ---------------------------------------------------------------------------

def validate_pipeline(eq_terms: list, schema_types=None) -> None:
    """Check sort and rank junctions across all equations."""
    if schema_types is None:
        schema: dict = {}
        for t in eq_terms:
            Equation.from_term(t).register_sorts(schema)
        schema_types = FrozenDict(schema)
    cs = []
    for up, down, slot in topo_edges(eq_terms):
        u, d = Equation.from_term(up), Equation.from_term(down)
        cs.append(TypeConstraint(
            sort_wrap(u.codomain_sort).type_,
            sort_wrap(d.domain_sort).type_,
            f"'{u.name}' codomain != '{d.name}' domain",
        ))
        up_einsum, down_einsum = u.einsum, d.einsum
        out_rank = len(up_einsum.split("->")[1].strip()) if up_einsum else None
        parts = down_einsum.split("->")[0].split(",") if down_einsum else []
        in_rank = len(parts[slot]) if slot < len(parts) else None
        if out_rank is not None and in_rank is not None and out_rank != in_rank:
            raise TypeError(
                f"Rank mismatch: '{u.name}' output rank {out_rank} != "
                f"'{d.name}' input rank {in_rank} at slot {slot}")
    _unify(cs, schema_types)


# ---------------------------------------------------------------------------
# Equation resolution
# ---------------------------------------------------------------------------

def _make_prim(prim_name, compute, coders, out_coder):
    """Dispatch a compute closure + coder list to prim1/prim2/prim3."""
    n = len(coders)
    if n == 1:
        return prim1(prim_name, compute, [], coders[0], out_coder)
    elif n == 2:
        return prim2(prim_name, compute, [], coders[0], coders[1], out_coder)
    elif n == 3:
        return prim3(prim_name, compute, [], coders[0], coders[1], coders[2], out_coder)
    else:
        raise ValueError(f"Primitive '{prim_name.value}': arity {n} exceeds max 3")


def resolve_equation(eq: Equation, backend: "Backend", ctx=None):
    """Compile an Equation to a Hydra Primitive."""
    has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, compiled, n_inputs, n_params = \
        ctx or eq.compile(backend)

    if not has_einsum and not has_nl:
        raise ValueError(f"Equation '{eq.name}' has neither einsum nor nonlinearity")
    if not has_einsum:
        n_inputs = 1

    total_arity = n_params + n_inputs
    if total_arity > 3:
        raise ValueError(
            f"Equation '{eq.name}': total arity {total_arity} "
            f"({n_params} params + {n_inputs} tensor inputs) exceeds max 3"
        )

    def _compute(*args):
        return contract_and_apply(compiled, list(args[n_params:]), sr, backend, nl_fn, args[:n_params])

    coders = [float32_coder()] * n_params + [in_coder] * n_inputs
    return _make_prim(prim_name, _compute, coders, out_coder)


def resolve_equation_as_merge(eq: Equation, backend: "Backend", ctx=None):
    """Compile an Equation as a list-consuming merge Primitive."""
    has_einsum, has_nl, nl_fn, in_coder, out_coder, prim_name, sr, compiled, n_inputs, _ = \
        ctx or eq.compile(backend)

    if has_einsum:
        if n_inputs not in (1, 2):
            raise ValueError(
                f"List-merge equation '{eq.name}': einsum must have 1 or 2 inputs, got {n_inputs}")

        def compute_merge(tensors):
            return contract_merge(compiled, tensors, sr, backend, nl_fn, n_inputs, eq.name)

        return _make_prim(prim_name, compute_merge, [list_coder(in_coder)], out_coder)

    elif has_nl:
        def compute_nl(tensors):
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return nl_fn(result)
        return _make_prim(prim_name, compute_nl, [list_coder(in_coder)], out_coder)

    else:
        raise ValueError(f"List-merge equation '{eq.name}' has neither einsum nor nonlinearity")


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class EquationPipeline:
    """Single-pass assembly over eq_terms producing schema, primitives, and eq_by_name.

    Calling validate_pipeline and resolving primitives in one sweep avoids
    iterating eq_terms multiple times.
    """

    def __init__(
        self,
        eq_terms: list[core.Term],
        backend: Backend,
        merge_names: set[str] = frozenset(),
        extra_sorts: list[core.Term] = (),
        semirings: dict[str, core.Term] | None = None,
    ):
        equations = [Equation.from_term(t) for t in eq_terms]
        seen: dict[str, int] = {}
        for i, eq in enumerate(equations):
            if eq.name in seen:
                raise ValueError(
                    f"Duplicate equation name '{eq.name}' (positions {seen[eq.name]} and {i})")
            seen[eq.name] = i
        self.eq_by_name: dict[str, Equation] = {eq.name: eq for eq in equations}

        schema: dict = {}
        self.primitives: dict = dict(standard_library())
        self.resolved_semirings: dict = {}
        self.coder = None

        if semirings:
            for name, sr_term in semirings.items():
                self.resolved_semirings[name] = Equation.resolve_semiring_term(sr_term, backend)

        for eq in equations:
            eq.register_sorts(schema)
            ctx = eq.compile(backend)
            sr_name = eq.semiring_name
            if sr_name and ctx[6] is not None:
                self.resolved_semirings.setdefault(sr_name, ctx[6])
            if self.coder is None:
                self.coder = ctx[3]
            prim = resolve_equation_as_merge(eq, backend, ctx) if eq.name in merge_names \
                else resolve_equation(eq, backend, ctx)
            self.primitives[prim.name] = prim

        for st in extra_sorts:
            if st is not None:
                sort_wrap(st).register_schema(schema)

        self.schema_types = FrozenDict(schema)
        validate_pipeline(eq_terms, self.schema_types)

    def validate_spec(self, spec) -> None:
        spec.validate(self.eq_by_name, self.schema_types)
