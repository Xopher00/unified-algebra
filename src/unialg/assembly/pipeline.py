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
from hydra.sources.libraries import standard_library
from hydra.typing import TypeConstraint
from hydra.unification import unify_type_constraints

from unialg.algebra.contraction import compile_einsum
from unialg.algebra.equation import Equation
from unialg.algebra.sort import sort_wrap
from unialg.assembly.resolver import resolve_equation, resolve_equation_as_merge

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
        up_compiled = compile_einsum(u.einsum) if u.einsum else None
        down_compiled = compile_einsum(d.einsum) if d.einsum else None
        out_rank = len(up_compiled.output_vars) if up_compiled is not None else None
        in_rank = (len(down_compiled.input_vars[slot])
                   if down_compiled is not None and slot < len(down_compiled.input_vars)
                   else None)
        if out_rank is not None and in_rank is not None and out_rank != in_rank:
            raise TypeError(
                f"Rank mismatch: '{u.name}' output rank {out_rank} != "
                f"'{d.name}' input rank {in_rank} at slot {slot}")
    _unify(cs, schema_types)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class EquationPipeline:
    """Single-pass assembly over eq_terms producing schema, primitives, and eq_by_name.

    The constructor orchestrates four phases: parse + dedupe → resolve user semirings
    → resolve equations (also accumulates schema sorts) → register extra sorts +
    validate. Each phase is its own private method.
    """

    def __init__(
        self,
        eq_terms: list[core.Term],
        backend: Backend,
        merge_names: set[str] = frozenset(),
        extra_sorts: list[core.Term] = (),
        semirings: dict[str, core.Term] | None = None,
    ):
        equations = self._parse_equations(eq_terms)
        self.eq_by_name: dict[str, Equation] = {eq.name: eq for eq in equations}
        self.primitives: dict = dict(standard_library())
        self.native_fns: dict = {}
        self.resolved_semirings: dict = self._resolve_user_semirings(semirings, backend)
        self.coder = None

        schema = self._resolve_equations(equations, backend, merge_names)
        for st in extra_sorts:
            if st is not None:
                sort_wrap(st).register_schema(schema)
        self.schema_types = FrozenDict(schema)

        validate_pipeline(eq_terms, self.schema_types)

    @staticmethod
    def _parse_equations(eq_terms: list) -> list[Equation]:
        """Wrap raw terms as Equations; raise on duplicate names."""
        equations = [Equation.from_term(t) for t in eq_terms]
        seen: dict[str, int] = {}
        for i, eq in enumerate(equations):
            if eq.name in seen:
                raise ValueError(
                    f"Duplicate equation name '{eq.name}' (positions {seen[eq.name]} and {i})")
            seen[eq.name] = i
        return equations

    @staticmethod
    def _resolve_user_semirings(semirings: dict | None, backend: Backend) -> dict:
        """Pre-resolve any user-provided semiring terms against the backend."""
        if not semirings:
            return {}
        return {name: Equation.resolve_semiring_term(t, backend) for name, t in semirings.items()}

    def _resolve_equations(self, equations: list[Equation], backend: Backend, merge_names: set[str]) -> dict:
        """Compile each equation, register its primitive + native_fn, accumulate sorts.

        Returns the populated schema dict (mutates self.primitives / native_fns / etc.).
        """
        schema: dict = {}
        for eq in equations:
            eq.register_sorts(schema)
            ctx = eq.compile(backend)
            sr_name = eq.semiring_name
            if sr_name and ctx[6] is not None:
                self.resolved_semirings.setdefault(sr_name, ctx[6])
            if self.coder is None:
                self.coder = ctx[3]
            resolver = resolve_equation_as_merge if eq.name in merge_names else resolve_equation
            prim, native_fn = resolver(eq, backend, ctx)
            self.primitives[prim.name] = prim
            self.native_fns[prim.name] = native_fn
        return schema

    def validate_spec(self, spec) -> None:
        spec.validate(self.eq_by_name, self.schema_types)
