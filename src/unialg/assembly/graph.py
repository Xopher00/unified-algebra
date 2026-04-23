"""DAG validation and graph assembly for equation sets.

Handles topological ordering, sort/rank junction checking, and
Hydra Graph construction from resolved equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.dsl.python import FrozenDict
import unialg.algebra as alg
import unialg.resolve as res
import unialg.composition as comp
import unialg.specs as sp
import unialg.views as vw
from unialg.resolve.morphism import Equation
from unialg.assembly.topology import validate_pipeline, _register_sort_components, _build_schema
from unialg.assembly.topology import validate_spec
from unialg.composition.catamorphism import fold, unfold
from unialg.algebra.fixpoint import fixpoint
from unialg.assembly.primitives import unfold_n_primitive, fixpoint_primitive, lens_fwd_primitive, lens_bwd_primitive

if TYPE_CHECKING:
    import hydra.core as core
    import hydra.graph
    from unialg.backend import Backend


def _register_residual_prims(
    specs: list,
    eq_by_name: dict,
    primitives: dict,
    backend: "Backend",
    semirings: dict | None = None,
) -> None:
    """Register ua.prim.residual_add.<sr_name> for any PathSpec with residual=True."""
    import hydra.core as core
    from hydra.dsl.prims import prim2
    from unialg.algebra.semiring import Semiring

    for spec in specs:
        if not isinstance(spec, sp.PathSpec) or not spec.residual:
            continue

        sr_name = spec.residual_semiring or "default"
        prim_name = core.Name(f"ua.prim.residual_add.{sr_name}")
        if prim_name in primitives:
            continue

        sr_term = None
        for eq_term in eq_by_name.values():
            v = Equation.from_term(eq_term)
            sr_field = v.semiring
            if isinstance(sr_field, core.TermRecord):
                sv = Semiring.from_term(sr_field)
                if sv.name == sr_name:
                    sr_term = sr_field
                    break

        if sr_term is None and semirings:
            sr_term = semirings.get(sr_name)

        if sr_term is None:
            raise ValueError(
                f"Residual path references semiring '{sr_name}' but no equation "
                f"uses it and it was not passed via semirings="
            )

        sr = Semiring.from_term(sr_term).resolve(backend)
        coder = alg.tensor_coder()

        def _make_compute(resolved_sr):
            return lambda a, b: resolved_sr.plus_elementwise(a, b)

        prim = prim2(prim_name, _make_compute(sr), [], coder, coder, coder)
        primitives[prim_name] = prim


def _build_lens_by_name(
    lenses: list[core.Term] | None,
    eq_by_name: dict[str, core.Term],
) -> dict[str, core.Term]:
    """Validate lenses and build lens_by_name lookup."""
    lens_by_name: dict[str, core.Term] = {}
    if lenses:
        for lens_term in lenses:
            lens_by_name[vw.LensView(lens_term).name] = lens_term
            comp.validate_lens(eq_by_name, lens_term)
    return lens_by_name


def _build_compositions(
    specs: list,
    eq_by_name: dict[str, core.Term],
    primitives: dict,
    bound_terms: dict,
    lens_by_name: dict[str, core.Term],
    schema_types,
) -> None:
    """Validate and build all composition specs. Mutates primitives and bound_terms."""
    for spec in specs:
        validate_spec(eq_by_name, spec, schema_types)

        if isinstance(spec, sp.PathSpec):
            results = [comp.path(spec.name, spec.eq_names, spec.params,
                            residual=spec.residual,
                            residual_semiring=spec.residual_semiring)]

        elif isinstance(spec, sp.FanSpec):
            results = [comp.fan(spec.name, spec.branch_names, spec.merge_name)]

        elif isinstance(spec, sp.FoldSpec):
            results = [fold(spec.name, spec.step_name, spec.init_term)]

        elif isinstance(spec, sp.UnfoldSpec):
            unfold_prim = unfold_n_primitive
            primitives.setdefault(unfold_prim.name, unfold_prim)
            results = [unfold(spec.name, spec.step_name, spec.n_steps)]

        elif isinstance(spec, sp.LensPathSpec):
            if any(vw.LensView(lens_by_name[ln]).residual_sort is not None for ln in spec.lens_names):
                for prim in (lens_fwd_primitive, lens_bwd_primitive):
                    primitives.setdefault(prim.name, prim)
            (fwd, bwd) = comp.lens_path(spec.name, spec.lens_names, lens_by_name, spec.params)
            results = [fwd, bwd]

        elif isinstance(spec, sp.LensFanSpec):
            (fwd, bwd) = comp.lens_fan(spec.name, spec.lens_names, spec.merge_lens_name, lens_by_name)
            results = [fwd, bwd]

        elif isinstance(spec, sp.FixpointSpec):
            fp_prim = fixpoint_primitive(spec.epsilon, spec.max_iter)
            primitives.setdefault(fp_prim.name, fp_prim)
            results = [fixpoint(spec.name, spec.step_name, spec.predicate_name, spec.epsilon, spec.max_iter)]

        else:
            raise TypeError(f"Unknown composition spec type: {type(spec).__name__}")

        for name, term in results:
            bound_terms[name] = term


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    sort_terms: list[core.Term],
    primitives: dict | None = None,
    bound_terms: dict | None = None,
) -> hydra.graph.Graph:
    """Assemble a Hydra Graph with sorts registered as schema_types.

    Args:
        sort_terms:  list of sort record terms (from sort())
        primitives:  optional dict of Name -> Primitive (for Phase 3+)
        bound_terms: optional dict of Name -> Term
    """
    import hydra.core as core
    import hydra.graph
    from hydra.dsl.python import FrozenDict, Nothing

    schema = {}
    terms = dict(bound_terms or {})

    # Register the tensor type
    tensor_name = core.Name("ua.tensor.NDArray")
    schema[tensor_name] = core.TypeScheme(
        (), core.TypeVariable(tensor_name), Nothing()
    )

    # Register each sort's structural component names so sort_type_from_term
    # results are ground (matches what _build_schema does in assemble_graph).
    for st in sort_terms:
        _register_sort_components(st, schema)

    return hydra.graph.Graph(
        bound_terms=FrozenDict(terms),
        bound_types=FrozenDict({}),
        class_constraints=FrozenDict({}),
        lambda_variables=frozenset(),
        metadata=FrozenDict({}),
        primitives=FrozenDict(primitives or {}),
        schema_types=FrozenDict(schema),
        type_variables=frozenset(),
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def assemble_graph(
    eq_terms: list[core.Term],
    backend: Backend,
    extra_sorts: list[core.Term] | None = None,
    specs: list | None = None,
    hyperparams: dict[str, core.Term] | None = None,
    lenses: list[core.Term] | None = None,
    semirings: dict[str, core.Term] | None = None,
) -> hydra.graph.Graph:
    """Resolve equation terms and assemble a Hydra Graph.

    Args:
        eq_terms:     list of equation record terms
        backend:      Backend providing op implementations
        extra_sorts:  additional sort terms to register
        specs:        list of composition specs (PathSpec, FanSpec, FoldSpec,
                      UnfoldSpec, LensPathSpec, FixpointSpec)
        hyperparams:  dict of param_name → scalar Term
        lenses:       list of lens record terms (from lens())
        semirings:    optional dict of semiring_name → semiring Term, used
                      as fallback when a residual path's semiring is not
                      referenced by any equation
    """
    validate_pipeline(eq_terms)

    all_specs = list(specs or [])

    merge_names: set[str] = {spec.merge_name for spec in all_specs if isinstance(spec, sp.FanSpec)}
    primitives, eq_by_name = res.resolve_all_primitives(eq_terms, backend, merge_names)

    schema_types = _build_schema(eq_terms)

    import hydra.core as core
    bound_terms: dict[core.Name, core.Term] = {}
    if hyperparams:
        for param_name, param_term in hyperparams.items():
            bound_terms[core.Name(f"ua.param.{param_name}")] = param_term

    lens_by_name = _build_lens_by_name(lenses, eq_by_name)
    _register_residual_prims(all_specs, eq_by_name, primitives, backend, semirings=semirings)
    _build_compositions(all_specs, eq_by_name, primitives, bound_terms, lens_by_name, schema_types)

    seen_sorts: dict[str, core.Term] = {}
    for eq_term in eq_terms:
        v = Equation.from_term(eq_term)
        for st in (v.domain_sort, v.codomain_sort):
            seen_sorts.setdefault(str(alg.sort_type_from_term(st)), st)  # sort_type_from_term stays in algebra
    all_sorts = list(seen_sorts.values()) + list(extra_sorts or [])
    return build_graph(all_sorts, primitives=primitives, bound_terms=bound_terms)


# ---------------------------------------------------------------------------
# Hyperparameter rebinding
# ---------------------------------------------------------------------------

def rebind_hyperparams(
    graph: hydra.graph.Graph,
    updates: dict[str, core.Term],
) -> hydra.graph.Graph:
    """Return a new Graph with hyperparameters substituted into term bodies.

    Uses Hydra's structural substitution to replace var("ua.param.X")
    references directly inside lambda bodies, respecting variable scoping.

    Args:
        graph:   existing Hydra Graph
        updates: dict of param_name → new scalar Term
                 (keys without "ua.param." prefix — it's added automatically)
    """
    import hydra.core as core
    import hydra.graph as hgraph
    import hydra.substitution as subst
    import hydra.typing

    param_updates = {
        core.Name(f"ua.param.{k}"): v for k, v in updates.items()
    }
    ts = hydra.typing.TermSubst(FrozenDict(param_updates))

    new_terms = {
        name: subst.substitute_in_term(ts, term)
        for name, term in graph.bound_terms.items()
    }
    # Also update bound_terms entries for the params themselves
    new_terms.update(param_updates)

    return hgraph.Graph(
        bound_terms=FrozenDict(new_terms),
        bound_types=graph.bound_types,
        class_constraints=graph.class_constraints,
        lambda_variables=graph.lambda_variables,
        metadata=graph.metadata,
        primitives=graph.primitives,
        schema_types=graph.schema_types,
        type_variables=graph.type_variables,
    )
