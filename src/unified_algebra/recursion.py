"""Fold (catamorphism) and unfold (anamorphism) as Hydra lambda terms.

    fold()   -> Hydra lambda: λseq. foldl(step, init, seq)
    unfold() -> Hydra lambda: λstate. unfold_n(step, n, state)

Fold uses Hydra's built-in `hydra.lib.lists.foldl` primitive — no custom
iteration logic needed. The step function is resolved through normal Hydra
reduction (var lookup → primitive/bound_term → beta-reduce).

Unfold uses a custom higher-order primitive `ua.prim.unfold_n` since Hydra
has no built-in unfold/iterate. It follows the same pattern as Hydra's own
higher-order primitives (foldl, map, etc.) — calling reduce_term internally.

Both are completely backend-agnostic. Weight tying is automatic: the same
step function term is applied at every recursive step.
"""

from __future__ import annotations

from unified_algebra.utils import record_fields, string_value
import hydra.core as core
import hydra.dsl.terms as Terms
import hydra.graph
from hydra.dsl import prims

from .sort import sort_type_from_term


# ---------------------------------------------------------------------------
# Fold (catamorphism) — uses Hydra's built-in foldl
# ---------------------------------------------------------------------------

def fold(
    name: str,
    step_name: str,
    init_term: core.Term,
    domain_sort: core.Term,
    state_sort: core.Term,
) -> tuple[core.Name, core.Term]:
    """Build a fold as a Hydra lambda term.

    Args:
        name:       identifier (e.g. "rnn")
        step_name:  name of a 2-input equation/path: step(state, element) → new_state
        init_term:  pre-encoded Hydra tensor term for initial state
        domain_sort: sort term for list elements
        state_sort:  sort term for the state (also the output)

    Returns:
        (Name("ua.fold.<name>"), lambda_term)
        The lambda term is: λseq. foldl(step, init, seq)
    """
    step_fn = Terms.var(f"ua.equation.{step_name}")

    body = Terms.apply(
        Terms.apply(
            Terms.apply(Terms.var("hydra.lib.lists.foldl"), step_fn),
            init_term,
        ),
        Terms.var("seq"),
    )
    term = Terms.lambda_("seq", body)
    return (core.Name(f"ua.fold.{name}"), term)


# ---------------------------------------------------------------------------
# Unfold (anamorphism) — custom higher-order Primitive
# ---------------------------------------------------------------------------

def _unfold_n_primitive() -> hydra.graph.Primitive:
    """Create the ua.prim.unfold_n higher-order Primitive.

    Signature: (state → state) → int32 → state → list<state>

    Iterates a step function n times from an initial state, collecting
    each intermediate state into a list. Uses Hydra's fun() TermCoder
    to bridge term-level step functions to native callables, following
    the same pattern as hydra.lib.lists.foldl in libraries.py.
    """
    from hydra.sources.libraries import fun

    prim_name = core.Name("ua.prim.unfold_n")
    a = prims.variable("a")
    _a = prims.v("a")

    def compute(step, n, init):
        # step: native callable (fun() bridges term → native via reduce_term)
        # n: Python int (int32 coder decodes it)
        # init: raw Term (variable coder passes through)
        outputs = []
        state = init
        for _ in range(n):
            state = step(state)
            outputs.append(state)
        return tuple(outputs)

    return prims.prim3(
        prim_name, compute, [_a],
        fun(a, a),          # step: a → a (bridged to native callable)
        prims.int32(),      # n_steps (decoded to Python int)
        a,                  # init state (passthrough Term)
        prims.list_(a),     # output: list<a> (encoded from tuple of Terms)
    )


def unfold(
    name: str,
    step_name: str,
    n_steps: int,
    domain_sort: core.Term,
    state_sort: core.Term,
) -> tuple[core.Name, core.Term]:
    """Build an unfold as a Hydra lambda term.

    Args:
        name:       identifier (e.g. "stream")
        step_name:  name of a 1-input equation/path: step(state) → new_state
        n_steps:    number of unfolding iterations
        domain_sort: sort term for the state
        state_sort:  sort term for the state (also list element type)

    Returns:
        (Name("ua.unfold.<name>"), lambda_term)
        The lambda term is: λstate. unfold_n(step, n, state)
    """
    step_fn = Terms.var(f"ua.equation.{step_name}")
    n_term = Terms.int32(n_steps)

    body = Terms.apply(
        Terms.apply(
            Terms.apply(Terms.var("ua.prim.unfold_n"), step_fn),
            n_term,
        ),
        Terms.var("state"),
    )
    term = Terms.lambda_("state", body)
    return (core.Name(f"ua.unfold.{name}"), term)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_fold(
    eq_terms_by_name: dict[str, core.Term],
    step_name: str,
    domain_sort: core.Term,
    state_sort: core.Term,
) -> None:
    """Validate sort junctions for a fold.

    Checks:
    1. Step equation exists
    2. Step's output sort matches state_sort (recurrence)
    """
    if step_name not in eq_terms_by_name:
        raise ValueError(f"Fold step equation '{step_name}' not found")

    step_fields = record_fields(eq_terms_by_name[step_name])
    step_codomain = sort_type_from_term(step_fields["codomainSort"])
    state_type = sort_type_from_term(state_sort)
    if step_codomain != state_type:
        raise TypeError(
            f"Fold step '{step_name}' codomain {step_codomain.value.value!r} != "
            f"state sort {state_type.value.value!r}"
        )


def validate_unfold(
    eq_terms_by_name: dict[str, core.Term],
    step_name: str,
    domain_sort: core.Term,
    state_sort: core.Term,
) -> None:
    """Validate sort junctions for an unfold.

    Checks:
    1. Step equation exists
    2. Step's input sort matches domain_sort (state recurrence)
    3. Step's output sort matches domain_sort (state recurrence)
    """
    if step_name not in eq_terms_by_name:
        raise ValueError(f"Unfold step equation '{step_name}' not found")

    step_fields = record_fields(eq_terms_by_name[step_name])

    step_domain = sort_type_from_term(step_fields["domainSort"])
    domain_type = sort_type_from_term(domain_sort)
    if step_domain != domain_type:
        raise TypeError(
            f"Unfold step '{step_name}' domain {step_domain.value.value!r} != "
            f"state sort {domain_type.value.value!r}"
        )

    step_codomain = sort_type_from_term(step_fields["codomainSort"])
    if step_codomain != domain_type:
        raise TypeError(
            f"Unfold step '{step_name}' codomain {step_codomain.value.value!r} != "
            f"state sort {domain_type.value.value!r}"
        )
