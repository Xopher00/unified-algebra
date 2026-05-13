"""Hydra primitive catalog used by the unialg compiler.

Hydra already names and registers standard-library primitives.  This module is
the single place where unialg chooses those backend targets, so encoding code
does not scatter raw ``Name("hydra.lib...")`` strings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import hydra.dsl.meta.phantoms as P
import hydra.dsl.meta.lib.lists as Lists
import hydra.dsl.meta.lib.maybes as Maybes
import hydra.dsl.terms as Terms
from hydra.core import Name, Term
from hydra.graph import Graph
from hydra.lib import names as Names
from hydra.phantoms import TTerm
import hydra.hoisting as Hoisting
import hydra.rewriting as Rewriting

class MonadDescriptor(Protocol):
    """Minimal monad protocol used at the Hydra term boundary."""

    bind_name: Name
    pure_name: Name


PAIRS_BIMAP = Names.pairs_bimap
PAIRS_FIRST = Names.pairs_first
PAIRS_SECOND = Names.pairs_second

EITHERS_BIMAP = Names.eithers_bimap
EITHERS_EITHER = Names.eithers_either

LISTS_BIND = Names.lists_bind
LISTS_PURE = Names.lists_pure
LISTS_MAP = Names.lists_map
LISTS_APPLY = Names.lists_apply

MAYBES_BIND = Names.maybes_bind
MAYBES_PURE = Names.maybes_pure
MAYBES_APPLY = Names.maybes_apply

# Hydra has structural TypeVoid, but its term-level case statement targets a
# nominal union type name.  This synthetic name is only used to encode the
# impossible eliminator for unialg's structural void domain.
VOID_CASE_TYPE = Name("hydra.core.Void")


def prim2(name: Name, a: TTerm, b: TTerm) -> TTerm:
    """Apply a binary Hydra primitive to two arguments."""
    return P.primitive2(name, a, b)


def term_lambda(name: str, body: Callable[[TTerm], TTerm]) -> TTerm:
    """Build a Hydra lambda term from a named variable."""
    x = P.var(name)
    return P.lam(name, body(x))


def lam2(name1: str, name2: str, body: Callable[[TTerm, TTerm], TTerm]) -> TTerm:
    """Build a curried two-argument Hydra lambda."""
    return term_lambda(name1, lambda x: term_lambda(name2, lambda y: body(x, y)))


def _variant_fields(t, variant: str, *fields, then):
    if type(t).__name__ != variant:
        return t
    v = getattr(t, "value", None)
    if v is None:
        return t
    out = tuple(getattr(v, field, None) for field in fields)
    if any(x is None for x in out):
        return t
    return then(*out)


def _structural_rewrite_once(t):
    """
    Peephole simplifications for Hydra terms emitted by this module.

    first(pair(a, b))  -> a
    second(pair(a, b)) -> b
    swap(pair(a, b))   -> pair(b, a)
    """
    def rewrite_pair_app(f, x):
        return _variant_fields(
            x, "TermPair", "left", "right",
            then=lambda a, b: (
                a if f == pair_first().value else
                b if f == pair_second().value else
                P.pair(TTerm(b), TTerm(a)).value if f == pair_swap().value else
                t
            ))
    return _variant_fields(t, "TermApplication", "function", "argument", then=rewrite_pair_app)


def optimize_term(term: TTerm | Term) -> TTerm:
    """Optimize a Hydra term after lowering from unialg morphism combinators."""
    def rule(recurse, t):
        return _structural_rewrite_once(recurse(t))

    raw = term.value if isinstance(term, TTerm) else term
    return TTerm(Rewriting.rewrite_term(rule, raw))


def normalize_term(term: TTerm | Term, graph: Graph | None = None) -> TTerm:
    """Run structural normalization passes on a realized Hydra term."""
    out = optimize_term(term)
    if graph is not None:
        out = TTerm(Hoisting.hoist_case_statements(graph, out.value))
    return out


def bind(monad: MonadDescriptor, value: TTerm, name: str,
    body: Callable[[TTerm], TTerm]) -> TTerm:
    """Hydra monadic bind for the configured monad descriptor."""
    f = term_lambda(name, body)
    if monad.bind_name == LISTS_BIND:
        return Lists.bind(value, f)
    if monad.bind_name == MAYBES_BIND:
        return Maybes.bind(value, f)
    return prim2(monad.bind_name, value, f)


def pure(monad: MonadDescriptor, value: TTerm) -> TTerm:
    """Hydra monadic pure for the configured monad descriptor."""
    if monad.pure_name == LISTS_PURE:
        return Lists.pure(value)
    if monad.pure_name == MAYBES_PURE:
        return Maybes.pure(value)
    return P.apply(P.primitive(monad.pure_name), value)


def apply_effect(monad: MonadDescriptor, ff: TTerm, fx: TTerm) -> TTerm:
    """Applicative apply ``ff <*> fx`` in the given monad context."""
    if monad.pure_name == LISTS_PURE:
        return prim2(LISTS_APPLY, ff, fx)
    if monad.pure_name == MAYBES_PURE:
        return prim2(MAYBES_APPLY, ff, fx)
    return bind(monad, ff, "ap_f", lambda f:
        bind(monad, fx, "ap_x", lambda x:
            pure(monad, P.apply(f, x))))


def map_effect(monad: MonadDescriptor, f: TTerm, fx: TTerm) -> TTerm:
    """Apply ``f`` over an effectful value ``fx``, preserving the monad wrapper."""
    if monad.pure_name == LISTS_PURE:
        return Lists.map(f, fx)
    if monad.pure_name == MAYBES_PURE:
        return Maybes.map(f, fx)
    return bind(monad, fx, "map_x", lambda x:
        pure(monad, P.apply(f, x)))


def lift2_effect(monad: MonadDescriptor, f: Callable[[TTerm, TTerm], TTerm],
    left: TTerm, right: TTerm, name1: str = "x", name2: str = "y") -> TTerm:
    """Lift a binary function ``f`` into an effectful context over ``left`` and ``right``."""
    ctor = lam2(name1, name2, f)
    return apply_effect(monad, apply_effect(monad, pure(monad, ctor), left), right)


def absurd() -> TTerm:
    """Term-level eliminator for the uninhabited object."""
    return P.match(VOID_CASE_TYPE, P.Nothing(), [])


def pairs_bimap(left: TTerm, right: TTerm) -> TTerm:
    """Hydra ``pairs.bimap`` partially applied to two component functions."""
    return prim2(PAIRS_BIMAP, left, right)


def pair_first() -> TTerm:
    """Hydra pair first projection."""
    return P.primitive(PAIRS_FIRST)


def pair_second() -> TTerm:
    """Hydra pair second projection."""
    return P.primitive(PAIRS_SECOND)


def eithers_bimap(left: TTerm, right: TTerm) -> TTerm:
    """Hydra ``eithers.bimap`` partially applied to two branch functions."""
    return prim2(EITHERS_BIMAP, left, right)


def eithers_either(left: TTerm, right: TTerm) -> TTerm:
    """Hydra ``eithers.either`` partially applied to two branch functions."""
    return prim2(EITHERS_EITHER, left, right)


def _injection(make_term: Callable) -> TTerm:
    return term_lambda("injected_", lambda x: TTerm(make_term(x.value)))


def left_injection() -> TTerm:
    """Function injecting a value into the left side of Hydra Either."""
    return _injection(Terms.left)


def right_injection() -> TTerm:
    """Function injecting a value into the right side of Hydra Either."""
    return _injection(Terms.right)


def lists_uncons(xs: TTerm) -> TTerm:
    """Hydra ``lists.uncons``: List(A) → Maybe(Pair(A, List(A)))."""
    return Lists.uncons(xs)


def lists_cons(head: TTerm, tail: TTerm) -> TTerm:
    """Hydra ``lists.cons``: A → List(A) → List(A)."""
    return Lists.cons(head, tail)


def lists_empty() -> TTerm:
    """Empty list literal."""
    return P.list_([])


def lists_foldr(f: TTerm, initial: TTerm, values: TTerm) -> TTerm:
    """Hydra ``lists.foldr``: (x → y → y) → y → list<x> → y."""
    return Lists.foldr(f, initial, values)


def lists_map(f: TTerm) -> TTerm:
    """Hydra ``lists.map`` partially applied to an element function."""
    return P.apply(P.primitive(LISTS_MAP), f)


def maybes_maybe(default: TTerm, f: TTerm, x: TTerm) -> TTerm:
    """Hydra ``maybes.maybe``: B → (A → B) → Maybe(A) → B."""
    return Maybes.maybe(default, f, x)


def maybes_nothing() -> TTerm:
    """Hydra Nothing literal."""
    return TTerm(Terms.nothing())


def maybes_just() -> TTerm:
    """Hydra Just constructor."""
    return term_lambda("just_value", lambda x: TTerm(Terms.just(x.value)))


def pair_swap() -> TTerm:
    """Swap a Hydra pair: A × B → B × A."""
    return term_lambda("p", lambda p:
        P.pair(P.second(p), P.first(p))
    )


def either_swap() -> TTerm:
    """Swap a Hydra Either: A + B → B + A."""
    return eithers_either(
        term_lambda("l", lambda l: P.apply(right_injection(), l)),
        term_lambda("r", lambda r: P.apply(left_injection(), r)),
    )


def pure_unit(monad: MonadDescriptor | None) -> TTerm:
    """Terminal morphism ``A → 1``, plain or lifted into the monad as ``A → T(1)``."""
    return P.constant(P.unit()) if monad is None else term_lambda("unit_x", lambda _: pure(monad, P.unit()))


def pure_identity(monad: MonadDescriptor | None) -> TTerm:
    """Identity morphism ``A → A``, plain or lifted into the monad as ``A → T(A)``."""
    return P.identity() if monad is None else term_lambda("const_x", lambda x: pure(monad, x))


def product_action(monad: MonadDescriptor | None, left_action: TTerm, right_action: TTerm) -> TTerm:
    """Parallel product morphism ``A×B → C×D`` by applying ``left_action`` and ``right_action`` component-wise.

    Without a monad, delegates to ``pairs_bimap``.  With a monad, assembles
    component results applicatively inside the effect.
    """
    if monad is None:
        return pairs_bimap(left_action, right_action)
    return term_lambda(
        "fm_x",
        lambda x: pair_effects(
            monad,
            P.apply(left_action, P.first(x)),
            P.apply(right_action, P.second(x)),
        ),
    )


def pair_effects(monad: MonadDescriptor | None, left: TTerm, right: TTerm) -> TTerm:
    """Pair two results, using applicative assembly when Hydra supports it."""
    if monad is None:
        return P.pair(left, right)
    return lift2_effect(monad, lambda l, r: P.pair(l, r), left, right, "pe_l", "pe_r")


def case_effects(monad: MonadDescriptor | None, branch_l: TTerm, branch_r: TTerm) -> TTerm:
    """Build sum action, mapping injections through concrete Hydra effects."""
    if monad is None:
        return eithers_bimap(branch_l, branch_r)
    lift_left = left_injection()
    lift_right = right_injection()
    return eithers_either(
        term_lambda("ce_lv", lambda lv:
            map_effect(monad, lift_left, P.apply(branch_l, lv))),
        term_lambda("ce_rv", lambda rv:
            map_effect(monad, lift_right, P.apply(branch_r, rv))),
    )


def list_effects(monad: MonadDescriptor | None, item_action: TTerm) -> TTerm:
    """Build list action, using applicative step assembly for concrete effects."""
    if monad is None:
        return lists_map(item_action)
    return term_lambda("xs", lambda xs:
        lists_foldr(
            lam2("x", "acc",
                lambda x, acc: lift2_effect(
                    monad, lists_cons, P.apply(item_action, x),
                    acc, "le_y", "le_ys",
                )),
            pure(monad, lists_empty()),
            xs,
        )
    )


def maybe_effects(monad: MonadDescriptor | None, item_action: TTerm) -> TTerm:
    """Build maybe action, preferring direct Hydra map/apply structure."""
    if monad is None:
        return term_lambda("mx", lambda mx: Maybes.map(item_action, mx))
    lift_just = maybes_just()
    branch = term_lambda(
        "x", lambda x: map_effect(monad, lift_just, P.apply(item_action, x)),
    )
    return term_lambda(
        "mx",
        lambda mx: maybes_maybe(
            pure(monad, maybes_nothing()),
            branch,
            mx,
        ),
    )
