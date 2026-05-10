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
from hydra.core import Name
from hydra.lib import names as Names
from hydra.phantoms import TTerm


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
MAYBES_BIND = Names.maybes_bind
MAYBES_PURE = Names.maybes_pure

# Hydra has structural TypeVoid, but its term-level case statement targets a
# nominal union type name.  This synthetic name is only used to encode the
# impossible eliminator for unialg's structural void domain.
VOID_CASE_TYPE = Name("hydra.core.Void")


def absurd() -> TTerm:
    """Term-level eliminator for the uninhabited object."""
    return P.match(VOID_CASE_TYPE, P.Nothing(), [])


def prim2(name: Name, a: TTerm, b: TTerm) -> TTerm:
    """Apply a binary Hydra primitive to two arguments."""
    return P.primitive2(name, a, b)


def term_lambda(name: str, body: Callable[[TTerm], TTerm]) -> TTerm:
    """Build a Hydra lambda term from a named variable."""
    x = P.var(name)
    return P.lam(name, body(x))


def bind(
    monad: MonadDescriptor,
    value: TTerm,
    name: str,
    body: Callable[[TTerm], TTerm],
) -> TTerm:
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


def pairs_bimap(left: TTerm, right: TTerm) -> TTerm:
    """Hydra ``pairs.bimap`` partially applied to two component functions."""
    return prim2(PAIRS_BIMAP, left, right)


def pair_first() -> TTerm:
    """Hydra pair first projection."""
    return P.primitive(PAIRS_FIRST)


def pair_second() -> TTerm:
    """Hydra pair second projection."""
    return P.primitive(PAIRS_SECOND)


def pair_effects(
    monad: MonadDescriptor | None,
    left: TTerm,
    right: TTerm,
) -> TTerm:
    """Pair two results, sequencing effects when a monad is present."""
    if monad is None:
        return P.pair(left, right)
    return bind(monad, left, "pe_l", lambda l:
        bind(monad, right, "pe_r", lambda r:
            pure(monad, P.pair(l, r))))


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


def case_effects(
    monad: MonadDescriptor | None,
    branch_l: TTerm,
    branch_r: TTerm,
) -> TTerm:
    """Build functor action for sums, traversing branch effects if needed."""
    if monad is None:
        return eithers_bimap(branch_l, branch_r)
    return eithers_either(
        term_lambda("ce_lv", lambda lv:
            bind(monad, P.apply(branch_l, lv), "ce_l", lambda lb:
                pure(monad, P.apply(left_injection(), lb)))),
        term_lambda("ce_rv", lambda rv:
            bind(monad, P.apply(branch_r, rv), "ce_r", lambda rb:
                pure(monad, P.apply(right_injection(), rb)))),
    )


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


def list_effects(monad: MonadDescriptor | None, item_action: TTerm) -> TTerm:
    """Build functor action for List, traversing element effects if needed."""
    if monad is None:
        return lists_map(item_action)

    return term_lambda("xs", lambda xs:
        lists_foldr(
            term_lambda("x", lambda x:
                term_lambda("acc", lambda acc:
                    bind(monad, P.apply(item_action, x), "le_y", lambda y:
                        bind(monad, acc, "le_ys", lambda ys:
                            pure(monad, lists_cons(y, ys)))))),
            pure(monad, lists_empty()),
            xs,
        )
    )


def maybes_maybe(default: TTerm, f: TTerm, x: TTerm) -> TTerm:
    """Hydra ``maybes.maybe``: B → (A → B) → Maybe(A) → B."""
    return Maybes.maybe(default, f, x)


def maybes_nothing() -> TTerm:
    """Hydra Nothing literal."""
    return TTerm(Terms.nothing())


def maybes_just() -> TTerm:
    """Hydra Just constructor."""
    return term_lambda("just_value", lambda x: TTerm(Terms.just(x.value)))


def maybe_effects(monad: MonadDescriptor | None, item_action: TTerm) -> TTerm:
    """Build functor action for Maybe, traversing element effects if needed."""
    if monad is None:
        return term_lambda("mx", lambda mx:
            maybes_maybe(
                maybes_nothing(),
                term_lambda("x", lambda x:
                    P.apply(maybes_just(), P.apply(item_action, x))),
                mx,
            )
        )

    return term_lambda("mx", lambda mx:
        maybes_maybe(
            pure(monad, maybes_nothing()),
            term_lambda("x", lambda x:
                bind(monad, P.apply(item_action, x), "me_y", lambda y:
                    pure(monad, P.apply(maybes_just(), y)))),
            mx,
        )
    )