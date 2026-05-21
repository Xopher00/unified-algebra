"""Optic behavioral (runtime) tests.

Verifies that Optic.act, act_forward, and act_backward compute correctly
when morphisms are realized and executed through the Hydra runtime.
Covers lens, prism, traversal (Maybe/List), composition, and parallel.
"""

import pytest

import hydra.lexical as L
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
from hydra.core import Name

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor
from unialg.semantics.optics import Optic, identity_optic
from unialg.objects import ProductType, SumType, TypeMaybe, TypeList
from unialg.main import run
from support.strategies import INT


UNIT = ops.TypeUnit()
ADD = Name("hydra.lib.math.add")
PAIR = ProductType(INT, INT)
SUM = SumType(INT, INT)


@pytest.fixture(scope="module")
def graph():
    primitives = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            primitives.extend(getattr(Libs, attr)().values())
    return L.graph_with_primitives(primitives, ())


@pytest.fixture(scope="module")
def ctx():
    return L.empty_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int(term) -> int:
    return term.value.value.value


def _pair_ints(term) -> tuple[int, int]:
    return _int(term.value[0]), _int(term.value[1])


def _add_one() -> ops.Morphism:
    x = P.var("x")
    raw = P.lam("x", P.primitive2(ADD, x, P.int32(1))).value
    return ops.Morphism(expr.Prim(raw, INT, INT))


# ---------------------------------------------------------------------------
# Lens: focus first of product  F = Prod(Id, Const(R))
# ---------------------------------------------------------------------------

def test_lens_fst_act(ctx, graph):
    lens = identity_optic(name="LensFst", functor=Functor("LensFst", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
    morphism = lens.act(_add_one())

    result = run(morphism, P.pair(P.int32(10), P.int32(20)).value, ctx, graph)
    assert _pair_ints(result) == (11, 20)


def test_lens_snd_act(ctx, graph):
    lens = identity_optic(name="LensSnd", functor=Functor("LensSnd", expr.Prod(expr.Const(INT), expr.Id())), focus=INT)
    morphism = lens.act(_add_one())

    result = run(morphism, P.pair(P.int32(10), P.int32(20)).value, ctx, graph)
    assert _pair_ints(result) == (10, 21)


# ---------------------------------------------------------------------------
# Lens: non-trivial decompose/reconstruct via symmetry (swap)
# ---------------------------------------------------------------------------

def test_lens_swap_focuses_second(ctx, graph):
    fwd = ops._symmetry(PAIR)
    bwd = ops._symmetry(PAIR)
    lens = Optic(
        functor=Functor("LensSwap", expr.Prod(expr.Id(), expr.Const(INT))),
        forward=fwd,
        backward=bwd,
    )
    morphism = lens.act(_add_one())

    result = run(morphism, P.pair(P.int32(10), P.int32(20)).value, ctx, graph)
    assert _pair_ints(result) == (10, 21)


# ---------------------------------------------------------------------------
# Prism left: focus left branch  F = Sum(Id, Const(R))
# ---------------------------------------------------------------------------

def test_prism_left_matching(ctx, graph):
    prism = identity_optic(name="PrismL", functor=Functor("PrismL", expr.Sum(expr.Id(), expr.Const(INT))), focus=INT)
    morphism = prism.act(_add_one())

    arg = Terms.left(P.int32(10).value)
    result = run(morphism, arg, ctx, graph)
    assert _int(result.value.value) == 11


def test_prism_left_nonmatching(ctx, graph):
    prism = identity_optic(name="PrismL", functor=Functor("PrismL", expr.Sum(expr.Id(), expr.Const(INT))), focus=INT)
    morphism = prism.act(_add_one())

    arg = Terms.right(P.int32(99).value)
    result = run(morphism, arg, ctx, graph)
    assert _int(result.value.value) == 99


# ---------------------------------------------------------------------------
# Prism right: focus right branch  F = Sum(Const(R), Id)
# ---------------------------------------------------------------------------

def test_prism_right_matching(ctx, graph):
    prism = identity_optic(name="PrismR", functor=Functor("PrismR", expr.Sum(expr.Const(INT), expr.Id())), focus=INT)
    morphism = prism.act(_add_one())

    arg = Terms.right(P.int32(10).value)
    result = run(morphism, arg, ctx, graph)
    assert _int(result.value.value) == 11


def test_prism_right_nonmatching(ctx, graph):
    prism = identity_optic(name="PrismR", functor=Functor("PrismR", expr.Sum(expr.Const(INT), expr.Id())), focus=INT)
    morphism = prism.act(_add_one())

    arg = Terms.left(P.int32(99).value)
    result = run(morphism, arg, ctx, graph)
    assert _int(result.value.value) == 99


# ---------------------------------------------------------------------------
# Traversal: Maybe(Id) — zero-or-one focus
# ---------------------------------------------------------------------------

def test_traversal_maybe_just(ctx, graph):
    trav = identity_optic(name="TravMaybe", functor=Functor("TravMaybe", expr.Maybe(expr.Id())), focus=INT)
    morphism = trav.act(_add_one())

    result = run(morphism, P.just(P.int32(10)).value, ctx, graph)
    assert _int(result.value.value) == 11


def test_traversal_maybe_nothing(ctx, graph):
    trav = identity_optic(name="TravMaybe", functor=Functor("TravMaybe", expr.Maybe(expr.Id())), focus=INT)
    morphism = trav.act(_add_one())

    result = run(morphism, P.nothing().value, ctx, graph)
    assert result == P.nothing().value


# ---------------------------------------------------------------------------
# Traversal: List(Id) — zero-or-many focus
# ---------------------------------------------------------------------------

def _list_ints(term) -> list[int]:
    return [_int(el) for el in term.value]


def test_traversal_list_nonempty(ctx, graph):
    trav = identity_optic(name="TravList", functor=Functor("TravList", expr.List(expr.Id())), focus=INT)
    morphism = trav.act(_add_one())

    arg = P.list_([P.int32(10), P.int32(20), P.int32(30)]).value
    result = run(morphism, arg, ctx, graph)
    assert _list_ints(result) == [11, 21, 31]


def test_traversal_list_empty(ctx, graph):
    trav = identity_optic(name="TravList", functor=Functor("TravList", expr.List(expr.Id())), focus=INT)
    morphism = trav.act(_add_one())

    arg = P.list_([]).value
    result = run(morphism, arg, ctx, graph)
    assert _list_ints(result) == []


# ---------------------------------------------------------------------------
# Composition: outer.compose(inner) focuses through two layers
# ---------------------------------------------------------------------------

def test_composed_lens_nested_focus(ctx, graph):
    TRIPLE = ProductType(PAIR, INT)
    outer = identity_optic(
        name="Outer", functor=Functor("Outer", expr.Prod(expr.Id(), expr.Const(INT))), focus=PAIR,
    )
    inner = identity_optic(
        name="Inner", functor=Functor("Inner", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT,
    )
    composed = outer.compose(inner)
    morphism = composed.act(_add_one())

    arg = P.pair(P.pair(P.int32(10), P.int32(20)), P.int32(30)).value
    result = run(morphism, arg, ctx, graph)
    inner_pair = _pair_ints(result.value[0])
    outer_residue = _int(result.value[1])
    assert inner_pair == (11, 20)
    assert outer_residue == 30


# ---------------------------------------------------------------------------
# Parallel: par acts on both product positions
# ---------------------------------------------------------------------------

def test_par_acts_on_both_positions(ctx, graph):
    lens_l = identity_optic(name="L", functor=Functor("L", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
    lens_r = identity_optic(name="R", functor=Functor("R", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
    par_optic = lens_l.par(lens_r)
    morphism = par_optic.act(_add_one())

    arg = P.pair(P.pair(P.int32(1), P.int32(2)), P.pair(P.int32(3), P.int32(4))).value
    result = run(morphism, arg, ctx, graph)
    left = _pair_ints(result.value[0])
    right = _pair_ints(result.value[1])
    assert left == (2, 2)
    assert right == (4, 4)


# ---------------------------------------------------------------------------
# act_forward ; backward == act
# ---------------------------------------------------------------------------

def test_act_forward_then_backward_equals_act(ctx, graph):
    fwd = ops._symmetry(PAIR)
    bwd = ops._symmetry(PAIR)
    lens = Optic(
        functor=Functor("LensSwap", expr.Prod(expr.Id(), expr.Const(INT))),
        forward=fwd,
        backward=bwd,
    )
    h = _add_one()
    via_act = lens.act(h)
    via_decomposed = ops.compose(lens.act_forward(h), lens.backward)

    arg = P.pair(P.int32(5), P.int32(15)).value
    result_act = run(via_act, arg, ctx, graph)
    result_decomposed = run(via_decomposed, arg, ctx, graph)
    assert _pair_ints(result_act) == _pair_ints(result_decomposed)
