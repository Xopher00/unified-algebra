"""Adversarial tests for the optic system.

Probes edge cases, rejection behavior, degenerate functors, type mismatches,
and runtime invariants that the happy-path behavioral tests do not cover.
"""

import pytest

import hydra.lexical as L
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
from hydra.core import Name, TypeVariable

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor, apply_poly
from unialg.semantics.optics import Optic, identity_optic
from unialg.semantics.morphisms import MorphismError
from unialg.objects import ProductType, SumType, TypeMaybe, TypeList, TypeUnit, VoidType
from unialg.main import run
from support.strategies import INT


UNIT = ops.TypeUnit()
ADD = Name("hydra.lib.math.add")
PAIR = ProductType(INT, INT)
SUM = SumType(INT, INT)
BOOL = TypeVariable(Name("Bool"))


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


def _int(term) -> int:
    return term.value.value.value


def _pair_ints(term) -> tuple[int, int]:
    return _int(term.value[0]), _int(term.value[1])


def _add_one() -> ops.Morphism:
    x = P.var("x")
    raw = P.lam("x", P.primitive2(ADD, x, P.int32(1))).value
    return ops.Morphism(expr.Prim(raw, INT, INT))


# ---------------------------------------------------------------------------
# Rejection: construction-time validation
# ---------------------------------------------------------------------------

class TestOpticRejection:

    def test_rejects_functor_without_id(self):
        f = Functor("NoId", expr.Const(INT))
        with pytest.raises(MorphismError, match="invalid optic forward codomain"):
            Optic(functor=f, forward=ops.identity(INT), backward=ops.identity(INT))

    def test_rejects_forward_codomain_mismatch(self):
        f = Functor("F", expr.Prod(expr.Id(), expr.Const(INT)))
        with pytest.raises(MorphismError, match="invalid optic forward codomain"):
            Optic(
                functor=f,
                forward=ops.identity(INT),
                backward=ops.identity(PAIR),
            )

    def test_rejects_backward_domain_mismatch(self):
        f = Functor("F", expr.Prod(expr.Id(), expr.Const(INT)))
        with pytest.raises(MorphismError, match="invalid optic backward domain"):
            Optic(
                functor=f,
                forward=ops.identity(PAIR),
                backward=ops.identity(INT),
            )

    def test_product_rejects_focus_mismatch(self):
        # product() requires both optics share the same focus type.
        # Using Id functor so focus == forward.cod() directly.  PAIR is a
        # concrete ProductType(INT, INT), distinct from INT in Hydra's type
        # equality — unlike TypeVariable("Bool") which is a free variable and
        # unifies with anything.
        f = Functor("Id", expr.Id())
        o1 = Optic(functor=f, forward=ops.identity(INT), backward=ops.identity(INT))
        o2 = Optic(functor=f, forward=ops.identity(PAIR), backward=ops.identity(PAIR))
        with pytest.raises(MorphismError, match="optic combine focus"):
            o1.product(o2)


# ---------------------------------------------------------------------------
# Identity action: act(identity) should be identity-like
# ---------------------------------------------------------------------------

class TestIdentityAction:

    def test_lens_act_identity_is_noop(self, ctx, graph):
        lens = identity_optic(name="L", functor=Functor("L", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
        morphism = lens.act(ops.identity(INT))

        result = run(morphism, P.pair(P.int32(42), P.int32(99)).value, ctx, graph)
        assert _pair_ints(result) == (42, 99)

    def test_prism_act_identity_left_noop(self, ctx, graph):
        prism = identity_optic(name="P", functor=Functor("P", expr.Sum(expr.Id(), expr.Const(INT))), focus=INT)
        morphism = prism.act(ops.identity(INT))

        result = run(morphism, Terms.left(P.int32(7).value), ctx, graph)
        assert _int(result.value.value) == 7

    def test_prism_act_identity_right_noop(self, ctx, graph):
        prism = identity_optic(name="P", functor=Functor("P", expr.Sum(expr.Id(), expr.Const(INT))), focus=INT)
        morphism = prism.act(ops.identity(INT))

        result = run(morphism, Terms.right(P.int32(7).value), ctx, graph)
        assert _int(result.value.value) == 7

    def test_traversal_maybe_act_identity_noop(self, ctx, graph):
        trav = identity_optic(name="T", functor=Functor("T", expr.Maybe(expr.Id())), focus=INT)
        morphism = trav.act(ops.identity(INT))

        result = run(morphism, P.just(P.int32(42)).value, ctx, graph)
        assert _int(result.value.value) == 42

    def test_traversal_list_act_identity_noop(self, ctx, graph):
        trav = identity_optic(name="T", functor=Functor("T", expr.List(expr.Id())), focus=INT)
        morphism = trav.act(ops.identity(INT))

        arg = P.list_([P.int32(1), P.int32(2), P.int32(3)]).value
        result = run(morphism, arg, ctx, graph)
        assert [_int(el) for el in result.value] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Double composition: three-level nesting
# ---------------------------------------------------------------------------

def test_triple_nested_lens_composition(ctx, graph):
    TRIPLE = ProductType(ProductType(PAIR, INT), INT)
    outer = identity_optic(name="O", functor=Functor("O", expr.Prod(expr.Id(), expr.Const(INT))), focus=ProductType(PAIR, INT))
    middle = identity_optic(name="M", functor=Functor("M", expr.Prod(expr.Id(), expr.Const(INT))), focus=PAIR)
    inner = identity_optic(name="I", functor=Functor("I", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)

    composed = outer.compose(middle).compose(inner)
    morphism = composed.act(_add_one())

    arg = P.pair(P.pair(P.pair(P.int32(10), P.int32(20)), P.int32(30)), P.int32(40)).value
    result = run(morphism, arg, ctx, graph)
    innermost = _pair_ints(result.value[0].value[0])
    mid_residue = _int(result.value[0].value[1])
    outer_residue = _int(result.value[1])
    assert innermost == (11, 20)
    assert mid_residue == 30
    assert outer_residue == 40


# ---------------------------------------------------------------------------
# Composed action: act with a non-trivial (composed) morphism
# ---------------------------------------------------------------------------

def test_lens_act_with_composed_morphism(ctx, graph):
    lens = identity_optic(name="L", functor=Functor("L", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
    double_add = ops.compose(_add_one(), _add_one())
    morphism = lens.act(double_add)

    result = run(morphism, P.pair(P.int32(10), P.int32(20)).value, ctx, graph)
    assert _pair_ints(result) == (12, 20)


# ---------------------------------------------------------------------------
# Lens on Unit residue (degenerate: F = Prod(Id, Const(Unit)) ≅ Id)
# ---------------------------------------------------------------------------

def test_lens_unit_residue(ctx, graph):
    f = Functor("Degen", expr.Prod(expr.Id(), expr.Const(UNIT)))
    source_type = ProductType(INT, UNIT)
    lens = Optic(
        functor=f,
        forward=ops.identity(source_type),
        backward=ops.identity(source_type),
    )
    morphism = lens.act(_add_one())

    result = run(morphism, P.pair(P.int32(10), P.unit()).value, ctx, graph)
    assert _int(result.value[0]) == 11
    assert result.value[1] == P.unit().value


# ---------------------------------------------------------------------------
# Prism: both branches exercised in sequence with same optic
# ---------------------------------------------------------------------------

def test_prism_same_optic_both_branches(ctx, graph):
    prism = identity_optic(name="P", functor=Functor("P", expr.Sum(expr.Id(), expr.Const(INT))), focus=INT)
    morphism = prism.act(_add_one())

    left_result = run(morphism, Terms.left(P.int32(10).value), ctx, graph)
    right_result = run(morphism, Terms.right(P.int32(10).value), ctx, graph)

    assert _int(left_result.value.value) == 11
    assert _int(right_result.value.value) == 10


# ---------------------------------------------------------------------------
# product with None carriers (should work; carrier=None in result)
# ---------------------------------------------------------------------------

def test_product_none_carriers_allowed(ctx, graph):
    f = Functor("F", expr.Prod(expr.Id(), expr.Const(INT)))
    o1 = Optic(functor=f, forward=ops.identity(PAIR), backward=ops.identity(PAIR))
    o2 = Optic(functor=f, forward=ops.identity(PAIR), backward=ops.identity(PAIR))
    par_optic = o1.product(o2)
    assert par_optic.carrier is None

    morphism = par_optic.act(_add_one())
    arg = P.pair(P.pair(P.int32(5), P.int32(6)), P.pair(P.int32(7), P.int32(8))).value
    result = run(morphism, arg, ctx, graph)
    assert _pair_ints(result.value[0]) == (6, 6)
    assert _pair_ints(result.value[1]) == (8, 8)


# ---------------------------------------------------------------------------
# Traversal: single-element list
# ---------------------------------------------------------------------------

def test_traversal_list_singleton(ctx, graph):
    trav = identity_optic(name="T", functor=Functor("T", expr.List(expr.Id())), focus=INT)
    morphism = trav.act(_add_one())

    result = run(morphism, P.list_([P.int32(99)]).value, ctx, graph)
    assert [_int(el) for el in result.value] == [100]


# ---------------------------------------------------------------------------
# Composition then product: (outer.compose(inner)).product(other)
# ---------------------------------------------------------------------------

def test_compose_then_product(ctx, graph):
    outer = identity_optic(name="O", functor=Functor("O", expr.Prod(expr.Id(), expr.Const(INT))), focus=PAIR)
    inner = identity_optic(name="I", functor=Functor("I", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)
    composed = outer.compose(inner)

    simple = identity_optic(name="S", functor=Functor("S", expr.Prod(expr.Id(), expr.Const(INT))), focus=INT)

    par_optic = composed.product(simple)
    morphism = par_optic.act(_add_one())

    left_arg = P.pair(P.pair(P.int32(10), P.int32(20)), P.int32(30))
    right_arg = P.pair(P.int32(40), P.int32(50))
    arg = P.pair(left_arg, right_arg).value
    result = run(morphism, arg, ctx, graph)

    left_inner = _pair_ints(result.value[0].value[0])
    left_residue = _int(result.value[0].value[1])
    right_pair = _pair_ints(result.value[1])
    assert left_inner == (11, 20)
    assert left_residue == 30
    assert right_pair == (41, 50)
