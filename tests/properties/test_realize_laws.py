import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import hydra.lexical as L
import hydra.reduction as R
import hydra.sources.libraries as Libs
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
from hydra.core import Name, TypeUnit
from hydra.dsl.python import Right
from hydra.phantoms import TTerm

from unialg import expressions as expr
from unialg import morphisms as ops
from unialg import functors as sem
from unialg.actions import poly_fmap
from unialg.realize import realize
from unialg.lowering import run
from unialg.space import MAYBE
from support.strategies import INT, UNIT, maybe_lax_morphisms, parametric_plain_morphisms, plain_morphisms, poly_values


pytestmark = [pytest.mark.semantics, pytest.mark.property]


ADD = Name("hydra.lib.math.add")
MUL = Name("hydra.lib.math.mul")


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


def reduce_term(ctx, graph, term):
    raw = term.value if hasattr(term, "value") else term
    result = R.reduce_term(ctx, graph, True, raw)
    assert isinstance(result, Right), f"reduction failed: {result!r}"
    return result.value


def apply_realized(ctx, graph, node, arg):
    return reduce_term(ctx, graph, P.apply(TTerm(realize(node)), TTerm(arg)))


def int_arg(value: int):
    return P.int32(value).value


def int_value(term) -> int:
    return term.value.value.value


def pair_value(term) -> tuple[int, int]:
    return int_value(term.value[0]), int_value(term.value[1])


def int_pair_arg(left: int, right: int):
    return P.pair(P.int32(left), P.int32(right)).value


def add_const_raw(amount: int):
    x = P.var("x")
    return P.lam("x", P.primitive2(ADD, x, P.int32(amount))).value


def mul_const_raw(amount: int):
    x = P.var("x")
    return P.lam("x", P.primitive2(MUL, x, P.int32(amount))).value


def maybe_add_const_raw(amount: int):
    x = P.var("x")
    value = P.primitive2(ADD, x, P.int32(amount))
    return P.lam("x", P.apply(P.primitive(MAYBE.pure_name), value)).value


def maybe_int_value(term) -> int:
    return int_value(term.value.value)


def has_exp(body: expr.PolyExpr) -> bool:
    if isinstance(body, expr.Exp):
        return True
    if isinstance(body, (expr.Sum, expr.Prod)):
        return has_exp(body.left) or has_exp(body.right)
    return False


small_ints = st.integers(min_value=-20, max_value=20)


@settings(max_examples=40)
@given(small_ints)
def test_realize_identity_copy_delete_laws(ctx, graph, value):
    assert int_value(apply_realized(ctx, graph, expr.Identity(INT), int_arg(value))) == value
    assert pair_value(apply_realized(ctx, graph, expr.Copy(INT), int_arg(value))) == (value, value)
    assert apply_realized(ctx, graph, expr.Delete(INT), int_arg(value)) == P.unit().value


@settings(max_examples=40)
@given(small_ints, small_ints)
def test_realize_projection_laws(ctx, graph, left, right):
    product = ops.ProductType(INT, INT)
    arg = int_pair_arg(left, right)

    assert int_value(apply_realized(ctx, graph, expr.First(product), arg)) == left
    assert int_value(apply_realized(ctx, graph, expr.Second(product), arg)) == right


@settings(max_examples=40)
@given(small_ints)
def test_realize_sum_injection_laws(ctx, graph, value):
    sum_type = ops.SumType(INT, INT)

    left = apply_realized(ctx, graph, expr.Left(sum_type), int_arg(value))
    right = apply_realized(ctx, graph, expr.Right(sum_type), int_arg(value))

    assert int_value(left.value.value) == value
    assert int_value(right.value.value) == value


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints)
def test_realize_assoc_law(ctx, graph, q, p, a):
    dom = ops.ProductType(ops.ProductType(INT, INT), INT)
    cod = ops.ProductType(INT, ops.ProductType(INT, INT))
    arg = P.pair(P.pair(P.int32(q), P.int32(p)), P.int32(a)).value

    result = apply_realized(ctx, graph, expr.Assoc(dom, cod), arg)

    assert int_value(result.value[0]) == q
    assert pair_value(result.value[1]) == (p, a)


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints)
def test_realize_plain_contextual_laws(ctx, graph, value, add_amount, mul_amount):
    add = ops.Morphism(expr.Prim(add_const_raw(add_amount), INT, INT))
    mul = ops.Morphism(expr.Prim(mul_const_raw(mul_amount), INT, INT))

    assert int_value(run(ops.compose(add, mul), int_arg(value), ctx, graph)) == (value + add_amount) * mul_amount
    assert pair_value(run(ops.pair(add, mul), int_arg(value), ctx, graph)) == (value + add_amount, value * mul_amount)


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints, small_ints)
def test_realize_plain_parallel_law(ctx, graph, left, right, add_amount, mul_amount):
    add = ops.Morphism(expr.Prim(add_const_raw(add_amount), INT, INT))
    mul = ops.Morphism(expr.Prim(mul_const_raw(mul_amount), INT, INT))

    assert pair_value(run(ops.par(add, mul), int_pair_arg(left, right), ctx, graph)) == (
        left + add_amount,
        right * mul_amount,
    )


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints)
def test_realize_plain_case_law(ctx, graph, value, add_amount, mul_amount):
    add = ops.Morphism(expr.Prim(add_const_raw(add_amount), INT, INT))
    mul = ops.Morphism(expr.Prim(mul_const_raw(mul_amount), INT, INT))
    cased = ops.case(add, mul)

    left = Terms.apply(realize(ops._inl(cased.dom()).node), int_arg(value))
    right = Terms.apply(realize(ops._inr(cased.dom()).node), int_arg(value))

    assert int_value(run(cased, left, ctx, graph)) == value + add_amount
    assert int_value(run(cased, right, ctx, graph)) == value * mul_amount


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints)
def test_realize_lax_compose_law(ctx, graph, value, first_amount, second_amount):
    first = ops.Morphism(
        expr.Prim(maybe_add_const_raw(first_amount), INT, MAYBE.wrap(INT)),
        monad=MAYBE,
    )
    second = ops.Morphism(
        expr.Prim(maybe_add_const_raw(second_amount), INT, MAYBE.wrap(INT)),
        monad=MAYBE,
    )

    assert maybe_int_value(run(ops.compose(first, second), int_arg(value), ctx, graph)) == (
        value + first_amount + second_amount
    )


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints, small_ints)
def test_realize_lax_case_law(ctx, graph, left_value, right_value, left_amount, right_amount):
    left_branch = ops.Morphism(
        expr.Prim(maybe_add_const_raw(left_amount), INT, MAYBE.wrap(INT)),
        monad=MAYBE,
    )
    right_branch = ops.Morphism(
        expr.Prim(maybe_add_const_raw(right_amount), INT, MAYBE.wrap(INT)),
        monad=MAYBE,
    )
    cased = ops.case(left_branch, right_branch)

    left = Terms.apply(realize(ops._inl(cased.dom()).node), int_arg(left_value))
    right = Terms.apply(realize(ops._inr(cased.dom()).node), int_arg(right_value))

    assert maybe_int_value(run(cased, left, ctx, graph)) == left_value + left_amount
    assert maybe_int_value(run(cased, right, ctx, graph)) == right_value + right_amount


def test_realize_rejects_unknown_morphism_node():
    with pytest.raises(TypeError):
        realize(expr.MorphismExpr())


def test_realize_prim_returns_raw_payload():
    raw = object()
    assert realize(expr.Prim(raw, INT, INT)) is raw


@settings(max_examples=80)
@given(poly_values(), plain_morphisms())
def test_poly_fmap_plain_type_law(body, morphism):
    lifted = poly_fmap(sem.Functor("_", body), morphism)

    assert lifted.param == morphism.param
    assert lifted.monad == morphism.monad
    assert lifted.aux_primitives == morphism.aux_primitives
    assert lifted.dom() == sem.apply_poly(body, morphism.dom())
    assert lifted.cod() == sem.apply_poly(body, morphism.cod())


@settings(max_examples=80)
@given(poly_values(), parametric_plain_morphisms())
def test_poly_fmap_parametric_type_law(body, morphism):
    lifted = poly_fmap(sem.Functor("_", body), morphism)

    assert lifted.param == morphism.param
    assert lifted.monad == morphism.monad
    assert lifted.dom() == sem.apply_poly(body, morphism.dom())
    assert lifted.cod() == sem.apply_poly(body, morphism.cod())


@settings(max_examples=80)
@given(poly_values(), maybe_lax_morphisms())
def test_poly_fmap_lax_type_law_or_exp_rejection(body, morphism):
    if has_exp(body):
        with pytest.raises(TypeError):
            poly_fmap(sem.Functor("_", body), morphism)
        return

    lifted = poly_fmap(sem.Functor("_", body), morphism)

    assert lifted.param == morphism.param
    assert lifted.monad == morphism.monad
    assert lifted.dom() == sem.apply_poly(body, morphism.dom())
    assert lifted.cod() == sem.apply_poly(body, morphism.cod())


@settings(max_examples=40)
@given(small_ints, small_ints, small_ints)
def test_poly_fmap_runtime_for_id_prod_and_sum(ctx, graph, left_value, right_value, amount):
    add = ops.Morphism(expr.Prim(add_const_raw(amount), INT, INT))

    assert int_value(run(poly_fmap(sem.Functor("_", expr.Id()), add), int_arg(left_value), ctx, graph)) == left_value + amount

    prod_body = expr.Prod(expr.Id(), expr.Id())
    assert pair_value(run(poly_fmap(sem.Functor("_", prod_body), add), int_pair_arg(left_value, right_value), ctx, graph)) == (
        left_value + amount,
        right_value + amount,
    )

    sum_body = expr.Sum(expr.Id(), expr.Id())
    lifted_sum = poly_fmap(sem.Functor("_", sum_body), add)
    left = Terms.apply(realize(ops._inl(lifted_sum.dom()).node), int_arg(left_value))
    right = Terms.apply(realize(ops._inr(lifted_sum.dom()).node), int_arg(right_value))

    assert int_value(run(lifted_sum, left, ctx, graph).value.value) == left_value + amount
    assert int_value(run(lifted_sum, right, ctx, graph).value.value) == right_value + amount
