import pytest
from hypothesis import given, settings

from unialg.syntax import expressions as expr
from support.strategies import (
    primitive_morphism_nodes,
    poly_values,
    product_type_values,
    sum_type_values,
    type_values,
)


pytestmark = [pytest.mark.semantics, pytest.mark.property]


@settings(max_examples=60)
@given(poly_values())
def test_all_generated_poly_nodes_are_polyexprs(body):
    assert isinstance(body, expr.PolyExpr)


@settings(max_examples=60)
@given(primitive_morphism_nodes())
def test_all_generated_morphism_nodes_are_morphismexprs(node):
    assert isinstance(node, expr.MorphismExpr)


@settings(max_examples=60)
@given(type_values())
def test_poly_leaf_constructors_are_class_sensitive(space):
    assert isinstance(expr.Const(space), expr.PolyExpr)
    assert expr.Zero() != expr.One()
    assert expr.Id() != expr.Const(space)


@settings(max_examples=60)
@given(poly_values(), poly_values())
def test_sum_and_product_order_is_structural(left, right):
    if left != right:
        assert expr.Sum(left, right) != expr.Sum(right, left)
        assert expr.Prod(left, right) != expr.Prod(right, left)


@settings(max_examples=60)
@given(poly_values(), poly_values())
def test_sum_and_product_nested_equality(left, right):
    assert expr.Sum(left, right) == expr.Sum(left, right)
    assert expr.Prod(left, right) == expr.Prod(left, right)


@settings(max_examples=60)
@given(poly_values(), poly_values())
def test_pretty_sum_is_recursive_in_children(left, right):
    assert expr.pretty(expr.Sum(left, right)) == f"{expr.pretty(left)} + {expr.pretty(right)}"


@settings(max_examples=60)
@given(poly_values(), poly_values())
def test_pretty_product_parenthesizes_sum_children(left, right):
    rendered = expr.pretty(expr.Prod(left, right))
    left_rendered = expr.pretty(left)
    right_rendered = expr.pretty(right)

    if isinstance(left, expr.Sum):
        assert rendered.startswith(f"({left_rendered}) * ")
    else:
        assert rendered.startswith(f"{left_rendered} * ")

    if isinstance(right, expr.Sum):
        assert rendered.endswith(f" * ({right_rendered})")
    else:
        assert rendered.endswith(f" * {right_rendered}")


@settings(max_examples=60)
@given(poly_values(), poly_values())
def test_pretty_exp_parenthesizes_sum_and_product_bodies(base, body):
    rendered = expr.pretty(expr.Exp(base, body))
    body_rendered = expr.pretty(body)
    base_rendered = expr.pretty(base)

    if isinstance(body, (expr.Sum, expr.Prod)):
        assert rendered == f"{base_rendered} -> ({body_rendered})"
    else:
        assert rendered == f"{base_rendered} -> {body_rendered}"


@settings(max_examples=60)
@given(type_values())
def test_pretty_basic_morphism_names(space):
    assert expr.pretty(expr.Identity(space)) == "id"
    assert expr.pretty(expr.Copy(space)) == "copy"
    assert expr.pretty(expr.Delete(space)) == "!"
    assert expr.pretty(expr.Absurd(space)) == "absurd"
    assert expr.pretty(expr.Prim(object(), space, space)) == "prim"


@settings(max_examples=60)
@given(product_type_values())
def test_pretty_product_projection_names(product):
    assert expr.pretty(expr.First(product)) == "π₁"
    assert expr.pretty(expr.Second(product)) == "π₂"


@settings(max_examples=60)
@given(sum_type_values())
def test_pretty_sum_injection_names(sum_type):
    assert expr.pretty(expr.Left(sum_type)) == "ι₁"
    assert expr.pretty(expr.Right(sum_type)) == "ι₂"


@settings(max_examples=60)
@given(primitive_morphism_nodes(), primitive_morphism_nodes(), type_values())
def test_pretty_contextual_morphism_composition_shapes(f, g, space):
    composed = expr.Compose(f, g, space, space, space, None, space, space)
    parallel = expr.Parallel(f, g, space, space, space, None, space, space)
    paired = expr.Pair(f, g, space, space, space, None, space, space)
    cased = expr.Case(f, g, space, space, space, None, space, space)

    assert expr.pretty(composed) == f"({expr.pretty(f)} ; {expr.pretty(g)})"
    assert expr.pretty(parallel) == f"({expr.pretty(f)} × {expr.pretty(g)})"
    assert expr.pretty(paired) == f"⟨{expr.pretty(f)}, {expr.pretty(g)}⟩"
    assert expr.pretty(cased) == f"[{expr.pretty(f)}, {expr.pretty(g)}]"
