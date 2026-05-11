import pytest
from hypothesis import assume, given, settings

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor, apply_poly
from unialg.semantics.optics import Optic
from support.strategies import (
    optic_values,
    poly_values,
    type_values,
)


pytestmark = [pytest.mark.semantics, pytest.mark.property]


# ---------------------------------------------------------------------------
# Optic type laws
# ---------------------------------------------------------------------------

@settings(max_examples=60)
@given(optic_values())
def test_optic_source_equals_forward_domain(o):
    assert o.source == o.forward.dom()


@settings(max_examples=60)
@given(optic_values())
def test_optic_target_equals_backward_codomain(o):
    assert o.target == o.backward.cod()


@settings(max_examples=60)
@given(optic_values())
def test_optic_focus_via_functor_unapply_forward_cod(o):
    assert o.focus == o.functor.unapply(o.forward.cod())


@settings(max_examples=60)
@given(optic_values())
def test_optic_replacement_via_functor_unapply_backward_dom(o):
    assert o.replacement == o.functor.unapply(o.backward.dom())


@settings(max_examples=60)
@given(optic_values())
def test_optic_forward_cod_equals_functor_apply_focus(o):
    assert o.forward.cod() == o.functor.apply(o.focus)


@settings(max_examples=60)
@given(optic_values())
def test_optic_backward_dom_equals_functor_apply_replacement(o):
    assert o.backward.dom() == o.functor.apply(o.replacement)


# ---------------------------------------------------------------------------
# Optic rejection laws
# ---------------------------------------------------------------------------

@settings(max_examples=60)
@given(type_values(), type_values(), type_values())
def test_optic_rejects_functor_without_id(s, c, t):
    with pytest.raises(ops.MorphismError):
        Optic(
            functor=Functor("_", expr.Const(c)),
            forward=ops.Morphism(expr.Prim(object(), s, c)),
            backward=ops.Morphism(expr.Prim(object(), c, t)),
        )


@settings(max_examples=60)
@given(poly_values(), type_values(), type_values(), type_values(), type_values())
def test_optic_rejects_incompatible_forward_codomain(body, s, wrong, b, t):
    from support.strategies import count_id
    assume(count_id(body) > 0)
    assume(not isinstance(body, expr.Id))
    functor = Functor("_", body)
    try:
        functor.unapply(wrong)
        assume(False)
    except TypeError:
        pass
    with pytest.raises(ops.MorphismError):
        Optic(
            functor=functor,
            forward=ops.Morphism(expr.Prim(object(), s, wrong)),
            backward=ops.Morphism(expr.Prim(object(), apply_poly(body, b), t)),
        )
