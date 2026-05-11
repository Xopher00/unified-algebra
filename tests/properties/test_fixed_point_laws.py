import pytest
from hypothesis import given, settings

from unialg.syntax import expressions as expr
from unialg.semantics import functors as sem
from unialg.semantics.morphisms import _identity, compose
from unialg.semantics.optics import Optic

from support.strategies import type_values


pytestmark = [pytest.mark.semantics, pytest.mark.property]


def carrier_optic(carrier):
    return Optic(
        functor=sem.Functor("Id", expr.Id()),
        forward=_identity(carrier),
        backward=_identity(carrier),
        carrier=carrier,
    )


# ---------------------------------------------------------------------------
# carrier optic boundary laws
# ---------------------------------------------------------------------------

@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_source_is_carrier(a):
    assert carrier_optic(a).source == a


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_target_is_carrier(a):
    assert carrier_optic(a).target == a


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_source_equals_target(a):
    fp = carrier_optic(a)
    assert fp.source == fp.target


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_forward_dom_is_carrier(a):
    assert carrier_optic(a).forward.dom() == a


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_forward_cod_is_functor_apply(a):
    fp = carrier_optic(a)
    assert fp.forward.cod() == fp.functor.apply(a)


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_backward_dom_is_functor_apply(a):
    fp = carrier_optic(a)
    assert fp.backward.dom() == fp.functor.apply(a)


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_backward_cod_is_carrier(a):
    assert carrier_optic(a).backward.cod() == a


@settings(max_examples=60)
@given(type_values())
def test_carrier_optic_forward_backward_compose(a):
    fp = carrier_optic(a)
    composed = compose(fp.forward, fp.backward)
    assert composed.dom() == a
    assert composed.cod() == a
