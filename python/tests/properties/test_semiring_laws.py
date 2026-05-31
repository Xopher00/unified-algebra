"""Property tests for Semiring field-selection invariants.

Contracts under test:
- op_env(adjoint=False) always selects (times, plus_reduce, zero) regardless
  of what other optional fields are present.
- op_env(adjoint=True) always selects (adjoint, times_reduce, one) regardless
  of what other optional fields are present.
- Error-raising behaviour is unconditional when required fields are absent.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hydra.core import TypeUnit

from unialg.tensors.semirings import Semiring
from support.strategies import plain_morphisms


pytestmark = [pytest.mark.semantics, pytest.mark.property]

UNIT = TypeUnit()


def _semiring(
    plus=None, times=None, zero=None, one=None, adjoint=None,
    plus_reduce=None, times_reduce=None, adjoint_reduce=None,
):
    """Build a Semiring with minimal defaults for fields not under test."""
    from unialg.semantics import morphisms as ops
    from unialg.syntax import expressions as expr
    stub = ops.Morphism(expr.Prim(object(), UNIT, UNIT))
    return Semiring(
        name="test",
        carrier=UNIT,
        plus=plus or stub,
        times=times or stub,
        zero=0.0 if zero is None else zero,
        one=1.0 if one is None else one,
        adjoint=adjoint,
        plus_reduce=plus_reduce,
        times_reduce=times_reduce,
        adjoint_reduce=adjoint_reduce,
    )


# ---------------------------------------------------------------------------
# Standard mode (adjoint=False) — op_env selects times / plus_reduce / zero
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_standard_product_is_always_times(times, plus_reduce, zero):
    sr = _semiring(times=times, plus_reduce=plus_reduce, zero=zero)
    env = sr.op_env(adjoint=False)
    assert env["product"] is times


@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_standard_fold_is_always_plus_reduce(times, plus_reduce, zero):
    sr = _semiring(times=times, plus_reduce=plus_reduce, zero=zero)
    env = sr.op_env(adjoint=False)
    assert env["fold"] is plus_reduce


@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_standard_seed_is_always_zero(times, plus_reduce, zero):
    sr = _semiring(times=times, plus_reduce=plus_reduce, zero=zero)
    env = sr.op_env(adjoint=False)
    assert env["seed"] is zero


@settings(max_examples=50)
@given(
    plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False),
    # extra optional fields — should be ignored by standard op_env
    plain_morphisms(), plain_morphisms(), plain_morphisms(),
)
def test_op_env_standard_unaffected_by_optional_fields(
    times, plus_reduce, zero, adjoint, times_reduce, adjoint_reduce
):
    sr = _semiring(
        times=times, plus_reduce=plus_reduce, zero=zero,
        adjoint=adjoint, times_reduce=times_reduce, adjoint_reduce=adjoint_reduce,
    )
    env = sr.op_env(adjoint=False)
    assert env["product"] is times
    assert env["fold"] is plus_reduce
    assert env["seed"] is zero


# ---------------------------------------------------------------------------
# Adjoint mode (adjoint=True) — op_env selects adjoint / times_reduce / one
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_adjoint_product_is_adjoint_morphism(adjoint, times_reduce, one):
    sr = _semiring(adjoint=adjoint, times_reduce=times_reduce, one=one)
    env = sr.op_env(adjoint=True)
    assert env["product"] is adjoint


@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_adjoint_fold_is_times_reduce(adjoint, times_reduce, one):
    sr = _semiring(adjoint=adjoint, times_reduce=times_reduce, one=one)
    env = sr.op_env(adjoint=True)
    assert env["fold"] is times_reduce


@settings(max_examples=50)
@given(plain_morphisms(), plain_morphisms(), st.floats(allow_nan=False))
def test_op_env_adjoint_seed_is_one(adjoint, times_reduce, one):
    sr = _semiring(adjoint=adjoint, times_reduce=times_reduce, one=one)
    env = sr.op_env(adjoint=True)
    assert env["seed"] is one


# ---------------------------------------------------------------------------
# Error cases — always raise when required fields are absent
# ---------------------------------------------------------------------------

@settings(max_examples=40)
@given(plain_morphisms())
def test_op_env_standard_always_raises_without_plus_reduce(times):
    sr = _semiring(times=times)  # plus_reduce=None
    with pytest.raises(ValueError):
        sr.op_env(adjoint=False)


@settings(max_examples=40)
@given(plain_morphisms())
def test_op_env_adjoint_always_raises_without_adjoint_morphism(times_reduce):
    sr = _semiring(times_reduce=times_reduce)  # adjoint=None
    with pytest.raises(ValueError):
        sr.op_env(adjoint=True)


@settings(max_examples=40)
@given(plain_morphisms())
def test_op_env_adjoint_always_raises_without_times_reduce(adjoint):
    sr = _semiring(adjoint=adjoint)  # times_reduce=None
    with pytest.raises(ValueError):
        sr.op_env(adjoint=True)
