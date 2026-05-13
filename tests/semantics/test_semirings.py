"""Tests for Semiring dataclass and op_env() method."""

import pytest

from hydra.core import TypeUnit

from unialg.semantics import morphisms as ops
from unialg.tensors.semirings import Semiring


pytestmark = pytest.mark.semantics

UNIT = TypeUnit()


def _m():
    """Return a minimal Morphism for use as a placeholder field."""
    return ops.identity(UNIT)


@pytest.fixture
def full_semiring():
    plus = _m()
    times = _m()
    zero = _m()
    one = _m()
    adjoint = _m()
    plus_r = _m()
    times_r = _m()
    adj_r = _m()
    return Semiring(
        name="test",
        carrier=UNIT,
        plus=plus,
        times=times,
        zero=zero,
        one=one,
        adjoint=adjoint,
        plus_reduce=plus_r,
        times_reduce=times_r,
        adjoint_reduce=adj_r,
    )


@pytest.fixture
def partial_semiring():
    """Semiring without reduce fields (not constructed via factory)."""
    return Semiring(
        name="partial",
        carrier=UNIT,
        plus=_m(),
        times=_m(),
        zero=_m(),
        one=_m(),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_semiring_fields_default_to_none():
    sr = Semiring(name="x", carrier=UNIT, plus=_m(), times=_m(),
                  zero=_m(), one=_m())
    assert sr.adjoint is None
    assert sr.plus_reduce is None
    assert sr.times_reduce is None
    assert sr.adjoint_reduce is None


def test_semiring_is_frozen():
    sr = Semiring(name="x", carrier=UNIT, plus=_m(), times=_m(),
                  zero=_m(), one=_m())
    with pytest.raises((AttributeError, TypeError)):
        sr.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# op_env — standard mode
# ---------------------------------------------------------------------------

def test_op_env_standard_returns_times_and_plus_reduce(full_semiring):
    env = full_semiring.op_env(adjoint=False)
    assert env["product"] is full_semiring.times
    assert env["fold"] is full_semiring.plus_reduce
    assert env["seed"] is full_semiring.zero


def test_op_env_default_is_standard(full_semiring):
    assert full_semiring.op_env() == full_semiring.op_env(adjoint=False)


def test_op_env_standard_raises_when_plus_reduce_missing(partial_semiring):
    with pytest.raises(ValueError, match="plus_reduce"):
        partial_semiring.op_env(adjoint=False)


# ---------------------------------------------------------------------------
# op_env — adjoint mode
# ---------------------------------------------------------------------------

def test_op_env_adjoint_returns_adjoint_and_times_reduce(full_semiring):
    env = full_semiring.op_env(adjoint=True)
    assert env["product"] is full_semiring.adjoint
    assert env["fold"] is full_semiring.times_reduce
    assert env["seed"] is full_semiring.one


def test_op_env_adjoint_raises_when_adjoint_morphism_missing(partial_semiring):
    with pytest.raises(ValueError, match="adjoint"):
        partial_semiring.op_env(adjoint=True)


def test_op_env_adjoint_raises_when_times_reduce_missing():
    sr = Semiring(
        name="x", carrier=UNIT,
        plus=_m(), times=_m(), zero=_m(), one=_m(),
        adjoint=_m(),       # adjoint present
        plus_reduce=_m(),   # but times_reduce absent
    )
    with pytest.raises(ValueError, match="times_reduce"):
        sr.op_env(adjoint=True)
