"""Tests for tensors/semiring_factory.py."""

import numpy as np
import pytest

from pathlib import Path

from unialg.emitters.backend import BackendOps
from unialg.tensors.semiring_factory import register_semiring_op, semiring_from_backend
from unialg.tensors.semirings import Semiring


SPEC = Path(__file__).parent.parent.parent / "src" / "unialg" / "emitters" / "backends" / "numpy.json"


@pytest.fixture
def ops():
    return BackendOps.from_spec(SPEC)


@pytest.fixture
def zero_morphism(ops):
    return ops["add"]   # placeholder Morphism; cod is FLOAT which is the carrier


@pytest.fixture
def one_morphism(ops):
    return ops["multiply"]


# ---------------------------------------------------------------------------
# semiring_from_backend
# ---------------------------------------------------------------------------

def test_semiring_from_backend_real(ops, zero_morphism, one_morphism):
    # divide has no reduce form in the numpy spec — adjoint_reduce will be None
    sr = semiring_from_backend("real", "add", "multiply", zero_morphism, one_morphism, ops,
                               adjoint_op="divide")
    assert isinstance(sr, Semiring)
    assert sr.name == "real"
    assert sr.plus is ops["add"]
    assert sr.times is ops["multiply"]
    assert sr.zero is zero_morphism
    assert sr.one is one_morphism
    assert sr.adjoint is ops["divide"]
    assert sr.plus_reduce is ops["reduce.add"]
    assert sr.times_reduce is ops["reduce.multiply"]
    assert sr.adjoint_reduce is None  # divide has no reduce.divide in backend spec


def test_semiring_from_backend_adjoint_reduce_populated_when_available(ops, zero_morphism, one_morphism):
    # logaddexp has both elementwise and reduce.logaddexp in the numpy spec
    sr = semiring_from_backend("smooth", "logaddexp", "add", zero_morphism, one_morphism, ops,
                               adjoint_op="logaddexp")
    assert sr.adjoint is ops["logaddexp"]
    assert sr.adjoint_reduce is ops["reduce.logaddexp"]


def test_semiring_from_backend_no_adjoint(ops, zero_morphism, one_morphism):
    sr = semiring_from_backend("tropical", "minimum", "add", zero_morphism, one_morphism, ops)
    assert sr.adjoint is None
    assert sr.adjoint_reduce is None
    assert sr.plus is ops["minimum"]
    assert sr.times is ops["add"]
    assert sr.plus_reduce is ops["reduce.minimum"]
    assert sr.times_reduce is ops["reduce.add"]


def test_semiring_from_backend_reduce_fields_populated(ops, zero_morphism, one_morphism):
    sr = semiring_from_backend("real", "add", "multiply", zero_morphism, one_morphism, ops)
    assert sr.plus_reduce is not None
    assert sr.times_reduce is not None


def test_semiring_from_backend_missing_op_raises(ops, zero_morphism, one_morphism):
    with pytest.raises(KeyError, match="nonexistent"):
        semiring_from_backend("bad", "nonexistent", "add", zero_morphism, one_morphism, ops)


def test_semiring_from_backend_missing_reduce_raises(ops, zero_morphism, one_morphism):
    # Register elementwise only — no reduce variant
    ops2 = BackendOps.from_spec(SPEC)
    register_semiring_op("myop", np.add, ops2)   # no reduce_fn
    with pytest.raises(KeyError, match="reduce.myop"):
        semiring_from_backend("x", "myop", "add", zero_morphism, one_morphism, ops2)


# ---------------------------------------------------------------------------
# register_semiring_op
# ---------------------------------------------------------------------------

def test_register_elementwise_only(ops):
    ops2 = BackendOps.from_spec(SPEC)
    register_semiring_op("custom_add", np.add, ops2)
    assert "custom_add" in ops2
    assert "reduce.custom_add" not in ops2


def test_register_with_reduce(ops):
    ops2 = BackendOps.from_spec(SPEC)
    register_semiring_op("custom_max", np.maximum, ops2,
                         reduce_fn=lambda x: np.maximum.reduce(x, axis=0))
    assert "custom_max" in ops2
    assert "reduce.custom_max" in ops2


def test_register_does_not_affect_other_instance(ops):
    ops2 = BackendOps.from_spec(SPEC)
    register_semiring_op("isolated_op", np.add, ops2)
    assert "isolated_op" not in ops   # original ops unchanged


def test_registered_op_usable_in_semiring_from_backend(ops, zero_morphism, one_morphism):
    ops2 = BackendOps.from_spec(SPEC)
    register_semiring_op(
        "smooth_plus",
        np.logaddexp,
        ops2,
        reduce_fn=lambda x: np.logaddexp.reduce(x, axis=0),
    )
    sr = semiring_from_backend(
        "logsumexp", "smooth_plus", "add", zero_morphism, one_morphism, ops2
    )
    assert sr.name == "logsumexp"
    assert sr.plus is ops2["smooth_plus"]
    assert sr.plus_reduce is ops2["reduce.smooth_plus"]
