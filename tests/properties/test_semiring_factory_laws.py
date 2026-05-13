"""Property tests for emitters/semiring_factory invariants.

Contracts under test:
- semiring_from_backend always wires the exact Morphisms from BackendOps into
  the right Semiring fields, for any valid op pair.
- The carrier is always zero.cod.
- register_semiring_op always makes the registered name available in ops.
- reduce registration is precisely conditional on reduce_fn being provided.
- Registration never leaks between independent BackendOps instances.
"""

import string
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from unialg.emitters.backend import BackendOps
from unialg.emitters.semiring_factory import register_semiring_op, semiring_from_backend
from support.strategies import plain_morphisms


pytestmark = [pytest.mark.property]

SPEC = Path(__file__).parent.parent.parent / "src" / "unialg" / "emitters" / "backends" / "numpy.json"

# Module-level ops for strategy sampling — shared, read-only
_OPS = BackendOps.from_spec(SPEC)

# Op names that have a paired reduce.* in the backend
_OPS_WITH_REDUCE = sorted(
    name for name in _OPS.keys()
    if not name.startswith("reduce.") and f"reduce.{name}" in _OPS
)

# Op names that do NOT have a reduce.* (e.g. divide)
_OPS_WITHOUT_REDUCE = sorted(
    name for name in _OPS.keys()
    if not name.startswith("reduce.") and f"reduce.{name}" not in _OPS
)

op_with_reduce = st.sampled_from(_OPS_WITH_REDUCE)
op_without_reduce = st.sampled_from(_OPS_WITHOUT_REDUCE) if _OPS_WITHOUT_REDUCE else st.nothing()

safe_name = st.text(
    alphabet=string.ascii_lowercase + "_",
    min_size=4,
    max_size=14,
).filter(lambda n: n not in _OPS and not n.startswith("reduce"))


# ---------------------------------------------------------------------------
# semiring_from_backend — field wiring invariants
# ---------------------------------------------------------------------------

@settings(max_examples=60)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_plus_morphism_is_always_ops_lookup(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.plus is ops[plus_op]


@settings(max_examples=60)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_times_morphism_is_always_ops_lookup(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.times is ops[times_op]


@settings(max_examples=60)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_plus_reduce_is_always_reduce_dot_plus(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.plus_reduce is ops[f"reduce.{plus_op}"]


@settings(max_examples=60)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_times_reduce_is_always_reduce_dot_times(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.times_reduce is ops[f"reduce.{times_op}"]


@settings(max_examples=60)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_zero_and_one_pass_through_unchanged(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.zero is zero
    assert sr.one is one


@settings(max_examples=40)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_carrier_is_always_zero_cod(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.carrier == zero.cod


# ---------------------------------------------------------------------------
# semiring_from_backend — adjoint wiring
# ---------------------------------------------------------------------------

@settings(max_examples=40)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms(), op_with_reduce)
def test_adjoint_morphism_is_ops_lookup_when_present(plus_op, times_op, zero, one, adj_op):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops, adjoint_op=adj_op)
    assert sr.adjoint is ops[adj_op]


@settings(max_examples=40)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms(), op_with_reduce)
def test_adjoint_reduce_populated_when_adjoint_op_has_reduce(plus_op, times_op, zero, one, adj_op):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops, adjoint_op=adj_op)
    assert sr.adjoint_reduce is ops[f"reduce.{adj_op}"]


@pytest.mark.skipif(not _OPS_WITHOUT_REDUCE, reason="no ops without reduce in this spec")
@settings(max_examples=30)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms(), op_without_reduce)
def test_adjoint_reduce_is_none_when_adjoint_op_has_no_reduce(plus_op, times_op, zero, one, adj_op):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops, adjoint_op=adj_op)
    assert sr.adjoint_reduce is None


@settings(max_examples=40)
@given(op_with_reduce, op_with_reduce, plain_morphisms(), plain_morphisms())
def test_no_adjoint_op_leaves_adjoint_fields_none(plus_op, times_op, zero, one):
    ops = BackendOps.from_spec(SPEC)
    sr = semiring_from_backend("sr", plus_op, times_op, zero, one, ops)
    assert sr.adjoint is None
    assert sr.adjoint_reduce is None


# ---------------------------------------------------------------------------
# register_semiring_op — availability invariants
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(safe_name)
def test_registered_name_always_becomes_available(name):
    ops = BackendOps.from_spec(SPEC)
    register_semiring_op(name, np.add, ops)
    assert name in ops


@settings(max_examples=50)
@given(safe_name)
def test_no_reduce_registered_without_reduce_fn(name):
    ops = BackendOps.from_spec(SPEC)
    register_semiring_op(name, np.add, ops)
    assert f"reduce.{name}" not in ops


@settings(max_examples=50)
@given(safe_name)
def test_reduce_registered_with_reduce_fn(name):
    ops = BackendOps.from_spec(SPEC)
    register_semiring_op(name, np.add, ops,
                         reduce_fn=lambda x: np.add.reduce(x, axis=0))
    assert f"reduce.{name}" in ops


@settings(max_examples=40)
@given(safe_name)
def test_registration_does_not_leak_to_other_instances(name):
    ops_a = BackendOps.from_spec(SPEC)
    ops_b = BackendOps.from_spec(SPEC)
    register_semiring_op(name, np.add, ops_a)
    assert name not in ops_b


@settings(max_examples=40)
@given(safe_name, safe_name)
def test_two_registrations_are_independent(name_a, name_b):
    assume(name_a != name_b)
    ops = BackendOps.from_spec(SPEC)
    register_semiring_op(name_a, np.add, ops)
    register_semiring_op(name_b, np.multiply, ops)
    assert name_a in ops
    assert name_b in ops
