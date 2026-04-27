"""Unit tests for equation arity packing."""

import numpy as np
import pytest

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    build_graph, Equation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


# ---------------------------------------------------------------------------
# Arity > 3: list-packing kicks in
# ---------------------------------------------------------------------------

class TestArityPacking:
    """Equations with n_params + n_inputs > 3 list-pack into Hydra prim slots
    while keeping the native callable variadic."""

    def test_four_tensor_inputs_native(self, real_sr, hidden, backend):
        """Einsum 'i,i,i,i->i' (arity 4): native_fn stays variadic."""
        eq = Equation("e4_native", "i,i,i,i->i", hidden, hidden, real_sr)
        prim, native_fn, *_ = eq.resolve(backend)
        # Hydra prim is packed to a single list_-coded slot
        assert prim.name.value == "ua.equation.e4_native"
        result = native_fn(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            np.array([5.0, 6.0]), np.array([7.0, 8.0]),
        )
        np.testing.assert_allclose(result, [1*3*5*7, 2*4*6*8])

    def test_four_tensor_inputs_via_program(self, real_sr, hidden, backend):
        """Full pipeline: compile_program + variadic call for arity 4."""
        from unialg import compile_program
        eq = Equation("e4_prog", "i,i,i,i->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)
        result = prog(
            "e4_prog",
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            np.array([5.0, 6.0]), np.array([7.0, 8.0]),
        )
        np.testing.assert_allclose(result, [105.0, 384.0])

    def test_three_inputs_with_two_params(self, real_sr, hidden, backend):
        """Arity 5: 3 tensor inputs + 2 scalar param_slots. Native stays variadic."""
        backend.unary_ops["scaled_add"] = lambda x, a, b: a * x + b
        eq = Equation("e5", "i,i,i->i", hidden, hidden, real_sr,
                      nonlinearity="scaled_add", param_slots=("a", "b"))
        prim, native_fn, *_ = eq.resolve(backend)
        # Variadic: 2 params then 3 tensors
        result = native_fn(2.0, 1.0, np.array([1.0]), np.array([2.0]), np.array([3.0]))
        # product 1*2*3=6; 2*6+1=13
        np.testing.assert_allclose(result, [13.0])
