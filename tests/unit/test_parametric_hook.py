"""Tests for parametric contraction hooks receiving runtime params.

A user registers a hook in CONTRACTION_REGISTRY.  The hook receives
(compute_sum, backend, params) where params is a tuple of float32 scalars
from the Equation's param_slots.

Two levels tested:
  1. Direct semiring_contract call — verifies the engine threads params to the hook.
  2. End-to-end via resolve_equation + native_fn — verifies the full assembly path.
"""

import numpy as np
import pytest

from unialg import Semiring, Sort, Equation
from unialg.algebra import CONTRACTION_REGISTRY
from unialg.algebra.contraction import compile_einsum, semiring_contract
from unialg.assembly._equation_resolution import resolve_equation


# ---------------------------------------------------------------------------
# Parametric hook: scales the real-semiring contraction result by params[0]
# ---------------------------------------------------------------------------

def _scale_hook(compute_sum, backend, params):
    """Compute the standard real dot product, then scale by params[0]."""
    import numpy as _np
    scale = params[0]
    result = compute_sum(lambda a, b: a * b, lambda t, dims: _np.sum(t, axis=dims))
    return scale * result


CONTRACTION_REGISTRY["scale_hook"] = _scale_hook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr_with_hook():
    """Real semiring that delegates contraction to scale_hook."""
    return Semiring("real_scale", plus="add", times="multiply",
                    zero=0.0, one=1.0, contraction="scale_hook")


@pytest.fixture
def hidden(real_sr_with_hook):
    return Sort("hidden", real_sr_with_hook)


# ---------------------------------------------------------------------------
# Level 1: direct semiring_contract call
# ---------------------------------------------------------------------------

class TestParametricHookDirect:
    """Verify params arrive at the hook when calling semiring_contract directly."""

    def test_hook_receives_params(self, real_sr_with_hook, backend):
        """Hook scales dot product by params[0]; result must equal scale * raw."""
        sr = real_sr_with_hook.resolve(backend, check_laws=False)
        einsum = compile_einsum("i,i->")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        raw_dot = float(np.dot(a, b))  # 4+10+18 = 32

        scale = 2.5
        result = semiring_contract(einsum, [a, b], sr, backend, params=(scale,))
        assert abs(float(result) - scale * raw_dot) < 1e-6

    def test_hook_receives_zero_scale(self, real_sr_with_hook, backend):
        """Scaling by 0 must give 0 regardless of inputs."""
        sr = real_sr_with_hook.resolve(backend, check_laws=False)
        einsum = compile_einsum("i,i->")
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = semiring_contract(einsum, [a, b], sr, backend, params=(0.0,))
        assert abs(float(result)) < 1e-9

    def test_hook_receives_matrix_params(self, real_sr_with_hook, backend):
        """Matrix-vector case: hook scales W @ x by params[0]."""
        sr = real_sr_with_hook.resolve(backend, check_laws=False)
        einsum = compile_einsum("ij,j->i")
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        expected = (W @ x) * 3.0
        result = semiring_contract(einsum, [W, x], sr, backend, params=(3.0,))
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Level 2: end-to-end via resolve_equation + native_fn
# ---------------------------------------------------------------------------

class TestParametricHookViaEquation:
    """Verify params flow from Equation.param_slots → native_fn → hook."""

    def test_param_slot_reaches_hook(self, real_sr_with_hook, hidden, backend):
        """Single param_slot 'scale' flows from native_fn call through to hook."""
        eq = Equation(
            "scaled_dot",
            "i,i->",
            hidden, hidden,
            real_sr_with_hook,
            param_slots=["scale"],
        )
        prim, native_fn, *_ = resolve_equation(eq, backend)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        raw_dot = float(np.dot(a, b))  # 32.0

        scale = 5.0
        result = native_fn(scale, a, b)
        assert abs(float(result) - scale * raw_dot) < 1e-6

    def test_different_param_values_produce_different_results(
            self, real_sr_with_hook, hidden, backend):
        """Different scale values must produce proportionally different results."""
        eq = Equation(
            "scaled_matmul",
            "ij,j->i",
            hidden, hidden,
            real_sr_with_hook,
            param_slots=["scale"],
        )
        prim, native_fn, *_ = resolve_equation(eq, backend)
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([2.0, 3.0])

        result_a = native_fn(1.0, W, x)
        result_b = native_fn(4.0, W, x)
        np.testing.assert_allclose(result_b, 4.0 * result_a)

    def test_prim_name_is_correct(self, real_sr_with_hook, hidden, backend):
        """Primitive is registered under the equation name."""
        eq = Equation(
            "named_scaled_op",
            "i,i->",
            hidden, hidden,
            real_sr_with_hook,
            param_slots=["scale"],
        )
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name.value == "ua.equation.named_scaled_op"
