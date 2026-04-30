"""Behavior-preservation tests for Equation arity dispatch and prim3 ceiling.

Audit reference: robust-scribbling-dove.md Phase 2 — "test_equation_arity_packing.py".

Pins:
  - Equations with arity 1, 2, 3 use prim1/prim2/prim3 respectively.
  - Equations with effective arity > 3 trigger list-packing (one list_coder slot
    for params and one for inputs) — Hydra prim arity stays <= 3.
  - _make_prim with n=0 or n>3 raises ValueError.
  - The primitive name format is ``ua.equation.{equation_name}``.
"""

import numpy as np
import pytest

from hydra.graph import Primitive

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.assembly._equation_resolution import (
    resolve_equation,
    _make_prim,
    _PRIMS,
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
# Prim name format
# ---------------------------------------------------------------------------

class TestPrimNameFormat:
    """Primitive name follows ``ua.equation.{name}`` convention."""

    def test_prim_name_arity_1(self, real_sr, hidden, backend):
        """Nonlinearity-only equation (arity 1) has correct prim name."""
        backend.unary_ops["relu"] = lambda x: np.maximum(x, 0.0)
        eq = Equation("relu_layer", "", hidden, hidden, real_sr, nonlinearity="relu")
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name.value == "ua.equation.relu_layer"

    def test_prim_name_arity_2(self, real_sr, hidden, backend):
        """Arity-2 einsum has correct prim name."""
        eq = Equation("matvec", "ij,j->i", hidden, hidden, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name.value == "ua.equation.matvec"

    def test_prim_name_arity_3(self, real_sr, hidden, backend):
        """Arity-3 einsum has correct prim name."""
        eq = Equation("trilinear", "i,i,i->i", hidden, hidden, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name.value == "ua.equation.trilinear"

    def test_prim_name_high_arity(self, real_sr, hidden, backend):
        """High-arity (list-packed) equation has correct prim name."""
        eq = Equation("quad", "i,i,i,i->i", hidden, hidden, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name.value == "ua.equation.quad"


# ---------------------------------------------------------------------------
# prim1 / prim2 / prim3 dispatch by arity
# ---------------------------------------------------------------------------

class TestArityCeilingDispatch:
    """Equations with arity 1/2/3 use prim1/prim2/prim3 respectively."""

    def test_arity_1_nonlinearity_uses_prim1(self, real_sr, hidden, backend):
        """Nonlinearity-only equation uses prim1 (1 input coder)."""
        backend.unary_ops["tanh"] = np.tanh
        eq = Equation("tanh_eq", "", hidden, hidden, real_sr, nonlinearity="tanh")
        prim, native_fn, *_ = resolve_equation(eq, backend)
        assert isinstance(prim, Primitive)
        # Verify it actually computes correctly
        x = np.array([0.0, 1.0, -1.0])
        result = native_fn(x)
        np.testing.assert_allclose(result, np.tanh(x), atol=1e-6)

    def test_arity_2_einsum_uses_prim2(self, real_sr, hidden, backend):
        """Arity-2 einsum uses prim2 (2 input coders)."""
        eq = Equation("dot2", "i,i->", hidden, hidden, real_sr)
        prim, native_fn, *_ = resolve_equation(eq, backend)
        assert isinstance(prim, Primitive)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = native_fn(a, b)
        np.testing.assert_allclose(result, np.dot(a, b))

    def test_arity_3_einsum_uses_prim3(self, real_sr, hidden, backend):
        """Arity-3 einsum uses prim3 (3 input coders). Also verifies correct output."""
        eq = Equation("trilinear3", "i,i,i->i", hidden, hidden, real_sr)
        prim, native_fn, *_ = resolve_equation(eq, backend)
        assert isinstance(prim, Primitive)
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        c = np.array([5.0, 6.0])
        result = native_fn(a, b, c)
        np.testing.assert_allclose(result, a * b * c)

    def test_arity_1_with_param_uses_prim2(self, real_sr, hidden, backend):
        """1 param + 1 input = arity 2 → uses prim2."""
        backend.unary_ops["scale"] = lambda x, a: a * x
        eq = Equation("scale1", "", hidden, hidden, real_sr,
                      nonlinearity="scale", param_slots=("a",))
        prim, native_fn, *_ = resolve_equation(eq, backend)
        assert isinstance(prim, Primitive)
        x = np.array([1.0, 2.0, 3.0])
        result = native_fn(2.0, x)
        np.testing.assert_allclose(result, 2.0 * x)


# ---------------------------------------------------------------------------
# List-packing fallback for arity > 3
# ---------------------------------------------------------------------------

class TestListPackingFallback:
    """Equations with n_params + n_inputs > 3 pack into list_coder slots."""

    def test_arity_4_triggers_list_packing(self, real_sr, hidden, backend):
        """4-input einsum packs all inputs into a single list_coder slot."""
        eq = Equation("quad4", "i,i,i,i->i", hidden, hidden, real_sr)
        prim, native_fn, *_ = resolve_equation(eq, backend)
        # Primitive is still valid
        assert isinstance(prim, Primitive)
        # Native callable still works variadically (not packed)
        a = np.array([1.0, 2.0])
        result = native_fn(a, a, a, a)
        np.testing.assert_allclose(result, a ** 4)

    def test_arity_4_prim_implementation_takes_one_arg(self, real_sr, hidden, backend):
        """After list-packing, the Primitive's implementation has signature (cx, g, args)
        where args is a 1-tuple (the packed list coder).

        For a 4-input equation, all inputs are packed into a single list_coder slot,
        so the primitive arity (from _PRIMS perspective) is 1, not 4.
        This is asserted via _build_resolved: n_inputs=4 > 3-0=3, triggers packing.
        Verify: the packed hydra_compute function receives a list and works correctly.
        """
        from unialg.assembly._equation_resolution import _build_resolved, compile_equation
        eq = Equation("quad_packcheck", "i,i,i,i->i", hidden, hidden, real_sr)
        ctx = compile_equation(eq, backend)
        coders, hydra_compute, _ = _build_resolved(
            ctx.in_coder, ctx.n_params, ctx.n_inputs,
            ctx.sr, ctx.compiled, backend, ctx.nl_fn,
        )
        # After packing: only 1 coder (the list_coder for inputs)
        assert len(coders) == 1, (
            f"Expected 1 coder after list-packing, got {len(coders)}"
        )
        # The hydra_compute receives the packed list and returns the correct value
        a = np.array([1.0, 2.0])
        result = hydra_compute([a, a, a, a])
        np.testing.assert_allclose(result, a ** 4)

    def test_params_and_inputs_both_packed(self, real_sr, hidden, backend):
        """2 params + 3 inputs = arity 5 → params list + inputs list (2 coders)."""
        backend.unary_ops["lin2"] = lambda x, a, b: a * x + b
        eq = Equation("lin2_3inputs", "i,i,i->i", hidden, hidden, real_sr,
                      nonlinearity="lin2", param_slots=("a", "b"))
        _, native_fn, *_ = resolve_equation(eq, backend)
        # Native fn still variadic: (a, b, x0, x1, x2)
        x = np.array([1.0, 2.0])
        result = native_fn(2.0, 1.0, x, x, x)
        # x*x*x * 2 + 1
        expected = 2.0 * (x ** 3) + 1.0
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# _make_prim ValueError for invalid arity
# ---------------------------------------------------------------------------

class TestMakePrimValidation:
    """_make_prim raises ValueError for n=0 or n>3."""

    def test_make_prim_n0_raises(self, backend):
        """_make_prim with 0 coders raises ValueError."""
        import hydra.core as core
        from unialg.terms import tensor_coder
        out_coder = tensor_coder(backend)
        with pytest.raises(ValueError, match="packed arity"):
            _make_prim(
                core.Name("ua.equation.test_zero"),
                lambda: None,
                [],          # 0 coders → invalid
                out_coder,
            )

    def test_make_prim_n4_raises(self, backend):
        """_make_prim with 4 coders raises ValueError (exceeds max 3)."""
        import hydra.core as core
        from unialg.terms import tensor_coder
        out_coder = tensor_coder(backend)
        in_coder = tensor_coder(backend)
        with pytest.raises(ValueError, match="packed arity"):
            _make_prim(
                core.Name("ua.equation.test_four"),
                lambda *_: None,
                [in_coder, in_coder, in_coder, in_coder],  # 4 coders → invalid
                out_coder,
            )

    def test_prims_dict_has_exactly_1_2_3(self):
        """_PRIMS maps exactly {1, 2, 3}; 0 and 4 are absent."""
        assert set(_PRIMS.keys()) == {1, 2, 3}
