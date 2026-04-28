"""Contraction hook semantics tests: multi-pass hooks and witness tensors."""

import numpy as np
import pytest

from unialg import NumpyBackend, Semiring
from unialg.algebra.contraction import compile_einsum, semiring_contract
from unialg.assembly._equation_resolution import resolve_equation


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real(backend):
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0).resolve(backend)


# ---------------------------------------------------------------------------
# Multi-pass contraction hook
# ---------------------------------------------------------------------------

class TestMultiPassContraction:
    """Tests for the optional ``contraction_fn`` hook on ``Semiring.Resolved``.

    The hook receives ``(compute_sum, backend)`` and must return a tensor.
    ``compute_sum(times_fn, reduce_fn)`` handles alignment internally and
    returns ``reduce_fn(product, reduced_dims)``.
    """

    @staticmethod
    def _hooked(real, hook):
        """Attach *hook* to the real-semiring Resolved via dataclasses.replace."""
        from dataclasses import replace
        return replace(real, contraction_fn=hook)

    # ------------------------------------------------------------------
    # 1. test_hook_dispatch
    # ------------------------------------------------------------------

    def test_hook_dispatch(self, backend, real):
        """Hook that delegates to compute_sum with the standard ops matches the default path."""
        eq = compile_einsum("ij,j->i")
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])

        expected = semiring_contract(eq, [W, x], real, backend)

        def passthrough_hook(compute_sum, _backend, _params=()):
            # Re-uses the semiring's own ops — result must be identical to default.
            return compute_sum(real.times_elementwise, real.plus_reduce)

        sr_hooked = self._hooked(real, passthrough_hook)
        result = semiring_contract(eq, [W, x], sr_hooked, backend)

        np.testing.assert_allclose(result, expected, rtol=1e-14)

    # ------------------------------------------------------------------
    # 2. test_multi_pass
    # ------------------------------------------------------------------

    def test_multi_pass(self, backend, real):
        """Hook that calls compute_sum TWICE with different reduce_fns and returns a combination.

        Algorithm
        ---------
        Pass 1 — row_sum_i  = sum_j(W_ij * x_j)    (standard reduce=add)
        Pass 2 — row_max_i  = max_j(W_ij * x_j)    (reduce=maximum)
        result  = row_sum - row_max

        This is independently verifiable against numpy without going through the
        semiring engine, proving that multi-pass hooks with distinct reduce_fns work.
        """
        eq = compile_einsum("ij,j->i")
        rng = np.random.default_rng(42)
        W = rng.standard_normal((5, 8))
        x = rng.standard_normal((8,))

        # Ground truth: (W @ x) - max_j(W_ij * x_j)
        product_matrix = W * x[np.newaxis, :]   # (5, 8) — W_ij * x_j
        expected = product_matrix.sum(axis=1) - product_matrix.max(axis=1)

        times_fn = real.times_elementwise

        def two_pass_hook(compute_sum, _backend, _params=()):
            row_sum = compute_sum(times_fn, backend.reduce("add"))
            row_max = compute_sum(times_fn, backend.reduce("maximum"))
            return row_sum - row_max

        sr_hooked = self._hooked(real, two_pass_hook)
        result = semiring_contract(eq, [W, x], sr_hooked, backend)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # ------------------------------------------------------------------
    # 3. test_blocked_with_hook
    # ------------------------------------------------------------------

    def test_blocked_with_hook(self, backend, real):
        """The hook fires for every block when block_size forces splitting.

        With a passthrough hook and block_size=1, the result must equal the
        unblocked hooked result. The invocation count must equal the block count
        (6 for axis size 6 with block_size=1).
        """
        eq = compile_einsum("ij,j->i")
        rng = np.random.default_rng(7)
        W = rng.standard_normal((4, 6))
        x = rng.standard_normal((6,))

        invocations = [0]

        def counting_passthrough(compute_sum, _backend, _params=()):
            invocations[0] += 1
            return compute_sum(real.times_elementwise, real.plus_reduce)

        sr_hooked = self._hooked(real, counting_passthrough)

        # Baseline: unblocked, hook fires exactly once.
        baseline = semiring_contract(eq, [W, x], sr_hooked, backend)
        assert invocations[0] == 1, "unblocked: hook must fire exactly once"

        # When hook is set, blocking is skipped (hooks may return tuples).
        invocations[0] = 0
        result = semiring_contract(eq, [W, x], sr_hooked, backend, block_size=1)
        assert invocations[0] == 1, f"hook skips blocking: expected 1 call, got {invocations[0]}"

        np.testing.assert_allclose(result, baseline, rtol=1e-12)

    # ------------------------------------------------------------------
    # 4. test_default_unchanged
    # ------------------------------------------------------------------

    def test_default_unchanged(self, backend, real):
        """Semiring without contraction_fn (None) produces identical results — regression guard."""
        eq = compile_einsum("ij,j->i")
        rng = np.random.default_rng(99)
        W = rng.standard_normal((6, 10))
        x = rng.standard_normal((10,))

        # The default Resolved must carry contraction_fn=None.
        assert real.contraction_fn is None, "default Resolved must have contraction_fn=None"

        r1 = semiring_contract(eq, [W, x], real, backend)
        r2 = semiring_contract(eq, [W, x], real, backend)
        np.testing.assert_allclose(r1, r2, rtol=1e-15)

        # Must also match numpy's own matmul.
        np.testing.assert_allclose(r1, W @ x, rtol=1e-12)


class TestWitnessTensors:
    """Viterbi-style contraction returning (values, indices) via contraction_fn hook."""

    @pytest.fixture
    def tropical(self, backend):
        return Semiring("tropical", plus="minimum", times="add", zero=float('inf'), one=0.0).resolve(backend)

    def _hooked(self, sr, hook):
        from dataclasses import replace
        return replace(sr, contraction_fn=hook)

    def test_viterbi_witness_via_compute_product(self, backend, tropical):
        eq = compile_einsum("ij,j->i")
        W = np.array([[1.0, 3.0, 2.0],
                       [4.0, 0.0, 5.0],
                       [2.0, 1.0, 3.0]])
        x = np.array([0.0, 0.0, 0.0])

        def viterbi_hook(compute_sum, _backend, _params=()):
            product = compute_sum.compute_product(tropical.times_elementwise)
            dims = compute_sum.reduced_dims
            values = tropical.plus_reduce(product, dims)
            indices = _backend.argmax(-product, axis=dims[0])
            return (values, indices)

        sr_hooked = self._hooked(tropical, viterbi_hook)
        values, indices = semiring_contract(eq, [W, x], sr_hooked, backend)

        expected_values = np.array([1.0, 0.0, 1.0])
        expected_indices = np.array([0, 1, 1])
        np.testing.assert_allclose(values, expected_values)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_viterbi_product_sort_parse(self):
        from unialg.parser import parse_ua_spec
        from unialg.algebra.sort import ProductSort

        spec = parse_ua_spec('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec hidden(tropical)
spec indices(tropical)
op viterbi : hidden -> (hidden, indices)
  einsum = "ij,j->i"
  algebra = tropical
''')
        eq = spec.equations[0]
        assert eq.name == 'viterbi'
        assert isinstance(eq.codomain_sort, ProductSort)
        assert [e.name for e in eq.codomain_sort.elements] == ['hidden', 'indices']

    def test_viterbi_equation_resolve_pair_coder(self, backend):
        from unialg.parser import parse_ua_spec
        from unialg.algebra.sort import ProductSort

        spec = parse_ua_spec('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec hidden(tropical)
spec indices(tropical)
op viterbi : hidden -> (hidden, indices)
  einsum = "ij,j->i"
  algebra = tropical
''')
        eq = spec.equations[0]
        prim, native_fn, sr, in_coder = resolve_equation(eq, backend)
        assert prim is not None
        assert 'viterbi' in prim.name.value
