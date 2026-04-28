"""Semiring contraction unit tests: semiring_contract over a user-provided backend."""

import numpy as np
import pytest

from unialg import NumpyBackend, Semiring
from unialg.algebra.contraction import compile_einsum, semiring_contract


@pytest.fixture
def real(backend):
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0).resolve(backend)


@pytest.fixture
def tropical(backend):
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0).resolve(backend)


@pytest.fixture
def fuzzy(backend):
    return Semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0,
                    bottom=0.0, top=1.0).resolve(backend)


class TestRealSemiring:

    def test_matrix_vector(self, real, backend):
        eq = compile_einsum("ij,j->i")
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        result = semiring_contract(eq, [W, x], real, backend)
        np.testing.assert_allclose(result, W @ x)

    def test_matrix_matrix(self, real, backend):
        eq = compile_einsum("ij,jk->ik")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = semiring_contract(eq, [A, B], real, backend)
        np.testing.assert_allclose(result, A @ B)

    def test_dot_product(self, real, backend):
        eq = compile_einsum("i,i->")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = semiring_contract(eq, [a, b], real, backend)
        np.testing.assert_allclose(result, np.dot(a, b))


class TestTropicalSemiring:

    def test_min_plus(self, tropical, backend):
        # Y_i = min_j(W_ij + x_j)
        eq = compile_einsum("ij,j->i")
        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])
        result = semiring_contract(eq, [W, x], tropical, backend)
        # Y_0 = min(1+1, 3+2) = min(2, 5) = 2
        # Y_1 = min(2+1, 0+2) = min(3, 2) = 2
        np.testing.assert_allclose(result, np.array([2.0, 2.0]))


class TestFuzzySemiring:

    def test_max_min(self, fuzzy, backend):
        # Y_i = max_j(min(W_ij, x_j))
        eq = compile_einsum("ij,j->i")
        W = np.array([[0.8, 0.3], [0.2, 0.9]])
        x = np.array([0.6, 0.7])
        result = semiring_contract(eq, [W, x], fuzzy, backend)
        # Y_0 = max(min(0.8,0.6), min(0.3,0.7)) = max(0.6, 0.3) = 0.6
        # Y_1 = max(min(0.2,0.6), min(0.9,0.7)) = max(0.2, 0.7) = 0.7
        np.testing.assert_allclose(result, np.array([0.6, 0.7]))


class TestBlockedContraction:
    """Blocked semiring_contract: results must be numerically identical to
    the unblocked path for all three semirings."""

    # ------------------------------------------------------------------
    # Real Semiring
    # ------------------------------------------------------------------

    def test_real_matvec_block2(self, real, backend):
        """Matrix-vector product blocked at size 2 matches unblocked."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(0).random((4, 6))
        x = np.random.default_rng(1).random((6,))
        expected = semiring_contract(eq, [W, x], real, backend)
        result = semiring_contract(eq, [W, x], real, backend, block_size=2)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_real_matvec_block1(self, real, backend):
        """block_size=1 (extreme: one element at a time) still correct."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(2).random((3, 5))
        x = np.random.default_rng(3).random((5,))
        expected = semiring_contract(eq, [W, x], real, backend)
        result = semiring_contract(eq, [W, x], real, backend, block_size=1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_real_matmul_block3(self, real, backend):
        """Matrix-matrix product with two reduction dims is blocked correctly."""
        eq = compile_einsum("ij,jk->ik")
        A = np.random.default_rng(4).random((3, 8))
        B = np.random.default_rng(5).random((8, 4))
        expected = semiring_contract(eq, [A, B], real, backend)
        result = semiring_contract(eq, [A, B], real, backend, block_size=3)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_real_block_larger_than_axis_is_noop(self, real, backend):
        """block_size > reduction axis size takes the fast path; result identical."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(6).random((4, 3))
        x = np.random.default_rng(7).random((3,))
        expected = semiring_contract(eq, [W, x], real, backend)
        # block_size=100 >> axis size 3 — should fall through to _contract_full
        result = semiring_contract(eq, [W, x], real, backend, block_size=100)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # ------------------------------------------------------------------
    # Tropical Semiring
    # ------------------------------------------------------------------

    def test_tropical_matvec_block2(self, tropical, backend):
        """Tropical (min-plus) blocked at 2 matches unblocked."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(8).random((4, 6))
        x = np.random.default_rng(9).random((6,))
        expected = semiring_contract(eq, [W, x], tropical, backend)
        result = semiring_contract(eq, [W, x], tropical, backend, block_size=2)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_tropical_matvec_block1(self, tropical, backend):
        """Tropical block_size=1 is still correct."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(10).random((3, 5))
        x = np.random.default_rng(11).random((5,))
        expected = semiring_contract(eq, [W, x], tropical, backend)
        result = semiring_contract(eq, [W, x], tropical, backend, block_size=1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # ------------------------------------------------------------------
    # Fuzzy Semiring
    # ------------------------------------------------------------------

    def test_fuzzy_matvec_block2(self, fuzzy, backend):
        """Fuzzy (max-min) blocked at 2 matches unblocked."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(12).random((4, 6))
        x = np.random.default_rng(13).random((6,))
        expected = semiring_contract(eq, [W, x], fuzzy, backend)
        result = semiring_contract(eq, [W, x], fuzzy, backend, block_size=2)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_fuzzy_matvec_block1(self, fuzzy, backend):
        """Fuzzy block_size=1 is still correct."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(14).random((3, 5))
        x = np.random.default_rng(15).random((5,))
        expected = semiring_contract(eq, [W, x], fuzzy, backend)
        result = semiring_contract(eq, [W, x], fuzzy, backend, block_size=1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_no_reduction_dims_block_ignored(self, real, backend):
        """When there are no reduced vars (output = all inputs), block_size is irrelevant."""
        # "ij->ij" — no reduction; result is just the input tensor
        eq = compile_einsum("ij->ij")
        A = np.random.default_rng(16).random((3, 4))
        expected = semiring_contract(eq, [A], real, backend)
        result = semiring_contract(eq, [A], real, backend, block_size=1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_block_size_none_unchanged(self, real, backend):
        """Passing block_size=None leaves behaviour identical to the original."""
        eq = compile_einsum("ij,j->i")
        W = np.random.default_rng(17).random((4, 5))
        x = np.random.default_rng(18).random((5,))
        expected = semiring_contract(eq, [W, x], real, backend)
        result = semiring_contract(eq, [W, x], real, backend, block_size=None)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestAutoBlockSize:
    """Auto block_size: identical results when available_memory forces blocking."""

    def test_auto_blocks_when_memory_tight(self, backend, real):
        """With tiny available_memory, auto-blocking kicks in and results match."""
        eq = compile_einsum("ij,jk->ik")
        A = np.random.RandomState(0).randn(10, 20).astype(np.float64)
        B = np.random.RandomState(1).randn(20, 15).astype(np.float64)

        expected = semiring_contract(eq, [A, B], real, backend, block_size=None)

        orig = backend.available_memory
        backend.available_memory = lambda: 64
        try:
            result = semiring_contract(eq, [A, B], real, backend)
        finally:
            backend.available_memory = orig

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_no_blocking_when_memory_abundant(self, backend, real):
        """With large available_memory, no blocking occurs (fast path)."""
        eq = compile_einsum("i,i->i")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        orig = backend.available_memory
        backend.available_memory = lambda: 10**12
        try:
            result = semiring_contract(eq, [a, b], real, backend)
        finally:
            backend.available_memory = orig

        np.testing.assert_allclose(result, a * b)

    def test_auto_block_ternary(self, backend, real):
        """Auto-blocking works for 3-operand einsums."""
        eq = compile_einsum("ik,jk,jl->il")
        A = np.random.RandomState(0).randn(5, 8).astype(np.float64)
        B = np.random.RandomState(1).randn(6, 8).astype(np.float64)
        C = np.random.RandomState(2).randn(6, 7).astype(np.float64)

        expected = semiring_contract(eq, [A, B, C], real, backend, block_size=1)

        orig = backend.available_memory
        backend.available_memory = lambda: 64
        try:
            result = semiring_contract(eq, [A, B, C], real, backend)
        finally:
            backend.available_memory = orig

        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestBackendArgmax:

    def test_argmax_matches_numpy(self):
        backend = NumpyBackend()
        x = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 4.0]])
        result = backend.argmax(x, axis=1)
        np.testing.assert_array_equal(result, np.argmax(x, axis=1))
