"""Phase 1 tests: semiring + contraction over a user-provided backend."""

import numpy as np
import pytest

from unified_algebra.backend import numpy_backend
from unified_algebra.semiring import semiring, resolve_semiring
from unified_algebra.contraction import compile_equation, semiring_contract


@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def real(backend):
    return resolve_semiring(
        semiring("real", plus="add", times="multiply", zero=0.0, one=1.0),
        backend,
    )


@pytest.fixture
def tropical(backend):
    return resolve_semiring(
        semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0),
        backend,
    )


@pytest.fixture
def fuzzy(backend):
    return resolve_semiring(
        semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0),
        backend,
    )


class TestRealSemiring:

    def test_matrix_vector(self, real, backend):
        eq = compile_equation("ij,j->i")
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        result = semiring_contract(eq, [W, x], real, backend)
        np.testing.assert_allclose(result, W @ x)

    def test_matrix_matrix(self, real, backend):
        eq = compile_equation("ij,jk->ik")
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = semiring_contract(eq, [A, B], real, backend)
        np.testing.assert_allclose(result, A @ B)

    def test_dot_product(self, real, backend):
        eq = compile_equation("i,i->")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = semiring_contract(eq, [a, b], real, backend)
        np.testing.assert_allclose(result, np.dot(a, b))


class TestTropicalSemiring:

    def test_min_plus(self, tropical, backend):
        # Y_i = min_j(W_ij + x_j)
        eq = compile_equation("ij,j->i")
        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])
        result = semiring_contract(eq, [W, x], tropical, backend)
        # Y_0 = min(1+1, 3+2) = min(2, 5) = 2
        # Y_1 = min(2+1, 0+2) = min(3, 2) = 2
        np.testing.assert_allclose(result, np.array([2.0, 2.0]))


class TestFuzzySemiring:

    def test_max_min(self, fuzzy, backend):
        # Y_i = max_j(min(W_ij, x_j))
        eq = compile_equation("ij,j->i")
        W = np.array([[0.8, 0.3], [0.2, 0.9]])
        x = np.array([0.6, 0.7])
        result = semiring_contract(eq, [W, x], fuzzy, backend)
        # Y_0 = max(min(0.8,0.6), min(0.3,0.7)) = max(0.6, 0.3) = 0.6
        # Y_1 = max(min(0.2,0.6), min(0.9,0.7)) = max(0.2, 0.7) = 0.7
        np.testing.assert_allclose(result, np.array([0.6, 0.7]))
