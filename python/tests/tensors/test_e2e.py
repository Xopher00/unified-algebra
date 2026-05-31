"""End-to-end tensor contraction execution tests."""

import numpy as np
from scipy.special import expit as scipy_sigmoid

from unialg import compile_program


def test_numpy_real_matmul_contract():
    prog = compile_program(
        """
        load numpy
        algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
        let matmul = contract[real]("ij,jk->ik")
        """
    )

    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    result = prog.run(left, right)

    assert np.allclose(result, left @ right)


def test_numpy_tropical_min_plus_matvec_contract():
    prog = compile_program(
        """
        load numpy
        algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
        let matvec = contract[tropical]("ij,j->i")
        """
    )

    matrix = np.array([[0.0, 4.0, 2.0], [5.0, 1.0, 3.0]])
    vector = np.array([7.0, 2.0, 6.0])

    result = prog.run(matrix, vector)
    expected = np.min(matrix + vector[None, :], axis=1)

    assert np.allclose(result, expected)


def test_numpy_tropical_matmul():
    """Tropical matrix-matrix product: C[i,k] = min_j(A[i,j] + B[j,k])."""
    prog = compile_program(
        """
        load numpy
        algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
        let tmatmul = contract[tropical]("ij,jk->ik")
        """
    )

    A = np.array([[0.0, 3.0], [2.0, 1.0]])
    B = np.array([[1.0, 4.0], [2.0, 0.0]])

    result = prog.run(A, B)
    # min_j(A[i,j] + B[j,k])
    expected = np.min(A[:, :, None] + B[None, :, :], axis=1)

    assert np.allclose(result, expected)


def test_numpy_tropical_inner_product():
    """Tropical inner product: min_i(a[i] + b[i])."""
    prog = compile_program(
        """
        load numpy
        algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
        let dot = contract[tropical]("i,i->")
        """
    )

    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    b = np.array([2.0, 7.0, 1.0, 8.0, 2.0])

    result = prog.run(a, b)
    expected = np.min(a + b)

    assert np.allclose(result, expected)


def test_numpy_matmul_then_sigmoid():
    """Tensor contraction composed with activation executes correctly."""
    prog = compile_program(
        """
        load numpy
        load extension tensors
        algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
        let f = contract[real]("ij,jk->ik") >> sigmoid
        """
    )

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[0.5, -0.5], [-1.0, 1.0]])

    result = prog.run(A, B)
    assert np.allclose(result, scipy_sigmoid(A @ B))


def test_numpy_matvec_then_softmax_with_inline_axis():
    """Tensor contraction composes with a typed configured operation."""
    prog = compile_program(
        """
        load numpy
        load extension tensors
        algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
        let f = contract[real]("ij,j->i") >> softmax(x, '-1')
        """
    )

    W = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
    x = np.array([1.0, 1.0, 1.0])

    result = prog.run(W, x)
    from scipy.special import softmax as scipy_softmax
    assert np.allclose(result, scipy_softmax(W @ x, axis=-1))
