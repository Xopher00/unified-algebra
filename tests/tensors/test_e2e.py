"""End-to-end tensor contraction execution tests."""

import numpy as np

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
