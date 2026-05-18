"""Phase D — diagonal/trace semantics tests.

Covers:
- Equation parsing accepts repeated labels
- diagonal_axes and post_diagonal_labels methods
- Axis adjustment for iterative diagonal extraction
- Numerical parity with np.einsum for trace, diagonal, mixed contractions
"""
import numpy as np
import pytest

from unialg import compile_program


class TestDiagonalNumerical:
    """End-to-end numerical tests via DSL compile_program."""

    def test_trace_identity_matrix(self):
        """ii-> on identity matrix returns n."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let tr = contract[real]("ii->")
        """)
        assert np.isclose(prog.run(np.eye(5)), 5.0)

    def test_trace_general_matrix(self):
        """ii-> matches np.einsum."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let tr = contract[real]("ii->")
        """)
        rng = np.random.default_rng(0)
        A = rng.random((4, 4))
        assert np.isclose(prog.run(A), np.einsum("ii->", A))

    def test_diagonal_extraction(self):
        """ii->i extracts the diagonal vector."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let diag = contract[real]("ii->i")
        """)
        A = np.diag([10.0, 20.0, 30.0]) + np.ones((3, 3))
        result = prog.run(A)
        assert np.allclose(result, np.einsum("ii->i", A))

    def test_mixed_contraction_with_diagonal(self):
        """ij,jj->i: matrix times diagonal of second operand."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let mixed = contract[real]("ij,jj->i")
        """)
        rng = np.random.default_rng(1)
        B = rng.random((3, 4))
        D = rng.random((4, 4))
        result = prog.run(B, D)
        assert np.allclose(result, np.einsum("ij,jj->i", B, D))

    def test_diagonal_first_operand_matmul(self):
        """ii,ij->j: diagonal of first operand times second."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let f = contract[real]("ii,ij->j")
        """)
        rng = np.random.default_rng(2)
        A = rng.random((4, 4))
        B = rng.random((4, 5))
        result = prog.run(A, B)
        assert np.allclose(result, np.einsum("ii,ij->j", A, B))

    def test_triple_diagonal(self):
        """iii->i: hyper-diagonal of a 3D tensor."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let hdiag = contract[real]("iii->i")
        """)
        arr = np.zeros((3, 3, 3))
        for i in range(3):
            arr[i, i, i] = (i + 1) * 10.0
        result = prog.run(arr)
        assert np.allclose(result, np.einsum("iii->i", arr))

    def test_non_adjacent_diagonal(self):
        """iji->ij: diagonal over non-adjacent axes (0 and 2)."""
        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let f = contract[real]("iji->ij")
        """)
        rng = np.random.default_rng(3)
        arr = rng.random((3, 4, 3))
        result = prog.run(arr)
        expected = np.einsum("iji->ij", arr)
        assert np.allclose(result, expected)


class TestAxisAdjustment:
    """Test the axis adjustment algorithm for iterative diagonals."""

    def test_single_pair_no_adjustment(self):
        from unialg.tensors.primitives import _adjust_diagonal_axes
        assert _adjust_diagonal_axes([(0, 1)], ndim=2) == [(0, 1)]

    def test_triple_repeat(self):
        from unialg.tensors.primitives import _adjust_diagonal_axes
        result = _adjust_diagonal_axes([(0, 1), (0, 2)], ndim=3)
        assert result == [(0, 1), (0, 1)]

    def test_two_separate_pairs(self):
        """iijj: two independent diagonal pairs."""
        from unialg.tensors.primitives import _adjust_diagonal_axes
        result = _adjust_diagonal_axes([(0, 1), (2, 3)], ndim=4)
        assert result == [(0, 1), (0, 1)]

    def test_non_adjacent_axes(self):
        """iji: diagonal over axes 0 and 2."""
        from unialg.tensors.primitives import _adjust_diagonal_axes
        result = _adjust_diagonal_axes([(0, 2)], ndim=3)
        assert result == [(0, 2)]
