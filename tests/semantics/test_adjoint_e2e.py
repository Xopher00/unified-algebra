"""E2E tests for the adjoint `'` suffix on einsum ops.

The adjoint swaps the semiring's times_elementwise for residual_elementwise
in the contraction, and uses times_reduce instead of plus_reduce.

For the real semiring with residual=divide:
  forward: out_i = sum_j(W_ij * x_j)           (multiply elementwise, add-reduce)
  adjoint: out_i = prod_j(W_ij / x_j)          (divide elementwise, multiply-reduce)

The `'` suffix on a cell equation reference synthesizes a new equation named
`{base}__adjoint`. Tests call equations directly via prog("op_name", W, x)
because cells wrapping einsum ops cannot pass multi-arg parameters through
Program.__call__ yet (known limitation in _morphism_compile._compose).
"""

import numpy as np
import pytest

from unialg import NumpyBackend, Semiring, parse_ua


_REAL_ADJ_BASE = """\
import numpy
algebra real_adj(plus=add, times=multiply, zero=0.0, one=1.0, residual=divide)
spec embed(real_adj)
"""

_W = np.array([[2.0, 4.0], [3.0, 6.0]])
_X = np.array([1.0, 2.0])


def _forward_oracle(W, x):
    return np.sum(W * x, axis=1)


def _adjoint_oracle(W, x):
    return np.prod(W / x, axis=1)


class TestWithAdjointErrorPath:

    def test_no_residual_raises(self):
        resolved = Semiring("real", plus="add", times="multiply",
                            zero=0.0, one=1.0).resolve(NumpyBackend())
        with pytest.raises(ValueError, match="residual"):
            resolved.with_adjoint("test_op")

    def test_residual_present_succeeds(self):
        resolved = Semiring("real_adj", plus="add", times="multiply",
                            zero=0.0, one=1.0, residual="divide").resolve(NumpyBackend())
        adj = resolved.with_adjoint("test_op")
        assert adj.contraction_fn is not None


class TestAdjointDSLExecution:

    _PROG = _REAL_ADJ_BASE + """\
op fwd_op : embed -> embed
  einsum = "ij,j->i"
  algebra = real_adj

cell adj : embed -> embed = fwd_op'
"""

    def test_adjoint_equation_synthesized(self):
        prog = parse_ua(self._PROG)
        eps = prog.entry_points()
        assert "fwd_op" in eps
        assert "fwd_op__adjoint" in eps

    def test_forward_matches_oracle(self):
        prog = parse_ua(self._PROG)
        result = prog("fwd_op", _W, _X)
        np.testing.assert_allclose(result, _forward_oracle(_W, _X), rtol=1e-10)

    def test_adjoint_matches_oracle(self):
        prog = parse_ua(self._PROG)
        result = prog("fwd_op__adjoint", _W, _X)
        np.testing.assert_allclose(result, _adjoint_oracle(_W, _X), rtol=1e-10)

    def test_forward_and_adjoint_differ(self):
        prog = parse_ua(self._PROG)
        fwd = prog("fwd_op", _W, _X)
        adj = prog("fwd_op__adjoint", _W, _X)
        assert not np.allclose(fwd, adj), "Forward and adjoint should differ"
