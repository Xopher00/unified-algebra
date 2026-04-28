"""Adjoint op suffix (*) and residual= semiring field."""

import numpy as np
import pytest
from dataclasses import replace

from unialg import NumpyBackend, Semiring
from unialg.algebra.contraction import compile_einsum, semiring_contract
from unialg import parse_ua


@pytest.fixture
def backend():
    return NumpyBackend()


# ---------------------------------------------------------------------------
# Direct semiring_contract tests (no DSL)
# ---------------------------------------------------------------------------

class TestAdjointContractDirect:
    """Verify adjoint dispatch using semiring_contract directly."""

    def test_residual_field_survives_round_trip(self, backend):
        """Semiring with residual= stores and resolves the residual op."""
        sr = Semiring("tropical", plus="minimum", times="add",
                      zero=float("inf"), one=0.0, residual="subtract")
        resolved = sr.resolve(backend)
        assert resolved.residual_name == "subtract"
        assert resolved.residual_elementwise is not None
        assert resolved.times_reduce is not None

    def test_adjoint_hook_uses_residual_and_times_reduce(self, backend):
        """Hook installed for adjoint=true uses residual_elementwise + times_reduce."""
        # Tropical semiring: plus=minimum, times=add.
        # Residual: subtract (b - a), so adjoint uses subtract + add_reduce.
        sr = Semiring("tropical", plus="minimum", times="add",
                      zero=float("inf"), one=0.0, residual="subtract")
        resolved = sr.resolve(backend)

        _res, _red = resolved.residual_elementwise, resolved.times_reduce
        sr_adj = replace(resolved, contraction_fn=lambda cs, be, p: cs(_res, _red))

        eq = compile_einsum("ij,kj->ik")
        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[4.0, 1.0], [2.0, 5.0]])

        result = semiring_contract(eq, [A, B], sr_adj, backend)

        # Adjoint: result[i,k] = add_reduce_j(subtract(A[i,j], B[k,j]))
        # = sum_j (A[i,j] - B[k,j])  [numpy subtract(a,b) = a - b]
        expected = np.array([
            [
                (A[0, 0] - B[0, 0]) + (A[0, 1] - B[0, 1]),
                (A[0, 0] - B[1, 0]) + (A[0, 1] - B[1, 1]),
            ],
            [
                (A[1, 0] - B[0, 0]) + (A[1, 1] - B[0, 1]),
                (A[1, 0] - B[1, 0]) + (A[1, 1] - B[1, 1]),
            ],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_adjoint_differs_from_forward_direct(self, backend):
        """Direct adjoint result differs from standard tropical contraction."""
        sr = Semiring("tropical", plus="minimum", times="add",
                      zero=float("inf"), one=0.0, residual="subtract")
        resolved = sr.resolve(backend)

        _res, _red = resolved.residual_elementwise, resolved.times_reduce
        sr_adj = replace(resolved, contraction_fn=lambda cs, be, p: cs(_res, _red))

        eq = compile_einsum("ij,kj->ik")
        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[4.0, 1.0], [2.0, 5.0]])

        forward = semiring_contract(eq, [A, B], resolved, backend)
        adjoint = semiring_contract(eq, [A, B], sr_adj, backend)

        assert not np.allclose(forward, adjoint), \
            "adjoint must differ from forward tropical contraction"

    def test_missing_residual_on_resolved_is_none(self, backend):
        """A semiring without residual= has residual_elementwise=None."""
        sr = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        resolved = sr.resolve(backend)
        assert resolved.residual_elementwise is None
        assert resolved.residual_name is None


# ---------------------------------------------------------------------------
# DSL end-to-end tests
# ---------------------------------------------------------------------------

class TestAdjointDSL:
    """Verify op* call-site adjoint annotation via parse_ua end-to-end."""

    def test_adjoint_parses_and_runs(self):
        """op* in a seq creates residuate__adj; calling it must return a valid result."""
        prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)

op join : mat -> mat
  einsum = "ij,jk->ik"
  algebra = tropical

op residuate : mat -> mat
  einsum = "ij,kj->ik"
  algebra = tropical

seq residuate_adj : mat -> mat = residuate*
''', NumpyBackend())

        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[4.0, 1.0], [2.0, 5.0]])
        result = prog('residuate__adj', A, B)
        assert result.shape == (2, 2)

    def test_adjoint_differs_from_forward(self):
        """The adjoint step must return a different result from the forward op."""
        prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)

op join : mat -> mat
  einsum = "ij,jk->ik"
  algebra = tropical

op residuate : mat -> mat
  einsum = "ij,kj->ik"
  algebra = tropical

seq residuate_adj : mat -> mat = residuate*
''', NumpyBackend())

        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[4.0, 1.0], [2.0, 5.0]])
        join_result = prog('join', A, B)
        adj_result = prog('residuate__adj', A, B)
        assert not np.allclose(join_result, adj_result), \
            "adjoint must differ from forward join"

    def test_adjoint_correct_values(self):
        """Adjoint result matches manual sum_j(A[i,j] - B[k,j]) for tropical."""
        prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)

op residuate : mat -> mat
  einsum = "ij,kj->ik"
  algebra = tropical

seq residuate_adj : mat -> mat = residuate*
''', NumpyBackend())

        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[4.0, 1.0], [2.0, 5.0]])
        result = prog('residuate__adj', A, B)

        # Adjoint: result[i,k] = sum_j(subtract(A[i,j], B[k,j]))
        # = sum_j(A[i,j] - B[k,j])  [numpy subtract(a,b) = a - b]
        expected = np.array([
            [
                (A[0, 0] - B[0, 0]) + (A[0, 1] - B[0, 1]),
                (A[0, 0] - B[1, 0]) + (A[0, 1] - B[1, 1]),
            ],
            [
                (A[1, 0] - B[0, 0]) + (A[1, 1] - B[0, 1]),
                (A[1, 0] - B[1, 0]) + (A[1, 1] - B[1, 1]),
            ],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_missing_residual_raises(self):
        """op* on a semiring without residual= must raise at compile time."""
        with pytest.raises((ValueError, AttributeError), match="adjoint|residual"):
            parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec vec(real)

op no_residual : vec -> vec
  einsum = "ij,j->i"
  algebra = real

seq no_residual_adj : vec -> vec = no_residual*
''', NumpyBackend())

    def test_residual_in_algebra_without_adjoint_op(self):
        """A semiring with residual= but no adjoint op runs normally (forward path)."""
        prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)

op join : mat -> mat
  einsum = "ij,jk->ik"
  algebra = tropical
''', NumpyBackend())

        A = np.array([[0.0, 1.0], [2.0, 0.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = prog('join', A, B)
        # Forward tropical matmul: result[i,k] = min_j(A[i,j] + B[j,k])
        expected = np.array([
            [min(0.0 + 1.0, 1.0 + 0.0), min(0.0 + 0.0, 1.0 + 1.0)],
            [min(2.0 + 1.0, 0.0 + 0.0), min(2.0 + 0.0, 0.0 + 1.0)],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_adjoint_field_on_equation(self):
        """op* in a seq creates a synthetic __adj equation with adjoint=True."""
        from unialg.parser import parse_ua_spec
        spec = parse_ua_spec('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)

op residuate : mat -> mat
  einsum = "ij,kj->ik"
  algebra = tropical

op join : mat -> mat
  einsum = "ij,jk->ik"
  algebra = tropical

seq residuate_adj : mat -> mat = residuate*
''')
        adj_eq = next(eq for eq in spec.equations if eq.name == 'residuate__adj')
        join_eq = next(eq for eq in spec.equations if eq.name == 'join')
        base_eq = next(eq for eq in spec.equations if eq.name == 'residuate')
        assert adj_eq.adjoint is True
        assert join_eq.adjoint is False
        assert base_eq.adjoint is False

    def test_residual_field_on_semiring(self):
        """Semiring.residual field is set when residual= is given in algebra declaration."""
        from unialg.parser import parse_ua_spec
        spec = parse_ua_spec('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0, residual=subtract)
spec mat(tropical)
op join : mat -> mat
  einsum = "ij,jk->ik"
  algebra = tropical
''')
        sr = spec.semirings['tropical']
        assert sr.residual == "subtract"
