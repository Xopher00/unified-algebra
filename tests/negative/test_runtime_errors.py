"""Runtime error tests: wrong shapes, wrong input counts at execution time."""

import numpy as np
import pytest

from unialg import (
    compile_program, parse_ua,
    Semiring, Sort, Equation, NumpyBackend,
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
# Shape errors
# ---------------------------------------------------------------------------

class TestRuntimeShapeErrors:

    def test_wrong_rank_raises(self, hidden, real_sr, backend):
        """Passing a 2D array where the einsum expects 1D raises ValueError.

        The einsum 'ij,j->i' expects a weight matrix (2D) and a vector (1D).
        Passing a 2D array as the second argument creates a rank mismatch when
        the contraction attempts to transpose/align dims.
        """
        eq = Equation("lin_re", "ij,j->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)
        W = np.eye(3)
        x_wrong = np.ones((3, 3))  # should be 1D, not 2D
        with pytest.raises(ValueError):
            prog("lin_re", W, x_wrong)

    def test_wrong_rank_error_message(self, hidden, real_sr, backend):
        """The ValueError from a rank mismatch is not swallowed — message is non-empty."""
        eq = Equation("lin_re2", "ij,j->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)
        W = np.eye(4)
        x_wrong = np.ones((4, 4))
        with pytest.raises(ValueError) as exc_info:
            prog("lin_re2", W, x_wrong)
        assert str(exc_info.value)


# ---------------------------------------------------------------------------
# Input count errors
# ---------------------------------------------------------------------------

class TestRuntimeInputCountErrors:

    def test_too_few_inputs_einsum_raises(self, hidden, real_sr, backend):
        """Calling an einsum entry point with too few arguments raises IndexError.

        The fast compiled path for 'ij,j->i' expects (W, x). Passing only W
        means the native function receives one argument but tries to index two.
        """
        eq = Equation("lin_ic", "ij,j->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)
        W = np.eye(3)
        with pytest.raises(IndexError):
            prog("lin_ic", W)  # missing x

    def test_too_few_inputs_nonlinearity_raises(self, hidden, real_sr, backend):
        """Calling a nonlinearity entry point with no arguments raises IndexError.

        The compiled path for a nonlinearity-only equation expects exactly one
        input tensor. Calling with no args leaves the native function with an
        empty positional argument list.
        """
        eq = Equation("relu_ic", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        with pytest.raises(IndexError):
            prog("relu_ic")  # missing input tensor


# ---------------------------------------------------------------------------
# DSL-level runtime errors
# ---------------------------------------------------------------------------

class TestDSLRuntimeErrors:

    def test_parse_ua_missing_input_raises(self):
        """Calling a compiled DSL program with no input tensor raises IndexError."""
        prog = parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu : hidden -> hidden
  nonlinearity = relu
""",
            NumpyBackend(),
        )
        with pytest.raises(IndexError):
            prog("relu")  # no input tensor supplied

    def test_parse_ua_wrong_rank_raises(self):
        """Passing a 2D array to a 1D-expecting DSL op raises ValueError."""
        prog = parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op lin : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
""",
            NumpyBackend(),
        )
        W = np.eye(3)
        x_wrong = np.ones((3, 3))
        with pytest.raises(ValueError):
            prog("lin", W, x_wrong)
