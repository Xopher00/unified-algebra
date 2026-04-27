"""Negative tests for Program: unknown entry points and error message quality."""

import numpy as np
import pytest

from unialg import (
    compile_program,
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
# Test 7: error path
# ---------------------------------------------------------------------------

class TestErrorPath:

    def test_unknown_entry_point_raises_valueerror(self, hidden, real_sr, backend):
        """Invoking an unknown entry point raises ValueError naming the entry."""
        eq = Equation("t7_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="nonexistent"):
            prog("nonexistent", np.array([1.0, 2.0]))

    def test_error_message_lists_available(self, hidden, real_sr, backend):
        """The ValueError message lists available entry points."""
        eq = Equation("t7b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="t7b_eq"):
            prog("wrong_name", np.array([1.0]))
