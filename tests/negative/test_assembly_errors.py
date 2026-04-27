"""Negative tests for assembly: sort name mismatches and semiring mismatches."""

import pytest

from unialg import (
    Semiring, Sort, Equation, validate_pipeline,
)


# ---------------------------------------------------------------------------
# Sort name mismatch
# ---------------------------------------------------------------------------

class TestSortNameMismatch:

    def test_different_sort_names_rejected(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        output = Sort("output", real)
        eq1 = Equation("eq1", "ij,j->i", hidden, hidden, real)
        eq2 = Equation("eq2", "ij,j->i", output, output, real, inputs=("eq1",))
        with pytest.raises(TypeError):
            validate_pipeline([eq1, eq2])


# ---------------------------------------------------------------------------
# Semiring mismatch
# ---------------------------------------------------------------------------

class TestSemiringMismatch:

    def test_different_semiring_rejected(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        tropical = Semiring("tropical", plus="minimum", times="add",
                            zero=float("inf"), one=0.0)
        hidden_real = Sort("hidden", real)
        hidden_trop = Sort("hidden", tropical)
        eq1 = Equation("real_eq", "ij,j->i", hidden_real, hidden_real, real)
        eq2 = Equation("trop_eq", "ij,j->i", hidden_trop, hidden_trop, tropical, inputs=("real_eq",))
        with pytest.raises(TypeError):
            validate_pipeline([eq1, eq2])
