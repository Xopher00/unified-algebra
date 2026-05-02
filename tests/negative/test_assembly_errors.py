"""Negative tests for assembly: sort name mismatches and semiring mismatches."""

import pytest

from unialg import Semiring, Sort, Equation
from unialg.assembly._validation import validate_pipeline


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


class TestDimensionMismatch:

    def test_sized_axes_mismatch_rejected(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        h128 = Sort("hidden", real, axes=("feature:128",))
        h64 = Sort("hidden", real, axes=("feature:64",))
        eq1 = Equation("eq1", "i,i->i", h128, h128, real)
        eq2 = Equation("eq2", "i,i->i", h64, h64, real, inputs=("eq1",))
        with pytest.raises(TypeError, match="Dimension mismatch.*size 128.*size 64"):
            validate_pipeline([eq1, eq2])

    def test_unsized_axes_skip_dim_check(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        h = Sort("hidden", real, axes=("feature",))
        eq1 = Equation("eq1", "i,i->i", h, h, real)
        eq2 = Equation("eq2", "i,i->i", h, h, real, inputs=("eq1",))
        validate_pipeline([eq1, eq2])

    def test_mixed_sized_unsized_skip_dim_check(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        h_sized = Sort("hidden", real, axes=("feature:128",))
        h_unsized = Sort("hidden", real, axes=("feature",))
        eq1 = Equation("eq1", "i,i->i", h_sized, h_sized, real)
        eq2 = Equation("eq2", "i,i->i", h_unsized, h_unsized, real, inputs=("eq1",))
        validate_pipeline([eq1, eq2])
