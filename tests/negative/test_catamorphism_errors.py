"""Catamorphism validation error tests: fold and unfold constraint enforcement."""

import pytest

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    FoldSpec, UnfoldSpec,
)
from conftest import build_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


# ---------------------------------------------------------------------------
# Fold validation
# ---------------------------------------------------------------------------

class TestFoldValidation:

    def test_valid_fold(self, real_sr, hidden):
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        ebn = {"step": eq_step}
        FoldSpec("_", "step", None, hidden, hidden).validate(ebn, build_schema(ebn))

    def test_fold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            FoldSpec("_", "missing", None, hidden, hidden).validate({}, build_schema({}))

    def test_fold_state_recurrence_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", "i,i->i", hidden, output_sort, real_sr)
        ebn = {"step": eq_step}
        with pytest.raises(TypeError, match="codomain.*state sort"):
            FoldSpec("_", "step", None, hidden, hidden).validate(ebn, build_schema(ebn, [hidden]))


# ---------------------------------------------------------------------------
# Unfold validation
# ---------------------------------------------------------------------------

class TestUnfoldValidation:

    def test_valid_unfold(self, real_sr, hidden):
        eq_step = Equation("step", None, hidden, hidden, nonlinearity="tanh")
        ebn = {"step": eq_step}
        UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, build_schema(ebn))

    def test_unfold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            UnfoldSpec("_", "missing", 0, hidden, hidden).validate({}, build_schema({}))

    def test_unfold_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", None, output_sort, hidden, nonlinearity="tanh")
        ebn = {"step": eq_step}
        with pytest.raises(TypeError, match="domain.*state sort"):
            UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, build_schema(ebn, [hidden]))

    def test_unfold_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", None, hidden, output_sort, nonlinearity="tanh")
        with pytest.raises(TypeError, match="codomain.*state sort"):
            ebn = {"step": eq_step}
            UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, build_schema(ebn))
