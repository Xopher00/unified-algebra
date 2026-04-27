"""Catamorphism validation error tests: fold and unfold constraint enforcement."""

import pytest

from hydra.dsl.python import FrozenDict

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    FoldSpec, UnfoldSpec,
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


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema(eq_by_name, extra_sorts=()):
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)


# ---------------------------------------------------------------------------
# Fold validation
# ---------------------------------------------------------------------------

class TestFoldValidation:

    def test_valid_fold(self, real_sr, hidden):
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        ebn = {"step": eq_step}
        FoldSpec("_", "step", None, hidden, hidden).validate(ebn, _schema(ebn))

    def test_fold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            FoldSpec("_", "missing", None, hidden, hidden).validate({}, _schema({}))

    def test_fold_state_recurrence_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", "i,i->i", hidden, output_sort, real_sr)
        ebn = {"step": eq_step}
        with pytest.raises(TypeError, match="codomain.*state sort"):
            FoldSpec("_", "step", None, hidden, hidden).validate(ebn, _schema(ebn, [hidden]))


# ---------------------------------------------------------------------------
# Unfold validation
# ---------------------------------------------------------------------------

class TestUnfoldValidation:

    def test_valid_unfold(self, real_sr, hidden):
        eq_step = Equation("step", None, hidden, hidden, nonlinearity="tanh")
        ebn = {"step": eq_step}
        UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, _schema(ebn))

    def test_unfold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            UnfoldSpec("_", "missing", 0, hidden, hidden).validate({}, _schema({}))

    def test_unfold_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", None, output_sort, hidden, nonlinearity="tanh")
        ebn = {"step": eq_step}
        with pytest.raises(TypeError, match="domain.*state sort"):
            UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, _schema(ebn, [hidden]))

    def test_unfold_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = Equation("step", None, hidden, output_sort, nonlinearity="tanh")
        with pytest.raises(TypeError, match="codomain.*state sort"):
            ebn = {"step": eq_step}
            UnfoldSpec("_", "step", 0, hidden, hidden).validate(ebn, _schema(ebn))
