"""Fixpoint validation error tests: constraint enforcement via FixpointSpec.validate()."""

import pytest

from hydra.dsl.python import FrozenDict

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    FixpointSpec,
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


# ===========================================================================
# Fixpoint validation
# ===========================================================================

class TestFixpointValidation:
    """validate_fixpoint enforces endomorphism and predicate domain constraints."""

    def test_validate_fixpoint_passes_for_valid_step_and_predicate(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint passes when step is endomorphism and predicate domain matches.

        The predicate codomain must differ from the state sort (it should be a scalar
        float32, not a tensor). output_sort stands in for a non-state codomain here;
        full float32 enforcement requires a prim1-level predicate (see TestFixpointEndToEnd).
        """
        step_eq = Equation("fp_step", None, hidden, hidden, nonlinearity="relu")
        pred_eq = Equation("fp_pred", None, hidden, output_sort, nonlinearity="abs")
        eq_by_name = {"fp_step": step_eq, "fp_pred": pred_eq}
        FixpointSpec("_", "fp_step", "fp_pred", 0.0, 0, hidden).validate(eq_by_name, _schema(eq_by_name))

    def test_validate_fixpoint_raises_for_endomorphism_predicate(
        self, hidden, real_sr
    ):
        """validate_fixpoint raises when predicate codomain == state sort (endomorphism)."""
        step_eq = Equation("endo_step", None, hidden, hidden, nonlinearity="relu")
        pred_eq = Equation("endo_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"endo_step": step_eq, "endo_pred": pred_eq}
        with pytest.raises(TypeError, match="scalar residual"):
            FixpointSpec("_", "endo_step", "endo_pred", 0.0, 0, hidden).validate(eq_by_name, _schema(eq_by_name))

    def test_validate_fixpoint_raises_for_non_endomorphism_step(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint raises when step maps hidden -> output (not endo)."""
        step_eq = Equation("bad_step", "i->j", hidden, output_sort, real_sr)
        pred_eq = Equation("bad_pred", None, hidden, output_sort, nonlinearity="abs")
        eq_by_name = {"bad_step": step_eq, "bad_pred": pred_eq}
        with pytest.raises(TypeError):
            FixpointSpec("_", "bad_step", "bad_pred", 0.0, 0, hidden).validate(eq_by_name, _schema(eq_by_name))

    def test_validate_fixpoint_raises_for_missing_predicate(self, hidden):
        """validate_fixpoint raises ValueError when predicate equation is not found."""
        step_eq = Equation("ms_step", None, hidden, hidden, nonlinearity="relu")
        eq_by_name = {"ms_step": step_eq}
        with pytest.raises(ValueError, match="op 'missing_pred' not found"):
            FixpointSpec("_", "ms_step", "missing_pred", 0.0, 0, hidden).validate(eq_by_name, _schema(eq_by_name))

    def test_validate_fixpoint_raises_for_missing_step(self, hidden):
        """validate_fixpoint raises ValueError when step equation is not found."""
        pred_eq = Equation("ms_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"ms_pred": pred_eq}
        with pytest.raises(ValueError, match="op 'missing_step' not found"):
            FixpointSpec("_", "missing_step", "ms_pred", 0.0, 0, hidden).validate(eq_by_name, _schema(eq_by_name))
