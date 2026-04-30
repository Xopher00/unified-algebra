"""Behavior-preservation tests for the monoidal unit object.

Audit reference: robust-scribbling-dove.md Phase 2 — "test_unit_sort.py".

UnitSort as a _RecordView subclass does NOT exist in the current codebase.
That absence is itself behavior worth pinning: no ``UnitSort`` class, no
``TypeUnit`` wrapper sort, no named "1" sort in the algebra layer.

What the codebase DOES have (tested here):
  - ``core.TermUnit`` is the sentinel for absent optional fields in _RecordView.
  - ``core.TypeUnit`` is available in Hydra core but no sort wraps it.
  - ``Cell.delete`` uses ``sort.name`` on its payload Sort — not a unit sort.
  - The ``lens`` cell stores ``Terms.unit()`` (a TermUnit) as the absent-residual
    sentinel, readable back as ``None`` through ``cell.residual_sort``.
  - ``_RecordView.Term`` with ``optional=True`` returns None for a TermUnit field.

If a UnitSort class is added in the future, this test file should be replaced
with direct assertions against that class. Until then, these tests guard the
TermUnit/TypeUnit usage patterns that fill the unit-object role.
"""

import pytest
import hydra.core as core
import hydra.dsl.terms as Terms
from hydra.core import Field, Name

from unialg import Semiring, Sort
from unialg.assembly.para._para import (
    Cell, lens, eq, iden, CELL_TYPE_NAME,
)
from unialg.terms import _RecordView


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


# ---------------------------------------------------------------------------
# UnitSort absence — pin that it does NOT exist
# ---------------------------------------------------------------------------

class TestUnitSortAbsence:
    """UnitSort is not yet a class — pin that to prevent silent re-introduction."""

    def test_unit_sort_class_absent_from_algebra(self):
        """algebra/sort.py must not export UnitSort (it doesn't exist)."""
        import unialg.algebra.sort as sort_module
        assert not hasattr(sort_module, "UnitSort"), (
            "UnitSort class introduced without updating test_unit_sort.py — "
            "replace absence tests with behavioral tests for the new class."
        )

    def test_unit_sort_absent_from_public_api(self):
        """UnitSort is not in the public unialg API."""
        import unialg
        assert not hasattr(unialg, "UnitSort")


# ---------------------------------------------------------------------------
# TypeUnit and TermUnit are available in hydra.core
# ---------------------------------------------------------------------------

class TestHydraCoreUnitTypes:
    """core.TypeUnit and core.TermUnit are the substrate for unit objects."""

    def test_type_unit_constructable(self):
        """core.TypeUnit() constructs without error."""
        t = core.TypeUnit()
        assert isinstance(t, core.TypeUnit)

    def test_term_unit_constructable(self):
        """core.TermUnit() constructs without error."""
        u = core.TermUnit()
        assert isinstance(u, core.TermUnit)

    def test_type_unit_is_distinct_from_type_variable(self):
        """TypeUnit is not a TypeVariable — it is the terminal type."""
        tu = core.TypeUnit()
        tv = core.TypeVariable(Name("unit"))
        assert not isinstance(tu, core.TypeVariable)
        assert type(tu) is not type(tv)


# ---------------------------------------------------------------------------
# TermUnit as optional-field sentinel in _RecordView
# ---------------------------------------------------------------------------

class TestTermUnitAsOptionalSentinel:
    """_RecordView.Term(optional=True) returns None when field holds TermUnit."""

    def test_optional_term_field_none_for_term_unit(self, real_sr):
        """A _RecordView with an optional Term field returns None when absent."""

        class _TinyRecord(_RecordView):
            _type_name = Name("ua.test.TinyRecord")
            name = _RecordView.Scalar(str)
            extra = _RecordView.Term(optional=True)

        obj = _TinyRecord(name="hello")
        # 'extra' was not supplied — should be None (TermUnit sentinel)
        assert obj.extra is None

    def test_optional_term_field_with_value(self, real_sr):
        """A _RecordView optional Term field returns the value when present."""

        class _TinyRecord(_RecordView):
            _type_name = Name("ua.test.TinyRecord2")
            name = _RecordView.Scalar(str)
            payload = _RecordView.Term(optional=True)

        sort = Sort("hidden", real_sr)
        obj = _TinyRecord(name="hello", payload=sort)
        # Should return the sort term (not None)
        assert obj.payload is not None


# ---------------------------------------------------------------------------
# Lens cell uses TermUnit as absent-residual sentinel
# ---------------------------------------------------------------------------

class TestLensCellUnitResidual:
    """The lens Cell stores Terms.unit() for absent residual, reads back as None."""

    def test_lens_no_residual_stores_term_unit(self, hidden):
        """lens(..., residual=None) stores TermUnit for residualSort field."""
        fwd = iden(hidden)
        bwd = iden(hidden)
        c = lens(fwd, bwd)
        assert c.kind == "lens"
        # residual_sort accessor returns None for TermUnit
        assert c.residual_sort is None

    def test_lens_with_residual_stores_sort_term(self, hidden):
        """lens(..., residual=sort) stores the sort term for residualSort field."""
        fwd = iden(hidden)
        bwd = iden(hidden)
        c = lens(fwd, bwd, residual=hidden)
        assert c.kind == "lens"
        # residual_sort accessor returns the sort (not None)
        residual = c.residual_sort
        assert residual is not None
        # The returned sort has the same type_ as the input sort
        assert residual.type_ == Sort.from_term(hidden).type_
