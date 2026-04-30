"""Behavior-preservation tests for _RecordView descriptor protocol.

Audit reference: robust-scribbling-dove.md Phase 2 — "test_terms_record_view.py".

Uses a synthetic _RecordView subclass (not production unialg classes) to pin:
  - Kwargs constructor populates fields correctly.
  - Each field accessor (Scalar, Term, TermList, ScalarList) returns the expected value.
  - from_term(instance.term) round-trips back to an equivalent object.
  - _fields returns a dict with the expected keys.
  - An optional Term field with no value returns None.
  - ScalarList with a default returns that default when field not provided.
"""

import pytest
import hydra.core as core
from hydra.core import Name

from unialg.terms import _RecordView
from unialg import Semiring, Sort


# ---------------------------------------------------------------------------
# Synthetic record — isolated from production classes
# ---------------------------------------------------------------------------

class _TinyView(_RecordView):
    """Minimal synthetic _RecordView for descriptor-protocol testing."""
    _type_name = Name("ua.test.TinyView")

    label       = _RecordView.Scalar(str)
    count       = _RecordView.Scalar(int, default=0)
    flag        = _RecordView.Scalar(bool, default=False)
    payload     = _RecordView.Term(optional=True)
    tags        = _RecordView.ScalarList(default=())


class _NestedView(_RecordView):
    """Record that holds a Term field pointing to another _RecordView."""
    _type_name = Name("ua.test.NestedView")

    name   = _RecordView.Scalar(str)
    inner  = _RecordView.Term(optional=True)
    items  = _RecordView.TermList(key="items")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def a_sort(real_sr):
    return Sort("alpha", real_sr)


# ---------------------------------------------------------------------------
# Scalar field tests
# ---------------------------------------------------------------------------

class TestScalarFields:
    """_RecordView.Scalar reads back primitive Python values."""

    def test_string_scalar(self):
        obj = _TinyView(label="hello")
        assert obj.label == "hello"

    def test_int_scalar(self):
        obj = _TinyView(label="x", count=42)
        assert obj.count == 42

    def test_bool_scalar(self):
        obj = _TinyView(label="x", flag=True)
        assert obj.flag is True

    def test_scalar_default_int(self):
        obj = _TinyView(label="x")
        assert obj.count == 0

    def test_scalar_default_bool(self):
        obj = _TinyView(label="x")
        assert obj.flag is False


# ---------------------------------------------------------------------------
# Optional Term field tests
# ---------------------------------------------------------------------------

class TestOptionalTermField:
    """_RecordView.Term(optional=True) returns None when absent."""

    def test_absent_returns_none(self):
        obj = _TinyView(label="x")
        assert obj.payload is None

    def test_present_returns_term(self, a_sort):
        obj = _TinyView(label="x", payload=a_sort)
        # Should return the term, not None
        assert obj.payload is not None

    def test_present_is_not_wrapped_record_view(self, a_sort):
        """Term field without a coerce returns the raw Hydra term (not a wrapper)."""
        obj = _TinyView(label="x", payload=a_sort)
        # payload has no coerce= on _TinyView, so it returns the raw term
        # (the Sort object's ._term)
        t = obj.payload
        assert isinstance(t, core.TermRecord), (
            f"Expected raw TermRecord, got {type(t).__name__}"
        )


# ---------------------------------------------------------------------------
# ScalarList field tests
# ---------------------------------------------------------------------------

class TestScalarListField:
    """_RecordView.ScalarList stores and retrieves lists of strings."""

    def test_tags_supplied(self):
        obj = _TinyView(label="x", tags=("a", "b", "c"))
        assert list(obj.tags) == ["a", "b", "c"]

    def test_tags_default_empty(self):
        obj = _TinyView(label="x")
        assert obj.tags == []

    def test_tags_round_trip_via_from_term(self):
        obj = _TinyView(label="x", tags=("x", "y"))
        obj2 = _TinyView.from_term(obj.term)
        assert list(obj2.tags) == ["x", "y"]


# ---------------------------------------------------------------------------
# TermList field tests
# ---------------------------------------------------------------------------

class TestTermListField:
    """_RecordView.TermList stores and retrieves lists of terms."""

    def test_items_empty(self, a_sort):
        obj = _NestedView(name="n", items=[])
        assert obj.items == []

    def test_items_one_element(self, a_sort):
        obj = _NestedView(name="n", items=[a_sort])
        items = obj.items
        assert len(items) == 1
        assert isinstance(items[0], core.TermRecord)

    def test_items_multiple_elements(self, real_sr):
        s1 = Sort("s1", real_sr)
        s2 = Sort("s2", real_sr)
        obj = _NestedView(name="n", items=[s1, s2])
        items = obj.items
        assert len(items) == 2


# ---------------------------------------------------------------------------
# _fields dict and from_term round-trip
# ---------------------------------------------------------------------------

class TestFieldsAndRoundTrip:
    """_fields returns a dict; from_term reconstructs equivalent object."""

    def test_fields_dict_has_expected_keys(self):
        obj = _TinyView(label="hello", count=7)
        fields = obj._fields
        assert isinstance(fields, dict)
        # label and count are explicit; count/flag/tags have defaults
        assert "label" in fields
        assert "count" in fields

    def test_from_term_round_trips_label(self):
        obj = _TinyView(label="roundtrip")
        obj2 = _TinyView.from_term(obj.term)
        assert obj2.label == "roundtrip"

    def test_from_term_round_trips_count(self):
        obj = _TinyView(label="x", count=99)
        obj2 = _TinyView.from_term(obj.term)
        assert obj2.count == 99

    def test_from_term_idempotent_on_instance(self):
        """from_term on an already-wrapped instance returns it unchanged."""
        obj = _TinyView(label="x")
        obj2 = _TinyView.from_term(obj)
        assert obj2._term is obj._term

    def test_term_property_returns_term_record(self):
        obj = _TinyView(label="x")
        assert isinstance(obj.term, core.TermRecord)

    def test_type_name_on_term(self):
        obj = _TinyView(label="x")
        assert obj.term.value.type_name == _TinyView._type_name
