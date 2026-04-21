"""Read-only view wrappers for Hydra record terms.

Each view wraps a raw Hydra Term and exposes its fields as @property
accessors. The Term stays the source of truth — views never copy data.
"""

from .utils import record_fields, string_value
import hydra.core as core


# ---------------------------------------------------------------------------
# Field descriptors — eliminate per-property boilerplate
# ---------------------------------------------------------------------------

class _StringField:
    """Descriptor: extracts a string field from the Hydra record."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return string_value(record_fields(obj._term)[self._key])

class _FloatField:
    """Descriptor: extracts a float field from the Hydra record."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return record_fields(obj._term)[self._key].value.value

class _TermField:
    """Descriptor: extracts a raw Term field (no conversion)."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return record_fields(obj._term)[self._key]


class _RecordView:
    """Base for all Hydra record views."""
    __slots__ = ("_term",)
    def __init__(self, term: core.Term):
        self._term = term
    @property
    def term(self) -> core.Term:
        return self._term


# ---------------------------------------------------------------------------
# Concrete views
# ---------------------------------------------------------------------------

class EquationView(_RecordView):
    """View over a ua.equation.Equation record term."""
    name = _StringField("name")
    einsum = _StringField("einsum")
    nonlinearity = _StringField("nonlinearity")
    domain_sort = _TermField("domainSort")
    codomain_sort = _TermField("codomainSort")
    semiring = _TermField("semiring")

    @property
    def inputs(self) -> list[str]:
        return [string_value(t) for t in record_fields(self._term)["inputs"].value]

    @property
    def param_slots(self) -> list[str]:
        ps = record_fields(self._term).get("paramSlots")
        if ps is None:
            return []
        if hasattr(ps, "value") and isinstance(ps.value, (list, tuple)):
            return [string_value(t) for t in ps.value]
        return []


class SortView(_RecordView):
    """View over a ua.sort.Sort record term."""
    name = _StringField("name")

    @property
    def semiring_name(self) -> str:
        return string_value(record_fields(record_fields(self._term)["semiring"])["name"])

    @property
    def batched(self) -> bool:
        b = record_fields(self._term).get("batched")
        if b is None:
            return False
        return hasattr(b, "value") and hasattr(b.value, "value") and b.value.value is True


class SemiringView(_RecordView):
    """View over a ua.semiring.Semiring record term."""
    name = _StringField("name")
    plus = _StringField("plus")
    times = _StringField("times")
    zero = _FloatField("zero")
    one = _FloatField("one")

    @property
    def residual(self) -> str:
        from hydra.dsl.meta.phantoms import string as phantom_string
        return string_value(record_fields(self._term).get("residual", phantom_string("").value))


class LensView(_RecordView):
    """View over a ua.lens.Lens record term."""
    name = _StringField("name")
    forward = _StringField("forward")
    backward = _StringField("backward")

    @property
    def residual_sort(self):
        rs = record_fields(self._term).get("residualSort")
        if rs is None or isinstance(rs, core.TermUnit):
            return None
        return rs


class ProductSortView(_RecordView):
    """View over a ua.sort.Product record term."""
    @property
    def elements(self) -> list[core.Term]:
        return list(record_fields(self._term)["sorts"].value)
