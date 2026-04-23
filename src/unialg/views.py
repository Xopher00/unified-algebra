"""Read-only view wrappers for Hydra record terms.

Each view wraps a raw Hydra Term and exposes its fields as @property
accessors. The Term stays the source of truth — views never copy data.
"""

from unialg.utils import record_fields, string_value
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

    @classmethod
    def from_term(cls, term):
        """Idempotent wrap: return term unchanged if already an instance of cls."""
        if isinstance(term, cls):
            return term
        obj = cls.__new__(cls)
        obj._term = term
        return obj

    @staticmethod
    def _unwrap(term):
        """Unwrap a _RecordView to its raw Hydra term; pass anything else through."""
        return term.term if isinstance(term, _RecordView) else term


# ---------------------------------------------------------------------------
# Concrete views
# ---------------------------------------------------------------------------


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


