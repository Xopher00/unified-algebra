"""Hydra term utilities: record views, field descriptors, and tensor coders."""

from __future__ import annotations

import hydra.core as core
import hydra.graph
from hydra.dsl.meta.phantoms import binary as phantom_binary
from hydra.dsl.python import Right
from hydra.extract.core import binary as extract_binary
from hydra.literals import float_value_to_bigfloat, integer_value_to_bigint


def tensor_coder(backend, type_=None) -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges arrays and Hydra Terms."""
    if type_ is None:
        type_ = core.TypeVariable(core.Name("ua.tensor.NDArray"))

    def encode(cx, graph, term):
        match extract_binary(graph, term):
            case Right(value=raw): pass
            case _: raw = literal_value(term)  # fallback for unwrapped LiteralBinary
        return Right(backend.from_wire(raw))

    def decode(cx, arr):
        return Right(phantom_binary(backend.to_wire(arr)).value)

    return hydra.graph.TermCoder(type=type_, encode=encode, decode=decode)


def record_fields(term) -> dict[str, object]:
    """Extract a Hydra record's fields as a {name_str: Term} dict."""
    return {f.name.value: f.term for f in term.value.fields}

def literal_value(term):
    """Extract a Python primitive from a Hydra TermLiteral via canonical extractors."""
    match term.value:
        case core.LiteralInteger(value=iv): return int(integer_value_to_bigint(iv))
        case core.LiteralFloat(value=fv): return float(float_value_to_bigfloat(fv))
        case core.LiteralString(value=s): return s
        case core.LiteralBoolean(value=b): return b
        case core.LiteralBinary(value=bs): return bs
    raise ValueError(f"Unknown literal kind: {type(term.value).__name__}")

def bind_composition(kind, name, var_name, body):
    """Wrap a body term in a lambda and return (Name, lambda_term)."""
    from hydra.dsl.meta.phantoms import lam, TTerm
    if not isinstance(body, TTerm):
        body = TTerm(body)
    term = lam(var_name, body).value
    return (core.Name(f"ua.{kind}.{name}"), term)


# ---------------------------------------------------------------------------
# Field descriptors — eliminate per-property boilerplate
# ---------------------------------------------------------------------------

class _TermField:
    """Descriptor: extracts a raw Term field (no conversion).

    With optional=True, returns None for missing fields or TermUnit values.
    """
    __slots__ = ("_key", "_optional")
    def __init__(self, key, optional: bool = False):
        self._key = key
        self._optional = optional
    def __get__(self, obj, cls=None):
        fields = record_fields(obj._term)
        if self._optional:
            t = fields.get(self._key)
            return None if t is None or isinstance(t, core.TermUnit) else t
        return fields[self._key]

class _ScalarField:
    __slots__ = ("_key", "_coerce", "_default", "_has_default")
    def __init__(self, key, coerce=None, default=None, *, has_default=False):
        self._key = key
        self._coerce = coerce
        self._default = default
        self._has_default = has_default or default is not None

    def __get__(self, obj, cls=None):
        fields = record_fields(obj._term)
        if self._key not in fields:
            if self._has_default:
                return self._default
            raise KeyError(self._key)
        value = literal_value(fields[self._key])
        return self._coerce(value) if self._coerce else value


class _StringListField:
    """Descriptor: extracts a list of string values from a TermList field."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return [literal_value(t) for t in record_fields(obj._term)[self._key].value]


class _TermListField:
    """Descriptor: extracts the raw Term list from a TermList field."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return list(record_fields(obj._term)[self._key].value)


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
