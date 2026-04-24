"""Hydra term utilities: record views, field descriptors, and tensor coders."""

from __future__ import annotations

import hydra.core as core
import hydra.graph
from hydra.dsl.meta.phantoms import binary as phantom_binary
from hydra.dsl.python import Right
from hydra.extract.core import binary as extract_binary


def tensor_coder(backend, type_=None) -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges arrays and Hydra Terms."""
    if type_ is None:
        type_ = core.TypeVariable(core.Name("ua.tensor.NDArray"))

    def encode(cx, graph, term):
        result = extract_binary(graph, term)
        match result:
            case Right(value=raw): pass
            case _: raw = term.value.value
        return Right(backend.from_wire(raw))

    def decode(cx, arr):
        return Right(phantom_binary(backend.to_wire(arr)).value)

    return hydra.graph.TermCoder(type=type_, encode=encode, decode=decode)


def record_fields(term) -> dict[str, object]:
    """Extract a Hydra record's fields as a {name_str: Term} dict."""
    return {f.name.value: f.term for f in term.value.fields}

def literal_value(term):
    """Extract a Python primitive from a Hydra TermLiteral."""
    v = term.value.value
    return v.value if hasattr(v, 'value') else v

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
    """Descriptor: extracts a raw Term field (no conversion)."""
    __slots__ = ("_key",)
    def __init__(self, key): self._key = key
    def __get__(self, obj, cls=None):
        return record_fields(obj._term)[self._key]

class _ScalarField:
    __slots__ = ("_key", "_coerce")
    def __init__(self, key, coerce=None):
        self._key = key
        self._coerce = coerce

    def __get__(self, obj, cls=None):
        value = literal_value(record_fields(obj._term)[self._key])
        return self._coerce(value) if self._coerce else value


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
