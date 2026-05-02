"""Hydra term utilities: record views, field descriptors, and tensor coders."""

from __future__ import annotations

import hydra.core as core
import hydra.graph
from hydra.dsl.meta.phantoms import (
    binary as phantom_binary, boolean, float64, int32, list_, record, string, TTerm, unit,
)
from hydra.dsl.python import Nothing, Right
from hydra.extract.core import binary as extract_binary
from hydra.literals import float_value_to_bigfloat, integer_value_to_bigint


_ENCODERS = {str: string, bool: boolean, float: float64, int: int32}



_TENSOR_NAME = core.Name("ua.tensor.NDArray")


def register_tensor_schema(schema: dict) -> None:
    schema[_TENSOR_NAME] = core.TypeScheme((), core.TypeVariable(_TENSOR_NAME), Nothing())


def tensor_coder(backend, type_=None) -> hydra.graph.TermCoder:
    """Create a TermCoder that bridges arrays and Hydra Terms."""
    if type_ is None:
        type_ = core.TypeVariable(_TENSOR_NAME)

    def encode(cx, graph, term):
        match extract_binary(graph, term):
            case Right(value=raw): pass
            case _: raw = _literal_value(term)
        return Right(backend.from_wire(raw))

    def decode(cx, arr):
        return Right(phantom_binary(backend.to_wire(arr)).value)

    return hydra.graph.TermCoder(type=type_, encode=encode, decode=decode)


def _literal_value(term):
    match term.value:
        case core.LiteralInteger(value=iv): return int(integer_value_to_bigint(iv))
        case core.LiteralFloat(value=fv): return float(float_value_to_bigfloat(fv))
        case core.LiteralString(value=s): return s
        case core.LiteralBoolean(value=b): return b
        case core.LiteralBinary(value=bs): return bs
    raise ValueError(f"Unknown literal kind: {type(term.value).__name__}")


# ---------------------------------------------------------------------------
# Record view base + field descriptors
# ---------------------------------------------------------------------------

class _RecordView:
    """Base for all Hydra term wrappers."""
    __slots__ = ("_term",)

    class Term:
        __slots__ = ("_key", "_optional", "_coerce")
        def __init__(self, *, key=None, optional=False, coerce=None):
            self._key = key
            self._optional = optional
            self._coerce = coerce

        def __set_name__(self, owner, name):
            if self._key is None:
                self._key = name

        def _extract(self, t):
            return t

        def _encode(self, value):
            if value is None and self._optional:
                return unit()
            return TTerm(_RecordView._unwrap(value))

        def __get__(self, obj, cls=None):
            fields = obj._fields
            if self._optional:
                t = fields.get(self._key)
                if t is None or isinstance(t, core.TermUnit):
                    return None
            else:
                t = fields[self._key]
            value = self._extract(t)
            return self._coerce(value) if self._coerce else value

    class Scalar(Term):
        __slots__ = ("_default",)
        def __init__(self, coerce=None, default=None, *, key=None, has_default=False):
            super().__init__(key=key, optional=(has_default or default is not None), coerce=coerce)
            self._default = default

        def _extract(self, t):
            return _literal_value(t)

        def _encode(self, value):
            encoder = _ENCODERS.get(self._coerce, string)
            return encoder(value if value is not None else self._default)

        def __get__(self, obj, cls=None):
            fields = obj._fields
            t = fields.get(self._key)
            if t is None or isinstance(t, core.TermUnit):
                if self._optional:
                    return self._default
                raise KeyError(self._key)
            value = self._extract(t)
            return self._coerce(value) if self._coerce else value

    class TermList(Term):
        __slots__ = ()
        def __init__(self, *, key=None, coerce=None):
            super().__init__(key=key, coerce=coerce)

        def _encode(self, values):
            return list_([TTerm(_RecordView._unwrap(v)) for v in (values or [])])

        def __get__(self, obj, cls=None):
            items = obj._fields[self._key].value
            extract = self._extract
            coerce = self._coerce
            if coerce:
                return [coerce(extract(t)) for t in items]
            return [extract(t) for t in items]

    class ScalarList(TermList):
        __slots__ = ("_default",)
        def __init__(self, *, key=None, default=None):
            super().__init__(key=key)
            self._default = default

        def _extract(self, t):
            return _literal_value(t)

        def _encode(self, values):
            return list_([string(v) for v in (values or self._default or [])])

        def __get__(self, obj, cls=None):
            t = obj._fields.get(self._key)
            if t is None:
                return list(self._default) if self._default is not None else []
            return [self._extract(item) for item in t.value]

    @classmethod
    def _descriptors(cls):
        seen = set()
        result = []
        for klass in reversed(cls.__mro__):
            for n, d in vars(klass).items():
                if isinstance(d, _RecordView.Term) and n not in seen:
                    seen.add(n)
                    result.append((n, d))
        return result

    def __init__(self, *args, **kwargs):
        descriptors = type(self)._descriptors()
        for i, (name, _) in enumerate(descriptors):
            if i < len(args):
                kwargs[name] = args[i]
        self._term = self._build_record(**kwargs)

    @property
    def term(self) -> core.Term:
        return self._term

    @property
    def _fields(self) -> dict[str, object]:
        return {f.name.value: f.term for f in self._term.value.fields}

    @classmethod
    def _build_record(cls, **values):
        fields = []
        for name, obj in cls._descriptors():
            fields.append(core.Name(obj._key) >> obj._encode(values.get(name)))
        return record(cls._type_name, fields).value

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
        return term.term if isinstance(term, _RecordView) else term

