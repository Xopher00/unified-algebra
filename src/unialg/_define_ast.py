"""Define expression AST nodes — Hydra union type ua.syntax.DefineExpr."""
from __future__ import annotations

import hydra.dsl.terms as Terms
from hydra.core import Name, TermInject

from unialg.terms import _RecordView


# ---------------------------------------------------------------------------
# Type & variant names
# ---------------------------------------------------------------------------

DEFINE_TYPE_NAME = Name("ua.syntax.DefineExpr")

_K_LIT  = Name("lit")
_K_VAR  = Name("var")
_K_CALL = Name("call")

# Record type name for the multi-field call payload
_CALL_REC = Name("ua.syntax.DefineCall")

unwrap = _RecordView._unwrap


def _define_term(term):
    term = unwrap(term)
    if not isinstance(term, TermInject) or term.value.type_name != DEFINE_TYPE_NAME:
        raise TypeError(
            f"DefineExpr: expected injection of {DEFINE_TYPE_NAME.value!r}, got {term!r}"
        )
    return term


# ---------------------------------------------------------------------------
# DefineExpr — wrapper over a TermInject of the define union
# ---------------------------------------------------------------------------

class DefineExpr(_RecordView):
    """Define expression — Hydra union type ua.syntax.DefineExpr."""
    __slots__ = ()

    def __init__(self, term):
        self._term = _define_term(term)

    @property
    def kind(self) -> str:
        return self._term.value.field.name.value

    @property
    def _payload(self):
        return self._term.value.field.term

    def _payload_record_fields(self) -> dict:
        return {f.name.value: f.term for f in self._payload.value.fields}

    def __eq__(self, other) -> bool:
        return isinstance(other, DefineExpr) and self._term == other._term

    def __hash__(self) -> int:
        return hash(self._term)

    def __repr__(self) -> str:
        return f"DefineExpr(kind={self.kind!r})"


# ---------------------------------------------------------------------------
# Constructors — each returns a DefineExpr wrapping a TermInject
# ---------------------------------------------------------------------------

def def_lit(value: float) -> DefineExpr:
    return DefineExpr(Terms.inject(DEFINE_TYPE_NAME, _K_LIT, Terms.float32(float(value))))


def def_var(name: str) -> DefineExpr:
    return DefineExpr(Terms.inject(DEFINE_TYPE_NAME, _K_VAR, Terms.string(name)))


def def_call(fn_name: str, args: list) -> DefineExpr:
    if len(args) > 2:
        raise ValueError(
            f"define: function '{fn_name}' called with {len(args)} args "
            f"— only 1 (unary) or 2 (binary) supported"
        )
    return DefineExpr(Terms.inject(DEFINE_TYPE_NAME, _K_CALL,
        Terms.record(_CALL_REC, [
            Terms.field("fn", Terms.string(fn_name)),
            Terms.field("args", Terms.list_([unwrap(a) for a in args])),
        ])))
