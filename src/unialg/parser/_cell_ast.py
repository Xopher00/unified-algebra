"""Cell expression AST nodes — Hydra union type ua.syntax.CellExpr."""
from __future__ import annotations

import hydra.dsl.terms as Terms
from hydra.core import Name, TermInject
from hydra.dsl.python import Nothing, Just

from unialg.terms import _RecordView, _literal_value


# ---------------------------------------------------------------------------
# Type & variant names
# ---------------------------------------------------------------------------

CELL_TYPE_NAME = Name("ua.syntax.CellExpr")

_K_EQ     = Name("eq")
_K_LIT    = Name("lit")
_K_COPY   = Name("copy")
_K_DELETE = Name("delete")
_K_IDEN   = Name("iden")
_K_SEQ    = Name("seq")
_K_PAR    = Name("par")
_K_LENS   = Name("lens")
_K_CATA   = Name("cata")
_K_ANA    = Name("ana")

# Record type names for multi-field payloads
_EQ_REC   = Name("ua.syntax.EqRef")
_LENS_REC = Name("ua.syntax.LensExpr")
_HOM_REC  = Name("ua.syntax.HomRef")

unwrap = _RecordView._unwrap


def _cell_term(term):
    term = unwrap(term)
    if not isinstance(term, TermInject) or term.value.type_name != CELL_TYPE_NAME:
        raise TypeError(
            f"CellExpr: expected injection of {CELL_TYPE_NAME.value!r}, got {term!r}"
        )
    return term


# ---------------------------------------------------------------------------
# CellExpr — wrapper over a TermInject of the cell union
# ---------------------------------------------------------------------------

class CellExpr(_RecordView):
    """Cell expression — Hydra union type ua.syntax.CellExpr."""
    __slots__ = ()

    def __init__(self, term):
        self._term = _cell_term(term)

    @property
    def kind(self) -> str:
        return self._term.value.field.name.value

    @property
    def _payload(self):
        return self._term.value.field.term

    def _payload_record_fields(self) -> dict:
        return {f.name.value: f.term for f in self._payload.value.fields}

    def __eq__(self, other) -> bool:
        return isinstance(other, CellExpr) and self._term == other._term

    def __hash__(self) -> int:
        return hash(self._term)

    def __repr__(self) -> str:
        return f"CellExpr(kind={self.kind!r})"


# ---------------------------------------------------------------------------
# Constructors — each returns a CellExpr wrapping a TermInject
# ---------------------------------------------------------------------------

def cell_eq(base_name: str, modifiers: str) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_EQ,
        Terms.record(_EQ_REC, [
            Terms.field("baseName", Terms.string(base_name)),
            Terms.field("modifiers", Terms.string(modifiers)),
        ])))


def cell_lit(value: float) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_LIT, Terms.float32(float(value))))


def cell_copy(sort_name: str) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_COPY, Terms.string(sort_name)))


def cell_delete(sort_name: str) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_DELETE, Terms.string(sort_name)))


def cell_iden(sort_name: str) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_IDEN, Terms.string(sort_name)))


def cell_seq(left: CellExpr, right: CellExpr) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_SEQ,
        Terms.pair(unwrap(left), unwrap(right))))


def cell_par(left: CellExpr, right: CellExpr) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_PAR,
        Terms.pair(unwrap(left), unwrap(right))))


def cell_lens(fwd: CellExpr, bwd: CellExpr, residual: str | None) -> CellExpr:
    res_term = Terms.nothing() if residual is None else Terms.just(Terms.string(residual))
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_LENS,
        Terms.record(_LENS_REC, [
            Terms.field("fwd", unwrap(fwd)),
            Terms.field("bwd", unwrap(bwd)),
            Terms.field("residual", res_term),
        ])))


def cell_cata(functor: str, args: tuple) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_CATA,
        Terms.record(_HOM_REC, [
            Terms.field("functor", Terms.string(functor)),
            Terms.field("args", Terms.list_([unwrap(a) for a in args])),
        ])))


def cell_ana(functor: str, args: tuple) -> CellExpr:
    return CellExpr(Terms.inject(CELL_TYPE_NAME, _K_ANA,
        Terms.record(_HOM_REC, [
            Terms.field("functor", Terms.string(functor)),
            Terms.field("args", Terms.list_([unwrap(a) for a in args])),
        ])))
