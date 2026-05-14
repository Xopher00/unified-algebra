"""Shared operator definitions and binary node constructors.

Single source of truth for morphism and functor operators. Used by both
the string parser (grammar _led) and the Python operator interface (Sym).

Rendering via hydra.serialization.print_expr on hydra.ast.Expr trees.
"""
from __future__ import annotations
from typing import Any

import hydra.ast as hast
import hydra.serialization as ser

from unialg.objects import TypeUnit, ProductType, SumType
from unialg.syntax.expressions import (
    MorphismExpr, PolyExpr,
    Compose, Parallel, Pair, Case,
    Prod, Sum, List as PolyList,
)

# ---------------------------------------------------------------------------
# Unit type placeholder — parser does not compute types
# ---------------------------------------------------------------------------

_U = TypeUnit()
_PU = ProductType(_U, _U)
_SU = SumType(_U, _U)

# ---------------------------------------------------------------------------
# Hydra Op objects — canonical operator metadata
# ---------------------------------------------------------------------------

def createOp(symbol: str, padding: bool, precedence: int, associativity: int):
    _SYM = ser.sym(symbol)
    if padding:
        _PAD = hast.Padding(hast.WsSpace(), hast.WsSpace())
    else:
        _PAD = hast.Padding(hast.WsNone(), hast.WsNone())
    _PREC = hast.Precedence(precedence)
    _ASSOC = (hast.Associativity.LEFT, hast.Associativity.RIGHT)[associativity]
    return hast.Op(_SYM, _PAD, _PREC, _ASSOC)
    

COMPOSE_OP = createOp(symbol=">>", padding=True, precedence=60, associativity=0) 
PAIR_OP    = createOp("&", True, 70, 0) 
PAR_OP     = createOp("||", True, 65, 0) 
CASE_OP    = createOp("|", True, 50, 0)

# Functor operators
FSTAR_OP = createOp(symbol="*", padding=False, precedence=80, associativity=0)
FPROD_OP = createOp("&", True, 70, 0)
FSUM_OP  = createOp("|", True, 60, 0)

# Token kind → Op mapping for morphism operators
_MORPHISM_OPS: dict[str, hast.Op] = {
    "COMPOSE": COMPOSE_OP,
    "PAIR":    PAIR_OP,
    "PAR":     PAR_OP,
    "CASE":    CASE_OP,
}

_FUNCTOR_OPS: dict[str, hast.Op] = {
    "STAR": FSTAR_OP,
    "PAIR": FPROD_OP,
    "CASE": FSUM_OP,
}


def _op_bp(op: hast.Op) -> tuple[int, int]:
    p = op.precedence.value
    match op.associativity:
        case hast.Associativity.LEFT:  return (p, p + 1)
        case hast.Associativity.RIGHT: return (p, p - 1)
        case _:                        return (p, p)


def morphism_bp() -> dict[str, tuple[int, int]]:
    return {k: _op_bp(op) for k, op in _MORPHISM_OPS.items()}


def functor_bp() -> dict[str, tuple[int, int]]:
    return {k: _op_bp(op) for k, op in _FUNCTOR_OPS.items()}


# ---------------------------------------------------------------------------
# Binary MorphismExpr constructors — shared by parser _led and Sym operators
# ---------------------------------------------------------------------------

def make_compose(f: MorphismExpr, g: MorphismExpr) -> Compose:
    return Compose(f=f, g=g, f_param=_U, g_param=_U,
                   param=_U, monad=None, dom=_U, cod=_U)


def make_pair(f: MorphismExpr, g: MorphismExpr) -> Pair:
    return Pair(f=f, g=g, f_param=_U, g_param=_U,
                param=_U, monad=None, dom=_U, cod=ProductType(_U, _U))


def make_par(f: MorphismExpr, g: MorphismExpr) -> Parallel:
    return Parallel(f=f, g=g, f_param=_U, g_param=_U,
                    param=_U, monad=None,
                    dom=ProductType(_U, _U), cod=ProductType(_U, _U))


def make_case(f: MorphismExpr, g: MorphismExpr) -> Case:
    return Case(f=f, g=g, f_param=_U, g_param=_U,
                param=_U, monad=None, dom=SumType(_U, _U), cod=_U)


_MORPHISM_BUILDERS: dict[str, Any] = {
    "COMPOSE": make_compose,
    "PAIR":    make_pair,
    "PAR":     make_par,
    "CASE":    make_case,
}


def make_binary(kind: str, left: MorphismExpr, right: MorphismExpr) -> MorphismExpr:
    return _MORPHISM_BUILDERS[kind](left, right)


# ---------------------------------------------------------------------------
# Functor binary constructors
# ---------------------------------------------------------------------------

def make_prod(l: PolyExpr, r: PolyExpr) -> Prod:
    return Prod(l, r)


def make_sum(l: PolyExpr, r: PolyExpr) -> Sum:
    return Sum(l, r)


def make_list(body: PolyExpr) -> PolyList:
    return PolyList(body)


# ---------------------------------------------------------------------------
# Hydra Expr builders — for rendering via print_expr
# ---------------------------------------------------------------------------

def atom_expr(name: str) -> hast.Expr:
    return hast.ExprConst(ser.sym(name))


def binary_expr(op: hast.Op, left: hast.Expr, right: hast.Expr) -> hast.Expr:
    return hast.ExprOp(hast.OpExpr(op, left, right))


def render(expr: hast.Expr) -> str:
    return ser.print_expr(ser.parenthesize(expr))
