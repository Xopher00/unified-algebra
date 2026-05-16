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

def createOp(symbol: str, padding: bool, precedence: int, associativity: str = "left"):
    """Create Hydra operator metadata for a surface syntax operator."""
    ws = hast.WsSpace if padding else hast.WsNone
    return hast.Op(
        ser.sym(symbol),
        hast.Padding(ws(), ws()),
        hast.Precedence(precedence),
        {"left": hast.Associativity.LEFT, "right": hast.Associativity.RIGHT}[associativity],
    )
    
COMPOSE_OP = createOp(symbol=">>", padding=True, precedence=60, associativity="left")
PAIR_OP    = createOp("&", True, 70, "left")
PAR_OP     = createOp("||", True, 65, "left")
CASE_OP    = createOp("|", True, 50, "left")
INDEX_OP   = createOp("[", False, 90, "left")
STAR_OP    = createOp("*", False, 90, "left")

# Functor operators
FSTAR_OP = createOp(symbol="*", padding=False, precedence=80, associativity="left")
FPROD_OP = createOp("&", True, 70, "left")
FSUM_OP  = createOp("|", True, 60, "left")

# Token kind → Op mapping for morphism operators
_MORPHISM_OPS: dict[str, hast.Op] = {
    "COMPOSE":  COMPOSE_OP,
    "PAIR":     PAIR_OP,
    "PAR":      PAR_OP,
    "CASE":     CASE_OP,
    "LBRACKET": INDEX_OP,
    "STAR":     STAR_OP,
}

_FUNCTOR_OPS: dict[str, hast.Op] = {
    "STAR": FSTAR_OP,
    "PAIR": FPROD_OP,
    "CASE": FSUM_OP,
}


def _op_bp(op: hast.Op) -> tuple[int, int]:
    """Convert Hydra operator metadata into Pratt left/right binding powers."""
    p = op.precedence.value
    match op.associativity:
        case hast.Associativity.LEFT:  return (p, p + 1)
        case hast.Associativity.RIGHT: return (p, p - 1)
        case _:                        return (p, p)


def morphism_bp() -> dict[str, tuple[int, int]]:
    """Return Pratt binding powers for morphism token kinds."""
    return {k: _op_bp(op) for k, op in _MORPHISM_OPS.items()}


def functor_bp() -> dict[str, tuple[int, int]]:
    """Return Pratt binding powers for functor token kinds."""
    return {k: _op_bp(op) for k, op in _FUNCTOR_OPS.items()}


# ---------------------------------------------------------------------------
# Binary MorphismExpr constructors — shared by parser _led and Sym operators
# ---------------------------------------------------------------------------

def make_compose(f: MorphismExpr, g: MorphismExpr) -> Compose:
    """Build a placeholder-typed sequential composition node."""
    return Compose(f=f, g=g, f_param=_U, g_param=_U,
                   param=_U, monad=None, dom=_U, cod=_U)


def make_pair(f: MorphismExpr, g: MorphismExpr) -> Pair:
    """Build a placeholder-typed product-introduction node."""
    return Pair(f=f, g=g, f_param=_U, g_param=_U,
                param=_U, monad=None, dom=_U, cod=ProductType(_U, _U))


def make_par(f: MorphismExpr, g: MorphismExpr) -> Parallel:
    """Build a placeholder-typed parallel product node."""
    return Parallel(f=f, g=g, f_param=_U, g_param=_U,
                    param=_U, monad=None,
                    dom=ProductType(_U, _U), cod=ProductType(_U, _U))


def make_case(f: MorphismExpr, g: MorphismExpr) -> Case:
    """Build a placeholder-typed coproduct elimination node."""
    return Case(f=f, g=g, f_param=_U, g_param=_U,
                param=_U, monad=None, dom=SumType(_U, _U), cod=_U)


_MORPHISM_BUILDERS: dict[str, Any] = {
    "COMPOSE": make_compose,
    "PAIR":    make_pair,
    "PAR":     make_par,
    "CASE":    make_case,
}


def make_binary(kind: str, left: MorphismExpr, right: MorphismExpr) -> MorphismExpr:
    """Dispatch a binary morphism token kind to its node constructor."""
    return _MORPHISM_BUILDERS[kind](left, right)


# ---------------------------------------------------------------------------
# Functor binary constructors
# ---------------------------------------------------------------------------

def make_prod(l: PolyExpr, r: PolyExpr) -> Prod:
    """Build a product polynomial functor node."""
    return Prod(l, r)


def make_sum(l: PolyExpr, r: PolyExpr) -> Sum:
    """Build a sum polynomial functor node."""
    return Sum(l, r)


def make_list(body: PolyExpr) -> PolyList:
    """Build a list polynomial functor node."""
    return PolyList(body)


# ---------------------------------------------------------------------------
# Hydra Expr builders — for rendering via print_expr
# ---------------------------------------------------------------------------

def atom_expr(name: str) -> hast.Expr:
    """Build a Hydra rendering expression for an atom name."""
    return hast.ExprConst(ser.sym(name))


def binary_expr(op: hast.Op, left: hast.Expr, right: hast.Expr) -> hast.Expr:
    """Build a Hydra rendering expression for a binary operation."""
    return hast.ExprOp(hast.OpExpr(op, left, right))


def render(expr: hast.Expr) -> str:
    """Render a Hydra expression with parser-compatible parentheses."""
    return ser.print_expr(ser.parenthesize(expr))
