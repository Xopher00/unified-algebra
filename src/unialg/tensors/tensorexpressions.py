
from __future__ import annotations
from dataclasses import dataclass
from functools import singledispatch

from unialg.objects import Type
from . import expressions as expr

@dataclass(frozen=True)
class Einsum(expr.MorphismExpr):
    """Structured tensor contraction with explicit arrow boundary."""
    equation: str
    dom: Type
    cod: Type