"""Typed AST nodes for top-level .ua declarations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._cell_ast import CellExpr
    from hydra.ast import Expr


class Decl:
    """Base class for top-level declaration AST nodes."""
    __slots__ = ()


@dataclass(frozen=True)
class ImportDecl(Decl):
    backend: str


@dataclass(frozen=True)
class AlgebraDecl(Decl):
    name: str
    kw_args: dict


@dataclass(frozen=True)
class SpecDecl(Decl):
    name: str
    sr_name: str
    batched: bool
    axes: tuple


@dataclass(frozen=True)
class OpDecl(Decl):
    name: str
    sig: tuple
    attrs: dict


@dataclass(frozen=True)
class ShareDecl(Decl):
    name: str
    op_names: list


@dataclass(frozen=True)
class DefineDecl(Decl):
    name: str
    arity: str
    params: list
    body: object  # raw define AST — opaque; consumed only by assembly/_define_lowering.py


@dataclass(frozen=True)
class FunctorDecl(Decl):
    name: str
    body: "Expr"
    attrs: dict


@dataclass(frozen=True)
class CellDecl(Decl):
    name: str
    sig: tuple
    expr: "CellExpr"
