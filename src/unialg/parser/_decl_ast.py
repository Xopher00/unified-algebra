"""Typed AST nodes for top-level .ua declarations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import hydra.core as core

if TYPE_CHECKING:
    from ._cell_ast import CellExpr
    from hydra.ast import Expr


class Decl:
    """Base class for top-level declaration AST nodes."""
    __slots__ = ()


@dataclass(frozen=True)
class ImportDecl(Decl):
    TYPE_ = core.Name("ua.syntax.ImportDecl")
    BACKEND = core.Name("backend")
    backend: str


@dataclass(frozen=True)
class AlgebraDecl(Decl):
    TYPE_ = core.Name("ua.syntax.AlgebraDecl")
    NAME = core.Name("name")
    KW_ARGS = core.Name("kwArgs")
    name: str
    kw_args: dict


@dataclass(frozen=True)
class SpecDecl(Decl):
    TYPE_ = core.Name("ua.syntax.SpecDecl")
    NAME = core.Name("name")
    SR_NAME = core.Name("srName")
    BATCHED = core.Name("batched")
    AXES = core.Name("axes")
    name: str
    sr_name: str
    batched: bool
    axes: tuple


@dataclass(frozen=True)
class OpDecl(Decl):
    TYPE_ = core.Name("ua.syntax.OpDecl")
    NAME = core.Name("name")
    SIG = core.Name("sig")
    ATTRS = core.Name("attrs")
    name: str
    sig: tuple
    attrs: dict


@dataclass(frozen=True)
class ShareDecl(Decl):
    TYPE_ = core.Name("ua.syntax.ShareDecl")
    NAME = core.Name("name")
    OP_NAMES = core.Name("opNames")
    name: str
    op_names: list


@dataclass(frozen=True)
class DefineDecl(Decl):
    TYPE_ = core.Name("ua.syntax.DefineDecl")
    NAME = core.Name("name")
    ARITY = core.Name("arity")
    PARAMS = core.Name("params")
    BODY = core.Name("body")
    name: str
    arity: str
    params: list
    body: object  # raw define AST — opaque; consumed only by algebra/expr.py


@dataclass(frozen=True)
class FunctorDecl(Decl):
    TYPE_ = core.Name("ua.syntax.FunctorDecl")
    NAME = core.Name("name")
    BODY = core.Name("body")
    ATTRS = core.Name("attrs")
    name: str
    body: "Expr"
    attrs: dict


@dataclass(frozen=True)
class CellDecl(Decl):
    TYPE_ = core.Name("ua.syntax.CellDecl")
    NAME = core.Name("name")
    SIG = core.Name("sig")
    EXPR = core.Name("expr")
    name: str
    sig: tuple
    expr: "CellExpr"
