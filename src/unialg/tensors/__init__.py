"""Tensor extension — domain-owned syntax, semantics, and structure.

Self-registers ``algebra`` keyword and ``contract`` expression form
with the core extension framework at import time.
"""
from __future__ import annotations

from .notation import SemiringDecl, ContractExpr, Equation, parse_algebra, parse_contract


def _register():
    from unialg.extensions import register_keyword, register_expr_form, register_domain, DomainProtocol
    from .semantics import construct, construct_expr, refs
    from .fusion import normalize_contracts

    register_keyword("algebra", parse_algebra)
    register_expr_form("contract", parse_contract)
    register_domain("tensors", DomainProtocol(
        construct=construct,
        construct_expr=construct_expr,
        refs=refs,
        finalize=normalize_contracts,
    ))


