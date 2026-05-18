"""Tensor extension — domain-owned syntax, semantics, and structure.

Self-registers ``algebra`` keyword and ``contract`` expression form
with the core extension framework at import time.
"""
from __future__ import annotations

from .notation import SemiringDecl, ContractExpr, Equation

from unialg.syntax._pratt import ParseError as _ParseError



def _parse_algebra(cursor, prog):
    """Parse ``algebra name(plus=op, times=op, zero=val, one=val, ...)``."""
    name_tok = cursor.expect("NAME", "semiring name")
    cursor.expect("LPAREN", "'('")

    fields: dict[str, str | float] = {}
    while cursor.peek()[0] != "RPAREN":
        key_tok = cursor.expect("NAME", "field name")
        cursor.expect("EQ", "'='")

        val_tok = cursor.advance()
        negate = False
        if val_tok[0] == "MINUS":
            negate = True
            val_tok = cursor.advance()

        if val_tok[0] == "NAME":
            val: str | float = val_tok[1]
            if negate:
                if val == "inf":
                    val = float("-inf")
                else:
                    raise _ParseError(f"cannot negate name {val!r}")
        elif val_tok[0] == "FLOAT":
            val = -val_tok[1] if negate else val_tok[1]
        elif val_tok[0] == "INT":
            val = float(-val_tok[1]) if negate else float(val_tok[1])
        else:
            raise _ParseError(f"expected value, got {val_tok[0]!r}")

        fields[key_tok[1]] = val

        if cursor.peek()[0] == "COMMA":
            cursor.advance()

    cursor.expect("RPAREN", "')'")

    required = {"plus", "times", "zero", "one"}
    missing = required - fields.keys()
    if missing:
        raise _ParseError(f"algebra {name_tok[1]!r} missing fields: {missing}")

    optional_str = {}
    if "adjoint" in fields:
        adj = fields["adjoint"]
        if not isinstance(adj, str):
            raise _ParseError("adjoint must be an op name, not a literal")
        optional_str["adjoint"] = adj

    decl = SemiringDecl(
        name=name_tok[1],
        plus=_require_str(fields, "plus"),
        times=_require_str(fields, "times"),
        zero=fields["zero"],
        one=fields["one"],
        **optional_str,
    )

    prog.extensions.setdefault("tensors", []).append(decl)


def _require_str(fields, key) -> str:
    v = fields[key]
    if not isinstance(v, str):
        raise _ParseError(f"{key} must be an op name, not a literal")
    return v


def _parse_contract(cursor):
    """Parse ``contract[sr]("eq")`` or ``contract[sr, adjoint]("eq")``.

    Called after the NAME("contract") token has been consumed by the
    expression grammar's nud lookup.
    """
    cursor.expect("LBRACKET", "'['")
    sr_tok = cursor.expect("NAME", "semiring name")
    sr_name = sr_tok[1]

    adjoint = False
    if cursor.peek()[0] == "COMMA":
        cursor.advance()
        adj_tok = cursor.expect("NAME", "'adjoint'")
        if adj_tok[1] != "adjoint":
            raise _ParseError(f"expected 'adjoint', got {adj_tok[1]!r}")
        adjoint = True

    cursor.expect("RBRACKET", "']'")
    cursor.expect("LPAREN", "'('")
    eq_tok = cursor.expect("STRING", "equation string")
    cursor.expect("RPAREN", "')'")

    return ContractExpr(
        semiring_name=sr_name,
        equation_str=eq_tok[1],
        adjoint=adjoint,
    )


def _lower_stub(spec, ctx):
    raise NotImplementedError("tensor lowering not yet implemented (Phase 5)")


def _register():
    from unialg.extensions import register_keyword, register_expr_form, register_domain, DomainProtocol
    from .semantics import construct, construct_expr, refs

    register_keyword("algebra", _parse_algebra)
    register_expr_form("contract", _parse_contract)
    register_domain("tensors", DomainProtocol(
        construct=construct,
        construct_expr=construct_expr,
        lower=_lower_stub,
        refs=refs,
    ))


_register()
