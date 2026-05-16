"""Tokenizer for the unialg surface syntax, built on hydra.parsers combinators.

  tokenize(src)          -> list[Token]   full token set (superset)
  tokenize_morphism(src) -> list[Token]   morphism expression tokens
  tokenize_functor(src)  -> list[Token]   functor expression tokens

Reserved keywords: 'route' → ROUTE, 'map' → MAP. These may not be used
as atom names inside expressions.

Imports: hydra.parsers only. No unialg imports.
"""
from __future__ import annotations

import hydra.parsers as P
from hydra.parsing import ParseResultSuccess

Token = tuple[str, object]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _ws_char():
    """Parse one whitespace character recognized by the surface language."""
    return P.satisfy(lambda c: chr(c) in " \t\n\r")

def _not_nl():
    """Parse one character that is not a line break."""
    return P.satisfy(lambda c: chr(c) not in "\n\r")

def _comment():
    """Parse a line comment beginning with ``#`` and discard its contents."""
    return P.bind(P.char(ord("#")), lambda _:
           P.bind(P.many(_not_nl()), lambda _:
           P.pure(None)))

def _skip():
    """Parse any amount of insignificant whitespace and comments."""
    return P.bind(
        P.many(P.alt(
            P.bind(_ws_char(), lambda _: P.pure(None)),
            _comment(),
        )),
        lambda _: P.pure(None),
    )

def _raw_ident():
    """Parse an identifier before keyword classification.

    Dots are allowed in the rest of an identifier so that backend op names
    such as ``reduce.add`` tokenize as a single NAME token.
    """
    id_start = P.satisfy(lambda c: chr(c).isalpha() or chr(c) == "_")
    id_rest  = P.many(P.satisfy(lambda c: chr(c).isalnum() or chr(c) in "_."))
    return P.bind(id_start, lambda c:
           P.bind(id_rest,  lambda cs:
           P.pure(chr(c) + "".join(chr(x) for x in cs))))

def _raw_int():
    """Parse a non-negative decimal integer."""
    return P.bind(
        P.some(P.satisfy(lambda c: chr(c).isdigit())),
        lambda ds: P.pure(int("".join(chr(c) for c in ds))),
    )

def _lit(text: str, kind: str, value: object = None):
    """Build a parser for a fixed literal token."""
    v = text if value is None else value
    return P.bind(P.string(text), lambda _: P.pure((kind, v)))

def _tokenize(raw_token):
    """Wrap a token parser with whitespace/comment skipping."""
    skip = _skip()
    return P.bind(skip, lambda _:
           P.many(P.bind(raw_token, lambda t:
           P.bind(skip,  lambda _:
           P.pure(t)))))

def _run(parser, src: str) -> list[Token]:
    """Run a tokenizer parser and require all non-whitespace input to be consumed."""
    result = P.run_parser(parser, src)
    if not isinstance(result, ParseResultSuccess):
        raise ValueError(f"tokenization failed near: {src[:60]!r}")
    parsed = result.value
    if parsed.remainder.strip():
        raise ValueError(f"unconsumed input: {parsed.remainder[:40]!r}")
    return list(parsed.value)


# ---------------------------------------------------------------------------
# Morphism token set
# Multi-char tokens (>>, ||) listed before their single-char prefixes.
# ---------------------------------------------------------------------------

# Reserved top-level keywords — cannot be used as expression atom names.
_KEYWORDS: dict[str, str] = {"route": "ROUTE", "map": "MAP", "load": "LOAD"}


def _morphism_token():
    """Return the token parser used for morphism expressions and programs."""
    return P.choice((
        _lit(">>>>", "SHARED_COMPOSE"),
        _lit(">>", "COMPOSE"),
        _lit("||", "PAR"),
        _lit("&",  "PAIR"),
        _lit("|",  "CASE"),
        _lit("*",  "STAR"),
        _lit("{",  "LBRACE"),
        _lit("}",  "RBRACE"),
        _lit("[",  "LBRACKET"),
        _lit("]",  "RBRACKET"),
        _lit("(",  "LPAREN"),
        _lit(")",  "RPAREN"),
        _lit(",",  "COMMA"),
        _lit("!",  "BANG"),
        _lit("?",  "QUESTION"),
        _lit("=",  "EQ"),
        _lit(";",  "ERROR", "use '>>' instead of ';'"),
        P.bind(_raw_int(),   lambda n: P.pure(("INT",  n))),
        P.bind(_raw_ident(), lambda s: P.pure((_KEYWORDS.get(s, "NAME"), s))),
    ))


def tokenize_morphism(src: str) -> list[Token]:
    """Tokenize a morphism expression or full program source."""
    return _run(_tokenize(_morphism_token()), src)


# Full token set — superset of morphism + functor; used for program parsing.
tokenize = tokenize_morphism


# ---------------------------------------------------------------------------
# Functor token set
# ---------------------------------------------------------------------------

def _functor_token():
    """Return the smaller token parser used for standalone functor expressions."""
    return P.choice((
        _lit("*",  "STAR"),
        _lit("&",  "PAIR"),
        _lit("|",  "CASE"),
        _lit("(",  "LPAREN"),
        _lit(")",  "RPAREN"),
        _lit("[",  "LBRACKET"),
        _lit("]",  "RBRACKET"),
        _lit("?",  "QUESTION"),
        P.bind(_raw_int(),   lambda n: P.pure(("INT",  n))),
        P.bind(_raw_ident(), lambda s: P.pure(("NAME", s))),
    ))

def tokenize_functor(src: str) -> list[Token]:
    """Tokenize a standalone polynomial functor expression."""
    return _run(_tokenize(_functor_token()), src)
