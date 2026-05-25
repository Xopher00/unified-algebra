"""Tokenizer for the unialg surface syntax, built on hydra.parsers combinators.

  tokenize(src)          -> list[Token]   full token set (superset)
  tokenize_morphism(src) -> list[Token]   morphism expression tokens
  tokenize_functor(src)  -> list[Token]   functor expression tokens

Reserved keywords include the declaration words ``let``, ``shape``, ``load``
and shape syntax markers such as ``fix`` and ``by``.

Imports: hydra.parsers only. No unialg imports.
"""
from __future__ import annotations

import hydra.parsers as P
from hydra.parsing import ParseResultSuccess, ParseSuccess, Parser as _Parser

Token = tuple[str, object]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _ws_char():
    """Parse one whitespace character recognized by the surface language."""
    return P.satisfy(lambda c: chr(c) in " \t\n\r")

def _comment():
    """Parse a line comment beginning with ``#`` and discard its contents."""
    def _skip_to_eol(remaining: str):
        i = 0
        while i < len(remaining) and remaining[i] not in "\n\r":
            i += 1
        return ParseResultSuccess(ParseSuccess(None, remaining[i:]))
    return P.bind(P.char(ord("#")), lambda _: _Parser(_skip_to_eol))

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


def _raw_number():
    """Parse INT or FLOAT: digits optionally followed by '.' digits."""
    digit = P.satisfy(lambda c: chr(c).isdigit())
    dot_frac = P.bind(
        P.char(ord(".")),
        lambda _: P.bind(P.some(digit), lambda frac: P.pure(frac)),
    )
    def _finish(int_part):
        int_str = "".join(chr(c) for c in int_part)
        def _with_frac(maybe_frac):
            if hasattr(maybe_frac, 'value'):
                frac_str = "".join(chr(c) for c in maybe_frac.value)
                return P.pure(("FLOAT", float(int_str + "." + frac_str)))
            return P.pure(("INT", int(int_str)))
        return P.bind(P.optional(dot_frac), _with_frac)
    return P.bind(P.some(digit), _finish)


def _raw_string():
    """Parse a double-quoted string literal."""
    inner_char = P.satisfy(lambda c: chr(c) != '"' and chr(c) != '\n')
    return P.bind(
        P.char(ord('"')), lambda _:
        P.bind(P.many(inner_char), lambda cs:
        P.bind(P.char(ord('"')), lambda _:
        P.pure("".join(chr(c) for c in cs)))))

def _raw_quoted_literal():
    """Parse a single-quoted payload for contextual morphism literals."""
    inner_char = P.satisfy(lambda c: chr(c) != "'" and chr(c) != '\n')
    return P.bind(
        P.char(ord("'")), lambda _:
        P.bind(P.many(inner_char), lambda cs:
        P.bind(P.char(ord("'")), lambda _:
        P.pure("".join(chr(c) for c in cs)))))

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
# Multi-char tokens (>>>>, <->, >>, ||) listed before their single-char prefixes.
# ---------------------------------------------------------------------------

# Reserved top-level keywords — cannot be used as expression atom names.
_KEYWORDS: dict[str, str] = {
    "let": "LET",
    "shape": "SHAPE",
    "load": "LOAD",
    "fix": "FIX",
    "by": "BY",
    "lens": "LENS",
    "prism": "PRISM",
    "traversal": "TRAVERSAL",
    "view": "VIEW",
}


def _morphism_token():
    """Return the token parser used for morphism expressions and programs."""
    return P.choice((
        _lit(">>>>", "SHARED_COMPOSE"),
        _lit("<->", "BIDIR"),
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
        _lit(":",  "COLON"),
        _lit("/",  "SLASH"),
        _lit("!",  "BANG"),
        _lit("?",  "QUESTION"),
        _lit("=",  "EQ"),
        _lit(";",  "ERROR", "use '>>' instead of ';'"),
        _lit("-",  "MINUS"),
        P.bind(_raw_quoted_literal(), lambda s: P.pure(("QUOTED", s))),
        P.bind(_raw_string(), lambda s: P.pure(("STRING", s))),
        _raw_number(),
        P.bind(_raw_ident(),  lambda s: P.pure((_KEYWORDS.get(s, "NAME"), s))),
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
        _lit(">>", "COMPOSE"),
        _lit("*",  "STAR"),
        _lit("&",  "PAIR"),
        _lit("|",  "CASE"),
        _lit("(",  "LPAREN"),
        _lit(")",  "RPAREN"),
        _lit("[",  "LBRACKET"),
        _lit("]",  "RBRACKET"),
        _lit(",",  "COMMA"),
        _lit("?",  "QUESTION"),
        P.bind(_raw_int(),   lambda n: P.pure(("INT",  n))),
        P.bind(_raw_ident(), lambda s: P.pure(("NAME", s))),
    ))

def tokenize_functor(src: str) -> list[Token]:
    """Tokenize a standalone polynomial functor expression."""
    return _run(_tokenize(_functor_token()), src)
