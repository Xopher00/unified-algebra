"""Generic Pratt parsing engine.

The grammar modules supply nud/led tables and a binding-power dict;
this module owns only the token cursor and parse loop.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Token = tuple[str, Any]
BindingPowers = Mapping[str, tuple[int, int]]
Nud = Callable[["PrattParser", Token], object]
Led = Callable[["PrattParser", object, Token, int], object]


class ParseError(Exception):
    """Raised when token streams do not match the active Pratt grammar."""
    pass


class TokenCursor:
    """Lightweight token stream cursor — peek, advance, expect."""

    def __init__(
        self,
        tokens: Sequence[Token],
        *,
        label: str = "",
        eof_token: Token = ("EOF", None),
    ):
        self._tokens = list(tokens)
        self._label = label
        self._eof_token = eof_token
        self._pos = 0

    @property
    def pos(self) -> int:
        """Return the current token index."""
        return self._pos

    def seek(self, pos: int) -> None:
        """Move the cursor to an absolute token index."""
        self._pos = pos

    def peek(self) -> Token:
        """Return the next token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else self._eof_token

    def advance(self) -> Token:
        """Consume and return the next token."""
        if self._pos >= len(self._tokens):
            raise ParseError(f"{self._label}: unexpected end of input")
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def expect(self, kind: str, msg: str = "") -> Token:
        """Consume the next token, requiring it to have ``kind``."""
        tok = self.advance()
        if tok[0] != kind:
            suffix = f": {msg}" if msg else ""
            raise ParseError(
                f"{self._label}: expected {kind!r}, got {tok[0]!r} ({tok[1]!r}){suffix}"
            )
        return tok


class PrattParser(TokenCursor):
    """Generic Pratt parser parameterized by grammar callbacks."""

    def __init__(
        self,
        tokens: Sequence[Token],
        *,
        label: str,
        binding_powers: BindingPowers,
        nud: Nud,
        led: Led,
        eof_token: Token = ("EOF", None),
    ):
        super().__init__(tokens, label=label, eof_token=eof_token)
        self._binding_powers = binding_powers
        self._nud = nud
        self._led = led

    def parse(self, min_bp: int = 0) -> object:
        """Parse an expression whose operators bind at least ``min_bp``."""
        tok = self.advance()
        if tok[0] == "ERROR":
            raise ParseError(f"{self._label}: {tok[1]}")
        left = self._nud(self, tok)
        while self._pos < len(self._tokens):
            tok = self.peek()
            if tok[0] == "ERROR":
                raise ParseError(f"{self._label}: {tok[1]}")
            if tok[0] not in self._binding_powers:
                break
            l_bp, r_bp = self._binding_powers[tok[0]]
            if l_bp < min_bp:
                break
            self.advance()
            left = self._led(self, left, tok, r_bp)
        return left

    def parse_args(self, *, close: str, sep: str = "COMMA") -> list[object]:
        """Parse a separator-delimited argument list ending with ``close``."""
        args = [self.parse(0)]
        while self._pos < len(self._tokens) and self.peek()[0] == sep:
            self.advance()
            args.append(self.parse(0))
        self.expect(close)
        return args

    def parse_all(self) -> object:
        """Parse one expression and reject trailing tokens."""
        result = self.parse(0)
        if self.peek()[0] != "EOF":
            raise ParseError(f"{self._label}: trailing token {self.peek()}")
        return result


def run_pratt(
    tokens: Sequence[Token],
    *,
    label: str,
    binding_powers: BindingPowers,
    nud: Nud,
    led: Led,
) -> object:
    """Convenience wrapper for one-shot Pratt parsing."""
    return PrattParser(
        tokens, label=label, binding_powers=binding_powers, nud=nud, led=led,
    ).parse_all()
