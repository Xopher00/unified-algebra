"""Small Pratt parser helper for tokenized parser sub-grammars."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

Token = tuple[str, Any]
BindingPowers = Mapping[str, tuple[int, int]]
Nud = Callable[["PrattParser", Token], object]
Led = Callable[["PrattParser", object, Token, int], object]


class PrattParser:
    """Shared Pratt parser state.

    Tokenization stays in ``_grammar.py`` because it depends on Hydra parser
    combinators. This helper only owns the precedence loop and small cursor
    operations used by both cell and functor expression parsers.
    """

    def __init__(
        self,
        tokens: Sequence[Token],
        *,
        label: str,
        binding_powers: BindingPowers,
        nud: Nud,
        led: Led,
        eof_token: Token,
    ):
        self._tokens = list(tokens)
        self._label = label
        self._binding_powers = binding_powers
        self._nud = nud
        self._led = led
        self._eof_token = eof_token
        self._pos = 0

    def peek(self) -> Token:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else self._eof_token

    def advance(self) -> Token:
        if self._pos >= len(self._tokens):
            raise ValueError(f"{self._label}: unexpected end of expression")
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def expect(self, token_type: str, msg: str = "") -> Token:
        if self._pos >= len(self._tokens):
            raise ValueError(
                f"{self._label}: expected {token_type} but reached end of expression: {msg}"
            )
        token = self.advance()
        if token[0] != token_type:
            raise ValueError(
                f"{self._label}: expected {token_type}, got {token}: {msg}"
            )
        return token

    def parse(self, min_bp: int = 0) -> object:
        left = self._nud(self, self.advance())
        while self._pos < len(self._tokens):
            token = self.peek()
            if token[0] == "ERROR":
                raise ValueError(f"{self._label}: {token[1]}")
            if token[0] not in self._binding_powers:
                break
            l_bp, r_bp = self._binding_powers[token[0]]
            if l_bp < min_bp:
                break
            self.advance()
            left = self._led(self, left, token, r_bp)
        return left

    def parse_args(self, *, close: str, sep: str) -> list[object]:
        args = [self.parse(0)]
        while self._pos < len(self._tokens) and self.peek()[0] == sep:
            self.advance()
            args.append(self.parse(0))
        self.expect(close, "closing )")
        return args

    def parse_all(self) -> object:
        result = self.parse(0)
        if self._pos < len(self._tokens):
            raise ValueError(f"{self._label}: trailing token {self.peek()}")
        return result


def parse_pratt(
    tokens: Sequence[Token],
    *,
    label: str,
    binding_powers: BindingPowers,
    nud: Nud,
    led: Led,
    eof_token: Token,
) -> object:
    return PrattParser(
        tokens,
        label=label,
        binding_powers=binding_powers,
        nud=nud,
        led=led,
        eof_token=eof_token,
    ).parse_all()
