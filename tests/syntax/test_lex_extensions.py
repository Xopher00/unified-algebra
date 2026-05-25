"""Tests for lexer additions: STRING, FLOAT, MINUS tokens."""
from unialg.syntax._lex import tokenize_morphism


def _tokens(src: str) -> list[tuple]:
    """Tokenize a morphism-level source string, drop EOF."""
    toks = tokenize_morphism(src)
    return [(t, v) for t, v in toks if t != "EOF"]


class TestStringToken:
    def test_simple_string(self):
        toks = _tokens('"hello"')
        assert toks == [("STRING", "hello")]

    def test_empty_string(self):
        toks = _tokens('""')
        assert toks == [("STRING", "")]

    def test_string_with_special_chars(self):
        toks = _tokens('"ij,jk->ik"')
        assert toks == [("STRING", "ij,jk->ik")]

    def test_string_among_other_tokens(self):
        toks = _tokens('f "abc" g')
        assert toks[0] == ("NAME", "f")
        assert toks[1] == ("STRING", "abc")
        assert toks[2] == ("NAME", "g")


class TestFloatToken:
    def test_zero_point_zero(self):
        toks = _tokens("0.0")
        assert toks == [("FLOAT", 0.0)]

    def test_one_point_five(self):
        toks = _tokens("1.5")
        assert toks == [("FLOAT", 1.5)]

    def test_large_float(self):
        toks = _tokens("123.456")
        assert toks == [("FLOAT", 123.456)]

    def test_int_not_confused_with_float(self):
        toks = _tokens("42")
        assert toks == [("INT", 42)]

    def test_float_among_other_tokens(self):
        toks = _tokens("x 0.0 y")
        assert toks[0] == ("NAME", "x")
        assert toks[1] == ("FLOAT", 0.0)
        assert toks[2] == ("NAME", "y")


class TestMinusToken:
    def test_minus_alone(self):
        toks = _tokens("-")
        assert toks == [("MINUS", "-")]

    def test_minus_before_name(self):
        toks = _tokens("-inf")
        assert toks == [("MINUS", "-"), ("NAME", "inf")]

    def test_minus_before_float(self):
        toks = _tokens("-1.0")
        assert toks == [("MINUS", "-"), ("FLOAT", 1.0)]

    def test_minus_before_int(self):
        toks = _tokens("-3")
        assert toks == [("MINUS", "-"), ("INT", 3)]


class TestExistingTokensUnchanged:
    """Verify the new tokens don't break existing tokenization."""

    def test_compose(self):
        toks = _tokens("f >> g")
        assert ("COMPOSE", ">>") in toks

    def test_coparallel(self):
        toks = _tokens("f && g")
        assert toks == [("NAME", "f"), ("COPAR", "&&"), ("NAME", "g")]

    def test_identity(self):
        toks = _tokens("id")
        assert toks == [("NAME", "id")]

    def test_bidir(self):
        toks = _tokens("A <-> B")
        assert ("BIDIR", "<->") in toks

    def test_int_still_works(self):
        toks = _tokens("3")
        assert toks == [("INT", 3)]

    def test_brackets(self):
        toks = _tokens("[f, g]")
        assert toks[0] == ("LBRACKET", "[")
        assert toks[-1] == ("RBRACKET", "]")

    def test_parens(self):
        toks = _tokens("(f)")
        assert toks[0] == ("LPAREN", "(")
        assert toks[-1] == ("RPAREN", ")")
