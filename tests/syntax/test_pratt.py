"""Tests for the Pratt parser in syntax/parse.py.

Covers:
- Morphism grammar: operators, precedence, left-associativity, atoms
- Functor grammar: operators, atoms
- Error tokens and parse errors
- Hypothesis: functor parse never crashes on valid token sequences
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unialg.syntax.parse import parse_morphism, parse_functor, ParseError
from unialg.syntax.expressions import (
    Compose, Pair, Parallel, Case,
    Identity, Delete, Copy, First, Second, Left, Right, Absurd, Assoc, Symmetry,
    Ref, PolyRef, PolyFmap,
    Zero, One, Id, Sum, Prod, List,
)


# ---------------------------------------------------------------------------
# Morphism grammar — operator shapes
# ---------------------------------------------------------------------------

def test_compose():
    r = parse_morphism("f >> g")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Ref) and r.g.name == "g"


def test_pair():
    r = parse_morphism("f & g")
    assert isinstance(r, Pair)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Ref) and r.g.name == "g"


def test_par():
    r = parse_morphism("f || g")
    assert isinstance(r, Parallel)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Ref) and r.g.name == "g"


def test_case():
    r = parse_morphism("f | g")
    assert isinstance(r, Case)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Ref) and r.g.name == "g"


# ---------------------------------------------------------------------------
# Precedence and associativity
# ---------------------------------------------------------------------------

def test_pair_binds_tighter_than_compose():
    # f & g >> h  →  Compose(Pair(f, g), h)
    r = parse_morphism("f & g >> h")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Pair)
    assert isinstance(r.f.f, Ref) and r.f.f.name == "f"
    assert isinstance(r.f.g, Ref) and r.f.g.name == "g"
    assert isinstance(r.g, Ref) and r.g.name == "h"


def test_par_binds_tighter_than_compose():
    # f || g >> h  →  Compose(Parallel(f, g), h)
    r = parse_morphism("f || g >> h")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Parallel)


def test_compose_left_associative():
    # f >> g >> h  →  Compose(Compose(f, g), h)
    r = parse_morphism("f >> g >> h")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Compose)
    assert isinstance(r.f.f, Ref) and r.f.f.name == "f"
    assert isinstance(r.f.g, Ref) and r.f.g.name == "g"
    assert isinstance(r.g, Ref) and r.g.name == "h"


def test_pair_left_associative():
    # f & g & h  →  Pair(Pair(f, g), h)
    r = parse_morphism("f & g & h")
    assert isinstance(r, Pair)
    assert isinstance(r.f, Pair)


def test_case_lowest_precedence():
    # f | g >> h  →  Case(f, Compose(g, h))? No: >> binds tighter than |
    # Actually: >> (60) > | (50), so g >> h groups first
    r = parse_morphism("f | g >> h")
    assert isinstance(r, Case)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Compose)


def test_grouping():
    # (f | g) >> h  →  Compose(Case(f, g), h)
    r = parse_morphism("(f | g) >> h")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Case)


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

def test_identity_x():
    r = parse_morphism("x")
    assert isinstance(r, Identity)


def test_identity_id():
    r = parse_morphism("id")
    assert isinstance(r, Identity)


def test_delete_bang():
    r = parse_morphism("!")
    assert isinstance(r, Delete)


def test_delete_name():
    r = parse_morphism("delete")
    assert isinstance(r, Delete)


def test_copy():
    r = parse_morphism("copy")
    assert isinstance(r, Copy)


def test_fst():
    r = parse_morphism("fst")
    assert isinstance(r, First)


def test_snd():
    r = parse_morphism("snd")
    assert isinstance(r, Second)


def test_inl():
    r = parse_morphism("inl")
    assert isinstance(r, Left)


def test_inr():
    r = parse_morphism("inr")
    assert isinstance(r, Right)


def test_absurd():
    r = parse_morphism("absurd")
    assert isinstance(r, Absurd)


def test_assoc():
    r = parse_morphism("assoc")
    assert isinstance(r, Assoc)


def test_sym():
    r = parse_morphism("sym")
    assert isinstance(r, Symmetry)


def test_ref_unknown_name():
    r = parse_morphism("unknown_fn")
    assert isinstance(r, Ref) and r.name == "unknown_fn"


def test_env_resolution():
    from unialg.syntax.expressions import Identity
    from unialg.objects import TypeUnit
    concrete = Identity(TypeUnit())
    r = parse_morphism("f", env={"f": concrete})
    assert r is concrete


def test_e_bracket():
    r = parse_morphism("E[linear_seq]")
    assert isinstance(r, Ref) and r.name == "linear_seq"


# ---------------------------------------------------------------------------
# PolyFmap (F{f} syntax)
# ---------------------------------------------------------------------------

def test_poly_fmap_x_star():
    # x*{f}  →  PolyFmap(List(Id()), Ref('f'))
    r = parse_morphism("x*{f}")
    assert isinstance(r, PolyFmap)
    assert isinstance(r.body, List)
    assert isinstance(r.body.body, Id)
    assert isinstance(r.f, Ref) and r.f.name == "f"


def test_poly_fmap_double_star():
    # x**{f}  →  PolyFmap(List(List(Id())), Ref('f'))
    r = parse_morphism("x**{f}")
    assert isinstance(r, PolyFmap)
    assert isinstance(r.body, List)
    assert isinstance(r.body.body, List)


def test_named_functor_fmap():
    # F{f}  →  PolyFmap(PolyRef('F'), Ref('f'))
    r = parse_morphism("F{f}")
    assert isinstance(r, PolyFmap)
    assert isinstance(r.body, PolyRef) and r.body.name == "F"
    assert isinstance(r.f, Ref) and r.f.name == "f"


def test_seq_ffn():
    # seq_ffn = x*{token_ffn}
    r = parse_morphism("x*{token_ffn}")
    assert isinstance(r, PolyFmap)
    assert isinstance(r.body, List)
    assert isinstance(r.f, Ref) and r.f.name == "token_ffn"


def test_dual_stream():
    # x*{left_block} || x*{right_block}
    r = parse_morphism("x*{left_block} || x*{right_block}")
    assert isinstance(r, Parallel)
    assert isinstance(r.f, PolyFmap)
    assert isinstance(r.g, PolyFmap)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_semicolon_error():
    with pytest.raises(ParseError, match="'>>'"):
        parse_morphism("f ; g")


def test_trailing_token():
    with pytest.raises(ParseError, match="trailing"):
        parse_morphism("f >> g h")


def test_unexpected_eof():
    with pytest.raises(ParseError):
        parse_morphism("f >>")


# ---------------------------------------------------------------------------
# Functor grammar
# ---------------------------------------------------------------------------

def test_functor_zero():
    r = parse_functor("0")
    assert isinstance(r, Zero)


def test_functor_one():
    r = parse_functor("1")
    assert isinstance(r, One)


def test_functor_id():
    r = parse_functor("x")
    assert isinstance(r, Id)


def test_functor_list():
    r = parse_functor("x*")
    assert isinstance(r, List)
    assert isinstance(r.body, Id)


def test_functor_maybe():
    # 1 | x  →  Sum(One, Id) = Maybe pattern
    r = parse_functor("1 | x")
    assert isinstance(r, Sum)
    assert isinstance(r.left, One)
    assert isinstance(r.right, Id)


def test_functor_sum_list():
    # 1 | x*  →  Sum(One, List(Id))
    r = parse_functor("1 | x*")
    assert isinstance(r, Sum)
    assert isinstance(r.right, List)


def test_functor_prod():
    # x* & x*  →  Prod(List(Id), List(Id))
    r = parse_functor("x* & x*")
    assert isinstance(r, Prod)
    assert isinstance(r.left, List)
    assert isinstance(r.right, List)


def test_functor_star_binds_tighter_than_amp():
    # x* & x  →  Prod(List(Id), Id)   not  List(Prod(Id, Id))
    r = parse_functor("x* & x")
    assert isinstance(r, Prod)
    assert isinstance(r.left, List)
    assert isinstance(r.right, Id)


def test_functor_grouping():
    # (1 | x)*  →  List(Sum(One, Id))
    r = parse_functor("(1 | x)*")
    assert isinstance(r, List)
    assert isinstance(r.body, Sum)


def test_functor_ref():
    r = parse_functor("MyFunctor")
    assert isinstance(r, PolyRef) and r.name == "MyFunctor"


# ---------------------------------------------------------------------------
# Hypothesis: functor parse never crashes on valid functor strings
# ---------------------------------------------------------------------------

_FUNCTOR_ATOMS = st.sampled_from(["x", "0", "1"])
_FUNCTOR_OPS   = st.sampled_from(["&", "|"])


@st.composite
def functor_src(draw, max_depth: int = 3):
    if max_depth == 0 or draw(st.booleans()):
        atom = draw(_FUNCTOR_ATOMS)
        stars = draw(st.integers(min_value=0, max_value=2))
        return atom + "*" * stars
    left  = draw(functor_src(max_depth=max_depth - 1))
    op    = draw(_FUNCTOR_OPS)
    right = draw(functor_src(max_depth=max_depth - 1))
    return f"({left} {op} {right})"


@given(functor_src())
@settings(max_examples=200)
def test_functor_parse_doesnt_crash(src: str):
    result = parse_functor(src)
    assert isinstance(result, (Zero, One, Id, Sum, Prod, List, PolyRef))


# ---------------------------------------------------------------------------
# Transformer proof-of-concept expressions (structural checks only)
# ---------------------------------------------------------------------------

def test_token_ffn():
    # token_ffn = E[token_up]{W1} >> gelu >> E[token_down]{W2}
    # E[name]{f} → PolyFmap(PolyRef(name), Ref(f))
    r = parse_morphism("E[token_up]{W1} >> gelu >> E[token_down]{W2}")
    assert isinstance(r, Compose)
    # Outermost is Compose(..., E[token_down]{W2})
    assert isinstance(r.g, PolyFmap)
    assert isinstance(r.g.body, PolyRef) and r.g.body.name == "token_down"


def test_residual_block():
    # (x & attention) >> add >> layer_norm
    r = parse_morphism("(x & attention) >> add >> layer_norm")
    assert isinstance(r, Compose)
    inner = r.f
    assert isinstance(inner, Compose)
    assert isinstance(inner.f, Pair)


def test_transformer():
    src = (
        "((x & attention) >> add >> layer_norm)"
        " >> "
        "((x & x*{token_ffn}) >> add >> layer_norm)"
    )
    r = parse_morphism(src)
    assert isinstance(r, Compose)
