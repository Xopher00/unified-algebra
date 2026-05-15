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

from unialg.syntax.parse import parse_morphism, parse_functor, parse_program, Program, ParseError
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


def test_copy_power_two():
    r = parse_morphism("*2")
    assert isinstance(r, Copy)


def test_copy_power_three_left_nested():
    r = parse_morphism("*3")
    assert isinstance(r, Pair)
    assert isinstance(r.f, Copy)
    assert isinstance(r.g, Identity)


def test_copy_power_is_not_fixed_arity():
    r = parse_morphism("*5")
    for _ in range(3):
        assert isinstance(r, Pair)
        assert isinstance(r.g, Identity)
        r = r.f
    assert isinstance(r, Copy)


def test_postfix_copy_power_two():
    r = parse_morphism("morph*2")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    assert isinstance(r.g, Copy)


def test_postfix_copy_power_is_not_fixed_arity():
    r = parse_morphism("morph*5")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    copied = r.g
    for _ in range(3):
        assert isinstance(copied, Pair)
        assert isinstance(copied.g, Identity)
        copied = copied.f
    assert isinstance(copied, Copy)


def test_postfix_copy_power_has_high_precedence():
    r = parse_morphism("f*2 >> g")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Compose)
    assert isinstance(r.f.f, Ref) and r.f.f.name == "f"
    assert isinstance(r.f.g, Copy)
    assert isinstance(r.g, Ref) and r.g.name == "g"


def test_postfix_copy_power_requires_at_least_two():
    with pytest.raises(ParseError, match="integer >= 2"):
        parse_morphism("morph*1")


def test_copy_power_requires_at_least_two():
    with pytest.raises(ParseError, match="integer >= 2"):
        parse_morphism("*1")


def test_copy_power_requires_integer():
    with pytest.raises(ParseError, match="copy power"):
        parse_morphism("*")


def test_fst():
    r = parse_morphism("fst")
    assert isinstance(r, First)


def test_postfix_projection_first():
    r = parse_morphism("morph[0]")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    assert isinstance(r.g, First)


def test_postfix_projection_second():
    r = parse_morphism("morph[1]")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    assert isinstance(r.g, Second)


def test_postfix_projection_on_grouped_pair():
    r = parse_morphism("(f & g)[0]")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Pair)
    assert isinstance(r.g, First)


def test_postfix_projection_has_high_precedence():
    r = parse_morphism("f & g[1]")
    assert isinstance(r, Pair)
    assert isinstance(r.f, Ref) and r.f.name == "f"
    assert isinstance(r.g, Compose)
    assert isinstance(r.g.f, Ref) and r.g.f.name == "g"
    assert isinstance(r.g.g, Second)


def test_postfix_projection_rejects_other_indexes():
    with pytest.raises(ParseError, match="0 or 1"):
        parse_morphism("morph[2]")


def test_snd():
    r = parse_morphism("snd")
    assert isinstance(r, Second)


def test_inl():
    r = parse_morphism("inl")
    assert isinstance(r, Left)


def test_case_injection_zero():
    r = parse_morphism("|0")
    assert isinstance(r, Left)


def test_case_injection_one():
    r = parse_morphism("|1")
    assert isinstance(r, Right)


def test_case_injection_rejects_other_indexes():
    with pytest.raises(ParseError, match="0 or 1"):
        parse_morphism("|2")


def test_case_injection_requires_integer():
    with pytest.raises(ParseError, match="prefix position"):
        parse_morphism("| f")


def test_postfix_case_injection_zero():
    r = parse_morphism("morph|0")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    assert isinstance(r.g, Left)


def test_postfix_case_injection_one():
    r = parse_morphism("morph|1")
    assert isinstance(r, Compose)
    assert isinstance(r.f, Ref) and r.f.name == "morph"
    assert isinstance(r.g, Right)


def test_postfix_case_injection_has_high_precedence():
    r = parse_morphism("f|0 | g")
    assert isinstance(r, Case)
    assert isinstance(r.f, Compose)
    assert isinstance(r.f.f, Ref) and r.f.f.name == "f"
    assert isinstance(r.f.g, Left)
    assert isinstance(r.g, Ref) and r.g.name == "g"


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


def test_dotted_name_is_single_ref():
    r = parse_morphism("reduce.add")
    assert isinstance(r, Ref) and r.name == "reduce.add"


def test_dotted_name_in_env():
    from unialg.syntax.expressions import Delete
    from unialg.objects import TypeUnit
    stub = Delete(TypeUnit())
    r = parse_morphism("reduce.add", env={"reduce.add": stub})
    assert r is stub


def test_del_alias():
    r = parse_morphism("del")
    assert isinstance(r, Delete)


def test_del_alias_with_n():
    r = parse_morphism("del(2)")
    assert isinstance(r, Delete)


def test_dup_alias():
    r = parse_morphism("dup(2)")
    assert isinstance(r, Copy)


def test_dup_alias_three():
    r = parse_morphism("dup(3)")
    assert isinstance(r, Pair)
    assert isinstance(r.f, Copy)
    assert isinstance(r.g, Identity)


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


# ---------------------------------------------------------------------------
# Program-level parsing (route / map)
# ---------------------------------------------------------------------------

def test_program_single_route():
    prog = parse_program("route copy_id = copy")
    assert isinstance(prog, Program)
    assert "copy_id" in prog.morphisms
    assert isinstance(prog.morphisms["copy_id"], Copy)


def test_program_single_map():
    prog = parse_program("map ListF = x*")
    assert "ListF" in prog.functors
    assert isinstance(prog.functors["ListF"], List)


def test_program_multiple_routes():
    src = "route f = copy\nroute g = delete"
    prog = parse_program(src)
    assert isinstance(prog.morphisms["f"], Copy)
    assert isinstance(prog.morphisms["g"], Delete)


def test_program_mixed():
    src = "route f = fst\nmap F = 1 | x\n"
    prog = parse_program(src)
    assert isinstance(prog.morphisms["f"], First)
    assert isinstance(prog.functors["F"], Sum)


def test_program_route_references_earlier_route():
    # second route references the first by name — env propagation
    src = "route f = copy\nroute g = f >> fst"
    prog = parse_program(src)
    g = prog.morphisms["g"]
    assert isinstance(g, Compose)
    assert isinstance(g.f, Copy)
    assert isinstance(g.g, First)


def test_program_map_references_earlier_map():
    src = "map A = x*\nmap B = A | 1"
    prog = parse_program(src)
    b = prog.functors["B"]
    assert isinstance(b, Sum)
    assert isinstance(b.left, List)


def test_program_route_fmap_references_earlier_map():
    src = "map Pair = x & x\nroute gated = Pair{tanh}"
    prog = parse_program(src)
    gated = prog.morphisms["gated"]
    assert isinstance(gated, PolyFmap)
    assert isinstance(gated.body, Prod)
    assert isinstance(gated.body.left, Id)
    assert isinstance(gated.body.right, Id)


def test_program_empty():
    prog = parse_program("")
    assert prog.morphisms == {}
    assert prog.functors == {}


def test_program_bad_keyword():
    with pytest.raises(ParseError, match="expected"):
        parse_program("define f = copy")


def test_program_missing_name():
    with pytest.raises(ParseError):
        parse_program("route = copy")


def test_program_missing_eq():
    with pytest.raises(ParseError):
        parse_program("route f copy")


# ---------------------------------------------------------------------------
# load directive — parser-only (stub handler, no real backend)
# ---------------------------------------------------------------------------

def _stub_handler(name: str):
    """Return fake Prim nodes for 'add' and 'multiply' regardless of backend."""
    from unialg.syntax.expressions import Prim
    from unialg.objects import TypeUnit
    tu = TypeUnit()
    return {"add": Prim(object(), tu, tu), "multiply": Prim(object(), tu, tu)}


def test_load_records_backend():
    prog = parse_program("load numpy", load_handler=_stub_handler)
    assert prog.loads == ("numpy",)


def test_load_binds_aliases():
    prog = parse_program("load numpy\nroute f = add", load_handler=_stub_handler)
    from unialg.syntax.expressions import Prim
    assert isinstance(prog.morphisms["f"], Prim)


def test_load_no_handler_records_only():
    # Without a handler, directive is still recorded but no env binding.
    prog = parse_program("load numpy\nroute g = unknown_op")
    assert prog.loads == ("numpy",)
    from unialg.syntax.expressions import Ref
    assert isinstance(prog.morphisms["g"], Ref)


def test_load_two_backends():
    prog = parse_program("load numpy\nload jax", load_handler=_stub_handler)
    assert prog.loads == ("numpy", "jax")


def test_load_then_route_composes():
    prog = parse_program("load numpy\nroute f = add >> multiply", load_handler=_stub_handler)
    from unialg.syntax.expressions import Compose, Prim
    assert isinstance(prog.morphisms["f"], Compose)
    assert isinstance(prog.morphisms["f"].f, Prim)
    assert isinstance(prog.morphisms["f"].g, Prim)


def test_load_bad_token():
    with pytest.raises(ParseError):
        parse_program("load >>", load_handler=_stub_handler)
