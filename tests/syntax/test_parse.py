"""Tests for the surface parser: parse_morphism, parse_functor, parse_program."""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from unialg.syntax.parse import parse_morphism, parse_functor, parse_program, ParseError
from unialg.syntax import expressions as expr


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_names = st.from_regex(r"[a-z][a-z0-9_.]*", fullmatch=True).filter(
    lambda n: n not in ("id", "identity", "copy", "dup", "delete", "drop", "del",
                        "absurd", "assoc", "sym", "symmetry", "x",
                        "route", "map", "load")
)

structural_keywords = st.sampled_from(["id", "identity", "copy", "absurd", "assoc", "sym"])

binary_ops = st.sampled_from([">>", "&", "||", "|"])


# ---------------------------------------------------------------------------
# parse_morphism — atoms
# ---------------------------------------------------------------------------

class TestMorphismAtoms:
    def test_ref_name(self):
        tree = parse_morphism("relu")
        assert isinstance(tree, expr.Ref)
        assert tree.name == "relu"

    def test_identity(self):
        assert isinstance(parse_morphism("id"), expr.Identity)
        assert isinstance(parse_morphism("identity"), expr.Identity)

    def test_copy(self):
        assert isinstance(parse_morphism("copy"), expr.Copy)

    def test_delete_keyword(self):
        assert isinstance(parse_morphism("delete"), expr.Delete)
        assert isinstance(parse_morphism("drop"), expr.Delete)
        assert isinstance(parse_morphism("del"), expr.Delete)

    def test_delete_bang(self):
        assert isinstance(parse_morphism("!"), expr.Delete)

    def test_absurd(self):
        assert isinstance(parse_morphism("absurd"), expr.Absurd)

    def test_int_0_absurd(self):
        assert isinstance(parse_morphism("0"), expr.Absurd)

    def test_int_1_delete(self):
        assert isinstance(parse_morphism("1"), expr.Delete)

    def test_assoc(self):
        assert isinstance(parse_morphism("assoc"), expr.Assoc)

    def test_symmetry(self):
        assert isinstance(parse_morphism("sym"), expr.Symmetry)
        assert isinstance(parse_morphism("symmetry"), expr.Symmetry)

    def test_projection_prefix(self):
        assert isinstance(parse_morphism("[0]"), expr.First)
        assert isinstance(parse_morphism("[1]"), expr.Second)

    def test_identity_x(self):
        assert isinstance(parse_morphism("x"), expr.Identity)

    @given(valid_names)
    @settings(max_examples=20)
    def test_arbitrary_name_produces_ref(self, name):
        tree = parse_morphism(name)
        assert isinstance(tree, expr.Ref)
        assert tree.name == name


# ---------------------------------------------------------------------------
# parse_morphism — binary operators
# ---------------------------------------------------------------------------

class TestMorphismBinary:
    def test_compose(self):
        tree = parse_morphism("add >> relu")
        assert isinstance(tree, expr.Compose)
        assert isinstance(tree.f, expr.Ref) and tree.f.name == "add"
        assert isinstance(tree.g, expr.Ref) and tree.g.name == "relu"

    def test_pair(self):
        tree = parse_morphism("add & mul")
        assert isinstance(tree, expr.Pair)
        assert isinstance(tree.f, expr.Ref) and tree.f.name == "add"
        assert isinstance(tree.g, expr.Ref) and tree.g.name == "mul"

    def test_par(self):
        tree = parse_morphism("add || mul")
        assert isinstance(tree, expr.Parallel)

    def test_case(self):
        tree = parse_morphism("add | mul")
        assert isinstance(tree, expr.Case)

    def test_precedence_pair_over_compose(self):
        tree = parse_morphism("a >> b & c")
        assert isinstance(tree, expr.Compose)
        assert isinstance(tree.g, expr.Pair)

    def test_precedence_compose_over_case(self):
        tree = parse_morphism("a | b >> c")
        assert isinstance(tree, expr.Case)
        assert isinstance(tree.g, expr.Compose)

    @given(valid_names, valid_names, binary_ops)
    @settings(max_examples=30)
    def test_binary_produces_contextual_binary(self, left, right, op):
        src = f"{left} {op} {right}"
        tree = parse_morphism(src)
        assert isinstance(tree, expr.ContextualBinary)


# ---------------------------------------------------------------------------
# parse_morphism — functor action
# ---------------------------------------------------------------------------

class TestMorphismFmap:
    def test_named_functor(self):
        tree = parse_morphism("F{add}")
        assert isinstance(tree, expr.PolyFmap)
        assert isinstance(tree.body, expr.PolyRef) and tree.body.name == "F"
        assert isinstance(tree.f, expr.Ref) and tree.f.name == "add"

    def test_identity_functor(self):
        tree = parse_morphism("x{add}")
        assert isinstance(tree, expr.PolyFmap)
        assert isinstance(tree.body, expr.Id)

    def test_nested_functor_action(self):
        tree = parse_morphism("F{G{h}}")
        assert isinstance(tree, expr.PolyFmap)
        inner = tree.f
        assert isinstance(inner, expr.PolyFmap)


# ---------------------------------------------------------------------------
# parse_morphism — postfix operators
# ---------------------------------------------------------------------------

class TestMorphismPostfix:
    def test_projection_postfix(self):
        tree = parse_morphism("f[0]")
        assert isinstance(tree, expr.Compose)
        assert isinstance(tree.g, expr.First)

    def test_copy_power(self):
        tree = parse_morphism("f*3")
        assert isinstance(tree, expr.Compose)

    def test_injection_postfix(self):
        tree = parse_morphism("f|0")
        assert isinstance(tree, expr.Compose)
        assert isinstance(tree.g, expr.Left)


# ---------------------------------------------------------------------------
# parse_functor
# ---------------------------------------------------------------------------

class TestFunctorParse:
    def test_id(self):
        assert isinstance(parse_functor("x"), expr.Id)
        assert isinstance(parse_functor("id"), expr.Id)

    def test_zero(self):
        assert isinstance(parse_functor("0"), expr.Zero)

    def test_one(self):
        assert isinstance(parse_functor("1"), expr.One)

    def test_prod(self):
        tree = parse_functor("x & 1")
        assert isinstance(tree, expr.Prod)
        assert isinstance(tree.left, expr.Id)
        assert isinstance(tree.right, expr.One)

    def test_sum(self):
        tree = parse_functor("x | 1")
        assert isinstance(tree, expr.Sum)

    def test_list(self):
        tree = parse_functor("List[x]")
        assert isinstance(tree, expr.List)
        assert isinstance(tree.body, expr.Id)

    def test_maybe(self):
        tree = parse_functor("Maybe[x]")
        assert isinstance(tree, expr.Maybe)

    def test_poly_ref(self):
        tree = parse_functor("MyFunctor")
        assert isinstance(tree, expr.PolyRef)
        assert tree.name == "MyFunctor"

    def test_nested(self):
        tree = parse_functor("List[x & 1]")
        assert isinstance(tree, expr.List)
        assert isinstance(tree.body, expr.Prod)


# ---------------------------------------------------------------------------
# parse_program
# ---------------------------------------------------------------------------

class TestProgramParse:
    def test_single_route(self):
        prog = parse_program("route f = add >> mul")
        assert "f" in prog.routes
        assert isinstance(prog.routes["f"], expr.Compose)

    def test_single_map(self):
        prog = parse_program("map F = x & 1")
        assert "F" in prog.functors
        assert isinstance(prog.functors["F"], expr.Prod)

    def test_multiple_declarations(self):
        prog = parse_program("route a = id\nroute b = a >> id\nmap F = x | 1")
        assert len(prog.routes) == 2
        assert len(prog.functors) == 1

    def test_load_directive(self):
        prog = parse_program("load numpy\nroute f = add")
        assert prog.loads == ("numpy",)
        assert "f" in prog.routes

    def test_multiple_loads(self):
        prog = parse_program("load numpy\nload torch\nroute f = add")
        assert prog.loads == ("numpy", "torch")

    def test_route_params(self):
        prog = parse_program("route f(theta, bias) = theta >> bias")
        assert prog.route_params["f"] == ("theta", "bias")

    def test_invalid_keyword_raises(self):
        with pytest.raises(ParseError):
            parse_program("invalid f = id")
