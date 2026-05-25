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
                        "absurd", "assoc", "sym", "symmetry", "distl", "distr", "merge", "x",
                        "let", "shape", "load", "fix", "by")
)

structural_keywords = st.sampled_from(["id", "identity", "copy", "absurd", "assoc", "sym", "distl", "distr"])

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

    def test_distl(self):
        assert isinstance(parse_morphism("distl"), expr.DistributeLeft)

    def test_distr(self):
        assert isinstance(parse_morphism("distr"), expr.DistributeRight)

    def test_projection_prefix(self):
        assert isinstance(parse_morphism("[0]"), expr.First)
        assert isinstance(parse_morphism("[1]"), expr.Second)

    def test_identity_x(self):
        assert isinstance(parse_morphism("x"), expr.Identity)

    def test_quoted_literal_argument_is_generic_syntax(self):
        tree = parse_morphism("foo(x, '-1')")
        assert isinstance(tree, expr.MorphismApp)
        assert isinstance(tree.args[1], expr.Literal)
        assert tree.args[1].text == "-1"
        assert tree.args[1].value is None

    @given(valid_names)
    @settings(max_examples=20)
    def test_arbitrary_name_produces_ref(self, name):
        tree = parse_morphism(name)
        assert isinstance(tree, expr.Ref)
        assert tree.name == name

    def test_cata_app(self):
        tree = parse_morphism("cata[nat](alg)")
        assert isinstance(tree, expr.RecursionApp)
        assert tree.kind == "cata"
        assert tree.focus == "nat"
        assert len(tree.args) == 1

    def test_hylo_app(self):
        tree = parse_morphism("hylo[nat](coalg, alg)")
        assert isinstance(tree, expr.RecursionApp)
        assert tree.kind == "hylo"
        assert tree.focus == "nat"
        assert len(tree.args) == 2

    def test_carrier_boundary(self):
        tree = parse_morphism("roll[nat]")
        assert isinstance(tree, expr.CarrierBoundary)
        assert tree.kind == "roll"
        assert tree.focus == "nat"

        tree = parse_morphism("unroll[nat]")
        assert isinstance(tree, expr.CarrierBoundary)
        assert tree.kind == "unroll"
        assert tree.focus == "nat"

    def test_monadic_lift(self):
        tree = parse_morphism("pure[Maybe](add1)")
        assert isinstance(tree, expr.MonadicLift)
        assert tree.monad == "Maybe"
        assert isinstance(tree.body, expr.Ref)


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

    def test_functor_composition(self):
        tree = parse_functor("F >> G")
        assert isinstance(tree, expr.PolyCompose)
        assert isinstance(tree.left, expr.PolyRef) and tree.left.name == "F"
        assert isinstance(tree.right, expr.PolyRef) and tree.right.name == "G"

    def test_nested(self):
        tree = parse_functor("List[x & 1]")
        assert isinstance(tree, expr.List)
        assert isinstance(tree.body, expr.Prod)


# ---------------------------------------------------------------------------
# parse_program
# ---------------------------------------------------------------------------

class TestProgramParse:
    def test_single_let(self):
        prog = parse_program("let f = add >> mul")
        assert "f" in prog.morphisms
        assert isinstance(prog.morphisms["f"], expr.Compose)

    def test_single_map(self):
        prog = parse_program("shape F = x & 1")
        assert "F" in prog.functors
        assert isinstance(prog.functors["F"], expr.Prod)

    def test_multiple_declarations(self):
        prog = parse_program("let a = id\nlet b = a >> id\nshape F = x | 1")
        assert len(prog.morphisms) == 2
        assert len(prog.functors) == 1

    def test_load_directive(self):
        prog = parse_program("load numpy\nlet f = add")
        assert prog.loads == ("numpy",)
        assert "f" in prog.morphisms

    def test_multiple_loads(self):
        prog = parse_program("load numpy\nload torch\nlet f = add")
        assert prog.loads == ("numpy", "torch")

    def test_let_params(self):
        prog = parse_program("let f(theta, bias) = theta >> bias")
        assert prog.morphism_params["f"] == ("theta", "bias")

    def test_focus_decl(self):
        prog = parse_program(
            "shape Id = x\n"
            "shape self : Id <-> Id by add1 / mul2\n"
            "let folded = cata[self](add1)"
        )
        assert prog.focuses["self"].functor == "Id"
        assert isinstance(prog.focuses["self"].forward, expr.Ref)
        assert isinstance(prog.morphisms["folded"], expr.RecursionApp)

    def test_explicit_optic_accepts_boundary_annotations(self):
        prog = parse_program("shape root : Tree(int) <-> int by extract / replace")
        assert prog.focuses["root"].functor == "Tree"
        assert isinstance(prog.focuses["root"].forward, expr.Ref)
        assert isinstance(prog.focuses["root"].backward, expr.Ref)

    def test_carrier_decl(self):
        prog = parse_program("shape NatF = 1 | x\nshape Nat = fix NatF")
        assert isinstance(prog.carriers["Nat"].functor, expr.PolyRef)
        assert prog.carriers["Nat"].functor.name == "NatF"

    def test_carrier_focus_decl(self):
        prog = parse_program(
            "shape NatF = 1 | x\n"
            "shape Nat = fix NatF"
        )
        assert "Nat" in prog.carriers
        assert prog.focuses == {}

    def test_focus_composition_decl(self):
        prog = parse_program(
            "shape NatF = 1 | x\n"
            "shape Nat = fix NatF\n"
            "shape two_layers = Nat >> Nat"
        )
        assert isinstance(prog.functors["two_layers"], expr.PolyCompose)

    def test_invalid_keyword_raises(self):
        with pytest.raises(ParseError):
            parse_program("invalid f = id")
