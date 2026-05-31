"""Tests for Rose and Tree first-class PolyExpr nodes."""
import pytest
from hydra.core import LiteralTypeInteger, IntegerType, TypeLiteral

from unialg.syntax import expressions as expr
from unialg.semantics.functors import Functor, apply_poly, compose_poly, rose, tree
from unialg.objects import ProductType, SumType, TypeList, TypeUnit

INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))


class TestApplyPoly:
    def test_rose_id_applied(self):
        # Rose(Id)(X) = X × List[X]
        result = apply_poly(expr.Rose(expr.Id()), INT)
        assert result == ProductType(INT, TypeList(INT))  # type: ignore[arg-type]

    def test_tree_id_applied(self):
        # Tree(Id)(X) = 1 + Rose(Id)(X) = 1 + X × List[X]
        result = apply_poly(expr.Tree(expr.Id()), INT)
        expected_inner = apply_poly(expr.Rose(expr.Id()), INT)
        assert result == SumType(TypeUnit(), expected_inner)  # type: ignore[arg-type]

    def test_rose_const_applied(self):
        # Rose(Const A)(X) = A × List[X]  (body does not use X)
        result = apply_poly(expr.Rose(expr.Const(INT)), INT)
        assert result == ProductType(INT, TypeList(INT))  # type: ignore[arg-type]

    def test_tree_one_applied(self):
        # Tree(One)(X) = 1 + 1 × List[X]
        result = apply_poly(expr.Tree(expr.One()), INT)
        inner = apply_poly(expr.Rose(expr.One()), INT)
        assert result == SumType(TypeUnit(), inner)  # type: ignore[arg-type]


class TestComposePoly:
    def test_rose_compose(self):
        # Rose(Id) ∘ F = Rose(F) — body is substituted
        f = expr.Const(INT)
        result = compose_poly(expr.Rose(expr.Id()), f)
        assert result == expr.Rose(expr.Const(INT))

    def test_tree_compose(self):
        # Tree(Id) ∘ F = Tree(F)
        f = expr.Const(INT)
        result = compose_poly(expr.Tree(expr.Id()), f)
        assert result == expr.Tree(expr.Const(INT))

    def test_tree_delegates_to_rose_on_composition(self):
        f = expr.Id()
        tree_composed = compose_poly(expr.Tree(expr.Id()), f)
        rose_composed = compose_poly(expr.Rose(expr.Id()), f)
        # Tree(Id) ∘ Id = Tree(Id), Rose(Id) ∘ Id = Rose(Id)
        assert isinstance(tree_composed, expr.Tree)
        assert isinstance(rose_composed, expr.Rose)


class TestXArity:
    def test_rose_id_arity(self):
        f = Functor("R", expr.Rose(expr.Id()))
        # body = Id + implicit List[Y] slot → 2 Id occurrences
        assert f.x_arity() == 2

    def test_tree_id_arity(self):
        f = Functor("T", expr.Tree(expr.Id()))
        assert f.x_arity() == 2

    def test_rose_const_arity_zero(self):
        f = Functor("R", expr.Rose(expr.Const(INT)))
        # Rose(Const A)(Y) = A × List[Y]: 1 Id slot from List[Y]
        assert f.x_arity() == 1

    def test_tree_const_arity_zero(self):
        f = Functor("T", expr.Tree(expr.Const(INT)))
        assert f.x_arity() == 1


class TestPretty:
    def test_rose_pretty(self):
        from unialg.syntax.expressions import pretty
        assert pretty(expr.Rose(expr.Id())) == "Rose[X]"

    def test_tree_pretty(self):
        from unialg.syntax.expressions import pretty
        assert pretty(expr.Tree(expr.Id())) == "Tree[X]"


class TestConstructors:
    def test_rose_constructor(self):
        assert rose(expr.Id()) == expr.Rose(expr.Id())

    def test_tree_constructor(self):
        assert tree(expr.Id()) == expr.Tree(expr.Id())
