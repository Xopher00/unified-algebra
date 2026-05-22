"""Parse tests for explicit lens / prism / traversal declarations."""
import pytest

from unialg.syntax import expressions as expr
from unialg.syntax.parse import parse_program


class TestLensParse:
    def test_lens_basic(self):
        prog = parse_program(
            "shape Id = x\n"
            "shape lens myLens : Int view Int by fwd / bwd"
        )
        decl = prog.focuses["myLens"]
        assert decl.kind == "lens"
        assert decl.functor == "Int"
        assert decl.residue == "Int"
        assert isinstance(decl.forward, expr.Ref)
        assert isinstance(decl.backward, expr.Ref)

    def test_lens_forward_ref(self):
        prog = parse_program("shape lens myLens : A view R by f / g")
        assert prog.focuses["myLens"].forward == expr.Ref("f")
        assert prog.focuses["myLens"].backward == expr.Ref("g")


class TestPrismParse:
    def test_prism_basic(self):
        prog = parse_program("shape prism myPrism : A view R by f / g")
        decl = prog.focuses["myPrism"]
        assert decl.kind == "prism"
        assert decl.functor == "A"
        assert decl.residue == "R"
        assert isinstance(decl.forward, expr.Ref)
        assert isinstance(decl.backward, expr.Ref)


class TestTraversalParse:
    def test_traversal_basic(self):
        prog = parse_program(
            "shape Id = x\n"
            "shape traversal myTrav : Id by f / g"
        )
        decl = prog.focuses["myTrav"]
        assert decl.kind == "traversal"
        assert decl.functor == "Id"
        assert isinstance(decl.forward, expr.Ref)
        assert isinstance(decl.backward, expr.Ref)


class TestBackwardsCompat:
    def test_untagged_shape_keeps_optic_kind(self):
        prog = parse_program(
            "shape Id = x\n"
            "shape myOptic : Id <-> Id by f / g"
        )
        assert prog.focuses["myOptic"].kind == "optic"
        assert prog.focuses["myOptic"].residue is None

    def test_fix_shape_unaffected(self):
        prog = parse_program("shape NatF = 1 | x\nshape Nat = fix NatF")
        assert "Nat" in prog.carriers
        assert prog.focuses == {}
