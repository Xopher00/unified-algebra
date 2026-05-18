"""Tests for tensor notation layer — Equation parsing and data structures."""
import pytest
from unialg.tensors.notation import Equation, AlignmentPlan, SemiringDecl, ContractExpr


class TestEquationParse:
    def test_matmul(self):
        eq = Equation.parse("ij,jk->ik")
        assert eq.inputs == (("i", "j"), ("j", "k"))
        assert eq.output == ("i", "k")
        assert eq.reduced == ("j",)

    def test_matvec(self):
        eq = Equation.parse("ij,j->i")
        assert eq.inputs == (("i", "j"), ("j",))
        assert eq.output == ("i",)
        assert eq.reduced == ("j",)

    def test_outer_product(self):
        eq = Equation.parse("i,j->ij")
        assert eq.inputs == (("i",), ("j",))
        assert eq.output == ("i", "j")
        assert eq.reduced == ()

    def test_trace_parses(self):
        eq = Equation.parse("ii->")
        assert eq.inputs == (("i", "i"),)
        assert eq.output == ()
        assert eq.reduced == ("i",)

    def test_diagonal_extraction_parses(self):
        eq = Equation.parse("ii->i")
        assert eq.inputs == (("i", "i"),)
        assert eq.output == ("i",)
        assert eq.reduced == ()

    def test_diagonal_axes(self):
        eq = Equation.parse("ii->i")
        assert eq.diagonal_axes(0) == [(0, 1)]
        eq2 = Equation.parse("iij->ij")
        assert eq2.diagonal_axes(0) == [(0, 1)]
        eq3 = Equation.parse("iji->ij")
        assert eq3.diagonal_axes(0) == [(0, 2)]

    def test_post_diagonal_labels(self):
        eq = Equation.parse("ii->i")
        assert eq.post_diagonal_labels(0) == ("i",)
        eq2 = Equation.parse("iij->ij")
        assert eq2.post_diagonal_labels(0) == ("j", "i")
        eq3 = Equation.parse("iji->ij")
        assert eq3.post_diagonal_labels(0) == ("j", "i")

    def test_no_diagonal_when_labels_unique(self):
        eq = Equation.parse("ij,jk->ik")
        assert eq.diagonal_axes(0) == []
        assert eq.diagonal_axes(1) == []
        assert eq.post_diagonal_labels(0) == ("i", "j")

    def test_batched_matmul(self):
        eq = Equation.parse("bij,bjk->bik")
        assert eq.inputs == (("b", "i", "j"), ("b", "j", "k"))
        assert eq.output == ("b", "i", "k")
        assert eq.reduced == ("j",)

    def test_three_operand(self):
        eq = Equation.parse("ij,jk,kl->il")
        assert eq.inputs == (("i", "j"), ("j", "k"), ("k", "l"))
        assert eq.output == ("i", "l")
        assert eq.reduced == ("j", "k")

    def test_whitespace_tolerance(self):
        eq = Equation.parse("  ij , jk -> ik  ")
        assert eq.inputs == (("i", "j"), ("j", "k"))
        assert eq.output == ("i", "k")

    def test_reduced_order_preserves_first_seen(self):
        eq = Equation.parse("ijk,jkl->il")
        assert eq.reduced == ("j", "k")

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="must contain '->'"):
            Equation.parse("ij,jk")

    def test_output_label_not_in_input_raises(self):
        with pytest.raises(ValueError, match="not in any input"):
            Equation.parse("ij->iz")


class TestTargetVars:
    def test_matmul(self):
        eq = Equation.parse("ij,jk->ik")
        assert eq.target_vars() == ("i", "k", "j")

    def test_no_reduced(self):
        eq = Equation.parse("i,j->ij")
        assert eq.target_vars() == ("i", "j")


class TestAlignmentPlan:
    def test_matmul_first_operand(self):
        eq = Equation.parse("ij,jk->ik")
        plan = eq.alignment_plan(0)
        # input (i,j), target (i,k,j) → unsqueeze k at axis 2, perm (0,2,1)
        assert plan.unsqueeze_axes == (2,)
        assert plan.perm == (0, 2, 1)

    def test_matmul_second_operand(self):
        eq = Equation.parse("ij,jk->ik")
        plan = eq.alignment_plan(1)
        # input (j,k), target (i,k,j) → unsqueeze i at axis 2, expanded=(j,k,i), perm (2,1,0)
        assert plan.unsqueeze_axes == (2,)
        assert plan.perm == (2, 1, 0)

    def test_outer_product_no_unsqueeze(self):
        eq = Equation.parse("ij->ij")
        plan = eq.alignment_plan(0)
        assert plan.unsqueeze_axes == ()
        assert plan.perm == (0, 1)


class TestReplaceInput:
    def test_fuse_inner_into_outer(self):
        inner = Equation.parse("ij,jk->ik")
        outer = Equation.parse("ik,kl->il")
        fused = outer.replace_input(
            slot=0,
            new_inputs=inner.inputs,
            new_reduced=inner.reduced,
        )
        assert fused.inputs == (("i", "j"), ("j", "k"), ("k", "l"))
        assert fused.output == ("i", "l")
        assert set(fused.reduced) == {"j", "k"}


class TestSemiringDecl:
    def test_fields(self):
        d = SemiringDecl(
            name="real",
            plus="add",
            times="multiply",
            zero=0.0,
            one=1.0,
        )
        assert d.plus == "add"
        assert d.adjoint is None

    def test_with_adjoint(self):
        d = SemiringDecl(
            name="smooth",
            plus="smooth_max",
            times="add",
            zero=float("-inf"),
            one=0.0,
            adjoint="smooth_res",
        )
        assert d.adjoint == "smooth_res"


class TestContractExpr:
    def test_fields(self):
        c = ContractExpr(semiring_name="real", equation_str="ij,jk->ik")
        assert c.semiring_name == "real"
        assert c.adjoint is False
        assert c._domain_tag == "tensors"

    def test_adjoint(self):
        c = ContractExpr(semiring_name="trop", equation_str="ij,j->i", adjoint=True)
        assert c.adjoint is True
