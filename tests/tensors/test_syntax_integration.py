"""Tests for tensor syntax integration — parser dispatches to tensor domain."""
import pytest
from unialg.syntax.parse import parse_program
from unialg.tensors.notation import SemiringDecl, ContractExpr


class TestAlgebraDeclaration:
    def test_basic_algebra(self):
        prog = parse_program(
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)'
        )
        assert "tensors" in prog.extensions
        decls = prog.extensions["tensors"]
        assert len(decls) == 1
        d = decls[0]
        assert isinstance(d, SemiringDecl)
        assert d.name == "real"
        assert d.plus == "add"
        assert d.times == "multiply"
        assert d.zero == 0.0
        assert d.one == 1.0
        assert d.adjoint is None

    def test_algebra_with_adjoint(self):
        prog = parse_program(
            'algebra smooth(plus=smooth_max, times=add, zero=-inf, one=0.0, adjoint=smooth_res)'
        )
        d = prog.extensions["tensors"][0]
        assert d.name == "smooth"
        assert d.zero == float("-inf")
        assert d.adjoint == "smooth_res"

    def test_algebra_integer_identity(self):
        prog = parse_program(
            'algebra trop(plus=minimum, times=add, zero=999, one=0)'
        )
        d = prog.extensions["tensors"][0]
        assert d.zero == 999.0
        assert d.one == 0.0

    def test_algebra_negative_float(self):
        prog = parse_program(
            'algebra test(plus=add, times=mul, zero=-1.0, one=1.0)'
        )
        d = prog.extensions["tensors"][0]
        assert d.zero == -1.0

    def test_algebra_missing_field_raises(self):
        with pytest.raises(Exception, match="missing fields"):
            parse_program('algebra bad(plus=add, times=multiply)')

    def test_multiple_algebras(self):
        prog = parse_program(
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)\n'
            'algebra trop(plus=minimum, times=add, zero=999, one=0)'
        )
        decls = prog.extensions["tensors"]
        assert len(decls) == 2
        assert decls[0].name == "real"
        assert decls[1].name == "trop"


class TestDeclarationDelimiter:
    def test_let_then_algebra(self):
        prog = parse_program(
            'let f = id\n'
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)'
        )
        assert "f" in prog.morphisms
        assert "tensors" in prog.extensions
        assert prog.extensions["tensors"][0].name == "real"

    def test_algebra_then_let(self):
        prog = parse_program(
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)\n'
            'let f = id'
        )
        assert "f" in prog.morphisms
        assert prog.extensions["tensors"][0].name == "real"


class TestFullPipelineParse:
    """Test that compile_program parses algebra + contract end-to-end."""

    def test_compile_program_with_contract(self):
        from unialg.main import compile_program
        from unialg.tensors.semantics import ContractSpec
        from unialg.syntax import expressions as expr

        # This exercises: parse → extension dispatch → domain construct →
        # construct_expr → contract_morphism. Compilation will hit the
        # lowering stub (NotImplementedError) so we stop before compile.
        from unialg.semantics.construct import construct_program
        prog = parse_program(
            'load numpy\n'
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)\n'
            'let matmul = contract[real]("ij,jk->ik")'
        )
        from unialg.main import load_backend, _resolve_backend_spec
        env = load_backend(_resolve_backend_spec("numpy"))
        constructed = construct_program(prog, env)
        m = constructed.morphisms["matmul"]
        assert isinstance(m.node, expr.Prim)
        assert isinstance(m.node.raw, ContractSpec)
        assert m.node.raw.equation.reduced == ("j",)


class TestContractExpression:
    def test_basic_contract(self):
        prog = parse_program(
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)\n'
            'let matmul = contract[real]("ij,jk->ik")'
        )
        node = prog.morphisms["matmul"]
        assert isinstance(node, ContractExpr)
        assert node.semiring_name == "real"
        assert node.equation_str == "ij,jk->ik"
        assert node.adjoint is False

    def test_contract_with_adjoint(self):
        prog = parse_program(
            'algebra trop(plus=minimum, times=add, zero=999, one=0)\n'
            'let relax = contract[trop, adjoint]("ij,j->i")'
        )
        node = prog.morphisms["relax"]
        assert isinstance(node, ContractExpr)
        assert node.semiring_name == "trop"
        assert node.adjoint is True

    def test_contract_alongside_regular_morphisms(self):
        prog = parse_program(
            'algebra real(plus=add, times=multiply, zero=0.0, one=1.0)\n'
            'let f = id\n'
            'let matmul = contract[real]("ij,jk->ik")\n'
            'let g = copy'
        )
        assert "f" in prog.morphisms
        assert "matmul" in prog.morphisms
        assert "g" in prog.morphisms
        assert isinstance(prog.morphisms["matmul"], ContractExpr)
