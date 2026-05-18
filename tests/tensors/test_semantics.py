"""Tests for tensor semantics layer — semiring resolution and contract_morphism."""
import pytest
from pathlib import Path

from unialg.syntax import expressions as expr
from unialg.objects import ProductType
from unialg.runtime import BackendOps
from unialg.semantics.morphisms import Morphism
from unialg.tensors.semantics import (
    BINARY,
    resolve_semiring,
    contract_morphism,
    construct,
    construct_expr,
)
from unialg.tensors.notation import SemiringDecl, ContractExpr
from unialg.main import load_backend


SPEC = str(Path(__file__).parent.parent.parent / "src" / "unialg" / "runtime" / "backends" / "numpy.json")


@pytest.fixture
def numpy_backend():
    return BackendOps.from_spec(SPEC)


@pytest.fixture
def numpy_env(numpy_backend):
    return {
        name: Morphism(
            node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
            aux_primitives=(bp.primitive,),
        )
        for name, bp in numpy_backend.primitives.items()
    }


@pytest.fixture
def real_semiring(numpy_env):
    decl = SemiringDecl(
        name="real",
        plus="add",
        times="multiply",
        zero=0.0,
        one=1.0,
    )
    return resolve_semiring(decl, numpy_env)


class TestResolveSemiring:
    def test_basic_fields(self, real_semiring):
        sr = real_semiring
        assert sr.name == "real"
        assert sr.carrier == BINARY
        assert sr.adjoint is None

    def test_plus_type(self, real_semiring):
        sr = real_semiring
        assert sr.plus.dom() == ProductType(BINARY, BINARY)
        assert sr.plus.cod() == BINARY

    def test_times_type(self, real_semiring):
        sr = real_semiring
        assert sr.times.dom() == ProductType(BINARY, BINARY)
        assert sr.times.cod() == BINARY

    def test_reduce_ops_derived(self, real_semiring):
        sr = real_semiring
        assert sr.plus_reduce is not None
        assert sr.times_reduce is not None
        assert sr.plus_reduce.dom() == BINARY
        assert sr.plus_reduce.cod() == BINARY

    def test_zero_is_float(self, real_semiring):
        assert real_semiring.zero == 0.0

    def test_one_is_float(self, real_semiring):
        assert real_semiring.one == 1.0

    def test_unknown_op_raises(self, numpy_env):
        decl = SemiringDecl(name="bad", plus="nonexistent", times="multiply", zero=0.0, one=1.0)
        with pytest.raises(Exception, match="unknown op 'nonexistent'"):
            resolve_semiring(decl, numpy_env)

    def test_with_adjoint(self, numpy_env):
        decl = SemiringDecl(
            name="with_adj",
            plus="add",
            times="multiply",
            zero=0.0,
            one=1.0,
            adjoint="divide",
        )
        sr = resolve_semiring(decl, numpy_env)
        assert sr.adjoint is not None

    def test_inf_identity(self, numpy_env):
        decl = SemiringDecl(
            name="trop",
            plus="minimum",
            times="add",
            zero=float("inf"),
            one=0.0,
        )
        sr = resolve_semiring(decl, numpy_env)
        assert sr.zero == float("inf")

    def test_op_env_standard(self, real_semiring):
        env = real_semiring.op_env()
        assert env["product"] is real_semiring.times
        assert env["fold"] is real_semiring.plus_reduce
        assert env["seed"] is real_semiring.zero


class TestContractMorphism:
    def test_matmul_types(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij,jk->ik", context=numpy_backend)
        assert m.dom() == ProductType(BINARY, BINARY)
        assert m.cod() == BINARY

    def test_matvec_types(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij,j->i", context=numpy_backend)
        assert m.dom() == ProductType(BINARY, BINARY)
        assert m.cod() == BINARY

    def test_single_input(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij->i", context=numpy_backend)
        assert m.dom() == BINARY
        assert m.cod() == BINARY

    def test_three_inputs(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij,jk,kl->il", context=numpy_backend)
        expected = ProductType(ProductType(BINARY, BINARY), BINARY)
        assert m.dom() == expected
        assert m.cod() == BINARY

    def test_node_is_composed_substrate_not_contract_spec(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij,jk->ik", context=numpy_backend)
        assert not isinstance(m.node, expr.Prim)
        assert isinstance(m.node, expr.Compose)

    def test_adjoint_mode_requires_adjoint(self, real_semiring, numpy_backend):
        with pytest.raises(Exception, match="no adjoint"):
            contract_morphism(real_semiring, "ij,jk->ik", context=numpy_backend, adjoint=True)

    def test_adjoint_mode_with_adjoint(self, numpy_env, numpy_backend):
        decl = SemiringDecl(
            name="with_adj", plus="add", times="multiply",
            zero=0.0, one=1.0, adjoint="divide",
        )
        sr = resolve_semiring(decl, numpy_env)
        m = contract_morphism(sr, "ij,jk->ik", context=numpy_backend, adjoint=True)
        assert m.dom() == ProductType(BINARY, BINARY)
        assert m.cod() == BINARY

    def test_aux_primitives_collected(self, real_semiring, numpy_backend):
        m = contract_morphism(real_semiring, "ij,jk->ik", context=numpy_backend)
        assert len(m.aux_primitives) > 0


class TestConstructProtocol:
    def test_construct_semirings(self, numpy_env):
        decls = [
            SemiringDecl(name="real", plus="add", times="multiply", zero=0.0, one=1.0),
        ]
        result = construct(decls, numpy_env)
        assert "semirings" in result
        assert "real" in result["semirings"]
        sr = result["semirings"]["real"]
        assert sr.name == "real"

    def test_construct_expr(self, numpy_env, numpy_backend):
        decls = [
            SemiringDecl(name="real", plus="add", times="multiply", zero=0.0, one=1.0),
        ]
        domain_data = construct(decls, numpy_env)
        node = ContractExpr(semiring_name="real", equation_str="ij,jk->ik")
        env = dict(numpy_env)
        env["_domain_data"] = {"tensors": domain_data}
        env["_domain_context"] = numpy_backend
        m = construct_expr(node, env)
        assert m.dom() == ProductType(BINARY, BINARY)
        assert m.cod() == BINARY

    def test_construct_expr_unknown_semiring(self, numpy_env):
        domain_data = construct([], numpy_env)
        node = ContractExpr(semiring_name="missing", equation_str="ij,jk->ik")
        env = dict(numpy_env)
        env["_domain_data"] = {"tensors": domain_data}
        with pytest.raises(Exception, match="unknown algebra 'missing'"):
            construct_expr(node, env)
