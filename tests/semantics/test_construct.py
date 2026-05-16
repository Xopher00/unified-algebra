"""Tests for semantic construction: construct() and construct_program()."""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import IntegerType, LiteralTypeInteger, Name, TypeLiteral

from unialg.syntax import expressions as expr
from unialg.syntax._ops import make_compose, make_pair, make_par, make_case
from unialg.syntax.parse import parse_morphism, parse_program
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.semantics.construct import construct, construct_program
from unialg.main import compile_morphism, run, load_backend, compile_program

def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
ADD = Name("hydra.lib.math.add")
MUL = Name("hydra.lib.math.mul")


@pytest.fixture
def ctx():
    return L.empty_context()


@pytest.fixture
def graph():
    primitives = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            primitives.extend(getattr(Libs, attr)().values())
    return L.graph_with_primitives(primitives, ())


@pytest.fixture
def int_env():
    add1_raw = P.lam("x", P.primitive2(ADD, P.var("x"), P.int32(1))).value
    mul2_raw = P.lam("x", P.primitive2(MUL, P.var("x"), P.int32(2))).value
    return {
        "add1": Morphism(node=expr.Prim(add1_raw, INT, INT)),
        "mul2": Morphism(node=expr.Prim(mul2_raw, INT, INT)),
    }


# ---------------------------------------------------------------------------
# construct() — basic resolution
# ---------------------------------------------------------------------------

class TestConstructBasic:
    def test_ref_resolves_from_env(self, int_env):
        m = construct(expr.Ref("add1"), int_env)
        assert m.dom() == INT
        assert m.cod() == INT

    def test_unresolved_ref_raises(self):
        with pytest.raises(MorphismError, match="unresolved reference"):
            construct(expr.Ref("nonexistent"), {})

    def test_prim_passthrough(self, int_env):
        raw = P.lam("x", P.var("x")).value
        node = expr.Prim(raw, INT, INT)
        m = construct(node, {})
        assert m.dom() == INT

    def test_identity(self):
        node = expr.Identity(INT)
        m = construct(node, {})
        assert m.dom() == INT
        assert m.cod() == INT


# ---------------------------------------------------------------------------
# construct() — combinators
# ---------------------------------------------------------------------------

class TestConstructCombinators:
    def test_compose(self, int_env):
        tree = make_compose(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        assert m.dom() == INT
        assert m.cod() == INT

    def test_pair(self, int_env):
        tree = make_pair(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        assert m.dom() == INT

    def test_par(self, int_env):
        from unialg.objects import ProductType
        tree = make_par(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        assert m.dom() == ProductType(INT, INT)

    def test_case(self, int_env):
        from unialg.objects import SumType
        tree = make_case(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        assert m.dom() == SumType(INT, INT)
        assert m.cod() == INT

    def test_nested_compose(self, int_env):
        tree = make_compose(make_compose(expr.Ref("add1"), expr.Ref("mul2")), expr.Ref("add1"))
        m = construct(tree, int_env)
        assert m.dom() == INT
        assert m.cod() == INT


# ---------------------------------------------------------------------------
# construct() — runtime verification
# ---------------------------------------------------------------------------

class TestConstructRuntime:
    def test_compose_runs(self, int_env, ctx, graph):
        tree = make_compose(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        result = run(m, P.int32(5).value, ctx, graph)
        assert result.value.value.value == 12  # (5+1)*2

    def test_pair_runs(self, int_env, ctx, graph):
        tree = make_pair(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, int_env)
        result = run(m, P.int32(5).value, ctx, graph)
        assert result.value[0].value.value.value == 6   # 5+1
        assert result.value[1].value.value.value == 10  # 5*2

    @given(st.integers(min_value=-100, max_value=100))
    @settings(max_examples=20)
    def test_compose_add_mul_property(self, value):
        add1_raw = P.lam("x", P.primitive2(ADD, P.var("x"), P.int32(1))).value
        mul2_raw = P.lam("x", P.primitive2(MUL, P.var("x"), P.int32(2))).value
        env = {
            "add1": Morphism(node=expr.Prim(add1_raw, INT, INT)),
            "mul2": Morphism(node=expr.Prim(mul2_raw, INT, INT)),
        }
        tree = make_compose(expr.Ref("add1"), expr.Ref("mul2"))
        m = construct(tree, env)
        ctx = L.empty_context()
        primitives = []
        for attr in dir(Libs):
            if attr.startswith("register_") and attr.endswith("_primitives"):
                primitives.extend(getattr(Libs, attr)().values())
        graph = L.graph_with_primitives(primitives, ())
        result = run(m, P.int32(value).value, ctx, graph)
        assert result.value.value.value == (value + 1) * 2


# ---------------------------------------------------------------------------
# construct_program() — route resolution
# ---------------------------------------------------------------------------

class TestConstructProgram:
    def test_route_ref(self, int_env):
        prog = parse_program("route a = add1\nroute b = a >> mul2")
        cp = construct_program(prog, int_env)
        assert "b" in cp.routes
        assert cp.routes["b"].dom() == INT

    def test_forward_ref(self, int_env):
        prog = parse_program("route b = a >> mul2\nroute a = add1")
        cp = construct_program(prog, int_env)
        assert "b" in cp.routes

    def test_cycle_detection(self, int_env):
        prog = parse_program("route a = b\nroute b = a")
        with pytest.raises(MorphismError, match="[Cc]ycl"):
            construct_program(prog, int_env)

    def test_functor_map_resolves(self, int_env):
        prog = parse_program("map F = x & 1\nroute g = F{add1}")
        cp = construct_program(prog, int_env)
        assert "g" in cp.routes

    def test_route_ref_runs(self, int_env, ctx, graph):
        prog = parse_program("route a = add1 >> mul2\nroute b = a >> add1")
        cp = construct_program(prog, int_env)
        result = run(cp.routes["b"], P.int32(5).value, ctx, graph)
        assert result.value.value.value == (5 + 1) * 2 + 1


# ---------------------------------------------------------------------------
# compile_program() — load directive integration
# ---------------------------------------------------------------------------

class TestLoadDirective:
    @pytest.mark.parametrize("backend", [
        "numpy",
        pytest.param("torch", marks=pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")),
        pytest.param("jax", marks=pytest.mark.skipif(not _has_module("jax"), reason="jax not installed")),
        pytest.param("cupy", marks=pytest.mark.skipif(not _has_module("cupy"), reason="cupy not installed")),
    ])
    def test_load_backend_compiles(self, backend):
        prog = compile_program(f"load {backend}\nroute f = tanh")
        assert prog.term is not None

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            compile_program("load nonexistent\nroute f = id")

    def test_loaded_ops_compose(self):
        prog = compile_program("load numpy\nroute f = add >> tanh")
        assert prog.term is not None

    @pytest.mark.parametrize("backend", [
        "numpy",
        pytest.param("torch", marks=pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")),
        pytest.param("jax", marks=pytest.mark.skipif(not _has_module("jax"), reason="jax not installed")),
        pytest.param("cupy", marks=pytest.mark.skipif(not _has_module("cupy"), reason="cupy not installed")),
    ])
    def test_all_backends_have_common_ops(self, backend):
        env = load_backend(
            f"src/unialg/emitters/backends/{backend}.json"
        )
        for op in ("add", "multiply", "tanh", "exp", "log"):
            assert op in env, f"{backend} missing {op}"
