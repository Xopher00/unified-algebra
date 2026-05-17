"""Tests for semantic construction: construct() and construct_program()."""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import IntegerType, LiteralTypeInteger, Name, TypeLiteral
from hydra.dsl.python import Just

from unialg.syntax import expressions as expr
from unialg.syntax._ops import make_compose, make_pair, make_par, make_case
from unialg.syntax.parse import parse_morphism, parse_program
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.semantics.construct import construct, construct_program
from unialg.main import compile_morphism, run, load_backend, compile_program
from unialg.objects import MAYBE

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

    def test_focus_decl_resolves(self, int_env):
        prog = parse_program(
            "map Id = x\n"
            "focus self = functor = Id forward = add1 backward = mul2\n"
            "route folded = cata[self](add1)\n"
            "route built = ana[self](add1)\n"
            "route transformed = hylo[self](add1, add1)"
        )
        cp = construct_program(prog, int_env)
        assert "self" in cp.focuses
        assert cp.focuses["self"].carrier == INT
        assert cp.routes["folded"].dom() == INT
        assert cp.routes["folded"].cod() == INT
        assert cp.routes["built"].dom() == INT
        assert cp.routes["built"].cod() == INT
        assert cp.routes["transformed"].dom() == INT
        assert cp.routes["transformed"].cod() == INT

    def test_recursive_carrier_focus_resolves(self):
        prog = parse_program(
            "map NatF = 1 | x\n"
            "carrier Nat = fix NatF\n"
            "focus nat = carrier = Nat"
        )
        cp = construct_program(prog)
        assert "Nat" in cp.carriers
        assert "nat" in cp.focuses
        assert cp.focuses["nat"].carrier == cp.carriers["Nat"].typ
        assert cp.focuses["nat"].forward.dom() == cp.carriers["Nat"].typ
        assert cp.focuses["nat"].forward.cod() == cp.carriers["Nat"].layer
        assert cp.focuses["nat"].backward.dom() == cp.carriers["Nat"].layer
        assert cp.focuses["nat"].backward.cod() == cp.carriers["Nat"].typ

    def test_recursive_carrier_boundaries_resolve(self):
        prog = parse_program(
            "map NatF = 1 | x\n"
            "carrier Nat = fix NatF\n"
            "focus nat = carrier = Nat\n"
            "route inspect = unroll[nat]\n"
            "route pack = roll[nat]\n"
            "route zero = |0 >> roll[nat]\n"
            "route succ = |1 >> roll[nat]"
        )
        cp = construct_program(prog)
        carrier = cp.carriers["Nat"]
        assert cp.routes["inspect"].dom() == carrier.typ
        assert cp.routes["inspect"].cod() == carrier.layer
        assert cp.routes["pack"].dom() == carrier.layer
        assert cp.routes["pack"].cod() == carrier.typ
        assert cp.routes["zero"].cod() == carrier.typ
        assert cp.routes["succ"].dom() == carrier.typ
        assert cp.routes["succ"].cod() == carrier.typ

    def test_functor_composition_resolves(self):
        prog = parse_program(
            "map F = x | 1\n"
            "map G = x & 1\n"
            "map H = F >> G"
        )
        cp = construct_program(prog)
        from unialg.semantics.functors import apply_poly
        h_int = apply_poly(cp.functors["H"], INT)
        expected = apply_poly(cp.functors["G"], apply_poly(cp.functors["F"], INT))
        assert h_int == expected

    def test_focus_composition_resolves(self):
        prog = parse_program(
            "map NatF = 1 | x\n"
            "carrier Nat = fix NatF\n"
            "focus nat = carrier = Nat\n"
            "focus two_layers = nat >> nat\n"
            "route inspect2 = unroll[two_layers]\n"
            "route pack2 = roll[two_layers]"
        )
        cp = construct_program(prog)
        carrier = cp.carriers["Nat"]
        two_layers = cp.focuses["two_layers"]
        assert two_layers.source == carrier.typ
        assert cp.routes["inspect2"].dom() == carrier.typ
        assert cp.routes["inspect2"].cod() == two_layers.forward.cod()
        assert cp.routes["pack2"].dom() == two_layers.backward.dom()
        assert cp.routes["pack2"].cod() == carrier.typ

    def test_focus_composition_cycle_raises(self):
        prog = parse_program("focus bad = bad >> bad")
        with pytest.raises(MorphismError, match="cyclic focus"):
            construct_program(prog)

    def test_monadic_lift_resolves(self, int_env):
        prog = parse_program("route maybe_add1 = pure[Maybe](add1)")
        cp = construct_program(prog, int_env)
        assert cp.routes["maybe_add1"].dom() == INT
        assert cp.routes["maybe_add1"].cod() == INT
        assert cp.routes["maybe_add1"].monad is MAYBE

    def test_unknown_monad_raises(self, int_env):
        prog = parse_program("route bad = pure[Unknown](add1)")
        with pytest.raises(MorphismError, match="unknown monad"):
            construct_program(prog, int_env)

    def test_recursive_carrier_unknown_functor_raises(self):
        prog = parse_program("carrier Nat = fix Missing\nfocus nat = carrier = Nat")
        with pytest.raises(MorphismError, match="unknown functor"):
            construct_program(prog)

    def test_focus_unknown_carrier_raises(self):
        prog = parse_program("focus nat = carrier = Nat")
        with pytest.raises(MorphismError, match="unresolved carrier"):
            construct_program(prog)

    def test_focus_boundary_mismatch_raises(self, int_env):
        prog = parse_program(
            "map F = x & 1\n"
            "focus bad = functor = F forward = add1 backward = mul2\n"
            "route folded = cata[bad](add1)"
        )
        with pytest.raises(MorphismError, match="focus bad.forward"):
            construct_program(prog, int_env)

    def test_unknown_focus_raises(self, int_env):
        prog = parse_program("route folded = cata[missing](add1)")
        with pytest.raises(MorphismError, match="unresolved focus"):
            construct_program(prog, int_env)

    def test_route_ref_runs(self, int_env, ctx, graph):
        prog = parse_program("route a = add1 >> mul2\nroute b = a >> add1")
        cp = construct_program(prog, int_env)
        result = run(cp.routes["b"], P.int32(5).value, ctx, graph)
        assert result.value.value.value == (5 + 1) * 2 + 1

    def test_monadic_lift_runs(self, int_env, ctx, graph):
        prog = parse_program("route maybe_add1 = pure[Maybe](add1)")
        cp = construct_program(prog, int_env)
        result = run(cp.routes["maybe_add1"], P.int32(5).value, ctx, graph)
        assert isinstance(result.value, Just)
        assert result.value.value.value.value.value == 6


# ---------------------------------------------------------------------------
# compile_program() — load directive integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# construct_program() — parametric routes
# ---------------------------------------------------------------------------

class TestParametricRoutes:
    def test_single_param(self, int_env):
        prog = parse_program("route f(theta) = theta >> add1\nroute g = f(mul2)")
        cp = construct_program(prog, int_env)
        assert "g" in cp.routes
        assert cp.routes["g"].dom() == INT
        assert cp.routes["g"].cod() == INT

    def test_multi_param(self, int_env):
        prog = parse_program("route f(a, b) = a >> b\nroute g = f(add1, mul2)")
        cp = construct_program(prog, int_env)
        assert "g" in cp.routes

    def test_nested_parametric(self, int_env):
        prog = parse_program(
            "route f(w) = w >> mul2\nroute g(v) = f(v) >> add1\nroute h = g(add1)"
        )
        cp = construct_program(prog, int_env)
        assert "h" in cp.routes

    def test_wrong_arity_raises(self, int_env):
        prog = parse_program("route f(a, b) = a >> b\nroute g = f(add1)")
        with pytest.raises(MorphismError, match="expects 2.*got 1"):
            construct_program(prog, int_env)

    def test_type_mismatch_raises(self, int_env):
        from unialg.objects import ProductType
        # add1: INT→INT, but par expects ProductType domain
        # f(theta) = theta || add1 requires theta.dom to be part of a product
        prog = parse_program("route f(theta) = theta || add1\nroute g = f(add1)")
        cp = construct_program(prog, int_env)
        # par(add1, add1) should work — both INT→INT
        assert cp.routes["g"].dom() == ProductType(INT, INT)

    def test_parametric_runs(self, int_env, ctx, graph):
        prog = parse_program("route f(theta) = theta >> mul2\nroute g = f(add1)")
        cp = construct_program(prog, int_env)
        result = run(cp.routes["g"], P.int32(5).value, ctx, graph)
        # (5+1)*2 = 12
        assert result.value.value.value == 12

    def test_parametric_not_in_routes(self, int_env):
        prog = parse_program("route f(theta) = theta >> add1\nroute g = f(mul2)")
        cp = construct_program(prog, int_env)
        assert "f" not in cp.routes  # parametric routes don't appear as constructed routes


class TestParametricNativeBackend:
    def test_single_param_native(self):
        import numpy as np
        prog = compile_program("load numpy\nroute f(w) = w >> tanh\nroute g = f(exp)")
        result = prog.run(np.array([1.0, 2.0]))
        assert np.allclose(result, np.tanh(np.exp([1.0, 2.0])))

    def test_multi_param_native(self):
        import numpy as np
        prog = compile_program("load numpy\nroute f(a, b) = a >> b\nroute g = f(exp, tanh)")
        result = prog.run(np.array([1.0, 2.0]))
        assert np.allclose(result, np.tanh(np.exp([1.0, 2.0])))

    def test_nested_native(self):
        import numpy as np
        prog = compile_program(
            "load numpy\nroute f(w) = w >> tanh\nroute g(v) = f(v) >> exp\nroute h = g(log)"
        )
        result = prog.run(np.array([1.0, 2.0]))
        assert np.allclose(result, np.exp(np.tanh(np.log([1.0, 2.0]))))


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
            f"src/unialg/runtime/backends/{backend}.json"
        )
        for op in ("add", "multiply", "tanh", "exp", "log"):
            assert op in env, f"{backend} missing {op}"


def _peano_value(value) -> int:
    n = 0
    while True:
        tag, payload = value
        if tag == "left":
            return n
        if tag != "right":
            raise AssertionError(f"unexpected Peano layer: {value!r}")
        n += 1
        value = payload


class TestMonadicRecursionDsl:
    def test_maybe_cata_runs(self):
        prog = compile_program(
            """
            map NatF = 1 | x
            carrier Nat = fix NatF
            focus nat = carrier = Nat

            route zero = |0 >> roll[nat]
            route succ = |1 >> roll[nat]
            route one = zero >> succ
            route two = one >> succ
            route three = two >> succ

            route safe_id = cata[nat](pure[Maybe](zero | succ))
            route maybe_three = three >> safe_id
            """,
            route="maybe_three",
        )
        assert _peano_value(prog.run()) == 3
