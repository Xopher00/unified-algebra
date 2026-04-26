"""Composition tests: path (sequential) and fan (parallel) composition via Hydra lambda terms."""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    build_graph, assemble_graph, Equation,
    PathSpec, FanSpec,
)
from unialg.assembly.compositions import PathComposition, FanComposition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def tropical_sort(tropical_sr):
    return Sort("tropic", tropical_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    result = coder.decode(None, arr)
    assert isinstance(result, Right)
    return result.value


def decode_term(coder, term):
    result = coder.encode(None, None, term)
    assert isinstance(result, Right)
    return result.value


def assert_reduce_ok(cx, graph, term):
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


def _schema(eq_by_name, extra_sorts=()):
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)


# ---------------------------------------------------------------------------
# Path: lambda term structure
# ---------------------------------------------------------------------------

class TestPathStructure:

    def test_path_returns_name_and_lambda(self, hidden):
        name, term = PathComposition("act", ["relu"]).to_lambda()
        assert name == Name("ua.path.act")
        assert isinstance(term.value, core.Lambda)

    def test_path_name_prefix(self, hidden):
        name, _ = PathComposition("ffn", ["a", "b", "c"]).to_lambda()
        assert name.value == "ua.path.ffn"

    def test_path_empty_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one equation"):
            PathComposition("bad", [])

    def test_path_single_step(self, hidden):
        """Single-equation path should be lambda x. eq(x)."""
        _, term = PathComposition("single", ["relu"]).to_lambda()
        # The body should be an application
        body = term.value.body
        assert isinstance(body.value, core.Application)

    def test_path_two_step(self, hidden):
        """Two-equation path: lambda x. b(a(x))."""
        _, term = PathComposition("two", ["a", "b"]).to_lambda()
        body = term.value.body
        # outer: apply(var("ua.equation.b"), ...)
        assert isinstance(body.value, core.Application)
        func = body.value.function
        assert isinstance(func, core.TermVariable)
        assert func.value.value == "ua.equation.b"


# ---------------------------------------------------------------------------
# Path: validation
# ---------------------------------------------------------------------------

class TestPathValidation:

    def test_valid_path(self, real_sr, hidden):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")
        ebn = {"a": eq_a, "b": eq_b}
        PathSpec("_", ["a", "b"], hidden, hidden).validate(ebn, _schema(ebn))

    def test_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="'a' domain mismatch"):
            ebn = {"a": eq_a}
            PathSpec("_", ["a"], output_sort, hidden).validate(ebn, _schema(ebn, [output_sort]))

    def test_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="'a' codomain mismatch"):
            ebn = {"a": eq_a}
            PathSpec("_", ["a"], hidden, output_sort).validate(ebn, _schema(ebn, [output_sort]))

    def test_junction_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="Attempted to unify schema names"):
            ebn = {"a": eq_a, "b": eq_b}
            PathSpec("_", ["a", "b"], hidden, hidden).validate(ebn, _schema(ebn))

    def test_cross_semiring_path(self, real_sr, hidden, tropical_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, tropical_sort, tropical_sort, nonlinearity="relu")
        with pytest.raises(TypeError, match="Attempted to unify schema names"):
            ebn = {"a": eq_a, "b": eq_b}
            PathSpec("_", ["a", "b"], hidden, tropical_sort).validate(ebn, _schema(ebn))


# ---------------------------------------------------------------------------
# Path: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestPathReduce:

    def test_pointwise_path(self, cx, real_sr, hidden, backend, coder):
        """PathComposition("act", ["relu", "tanh"]).to_lambda() == tanh(relu(x))."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")

        prim_relu, *_ = eq_relu.resolve(backend)
        prim_tanh, *_ = eq_tanh.resolve(backend)

        p_name, p_term = PathComposition("act", ["relu", "tanh"]).to_lambda()

        graph = build_graph(
            [],
            primitives={prim_relu.name: prim_relu, prim_tanh.name: prim_tanh},
            bound_terms={p_name: p_term},
        )

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_enc = encode_array(coder, x)

        # Via path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.act"), x_enc)
        ))

        # Relative oracle: matches chained individual calls
        step1 = assert_reduce_ok(cx, graph, apply(var("ua.equation.relu"), x_enc))
        chained = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.tanh"), step1)
        ))
        np.testing.assert_allclose(out, chained)

        # Independent numpy oracle
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)))

    def test_three_step_path(self, cx, real_sr, hidden, backend, coder):
        """relu -> tanh -> sigmoid chain matches sequential individual calls."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")

        prims = {}
        for eq in [eq_relu, eq_tanh, eq_sig]:
            p, *_ = eq.resolve(backend)
            prims[p.name] = p

        p_name, p_term = PathComposition("deep", ["relu", "tanh", "sigmoid"]).to_lambda()

        graph = build_graph(
            [], primitives=prims, bound_terms={p_name: p_term}
        )

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        x_enc = encode_array(coder, x)

        # Ground truth: chain individually
        s1 = assert_reduce_ok(cx, graph, apply(var("ua.equation.relu"), x_enc))
        s2 = assert_reduce_ok(cx, graph, apply(var("ua.equation.tanh"), s1))
        expected = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.sigmoid"), s2)
        ))

        # Via path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.deep"), x_enc)
        ))
        np.testing.assert_allclose(out, expected)

    def test_single_step_path(self, cx, real_sr, hidden, backend, coder):
        """Degenerate path with one equation == calling the equation directly."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim, *_ = eq_relu.resolve(backend)

        p_name, p_term = PathComposition("just_relu", ["relu"]).to_lambda()

        graph = build_graph(
            [], primitives={prim.name: prim}, bound_terms={p_name: p_term}
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        # Direct call
        expected = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.relu"), x_enc)
        ))

        # Via path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.just_relu"), x_enc)
        ))
        np.testing.assert_allclose(out, expected)

    def test_path_weight_tying(self, cx, real_sr, hidden, backend, coder):
        """Repeated equation name in path = same primitive applied multiple times.

        PathComposition("triple", ["tanh", "tanh", "tanh"]).to_lambda() == tanh(tanh(tanh(x)))
        All three references resolve to the same Hydra primitive via name lookup.
        """
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        prim, *_ = eq_tanh.resolve(backend)

        p_name, p_term = PathComposition("triple", ["tanh", "tanh", "tanh"]).to_lambda()
        graph = build_graph(
            [], primitives={prim.name: prim}, bound_terms={p_name: p_term}
        )

        x = np.array([1.0, -0.5, 2.0])
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.triple"), encode_array(coder, x))
        ))
        np.testing.assert_allclose(out, np.tanh(np.tanh(np.tanh(x))))

    def test_path_with_params(self, cx, real_sr, hidden, backend, coder):
        """Path with pre-bound weights: linear(W, x) then relu."""
        eq_lin = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")

        prim_lin, *_ = eq_lin.resolve(backend)
        prim_relu, *_ = eq_relu.resolve(backend)

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        W_enc = encode_array(coder, W)

        p_name, p_term = PathComposition(
            "lr", ["linear", "relu"],
            params={"linear": [W_enc]},
        ).to_lambda()

        graph = build_graph(
            [],
            primitives={prim_lin.name: prim_lin, prim_relu.name: prim_relu},
            bound_terms={p_name: p_term},
        )

        x = np.array([1.0, 1.0])
        x_enc = encode_array(coder, x)

        # Via path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.lr"), x_enc)
        ))

        # Relative oracle
        s1 = assert_reduce_ok(cx, graph, apply(apply(
            var("ua.equation.linear"), W_enc), x_enc))
        chained = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.relu"), s1)
        ))
        np.testing.assert_allclose(out, chained)

        # Independent numpy oracle
        np.testing.assert_allclose(out, np.maximum(0, W @ x))


# ---------------------------------------------------------------------------
# Fan: lambda term structure
# ---------------------------------------------------------------------------

class TestFanStructure:

    def test_fan_returns_name_and_lambda(self, hidden):
        name, term = FanComposition("f", ["a", "b"], "m").to_lambda()
        assert name == Name("ua.fan.f")
        assert isinstance(term.value, core.Lambda)

    def test_fan_empty_branches_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one branch"):
            FanComposition("bad", [], "m")

    def test_fan_many_branches_allowed(self, hidden):
        """Fan arity is unbounded — list-based merge handles any branch count."""
        _, term = FanComposition("wide", ["a", "b", "c", "d", "e"], "m").to_lambda()
        # Should not raise — produces a valid lambda term
        assert term is not None


# ---------------------------------------------------------------------------
# Fan: validation
# ---------------------------------------------------------------------------

class TestFanValidation:

    def test_valid_fan(self, real_sr, hidden):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")
        eq_m = Equation("m", "ij,ij->ij", hidden, hidden, real_sr)
        ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
        FanSpec("_", ["a", "b"], "m", hidden, hidden).validate(ebn, _schema(ebn))

    def test_branch_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, output_sort, hidden, nonlinearity="tanh")
        eq_m = Equation("m", "ij,ij->ij", hidden, hidden, real_sr)
        with pytest.raises(TypeError, match="'b' domain mismatch"):
            ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
            FanSpec("_", ["a", "b"], "m", hidden, hidden).validate(ebn, _schema(ebn))

    def test_merge_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_m = Equation("m", None, hidden, output_sort, nonlinearity="relu")
        with pytest.raises(TypeError, match="'m' codomain mismatch"):
            ebn = {"a": eq_a, "m": eq_m}
            FanSpec("_", ["a"], "m", hidden, hidden).validate(ebn, _schema(ebn))


# ---------------------------------------------------------------------------
# Fan: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestFanReduce:

    def test_two_branch_fan(self, cx, real_sr, hidden, backend, coder):
        """Fan: merge([relu(x), tanh(x)]) where merge is einsum "i,i->i" (Hadamard)."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("merge", "i,i->i", hidden, hidden, real_sr)

        prims = {}
        for eq in [eq_relu, eq_tanh]:
            p, *_ = eq.resolve(backend)
            prims[p.name] = p
        # Merge is resolved as list-merge (prim1 over list<tensor>)
        merge_prim, *_ = eq_merge.resolve_as_merge(backend)
        prims[merge_prim.name] = merge_prim

        f_name, f_term = FanComposition("res", ["relu", "tanh"], "merge").to_lambda()

        graph = build_graph(
            [], primitives=prims, bound_terms={f_name: f_term}
        )

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        # Via fan
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.res"), x_enc)
        ))

        # Independent numpy oracle: "i,i->i" with times=multiply → Hadamard product
        np.testing.assert_allclose(out, np.maximum(0, x) * np.tanh(x))

    def test_single_branch_fan(self, cx, real_sr, hidden, backend, coder):
        """Degenerate fan: one branch + identity merge == branch alone."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_ident = Equation("ident", "i->i", hidden, hidden, real_sr)

        prims = {}
        p, *_ = eq_relu.resolve(backend)
        prims[p.name] = p
        # Merge is resolved as list-merge (unary: 1-element list passthrough)
        p, *_ = eq_ident.resolve_as_merge(backend)
        prims[p.name] = p

        f_name, f_term = FanComposition("single", ["relu"], "ident").to_lambda()

        graph = build_graph(
            [], primitives=prims, bound_terms={f_name: f_term}
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        # Ground truth: relu directly
        expected = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.relu"), x_enc)
        ))

        # Via fan
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.single"), x_enc)
        ))
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# assemble_graph integration
# ---------------------------------------------------------------------------

class TestAssembleWithComposition:

    def test_assemble_with_path(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with a path: verify via individual primitive calls."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh], backend,
            specs=[PathSpec("act", ["relu", "tanh"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        # Ground truth: chain primitives
        s1 = assert_reduce_ok(cx, graph, apply(var("ua.equation.relu"), x_enc))
        expected = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.tanh"), s1)
        ))

        # Via path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.act"), x_enc)
        ))
        np.testing.assert_allclose(out, expected)

    def test_assemble_with_fan(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with a fan: merge([relu(x), tanh(x)]) = Hadamard product."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("merge", "i,i->i", hidden, hidden, real_sr)

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh, eq_merge], backend,
            specs=[FanSpec("res", ["relu", "tanh"], "merge", hidden, hidden)],
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        # Via fan
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.res"), x_enc)
        ))

        # Numpy oracle: "i,i->i" with real semiring times=multiply → Hadamard
        np.testing.assert_allclose(out, np.maximum(0, x) * np.tanh(x))

    def test_assemble_mixed(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with both a path and a fan in the same graph."""
        from scipy.special import expit

        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        eq_merge = Equation("merge", "i,i->i", hidden, hidden, real_sr)

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh, eq_sig, eq_merge], backend,
            specs=[
                PathSpec("act", ["relu", "tanh"], hidden, hidden),
                FanSpec("split", ["relu", "sigmoid"], "merge", hidden, hidden),
            ],
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        # Path: tanh(relu(x))
        path_out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.act"), x_enc)
        ))
        np.testing.assert_allclose(path_out, np.tanh(np.maximum(0, x)))

        # Fan: relu(x) * sigmoid(x)  (Hadamard via "i,i->i")
        fan_out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.split"), x_enc)
        ))
        np.testing.assert_allclose(fan_out, np.maximum(0, x) * expit(x))


# ---------------------------------------------------------------------------
# Nesting: path containing a fan (both in bound_terms)
# ---------------------------------------------------------------------------

class TestNesting:

    def test_path_of_fan_and_equation(self, cx, real_sr, hidden, backend, coder):
        """Nested composition: sigmoid(FanComposition(x)) via transitive bound_terms resolution.

        FanComposition("split", [relu, tanh], merge)  then  sigmoid
        Tests that reduce_term resolves bound_terms transitively.
        """
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        eq_merge = Equation("merge", "i,i->i", hidden, hidden, real_sr)

        prims = {}
        for eq in [eq_relu, eq_tanh, eq_sig]:
            p, *_ = eq.resolve(backend)
            prims[p.name] = p
        # Merge resolved as list-merge for fan compatibility
        p, *_ = eq_merge.resolve_as_merge(backend)
        prims[p.name] = p

        f_name, f_term = FanComposition("split", ["relu", "tanh"], "merge").to_lambda()

        # lambda x. sigmoid(split(x))
        import hydra.dsl.terms as Terms
        nested_body = Terms.apply(
            Terms.var("ua.equation.sigmoid"),
            Terms.apply(Terms.var("ua.fan.split"), Terms.var("x")),
        )
        nested_term = Terms.lambda_("x", nested_body)
        nested_name = Name("ua.path.nested")

        graph = build_graph(
            [],
            primitives=prims,
            bound_terms={f_name: f_term, nested_name: nested_term},
        )

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        # Ground truth: run fan individually, then sigmoid
        fan_out = assert_reduce_ok(cx, graph, apply(var("ua.fan.split"), x_enc))
        expected = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.sigmoid"), fan_out)
        ))

        # Via nested path
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.nested"), x_enc)
        ))
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Council fixes: order sensitivity, 3-branch fan, negative integration,
# branch-codomain-to-merge-domain validation
# ---------------------------------------------------------------------------

class TestOrderSensitivity:

    def test_path_order_matters(self, cx, real_sr, hidden, backend, coder):
        """sigmoid then relu != relu then sigmoid on negative inputs.

        sigmoid(-0.5) = 0.378, relu(0.378) = 0.378
        relu(-0.5) = 0.0, sigmoid(0.0) = 0.5
        These are different, so swapping order must produce different results.
        """
        from scipy.special import expit

        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")

        prims = {}
        for eq in [eq_relu, eq_sig]:
            p, *_ = eq.resolve(backend)
            prims[p.name] = p

        # Path A: sigmoid then relu
        pa_name, pa_term = PathComposition("sig_relu", ["sigmoid", "relu"]).to_lambda()
        # Path B: relu then sigmoid
        pb_name, pb_term = PathComposition("relu_sig", ["relu", "sigmoid"]).to_lambda()

        graph = build_graph(
            [], primitives=prims,
            bound_terms={pa_name: pa_term, pb_name: pb_term},
        )

        x = np.array([-2.0, -0.5, 0.0, 0.5])
        x_enc = encode_array(coder, x)

        out_a = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.sig_relu"), x_enc)
        ))
        out_b = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.relu_sig"), x_enc)
        ))

        # They must differ (order matters)
        assert not np.allclose(out_a, out_b), (
            f"Paths with different order produced identical results: {out_a}"
        )

        # Independent numpy oracles for each
        np.testing.assert_allclose(out_a, np.maximum(0, expit(x)))
        np.testing.assert_allclose(out_b, expit(np.maximum(0, x)))


class TestThreeBranchFan:

    def test_three_branch_fan_end_to_end(self, cx, real_sr, hidden, backend, coder):
        """3-branch fan at the prim3 boundary: merge3(relu(x), tanh(x), sigmoid(x))."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        # Binary merge via "i,i->i" — folded over 3 branches (triple Hadamard)
        eq_merge3 = Equation("merge3", "i,i->i", hidden, hidden, real_sr)

        prims = {}
        for eq in [eq_relu, eq_tanh, eq_sig]:
            p, *_ = eq.resolve(backend)
            prims[p.name] = p
        # Merge resolved as list-merge
        p, *_ = eq_merge3.resolve_as_merge(backend)
        prims[p.name] = p

        f_name, f_term = FanComposition(
            "triple", ["relu", "tanh", "sigmoid"], "merge3"
        ).to_lambda()

        graph = build_graph(
            [], primitives=prims, bound_terms={f_name: f_term}
        )

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        # Via fan
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.triple"), x_enc)
        ))

        # Independent numpy oracle: triple Hadamard product (fold of binary multiply)
        from scipy.special import expit
        np.testing.assert_allclose(
            out, np.maximum(0, x) * np.tanh(x) * expit(x), rtol=1e-6
        )


class TestNegativeIntegration:

    def test_assemble_rejects_invalid_path(self, real_sr, hidden, output_sort, backend):
        """assemble_graph raises TypeError when path has sort junction mismatch."""
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")

        with pytest.raises(TypeError):
            assemble_graph(
                [eq_a, eq_b], backend,
                specs=[PathSpec("bad", ["a", "b"], hidden, hidden)],
            )

    def test_assemble_rejects_invalid_fan(self, real_sr, hidden, output_sort, backend):
        """assemble_graph raises TypeError when fan branch codomain mismatches merge domain."""
        # Pipeline order [a, m, b] passes linear validation (all hidden→hidden junctions)
        # but fan validation catches branch 'b' codomain (output_sort) != merge domain (hidden)
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, output_sort, nonlinearity="tanh")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'b' codomain != merge domain"):
            assemble_graph(
                [eq_a, eq_m, eq_b], backend,
                specs=[FanSpec("bad", ["a", "b"], "m", hidden, hidden)],
            )


class TestBranchCodomainValidation:

    def test_branch_codomain_must_match_merge_domain(self, real_sr, hidden, output_sort):
        """validate_fan catches when branch codomains don't match merge's domain."""
        # Branch 'a' outputs to output_sort, but merge expects hidden as input
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'a' codomain != merge domain"):
            ebn = {"a": eq_a, "m": eq_m}
            FanSpec("_", ["a"], "m", hidden, hidden).validate(ebn, _schema(ebn))

    def test_mixed_branch_codomains_rejected(self, real_sr, hidden, output_sort):
        """Two branches with different codomains — first mismatch is caught."""
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, output_sort, nonlinearity="tanh")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'b' codomain != merge domain"):
            ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
            FanSpec("_", ["a", "b"], "m", hidden, hidden).validate(ebn, _schema(ebn))
