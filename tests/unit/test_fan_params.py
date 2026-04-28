"""Fan and dynamic parameter tests: list-based fan (unbounded arity) + dynamic hyperparameters."""

import numpy as np
import pytest
from scipy.special import expit

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    PathSpec, FanSpec,
)
from unialg.terms import tensor_coder
from unialg.assembly.graph import build_graph, assemble_graph, rebind_params
from unialg.assembly.compositions import PathComposition, FanComposition
from unialg.assembly._equation_resolution import resolve_equation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
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


# ===========================================================================
# Part A: List-based fan with unbounded arity
# ===========================================================================

class TestListFanArity:
    """Verify fan works with >3 branches via list-merge."""

    def test_four_branch_fan(self, cx, real_sr, hidden, backend, coder):
        """4-branch fan: relu * tanh * sigmoid * identity (Hadamard fold)."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_sig = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        eq_ident = Equation("ident", None, hidden, hidden, nonlinearity="abs")
        eq_merge = Equation("merge4", "i,i->i", hidden, hidden, real_sr)

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh, eq_sig, eq_ident, eq_merge], backend,
            specs=[FanSpec("quad", ["relu", "tanh", "sigmoid", "ident"], ["merge4"], hidden, hidden)],
        )

        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.quad"), x_enc)
        ))

        # Numpy oracle: fold of binary multiply over [relu(x), tanh(x), sigmoid(x), abs(x)]
        branches = [np.maximum(0, x), np.tanh(x), expit(x), np.abs(x)]
        expected = branches[0]
        for b in branches[1:]:
            expected = expected * b
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_eight_branch_fan(self, cx, real_sr, hidden, backend, coder):
        """8-branch fan simulating multi-head attention merge."""
        # All branches are the same equation (relu) with different names
        eqs = []
        for i in range(8):
            eqs.append(Equation(f"head{i}", None, hidden, hidden, nonlinearity="relu"))
        eq_merge = Equation("merge8", "i,i->i", hidden, hidden, real_sr)

        graph, *_ = assemble_graph(
            eqs + [eq_merge], backend,
            specs=[FanSpec("mha", [f"head{i}" for i in range(8)], ["merge8"], hidden, hidden)],
        )

        x = np.array([0.5, 1.0, -1.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.mha"), x_enc)
        ))

        # Numpy oracle: relu(x)^8 (fold of multiply over 8 identical relu branches)
        expected = np.maximum(0, x) ** 8
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_single_branch_list_fan(self, cx, real_sr, hidden, backend, coder):
        """Single branch fan with binary merge — merge gets 1-element list."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_merge = Equation("merge1", "i,i->i", hidden, hidden, real_sr)

        graph, *_ = assemble_graph(
            [eq_relu, eq_merge], backend,
            specs=[FanSpec("single", ["relu"], ["merge1"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.0, 1.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.single"), x_enc)
        ))

        # Single branch: merge([relu(x)]) = relu(x) (no fold iterations)
        np.testing.assert_allclose(out, np.maximum(0, x))


# ===========================================================================
# Part B: Dynamic hyperparameters
# ===========================================================================

class TestHyperparams:
    """Test hyperparameter declaration, injection, and rebinding.

    The mechanism: params are bound_terms (scalars). Equations that
    consume them are 2-input primitives (einsum "i,i->i" etc). The path's
    `params` dict pre-applies the scalar from bound_terms via var("ua.param.X").
    """

    def test_params_in_assemble_graph(self, cx, real_sr, hidden, backend, coder):
        """params dict creates bound_terms accessible during reduction.

        Pattern: relu → scale(alpha, x) where scale is einsum "i,->i"
        (vector times scalar broadcast). alpha comes from bound_terms.
        """
        # "scale" equation: einsum ",i->i" means scalar * vector
        # This is a 2-input einsum: input0=scalar (rank 0), input1=vector (rank 1)
        # We use a simpler approach: relu then multiply by a constant vector
        # Actually simplest: just test that bound_terms are accessible

        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh], backend,
            params={"temp": Terms.float32(1.0)},
            specs=[PathSpec("pipe", ["relu", "tanh"], hidden, hidden)],
        )

        # Verify the param is in bound_terms
        assert core.Name("ua.param.temp") in graph.bound_terms

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        # Path still works normally
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.pipe"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)))

    def test_rebind_params(self, cx, real_sr, hidden, backend, coder):
        """rebind_params changes bound_term values without re-resolving primitives.

        Test by using the param as a pre-applied weight to a 2-input equation.
        """
        # scale equation: "i,i->i" with times=multiply, so scale(w, x) = w * x
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_scale = Equation("scale", "i,i->i", hidden, hidden, real_sr)

        # Encode a "weight" vector as the param
        weight1 = np.array([2.0, 2.0, 2.0, 2.0])
        w1_enc = encode_array(coder, weight1)

        graph, *_ = assemble_graph(
            [eq_relu, eq_scale], backend,
            params={"weight": w1_enc},
            specs=[PathSpec("scaled_relu", ["relu", "scale"], hidden, hidden,
                    {"scale": [var("ua.param.weight")]})],
        )

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        # With weight=[2,2,2,2]: scale(w, relu(x)) = w * relu(x) = 2*relu(x)
        out1 = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.scaled_relu"), x_enc)
        ))
        np.testing.assert_allclose(out1, np.maximum(0, x) * 2.0)

        # Rebind to weight=[0.5, 0.5, 0.5, 0.5]
        weight2 = np.array([0.5, 0.5, 0.5, 0.5])
        w2_enc = encode_array(coder, weight2)
        graph2 = rebind_params(graph, {"weight": w2_enc})

        out2 = decode_term(coder, assert_reduce_ok(
            cx, graph2, apply(var("ua.path.scaled_relu"), x_enc)
        ))
        np.testing.assert_allclose(out2, np.maximum(0, x) * 0.5)

        # Verify they're different
        assert not np.allclose(out1, out2)

    def test_rebind_preserves_other_terms(self, cx, real_sr, hidden, backend, coder):
        """rebind_params doesn't affect other bound_terms or primitives."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_scale = Equation("scale", "i,i->i", hidden, hidden, real_sr)

        weight = np.array([3.0, 3.0, 3.0])
        w_enc = encode_array(coder, weight)

        graph, *_ = assemble_graph(
            [eq_relu, eq_scale], backend,
            params={"weight": w_enc},
            specs=[PathSpec("scaled_relu", ["relu", "scale"], hidden, hidden,
                    {"scale": [var("ua.param.weight")]})],
        )

        # Rebind weight
        weight2 = np.array([5.0, 5.0, 5.0])
        w2_enc = encode_array(coder, weight2)
        graph2 = rebind_params(graph, {"weight": w2_enc})

        # Primitives unchanged
        assert graph2.primitives == graph.primitives

        # Path bound_term still present
        assert core.Name("ua.path.scaled_relu") in graph2.bound_terms

        # Only ua.param.weight changed
        old_w = graph.bound_terms[core.Name("ua.param.weight")]
        new_w = graph2.bound_terms[core.Name("ua.param.weight")]
        assert old_w != new_w

    def test_scalar_hyperparam_via_einsum(self, cx, real_sr, hidden, backend, coder):
        """Scalar hyperparameter via broadcast einsum: ",i->i" (scalar * vector)."""
        # Define a scalar sort for the temperature-like param
        scalar_sort = Sort("scalar", real_sr)

        # Equation: multiply scalar by vector. Einsum ",i->i"
        # Input0 = scalar (rank 0), Input1 = vector (rank 1)
        eq_scale = Equation("scale", ",i->i", scalar_sort, hidden, real_sr)

        # This is a 2-input equation — resolve as normal prim2
        prim, *_ = resolve_equation(eq_scale, backend)
        assert prim.name == core.Name("ua.equation.scale")


class TestParamSlots:
    """Test param_slots: scalar hyperparameters passed to user-defined parametric ops.

    The pattern: a user extends the backend with a multi-arg nonlinearity
    (e.g. temperature-scaled softplus), and param_slots declares the scalar
    arguments that precede the tensor input in the resolved Primitive.
    """

    @pytest.fixture
    def temp_backend(self):
        """Backend extended with a temperature-scaled softplus: log(1 + exp(x/temp))."""
        b = NumpyBackend()
        # Parametric op: (tensor, temperature) -> tensor
        b.unary_ops["temp_softplus"] = lambda x, temp: np.log1p(np.exp(x / temp))
        return b

    def test_param_slots_pointwise(self, cx, real_sr, hidden, coder, temp_backend):
        """param_slots equation with a user-defined parametric nonlinearity."""
        eq = Equation("tsoftplus", None, hidden, hidden,
                      nonlinearity="temp_softplus", param_slots=("temperature",))

        prim, *_ = resolve_equation(eq, temp_backend)
        assert prim.name == core.Name("ua.equation.tsoftplus")

        # Build graph and test via reduce_term
        from hydra.dsl.prims import float32 as float32_coder
        from hydra.sources.libraries import standard_library

        primitives = dict(standard_library())
        primitives[prim.name] = prim

        graph = build_graph([], primitives=primitives)

        x = np.array([0.0, 1.0, 2.0])
        x_enc = encode_array(coder, x)

        # Apply with temperature=1.0: standard softplus
        temp_term = Terms.float32(1.0)
        result = assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.tsoftplus"), temp_term), x_enc)
        )
        out = decode_term(coder, result)
        np.testing.assert_allclose(out, np.log1p(np.exp(x / 1.0)), rtol=1e-6)

    def test_param_slots_different_temperatures(self, cx, real_sr, hidden, coder, temp_backend):
        """Different temperature values produce different outputs."""
        eq = Equation("tsoftplus", None, hidden, hidden,
                      nonlinearity="temp_softplus", param_slots=("temperature",))

        prim, *_ = resolve_equation(eq, temp_backend)
        from hydra.sources.libraries import standard_library
        primitives = dict(standard_library())
        primitives[prim.name] = prim
        graph = build_graph([], primitives=primitives)

        x = np.array([1.0, 2.0, 3.0])
        x_enc = encode_array(coder, x)

        # temp=0.5: sharper
        out_sharp = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.tsoftplus"), Terms.float32(0.5)), x_enc)
        ))
        # temp=5.0: smoother
        out_smooth = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.tsoftplus"), Terms.float32(5.0)), x_enc)
        ))

        np.testing.assert_allclose(out_sharp, np.log1p(np.exp(x / 0.5)), rtol=1e-6)
        np.testing.assert_allclose(out_smooth, np.log1p(np.exp(x / 5.0)), rtol=1e-6)
        assert not np.allclose(out_sharp, out_smooth)

    def test_param_slots_in_path_with_param(self, cx, real_sr, hidden, coder, temp_backend):
        """param_slots equation in a path, with temperature from bound_terms."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tsoftplus = Equation("tsoftplus", None, hidden, hidden,
                                nonlinearity="temp_softplus", param_slots=("temperature",))

        graph, *_ = assemble_graph(
            [eq_relu, eq_tsoftplus], temp_backend,
            params={"temperature": Terms.float32(2.0)},
            specs=[PathSpec("pipe", ["relu", "tsoftplus"], hidden, hidden,
                    {"tsoftplus": [var("ua.param.temperature")]})],
        )

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.pipe"), x_enc)
        ))

        # relu then temp_softplus with temp=2.0
        relu_x = np.maximum(0, x)
        expected = np.log1p(np.exp(relu_x / 2.0))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_param_slots_rebind_temperature(self, cx, real_sr, hidden, coder, temp_backend):
        """Rebinding temperature changes the output of param_slots equations."""
        eq_tsoftplus = Equation("tsoftplus", None, hidden, hidden,
                                nonlinearity="temp_softplus", param_slots=("temperature",))

        graph, *_ = assemble_graph(
            [eq_tsoftplus], temp_backend,
            params={"temperature": Terms.float32(1.0)},
            specs=[PathSpec("smooth", ["tsoftplus"], hidden, hidden,
                    {"tsoftplus": [var("ua.param.temperature")]})],
        )

        x = np.array([1.0, 2.0, 3.0])
        x_enc = encode_array(coder, x)

        # With temp=1.0
        out1 = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.smooth"), x_enc)
        ))

        # Rebind to temp=0.1
        graph2 = rebind_params(graph, {"temperature": Terms.float32(0.1)})
        out2 = decode_term(coder, assert_reduce_ok(
            cx, graph2, apply(var("ua.path.smooth"), x_enc)
        ))

        np.testing.assert_allclose(out1, np.log1p(np.exp(x / 1.0)), rtol=1e-6)
        np.testing.assert_allclose(out2, np.log1p(np.exp(x / 0.1)), rtol=1e-6)
        assert not np.allclose(out1, out2)


class TestHyperparamsValidation:
    """Edge cases and validation for hyperparameters."""

    def test_multiple_params(self, cx, real_sr, hidden, backend, coder):
        """Multiple params can coexist in the same graph."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")

        w1 = encode_array(coder, np.array([1.0, 1.0]))
        w2 = encode_array(coder, np.array([2.0, 2.0]))

        graph, *_ = assemble_graph(
            [eq_relu], backend,
            params={"alpha": w1, "beta": w2},
        )

        assert core.Name("ua.param.alpha") in graph.bound_terms
        assert core.Name("ua.param.beta") in graph.bound_terms

    def test_rebind_nonexistent_creates_new(self, cx, real_sr, hidden, backend, coder):
        """rebind_params can add new params that didn't exist before."""
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")

        graph, *_ = assemble_graph([eq_relu], backend)

        w = encode_array(coder, np.array([1.0]))
        graph2 = rebind_params(graph, {"new_param": w})
        assert core.Name("ua.param.new_param") in graph2.bound_terms
