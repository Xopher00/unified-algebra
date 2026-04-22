"""Phase 7 tests: fold (catamorphism) and unfold (anamorphism) via Hydra primitives."""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var, list_
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, semiring, sort, tensor_coder,
    equation, resolve_equation, fold, unfold,
    validate_spec, build_graph, assemble_graph,
    PathSpec, FoldSpec, UnfoldSpec,
)
from unialg.assembly.primitives import unfold_n_primitive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def real_sr():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return sort("output", real_sr)


@pytest.fixture
def coder():
    return tensor_coder()


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


def make_graph_with_stdlib(primitives=None, bound_terms=None):
    """Build a graph with Hydra standard library primitives included."""
    from hydra.sources.libraries import standard_library
    all_prims = dict(standard_library())
    if primitives:
        all_prims.update(primitives)
    return build_graph([], primitives=all_prims, bound_terms=bound_terms)


# ---------------------------------------------------------------------------
# Fold: structure
# ---------------------------------------------------------------------------

class TestFoldStructure:

    def test_fold_returns_name_and_lambda(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, term = fold("rnn", "step", init)
        assert name == Name("ua.fold.rnn")
        assert isinstance(term.value, core.Lambda)

    def test_fold_name_prefix(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, _ = fold("test", "step", init)
        assert name.value == "ua.fold.test"


# ---------------------------------------------------------------------------
# Fold: validation
# ---------------------------------------------------------------------------

class TestFoldValidation:

    def test_valid_fold(self, real_sr, hidden):
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        validate_spec({"step": eq_step}, FoldSpec("_", "step", None, hidden, hidden))

    def test_fold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            validate_spec({}, FoldSpec("_", "missing", None, hidden, hidden))

    def test_fold_state_recurrence_mismatch(self, real_sr, hidden, output_sort):
        eq_step = equation("step", "i,i->i", hidden, output_sort, real_sr)
        with pytest.raises(TypeError, match="codomain.*state sort"):
            validate_spec({"step": eq_step}, FoldSpec("_", "step", None, hidden, hidden))


# ---------------------------------------------------------------------------
# Fold: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestFoldReduce:

    def test_fold_hadamard(self, cx, real_sr, hidden, backend, coder):
        """Fold a list of vectors with Hadamard product step.

        step(state, element) = state * element  (einsum "i,i->i")
        init = [1, 1, 1]
        fold([x1, x2, x3]) = x1 * x2 * x3 (elementwise)
        """
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step = resolve_equation(eq_step, backend)

        init = np.ones(3)
        init_term = encode_array(coder, init)

        f_name, f_term = fold("prod", "step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        x1 = np.array([2.0, 3.0, 4.0])
        x2 = np.array([0.5, 2.0, 1.0])
        x3 = np.array([3.0, 0.1, 2.0])

        seq = list_([encode_array(coder, x) for x in [x1, x2, x3]])

        # Via fold
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.prod"), seq)
        ))

        # Relative oracle: manual loop through reduce_term
        state = init_term
        for x in [x1, x2, x3]:
            x_enc = encode_array(coder, x)
            state = assert_reduce_ok(
                cx, graph,
                apply(apply(var("ua.equation.step"), state), x_enc)
            )
        expected = decode_term(coder, state)
        np.testing.assert_allclose(out, expected)

        # Independent numpy oracle
        np.testing.assert_allclose(out, x1 * x2 * x3)

    def test_fold_single_element(self, cx, real_sr, hidden, backend, coder):
        """Fold over length-1 list = single step application."""
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step = resolve_equation(eq_step, backend)

        init = np.ones(2)
        init_term = encode_array(coder, init)
        f_name, f_term = fold("single", "step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        x = np.array([3.0, 5.0])
        seq = list_([encode_array(coder, x)])

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.single"), seq)
        ))

        # Single step: init * x = [1,1] * [3,5] = [3,5]
        np.testing.assert_allclose(out, init * x)

    def test_fold_weight_tying(self, cx, real_sr, hidden, backend, coder):
        """Verify weight tying: same step function at every iteration.

        If weights were different at each step, the result would differ
        from a manual loop using the same step.
        """
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step = resolve_equation(eq_step, backend)

        init = np.array([1.0, 1.0, 1.0])
        init_term = encode_array(coder, init)
        f_name, f_term = fold("tied", "step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        vectors = [np.array([2.0, 0.5, 3.0]), np.array([0.5, 4.0, 0.1])]
        seq = list_([encode_array(coder, v) for v in vectors])

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.tied"), seq)
        ))

        # Manual numpy loop with same operation
        state = init.copy()
        for v in vectors:
            state = state * v  # same Hadamard product at each step
        np.testing.assert_allclose(out, state)


# ---------------------------------------------------------------------------
# Unfold: structure
# ---------------------------------------------------------------------------

class TestUnfoldStructure:

    def test_unfold_returns_name_and_lambda(self, hidden):
        name, term = unfold("stream", "step", 3)
        assert name == Name("ua.unfold.stream")
        assert isinstance(term.value, core.Lambda)


# ---------------------------------------------------------------------------
# Unfold: validation
# ---------------------------------------------------------------------------

class TestUnfoldValidation:

    def test_valid_unfold(self, real_sr, hidden):
        eq_step = equation("step", None, hidden, hidden, nonlinearity="tanh")
        validate_spec({"step": eq_step}, UnfoldSpec("_", "step", 0, hidden, hidden))

    def test_unfold_step_not_found(self, hidden):
        with pytest.raises(ValueError, match="not found"):
            validate_spec({}, UnfoldSpec("_", "missing", 0, hidden, hidden))

    def test_unfold_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = equation("step", None, output_sort, hidden, nonlinearity="tanh")
        with pytest.raises(TypeError, match="domain.*state sort"):
            validate_spec({"step": eq_step}, UnfoldSpec("_", "step", 0, hidden, hidden))

    def test_unfold_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_step = equation("step", None, hidden, output_sort, nonlinearity="tanh")
        with pytest.raises(TypeError, match="codomain.*state sort"):
            validate_spec({"step": eq_step}, UnfoldSpec("_", "step", 0, hidden, hidden))


# ---------------------------------------------------------------------------
# Unfold: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestUnfoldReduce:

    def test_unfold_tanh(self, cx, real_sr, hidden, backend, coder):
        """Unfold tanh for 3 steps: s0 → tanh(s0) → tanh(tanh(s0)) → ..."""
        eq_step = equation("step", None, hidden, hidden, nonlinearity="tanh")
        prim_step = resolve_equation(eq_step, backend)

        unfold_prim = unfold_n_primitive
        u_name, u_term = unfold("stream", "step", 3)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step, unfold_prim.name: unfold_prim},
            bound_terms={u_name: u_term},
        )

        s0 = np.array([1.0, -0.5, 2.0])
        s0_enc = encode_array(coder, s0)

        # Via unfold — returns a TermList
        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.stream"), s0_enc)
        )
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == 3

        # Relative oracle: manual loop through reduce_term
        state = s0_enc
        expected = []
        for _ in range(3):
            state = assert_reduce_ok(
                cx, graph, apply(var("ua.equation.step"), state)
            )
            expected.append(decode_term(coder, state))

        for out, exp in zip(outputs, expected):
            np.testing.assert_allclose(out, exp)

        # Independent numpy oracle
        s = s0.copy()
        for i in range(3):
            s = np.tanh(s)
            np.testing.assert_allclose(outputs[i], s)

    def test_unfold_single_step(self, cx, real_sr, hidden, backend, coder):
        """Unfold with n_steps=1."""
        eq_step = equation("step", None, hidden, hidden, nonlinearity="relu")
        prim_step = resolve_equation(eq_step, backend)

        unfold_prim = unfold_n_primitive
        u_name, u_term = unfold("one", "step", 1)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step, unfold_prim.name: unfold_prim},
            bound_terms={u_name: u_term},
        )

        s0 = np.array([-1.0, 0.0, 1.0])
        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.one"), encode_array(coder, s0))
        )
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == 1
        np.testing.assert_allclose(outputs[0], np.maximum(0, s0))


# ---------------------------------------------------------------------------
# assemble_graph integration
# ---------------------------------------------------------------------------

class TestAssembleWithRecursion:

    def test_assemble_with_fold(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with folds parameter produces a working graph."""
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        init = encode_array(coder, np.ones(2))

        graph = assemble_graph(
            [eq_step], backend,
            specs=[FoldSpec("prod", "step", init, hidden, hidden)],
        )

        x1 = np.array([2.0, 3.0])
        x2 = np.array([4.0, 5.0])
        seq = list_([encode_array(coder, x) for x in [x1, x2]])

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.prod"), seq)
        ))
        np.testing.assert_allclose(out, x1 * x2)

    def test_assemble_with_unfold(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with unfolds parameter produces a working graph."""
        eq_step = equation("step", None, hidden, hidden, nonlinearity="tanh")

        graph = assemble_graph(
            [eq_step], backend,
            specs=[UnfoldSpec("stream", "step", 2, hidden, hidden)],
        )

        s0 = np.array([1.0, -1.0])
        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.stream"), encode_array(coder, s0))
        )
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == 2

        np.testing.assert_allclose(outputs[0], np.tanh(s0))
        np.testing.assert_allclose(outputs[1], np.tanh(np.tanh(s0)))

    def test_assemble_mixed(self, cx, real_sr, hidden, backend, coder):
        """Equations + paths + folds coexisting in the same graph."""
        eq_relu = equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_step = equation("step", "i,i->i", hidden, hidden, real_sr)
        init = encode_array(coder, np.ones(3))

        graph = assemble_graph(
            [eq_relu, eq_tanh, eq_step], backend,
            specs=[
                PathSpec("act", ["relu", "tanh"], hidden, hidden),
                FoldSpec("prod", "step", init, hidden, hidden),
            ],
        )

        # Test path still works
        x = np.array([-1.0, 0.0, 1.0])
        path_out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.act"), encode_array(coder, x))
        ))
        np.testing.assert_allclose(path_out, np.tanh(np.maximum(0, x)))

        # Test fold works
        v1 = np.array([2.0, 3.0, 4.0])
        v2 = np.array([0.5, 0.5, 0.5])
        seq = list_([encode_array(coder, v) for v in [v1, v2]])
        fold_out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.prod"), seq)
        ))
        np.testing.assert_allclose(fold_out, v1 * v2)
