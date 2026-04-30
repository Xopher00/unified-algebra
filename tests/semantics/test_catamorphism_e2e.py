"""Catamorphism end-to-end tests: fold and unfold with reduce_term dispatch."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Uses legacy seq/branch/parallel/scan/unroll/fixpoint/lens grammar — migrate to cell-DSL (operator syntax) or reconstruct via Python API."
)
import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.terms import apply, var, list_

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    PathSpec, FoldSpec, UnfoldSpec,
)
from unialg.assembly.graph import assemble_graph
from unialg.assembly.legacy.compositions import FoldComposition, UnfoldComposition
from unialg.assembly.legacy._primitives import unfold_n_primitive
from unialg.assembly._equation_resolution import resolve_equation

from conftest import encode_array, decode_term, assert_reduce_ok, make_graph_with_stdlib


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


# ---------------------------------------------------------------------------
# Fold: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestFoldReduce:

    def test_fold_hadamard(self, cx, real_sr, hidden, backend, coder):
        """Fold a list of vectors with Hadamard product step.

        step(state, element) = state * element  (einsum "i,i->i")
        init = [1, 1, 1]
        FoldComposition([x1, x2, x3]) = x1 * x2 * x3 (elementwise)
        """
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step, *_ = resolve_equation(eq_step, backend)

        init = np.ones(3)
        init_term = encode_array(coder, init)

        f_name, f_term = FoldComposition("prod", "step", init_term).to_lambda()

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
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step, *_ = resolve_equation(eq_step, backend)

        init = np.ones(2)
        init_term = encode_array(coder, init)
        f_name, f_term = FoldComposition("single", "step", init_term).to_lambda()

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
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        prim_step, *_ = resolve_equation(eq_step, backend)

        init = np.array([1.0, 1.0, 1.0])
        init_term = encode_array(coder, init)
        f_name, f_term = FoldComposition("tied", "step", init_term).to_lambda()

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
# Unfold: end-to-end with reduce_term
# ---------------------------------------------------------------------------

class TestUnfoldReduce:

    def test_unfold_tanh(self, cx, real_sr, hidden, backend, coder):
        """Unfold tanh for 3 steps: s0 → tanh(s0) → tanh(tanh(s0)) → ..."""
        eq_step = Equation("step", None, hidden, hidden, nonlinearity="tanh")
        prim_step, *_ = resolve_equation(eq_step, backend)

        unfold_prim = unfold_n_primitive
        u_name, u_term = UnfoldComposition("stream", "step", 3).to_lambda()

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
        eq_step = Equation("step", None, hidden, hidden, nonlinearity="relu")
        prim_step, *_ = resolve_equation(eq_step, backend)

        unfold_prim = unfold_n_primitive
        u_name, u_term = UnfoldComposition("one", "step", 1).to_lambda()

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
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        init = encode_array(coder, np.ones(2))

        graph, *_ = assemble_graph(
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
        eq_step = Equation("step", None, hidden, hidden, nonlinearity="tanh")

        graph, *_ = assemble_graph(
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
        eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_step = Equation("step", "i,i->i", hidden, hidden, real_sr)
        init = encode_array(coder, np.ones(3))

        graph, *_ = assemble_graph(
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


class TestUnfoldFromParser:

    def test_unfold_produces_sequence(self):
        """unroll with steps=3 applies the step op 3 times, returning a tuple of states."""
        backend = NumpyBackend()
        backend.unary_ops["double"] = lambda x: x * 2.0
        from unialg import parse_ua
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op double : hidden -> hidden
  nonlinearity = double

unroll stream : hidden -> hidden
  step = double
  steps = 3
''', backend)
        x = np.array([1.0, 2.0])
        result = prog('stream', x)
        # Unfold returns a tuple: (step(x), step²(x), step³(x)) = (2x, 4x, 8x)
        assert len(result) == 3
        np.testing.assert_allclose(result[0], x * 2.0)
        np.testing.assert_allclose(result[1], x * 4.0)
        np.testing.assert_allclose(result[2], x * 8.0)

    def test_unfold_single_step(self):
        """unroll with steps=1 returns a 1-tuple containing one application of the step op."""
        backend = NumpyBackend()
        from unialg import parse_ua
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op relu_step : hidden -> hidden
  nonlinearity = relu

unroll one_step : hidden -> hidden
  step = relu_step
  steps = 1
''', backend)
        x = np.array([-1.0, 0.0, 2.0])
        result = prog('one_step', x)
        assert len(result) == 1
        np.testing.assert_allclose(result[0], np.maximum(0, x))


class TestFoldFromParser:

    def test_fold_default_init(self):
        """scan with no init= uses 0.0 default."""
        from unialg import parse_ua, NumpyBackend
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op step : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

scan acc : hidden -> hidden
  step = step
''', NumpyBackend())
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        result = prog('acc', x)
        np.testing.assert_allclose(result, np.zeros(2))

    def test_fold_explicit_init(self):
        """scan with init=1.0 starts accumulation from 1."""
        from unialg import parse_ua, NumpyBackend
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op step : hidden -> hidden
  einsum = "i,i->i"
  algebra = real

scan prod : hidden -> hidden
  step = step
  init = 1.0
''', NumpyBackend())
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        result = prog('prod', x)
        np.testing.assert_allclose(result, np.array([8.0, 15.0]))
