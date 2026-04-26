"""RNN tests: RNN architecture demonstrated via fold/unfold primitives.

This module shows how to express recurrent neural networks using the unified-
algebra DSL's existing fold and unfold primitives.  No new DSL code is needed.

Two complementary demonstrations:

1. Weight-tied recurrent application via unfold:
   h_0, h_1, ..., h_n  where  h_{t+1} = tanh(h_t)
   This is the Elman "echo" RNN with no external input.

2. Elman-style fold over an input sequence:
   h_t = tanh(h_{t-1} * x_t)   (elementwise, real semiring)
   Fold step: Equation("rnn_step", "i,i->i", hidden, hidden, real_sr,
                        nonlinearity="tanh")
   This is a simplified Elman where x_t plays the role of the combined
   input-plus-weight signal.

The two patterns together cover the essential structure of any RNN:
  - unfold  = state propagation without external input (autonomous dynamics)
  - fold    = state update driven by an input sequence (supervised recurrence)

Tropical semiring test: swap real arithmetic for (min, add) to get a
shortest-path-through-time recurrence — the Viterbi recurrence in disguise.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var, list_
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    Equation,
    build_graph, assemble_graph,
    FoldSpec, UnfoldSpec,
)
from unialg.assembly.compositions import FoldComposition, UnfoldComposition
from unialg.assembly import unfold_n_primitive


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
    # (min, add) tropical semiring — shortest-path algebra
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def hidden_trop(tropical_sr):
    return Sort("hidden_trop", tropical_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_phase7.py)
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
    from hydra.sources.libraries import standard_library
    all_prims = dict(standard_library())
    if primitives:
        all_prims.update(primitives)
    return build_graph([], primitives=all_prims, bound_terms=bound_terms)


def _schema(eq_by_name, extra_sorts=()):
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)


# ---------------------------------------------------------------------------
# TestRNN
# ---------------------------------------------------------------------------

class TestRNN:

    # ------------------------------------------------------------------
    # 1. Assembly: the step equation and fold/unfold spec assemble cleanly
    # ------------------------------------------------------------------

    def test_rnn_step_assembles(self, real_sr, hidden, backend, coder):
        """A 2-input fold step and a 1-input unfold step both resolve without error."""
        # Fold step: (h_prev, x_t) -> tanh(h_prev * x_t)
        eq_fold_step = Equation(
            "rnn_fold_step", "i,i->i", hidden, hidden, real_sr,
            nonlinearity="tanh",
        )
        prim_fold, *_ = eq_fold_step.resolve(backend)
        assert prim_fold.name == core.Name("ua.equation.rnn_fold_step")

        # Unfold step: h -> tanh(h)
        eq_unfold_step = Equation(
            "rnn_unfold_step", None, hidden, hidden,
            nonlinearity="tanh",
        )
        prim_unfold, *_ = eq_unfold_step.resolve(backend)
        assert prim_unfold.name == core.Name("ua.equation.rnn_unfold_step")

        # Fold and unfold lambdas build without error
        init = encode_array(coder, np.zeros(4))
        f_name, f_term = FoldComposition.build("rnn", "rnn_fold_step", init)
        assert f_name == core.Name("ua.fold.rnn")
        assert isinstance(f_term.value, core.Lambda)

        u_name, u_term = UnfoldComposition.build("rnn_echo", "rnn_unfold_step", 3)
        assert u_name == core.Name("ua.unfold.rnn_echo")
        assert isinstance(u_term.value, core.Lambda)

    def test_rnn_step_validate_fold_spec(self, real_sr, hidden):
        """FoldSpec validation passes for a correctly typed RNN step."""
        eq_step = Equation(
            "fold_step_v", "i,i->i", hidden, hidden, real_sr,
            nonlinearity="tanh",
        )
        # Should not raise
        ebn = {"fold_step_v": eq_step}
        FoldSpec("rnn_v", "fold_step_v", None, hidden, hidden).validate(ebn, _schema(ebn))

    def test_rnn_step_validate_unfold_spec(self, real_sr, hidden):
        """UnfoldSpec validation passes for a pure-nonlinearity RNN step."""
        eq_step = Equation(
            "unfold_step_v", None, hidden, hidden,
            nonlinearity="tanh",
        )
        ebn = {"unfold_step_v": eq_step}
        UnfoldSpec("rnn_echo_v", "unfold_step_v", 3, hidden, hidden).validate(ebn, _schema(ebn))

    # ------------------------------------------------------------------
    # 2. Unfold produces a sequence of states
    # ------------------------------------------------------------------

    def test_rnn_produces_sequence_of_states(self, cx, real_sr, hidden, backend, coder):
        """Unfold with n=3 produces 3 states h_1, h_2, h_3 where h_t = tanh(h_{t-1}).

        This is the autonomous Elman RNN: no external input, just iterated
        nonlinear state propagation.  The result is a TermList of length 3.
        """
        eq_step = Equation(
            "echo_step", None, hidden, hidden,
            nonlinearity="tanh",
        )
        prim_step, *_ = eq_step.resolve(backend)

        u_name, u_term = UnfoldComposition.build("echo3", "echo_step", 3)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step, unfold_n_primitive.name: unfold_n_primitive},
            bound_terms={u_name: u_term},
        )

        h0 = np.array([0.5, -1.0, 2.0, 0.0])
        h0_enc = encode_array(coder, h0)

        result_term = assert_reduce_ok(cx, graph, apply(var("ua.unfold.echo3"), h0_enc))

        assert isinstance(result_term, core.TermList), \
            f"Expected TermList, got {type(result_term)}"
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == 3, f"Expected 3 states, got {len(outputs)}"

        # Independent numpy oracle: h_t = tanh^t(h0)
        h = h0.copy()
        for i, out in enumerate(outputs):
            h = np.tanh(h)
            np.testing.assert_allclose(out, h, rtol=1e-6,
                err_msg=f"State at step {i+1} does not match tanh^{i+1}(h0)")

    def test_rnn_fold_driven_by_input_sequence(self, cx, real_sr, hidden, backend, coder):
        """Fold-based RNN: h_t = tanh(h_{t-1} * x_t) over an input sequence.

        The fold step (state, element) -> tanh(state * element) models
        a simplified Elman update where x_t is the pre-transformed input.

        With h_0 = [1, 1, 1] and inputs x_1, x_2, x_3, the expected
        output is tanh(tanh(tanh(h0 * x1) * x2) * x3).
        """
        eq_step = Equation(
            "elman_step", "i,i->i", hidden, hidden, real_sr,
            nonlinearity="tanh",
        )
        prim_step, *_ = eq_step.resolve(backend)

        h0 = np.ones(3)
        init_term = encode_array(coder, h0)

        f_name, f_term = FoldComposition.build("elman", "elman_step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        x1 = np.array([0.5, -0.3,  1.2])
        x2 = np.array([1.0,  2.0, -0.5])
        x3 = np.array([0.8, -1.0,  0.3])

        seq = list_([encode_array(coder, x) for x in [x1, x2, x3]])
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.elman"), seq)
        ))

        # Relative oracle: step through reduce_term manually
        state = init_term
        for x in [x1, x2, x3]:
            x_enc = encode_array(coder, x)
            state = assert_reduce_ok(
                cx, graph,
                apply(apply(var("ua.equation.elman_step"), state), x_enc),
            )
        expected = decode_term(coder, state)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

        # Independent numpy oracle
        h = h0.copy()
        for x in [x1, x2, x3]:
            h = np.tanh(h * x)
        np.testing.assert_allclose(out, h, rtol=1e-6)

    # ------------------------------------------------------------------
    # 3. Weight tying
    # ------------------------------------------------------------------

    def test_weight_tying(self, cx, real_sr, hidden, backend, coder):
        """The same step function is applied at every unfold iteration.

        If weight tying were broken, step i would differ from step j.
        We verify by comparing unfold output to a manual loop using the
        same (single) step primitive at every iteration.
        """
        eq_step = Equation(
            "tied_step", None, hidden, hidden,
            nonlinearity="relu",
        )
        prim_step, *_ = eq_step.resolve(backend)

        n_steps = 4
        u_name, u_term = UnfoldComposition.build("tied_rnn", "tied_step", n_steps)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step, unfold_n_primitive.name: unfold_n_primitive},
            bound_terms={u_name: u_term},
        )

        h0 = np.array([-2.0, 0.5, 1.0, -0.1])
        h0_enc = encode_array(coder, h0)

        result_term = assert_reduce_ok(cx, graph, apply(var("ua.unfold.tied_rnn"), h0_enc))
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == n_steps

        # Manual loop through the same step primitive via reduce_term
        state = h0_enc
        expected = []
        for _ in range(n_steps):
            state = assert_reduce_ok(cx, graph, apply(var("ua.equation.tied_step"), state))
            expected.append(decode_term(coder, state))

        for i, (out, exp) in enumerate(zip(outputs, expected)):
            np.testing.assert_allclose(out, exp, rtol=1e-6,
                err_msg=f"Step {i}: unfold output differs from manual loop (weight tying broken?)")

        # Also verify against numpy oracle (relu^n)
        h = h0.copy()
        for i in range(n_steps):
            h = np.maximum(0, h)
            np.testing.assert_allclose(outputs[i], h, rtol=1e-6)

    def test_weight_tying_fold(self, cx, real_sr, hidden, backend, coder):
        """Weight tying holds for fold: the same step is applied every iteration.

        Verified by comparing fold output to a manual loop using the same
        step primitive, and independently to a numpy reference.
        """
        eq_step = Equation(
            "tied_fold_step", "i,i->i", hidden, hidden, real_sr,
        )
        prim_step, *_ = eq_step.resolve(backend)

        init = np.array([1.0, 1.0, 1.0])
        init_term = encode_array(coder, init)
        f_name, f_term = FoldComposition.build("tied_fold", "tied_fold_step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        vectors = [
            np.array([2.0, 0.5, 3.0]),
            np.array([0.5, 4.0, 0.1]),
            np.array([1.5, 2.0, 0.2]),
        ]
        seq = list_([encode_array(coder, v) for v in vectors])

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.tied_fold"), seq)
        ))

        # Manual loop oracle
        state = init_term
        for v in vectors:
            v_enc = encode_array(coder, v)
            state = assert_reduce_ok(
                cx, graph,
                apply(apply(var("ua.equation.tied_fold_step"), state), v_enc),
            )
        expected = decode_term(coder, state)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

        # Numpy oracle: elementwise product (real ⊗ = multiply, reduce = multiply)
        h = init.copy()
        for v in vectors:
            h = h * v
        np.testing.assert_allclose(out, h, rtol=1e-6)

    # ------------------------------------------------------------------
    # 4. Tropical semiring RNN
    # ------------------------------------------------------------------

    def test_tropical_rnn_unfold(self, cx, tropical_sr, hidden_trop, backend, coder):
        """Tropical unfold: h_t = tanh(h_{t-1}) under (min, add) algebra.

        The tropical semiring changes the contraction semantics, but a pure
        nonlinearity step is semiring-independent.  This test verifies that
        the same unfold pattern works without modification when the sort's
        semiring changes.
        """
        eq_step = Equation(
            "trop_echo_step", None, hidden_trop, hidden_trop,
            nonlinearity="tanh",
        )
        prim_step, *_ = eq_step.resolve(backend)

        n_steps = 3
        u_name, u_term = UnfoldComposition.build("trop_echo", "trop_echo_step", n_steps)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step, unfold_n_primitive.name: unfold_n_primitive},
            bound_terms={u_name: u_term},
        )

        h0 = np.array([0.3, -0.7, 1.5])
        h0_enc = encode_array(coder, h0)

        result_term = assert_reduce_ok(cx, graph, apply(var("ua.unfold.trop_echo"), h0_enc))
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == n_steps

        # Numpy oracle: tanh^t(h0) — same as real case since step is pure nonlinearity
        h = h0.copy()
        for i, out in enumerate(outputs):
            h = np.tanh(h)
            np.testing.assert_allclose(out, h, rtol=1e-6,
                err_msg=f"Tropical unfold: step {i+1} mismatch")

    def test_tropical_rnn_fold(self, cx, tropical_sr, hidden_trop, backend, coder):
        """Tropical fold: h_t = min(h_{t-1} + x_t) — shortest path through time.

        Under the (min, add) tropical Semiring, the fold step "i,i->i"
        computes h_t[i] = h_{t-1}[i] + x_t[i] (add = ⊗, min = ⊕ for reduction).

        For a 1D einsum "i,i->i" with tropical semiring:
          - ⊗ = add (elementwise)
          - The einsum aligns and combines: result[i] = h[i] + x[i]
        This gives a min-plus recurrence accumulating path costs over time.
        """
        eq_step = Equation(
            "trop_step", "i,i->i", hidden_trop, hidden_trop, tropical_sr,
        )
        prim_step, *_ = eq_step.resolve(backend)

        h0 = np.array([0.0, 0.0, 0.0])
        init_term = encode_array(coder, h0)

        f_name, f_term = FoldComposition.build("trop_rnn", "trop_step", init_term)

        graph = make_graph_with_stdlib(
            primitives={prim_step.name: prim_step},
            bound_terms={f_name: f_term},
        )

        # Each x_t is an "edge cost" vector at time t
        x1 = np.array([1.0, 2.0, 0.5])
        x2 = np.array([0.3, 1.0, 2.0])
        x3 = np.array([0.5, 0.5, 0.5])

        seq = list_([encode_array(coder, x) for x in [x1, x2, x3]])
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.trop_rnn"), seq)
        ))

        # Relative oracle via manual reduce_term loop
        state = init_term
        for x in [x1, x2, x3]:
            x_enc = encode_array(coder, x)
            state = assert_reduce_ok(
                cx, graph,
                apply(apply(var("ua.equation.trop_step"), state), x_enc),
            )
        expected = decode_term(coder, state)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

        # Independent numpy oracle: tropical "i,i->i" = elementwise add (⊗)
        # No reduction axes, so contraction is pure elementwise ⊗ = np.add
        h = h0.copy()
        for x in [x1, x2, x3]:
            h = h + x   # tropical ⊗ = add
        np.testing.assert_allclose(out, h, rtol=1e-6)

    def test_tropical_rnn_semiring_polymorphism(self, cx, real_sr, tropical_sr, hidden, hidden_trop, backend, coder):
        """The same fold pattern works for both real and tropical semirings.

        Demonstrates semiring polymorphism: swapping the semiring changes the
        contraction semantics without touching the fold structure or step arity.
        """
        # Real fold: h_t = h_{t-1} * x_t
        eq_real = Equation("poly_real_step", "i,i->i", hidden, hidden, real_sr)
        prim_real, *_ = eq_real.resolve(backend)

        h0_real = np.array([1.0, 1.0, 1.0])
        init_real = encode_array(coder, h0_real)
        r_name, r_term = FoldComposition.build("poly_real", "poly_real_step", init_real)

        # Tropical fold: h_t = h_{t-1} + x_t
        eq_trop = Equation("poly_trop_step", "i,i->i", hidden_trop, hidden_trop, tropical_sr)
        prim_trop, *_ = eq_trop.resolve(backend)

        h0_trop = np.array([0.0, 0.0, 0.0])
        init_trop = encode_array(coder, h0_trop)
        t_name, t_term = FoldComposition.build("poly_trop", "poly_trop_step", init_trop)

        graph = make_graph_with_stdlib(
            primitives={
                prim_real.name: prim_real,
                prim_trop.name: prim_trop,
            },
            bound_terms={r_name: r_term, t_name: t_term},
        )

        x1 = np.array([2.0, 3.0, 4.0])
        x2 = np.array([0.5, 0.5, 0.5])
        seq = list_([encode_array(coder, x) for x in [x1, x2]])

        # Real result: 1 * 2 * 0.5 = 1, 1 * 3 * 0.5 = 1.5, 1 * 4 * 0.5 = 2
        out_real = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.poly_real"), seq)
        ))
        np.testing.assert_allclose(out_real, h0_real * x1 * x2, rtol=1e-6)

        # Tropical result: 0 + 2 + 0.5 = 2.5, 0 + 3 + 0.5 = 3.5, 0 + 4 + 0.5 = 4.5
        out_trop = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.poly_trop"), seq)
        ))
        np.testing.assert_allclose(out_trop, h0_trop + x1 + x2, rtol=1e-6)

        # The two results must differ (different algebras, same structure)
        assert not np.allclose(out_real, out_trop), \
            "Real and tropical results should differ — semiring is not being respected"

    # ------------------------------------------------------------------
    # 5. assemble_graph integration
    # ------------------------------------------------------------------

    def test_rnn_assemble_graph(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with FoldSpec produces a working Elman RNN graph."""
        eq_step = Equation(
            "asm_elman_step", "i,i->i", hidden, hidden, real_sr,
            nonlinearity="tanh",
        )
        h0 = encode_array(coder, np.ones(3))

        graph, *_ = assemble_graph(
            [eq_step], backend,
            specs=[FoldSpec("asm_elman", "asm_elman_step", h0, hidden, hidden)],
        )

        x1 = np.array([0.5, -1.0,  0.3])
        x2 = np.array([0.8,  0.2, -0.5])
        seq = list_([encode_array(coder, x) for x in [x1, x2]])

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.asm_elman"), seq)
        ))

        # Numpy oracle
        h = np.ones(3)
        h = np.tanh(h * x1)
        h = np.tanh(h * x2)
        np.testing.assert_allclose(out, h, rtol=1e-6)

    def test_rnn_unfold_assemble_graph(self, cx, real_sr, hidden, backend, coder):
        """assemble_graph with UnfoldSpec produces a working echo-state graph."""
        eq_step = Equation(
            "asm_echo_step", None, hidden, hidden,
            nonlinearity="tanh",
        )

        graph, *_ = assemble_graph(
            [eq_step], backend,
            specs=[UnfoldSpec("asm_echo", "asm_echo_step", 3, hidden, hidden)],
        )

        h0 = np.array([1.0, -0.5, 0.0])
        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.asm_echo"), encode_array(coder, h0))
        )
        assert isinstance(result_term, core.TermList)
        outputs = [decode_term(coder, t) for t in result_term.value]
        assert len(outputs) == 3

        h = h0.copy()
        for i, out in enumerate(outputs):
            h = np.tanh(h)
            np.testing.assert_allclose(out, h, rtol=1e-6)
