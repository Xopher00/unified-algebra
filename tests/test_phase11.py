"""Phase 11 tests: update rules — closing iteration loops.

An update rule is a morphism P × P' → P: given current state and backward
feedback, produce new state. This is general — it covers gradient descent
(real semiring), Bellman relaxation (tropical), Galois closure (max-min), etc.

These tests demonstrate that update rules are ALREADY expressible with
existing DSL primitives. No new infrastructure was added in this phase:
  - 2-input equations ("i,i->i") serve as update morphisms
  - Lenses provide bidirectional forward/backward pairing
  - fold() iterates the update over a dataset
  - unfold() iterates the update for fixed steps
  - rebind_hyperparams() adjusts learning rates between iterations
  - Custom backend ops enable any user-defined update function

The same update structure (lens + equation + iteration) works across
semirings — only the operations change.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, semiring, sort, tensor_coder, sort_coder,
    equation, resolve_equation,
    build_graph, assemble_graph, rebind_hyperparams,
    lens, lens_path, fold, unfold,
    PathSpec, FoldSpec, UnfoldSpec, LensPathSpec,
)
from unialg.backend import BinaryOp, UnaryOp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def real_sr():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


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
# Group 1: Update equations — the morphism P × P' → P
# ===========================================================================

class TestUpdateEquations:
    """Update rules expressed as 2-input equations with custom semirings."""

    def test_gradient_descent_update(self, cx, backend, coder):
        """SGD update: w_new = w - 0.1 * grad via custom binary op as semiring times."""
        # Custom binary op: sgd_step(w, g) = w - 0.1 * g
        backend.binary_ops["sgd_step"] = BinaryOp(
            elementwise=lambda w, g: w - 0.1 * g,
            reduce=lambda arr, axis: np.sum(arr, axis=axis),
        )
        sgd_sr = semiring("sgd", plus="add", times="sgd_step", zero=0.0, one=1.0)
        param_sort = sort("param", sgd_sr)

        # "i,i->i" with sgd semiring: no reduction indices → elementwise ⊗ = sgd_step
        eq_update = equation("sgd_update", "i,i->i", param_sort, param_sort, sgd_sr)

        graph = assemble_graph([eq_update], backend)

        w = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.5, 1.0, 0.2])
        w_enc = encode_array(coder, w)
        grad_enc = encode_array(coder, grad)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.sgd_update"), w_enc), grad_enc)
        ))
        # w - 0.1 * grad = [0.95, 1.9, 2.98]
        np.testing.assert_allclose(out, w - 0.1 * grad)

    def test_bellman_relaxation_update(self, cx, backend, coder):
        """Bellman relaxation: update(dist, new_dist) = min(dist, new_dist)."""
        # Semiring where times=minimum: "i,i->i" → elementwise min
        bellman_sr = semiring("bellman", plus="minimum", times="minimum",
                             zero=float("inf"), one=float("inf"))
        dist_sort = sort("dist", bellman_sr)

        eq_relax = equation("bellman", "i,i->i", dist_sort, dist_sort, bellman_sr)
        graph = assemble_graph([eq_relax], backend)

        current = np.array([5.0, 3.0, 7.0, 1.0])
        candidate = np.array([4.0, 8.0, 2.0, 1.0])
        cur_enc = encode_array(coder, current)
        cand_enc = encode_array(coder, candidate)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.bellman"), cur_enc), cand_enc)
        ))
        np.testing.assert_allclose(out, np.minimum(current, candidate))

    def test_maxmin_closure_update(self, cx, backend, coder):
        """Max-min closure: update(current, candidate) = max(current, candidate)."""
        maxmin_sr = semiring("maxmin", plus="maximum", times="maximum",
                            zero=0.0, one=1.0)
        cap_sort = sort("capacity", maxmin_sr)

        eq_close = equation("closure", "i,i->i", cap_sort, cap_sort, maxmin_sr)
        graph = assemble_graph([eq_close], backend)

        current = np.array([0.3, 0.7, 0.1])
        candidate = np.array([0.5, 0.2, 0.8])
        cur_enc = encode_array(coder, current)
        cand_enc = encode_array(coder, candidate)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(apply(var("ua.equation.closure"), cur_enc), cand_enc)
        ))
        np.testing.assert_allclose(out, np.maximum(current, candidate))


# ===========================================================================
# Group 2: Lens backward → update composition
# ===========================================================================

class TestLensUpdateComposition:
    """Forward pass, backward pass, then update — all via reduce_term."""

    def test_forward_backward_update_real(self, cx, real_sr, hidden, backend, coder):
        """Compose forward (relu) → backward (tanh) → SGD update on a real semiring.

        This manually threads the lens backward output into the update equation,
        demonstrating the full forward-backward-update pipeline.

        The lens equations and the update equation use the same sort (hidden/real),
        since validate_pipeline checks sort junctions across all equations.
        The update equation uses a custom binary op registered on the same backend.
        """
        # Forward and backward equations form a lens
        eq_fwd = equation("act_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("act_bwd", None, hidden, hidden, nonlinearity="tanh")
        l = lens("activation", "act_fwd", "act_bwd")

        # SGD update uses the same sort but a custom binary op
        backend.binary_ops["sgd_step"] = BinaryOp(
            elementwise=lambda w, g: w - 0.1 * g,
            reduce=lambda arr, axis: np.sum(arr, axis=axis),
        )
        sgd_sr = semiring("sgd", plus="add", times="sgd_step", zero=0.0, one=1.0)
        upd_sort = sort("hidden", sgd_sr)  # same name "hidden", different semiring
        eq_update = equation("sgd", "i,i->i", upd_sort, upd_sort, sgd_sr)

        # Build two separate graphs: one for the lens, one for the update.
        # In practice a user would wire these in a single graph if the sorts
        # are compatible; here the update uses a different semiring, so we
        # assemble them separately to avoid sort junction validation.
        lens_graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("act_pipe", ["activation"], hidden, hidden)],
        )
        update_graph = assemble_graph([eq_update], backend)

        x = np.array([-1.0, 0.5, 2.0, -0.3])
        x_enc = encode_array(coder, x)

        # Step 1: forward pass
        fwd_out = decode_term(coder, assert_reduce_ok(
            cx, lens_graph, apply(var("ua.path.act_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(fwd_out, np.maximum(0, x))

        # Step 2: backward pass (using forward output as input to backward)
        fwd_enc = encode_array(coder, fwd_out)
        bwd_out = decode_term(coder, assert_reduce_ok(
            cx, lens_graph, apply(var("ua.path.act_pipe.bwd"), fwd_enc)
        ))
        np.testing.assert_allclose(bwd_out, np.tanh(fwd_out))

        # Step 3: update weights using backward signal
        w = np.array([1.0, 1.0, 1.0, 1.0])
        w_enc = encode_array(coder, w)
        bwd_enc = encode_array(coder, bwd_out)

        w_new = decode_term(coder, assert_reduce_ok(
            cx, update_graph, apply(apply(var("ua.equation.sgd"), w_enc), bwd_enc)
        ))
        np.testing.assert_allclose(w_new, w - 0.1 * bwd_out)

    def test_forward_backward_update_tropical(self, cx, backend, coder):
        """Same pipeline with tropical semiring: forward contraction, backward recovery, Bellman update.

        The lens uses tropical semiring; the Bellman update uses a separate
        semiring (times=minimum). Assembled in separate graphs to avoid
        sort junction validation across different semirings.
        """
        tropical_sr = semiring("tropical", plus="minimum", times="add",
                               zero=float("inf"), one=0.0)
        trop_sort = sort("trop", tropical_sr)

        # Forward: identity-like "i->i" (tropical times=add, no reduction → identity)
        eq_fwd = equation("trop_fwd", "i->i", trop_sort, trop_sort, tropical_sr)
        eq_bwd = equation("trop_bwd", "i->i", trop_sort, trop_sort, tropical_sr)
        l = lens("trop_lens", "trop_fwd", "trop_bwd")

        lens_graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("trop_pipe", ["trop_lens"], trop_sort, trop_sort)],
        )

        # Bellman relaxation update (separate graph, different semiring)
        bellman_sr = semiring("bellman", plus="minimum", times="minimum",
                             zero=float("inf"), one=float("inf"))
        bell_sort = sort("bell", bellman_sr)
        eq_relax = equation("relax", "i,i->i", bell_sort, bell_sort, bellman_sr)
        update_graph = assemble_graph([eq_relax], backend)

        dist = np.array([5.0, 3.0, 7.0])
        dist_enc = encode_array(coder, dist)

        # Forward pass
        fwd_out = decode_term(coder, assert_reduce_ok(
            cx, lens_graph, apply(var("ua.path.trop_pipe.fwd"), dist_enc)
        ))
        np.testing.assert_allclose(fwd_out, dist)

        # Backward pass
        bwd_out = decode_term(coder, assert_reduce_ok(
            cx, lens_graph, apply(var("ua.path.trop_pipe.bwd"), dist_enc)
        ))
        np.testing.assert_allclose(bwd_out, dist)

        # Bellman update: min(current, candidate)
        new_dist = np.array([4.0, 8.0, 2.0])
        new_enc = encode_array(coder, new_dist)
        relaxed = decode_term(coder, assert_reduce_ok(
            cx, update_graph, apply(apply(var("ua.equation.relax"), dist_enc), new_enc)
        ))
        np.testing.assert_allclose(relaxed, np.minimum(dist, new_dist))


# ===========================================================================
# Group 3: Iteration via fold / unfold
# ===========================================================================

class TestIteratedUpdate:
    """Update rules iterated via fold (over data) and unfold (fixed steps)."""

    def test_fold_additive_update_over_dataset(self, cx, backend, coder):
        """Fold an additive update over a dataset: accumulate running sum.

        step(state, element) = state + element via additive semiring.
        """
        add_sr = semiring("add", plus="add", times="add", zero=0.0, one=0.0)
        acc_sort = sort("acc", add_sr)

        # 2-input equation: "i,i->i" with times=add → elementwise addition
        eq_step = equation("add_step", "i,i->i", acc_sort, acc_sort, add_sr)

        init = encode_array(coder, np.zeros(3))
        graph = assemble_graph(
            [eq_step], backend,
            specs=[FoldSpec("accumulate", "add_step", init, acc_sort, acc_sort)],
        )

        data = [
            encode_array(coder, np.array([1.0, 2.0, 3.0])),
            encode_array(coder, np.array([0.5, 0.5, 0.5])),
            encode_array(coder, np.array([2.0, 1.0, 0.5])),
        ]
        seq = Terms.list_(data)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.accumulate"), seq)
        ))
        np.testing.assert_allclose(out, np.array([3.5, 3.5, 4.0]))

    def test_unfold_contractive_decay(self, cx, backend, coder):
        """Unfold a contractive map N times: x → 0.9 * x converges toward 0.

        After 10 steps from x0, result ≈ 0.9^10 * x0.
        """
        backend.unary_ops["decay"] = UnaryOp(fn=lambda x: 0.9 * x)
        real_sr = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = sort("state", real_sr)

        eq_decay = equation("decay", None, state_sort, state_sort, nonlinearity="decay")

        graph = assemble_graph(
            [eq_decay], backend,
            specs=[UnfoldSpec("converge", "decay", 10, state_sort, state_sort)],
        )

        x0 = np.array([10.0, 5.0, 1.0])
        x0_enc = encode_array(coder, x0)

        out_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.converge"), x0_enc)
        )
        # unfold returns a list of states; the last one is the final state
        # Decode each element
        last = out_term.value[-1]
        final = decode_term(coder, last)
        np.testing.assert_allclose(final, x0 * 0.9**10, rtol=1e-6)

    def test_fold_sgd_updates_move_weights(self, cx, backend, coder):
        """Fold SGD updates over a sequence of gradients — weights converge.

        Each gradient points the same direction, so accumulated updates
        should shift weights consistently.
        """
        backend.binary_ops["sgd_fold"] = BinaryOp(
            elementwise=lambda w, g: w - 0.01 * g,
            reduce=lambda arr, axis: np.sum(arr, axis=axis),
        )
        sgd_sr = semiring("sgd_fold", plus="add", times="sgd_fold", zero=0.0, one=1.0)
        w_sort = sort("weight", sgd_sr)

        eq_step = equation("sgd_step", "i,i->i", w_sort, w_sort, sgd_sr)

        init_w = encode_array(coder, np.array([1.0, 1.0, 1.0]))
        graph = assemble_graph(
            [eq_step], backend,
            specs=[FoldSpec("train", "sgd_step", init_w, w_sort, w_sort)],
        )

        # 5 identical gradients pushing weights down
        grad = np.array([1.0, 2.0, 0.5])
        grads = [encode_array(coder, grad) for _ in range(5)]
        seq = Terms.list_(grads)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.train"), seq)
        ))
        # After 5 steps: w - 5 * 0.01 * grad = [0.95, 0.9, 0.975]
        np.testing.assert_allclose(out, np.array([1.0, 1.0, 1.0]) - 5 * 0.01 * grad)


# ===========================================================================
# Group 4: Dynamic hyperparameters in update loops
# ===========================================================================

class TestHyperparamsInUpdate:
    """Learning rate and other hyperparams compose with update rules."""

    def test_rebind_learning_rate(self, cx, real_sr, hidden, backend, coder):
        """rebind_hyperparams changes the effective learning rate in a scaled update.

        Path: scale(lr_vector, x) where lr_vector is a hyperparam.
        Two different lr_vectors produce different scaling.
        """
        eq_scale = equation("lr_scale", "i,i->i", hidden, hidden, real_sr)

        lr1 = encode_array(coder, np.array([0.1, 0.1, 0.1]))
        lr2 = encode_array(coder, np.array([0.01, 0.01, 0.01]))

        graph = assemble_graph(
            [eq_scale], backend,
            hyperparams={"lr": lr1},
            specs=[PathSpec("scaled", ["lr_scale"], hidden, hidden,
                            {"lr_scale": [var("ua.param.lr")]})],
        )

        grad = np.array([5.0, 10.0, 2.0])
        grad_enc = encode_array(coder, grad)

        # lr=0.1: scale * grad
        out1 = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.scaled"), grad_enc)
        ))
        np.testing.assert_allclose(out1, 0.1 * grad)

        # Rebind to lr=0.01
        graph2 = rebind_hyperparams(graph, {"lr": lr2})
        out2 = decode_term(coder, assert_reduce_ok(
            cx, graph2, apply(var("ua.path.scaled"), grad_enc)
        ))
        np.testing.assert_allclose(out2, 0.01 * grad)

        # Different results
        assert not np.allclose(out1, out2)

    def test_param_slots_scaled_update(self, cx, backend, coder):
        """param_slots for a temperature-scaled update: user-defined parametric op."""
        backend.unary_ops["scaled_decay"] = UnaryOp(
            fn=lambda x, rate: rate * x
        )
        real_sr = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = sort("state", real_sr)

        eq = equation("scaled_decay", None, state_sort, state_sort,
                      nonlinearity="scaled_decay", param_slots=("rate",))

        graph = assemble_graph(
            [eq], backend,
            hyperparams={"rate": Terms.float32(0.5)},
            specs=[PathSpec("decay_path", ["scaled_decay"], state_sort, state_sort,
                            {"scaled_decay": [var("ua.param.rate")]})],
        )

        x = np.array([4.0, 6.0, 8.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.decay_path"), x_enc)
        ))
        np.testing.assert_allclose(out, 0.5 * x)


# ===========================================================================
# Group 5: Semiring-polymorphic update
# ===========================================================================

class TestSemiringPolymorphicUpdate:
    """Same update structure, three semirings — the architecture doesn't change."""

    def test_same_structure_three_semirings(self, cx, backend, coder):
        """One 2-input equation pattern instantiated with three different semirings.

        "i,i->i" with no reduction indices uses ⊗ elementwise:
          - real (times=multiply): elementwise product
          - min-semiring (times=minimum): elementwise min
          - max-semiring (times=maximum): elementwise max

        Same inputs, same equation string, different semirings → different results.
        """
        state = np.array([1.0, 3.0, 2.0])
        feedback = np.array([2.0, 1.0, 4.0])
        s_enc = encode_array(coder, state)
        f_enc = encode_array(coder, feedback)

        results = {}
        for sr_name, plus_op, times_op, zero, one, expected in [
            ("real", "add", "multiply", 0.0, 1.0, state * feedback),
            ("min_sr", "minimum", "minimum", float("inf"), float("inf"),
             np.minimum(state, feedback)),
            ("max_sr", "maximum", "maximum", 0.0, 0.0,
             np.maximum(state, feedback)),
        ]:
            sr = semiring(sr_name, plus=plus_op, times=times_op, zero=zero, one=one)
            s = sort(f"s_{sr_name}", sr)
            eq = equation(f"update_{sr_name}", "i,i->i", s, s, sr)

            graph = assemble_graph([eq], backend)

            out = decode_term(coder, assert_reduce_ok(
                cx, graph,
                apply(apply(var(f"ua.equation.update_{sr_name}"), s_enc), f_enc)
            ))
            np.testing.assert_allclose(out, expected)
            results[sr_name] = out

        # All three produced different results from the same inputs
        assert not np.allclose(results["real"], results["min_sr"])
        assert not np.allclose(results["real"], results["max_sr"])
        assert not np.allclose(results["min_sr"], results["max_sr"])
