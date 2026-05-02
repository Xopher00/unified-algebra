"""Update-rule tests that survive strict semiring validation.

The DSL originally exercised "update rules" (SGD, Bellman, max-min closure,
additive merges) by constructing fake Semiring instances whose times was a
custom binary op (e.g. sgd_step) -- relying on the contraction engine to
treat the fake semiring's times as element-wise application. Those tests
were deleted because Semiring.resolve() now validates the semiring axioms
on instantiation; a fake semiring fails law checks and is rejected.

What remains here are the genuine semiring-based iteration patterns:
applying a contractive decay repeatedly, and rebinding params in a scaled update.
"""

import numpy as np
import pytest

from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms

from unialg import (
    Semiring, Sort,
    Equation,
    compile_program,
)
from unialg.assembly.graph import assemble_graph, rebind_params

from conftest import encode_array, decode_term, assert_reduce_ok


# ===========================================================================
# Iteration via repeated calls (semiring-valid)
# ===========================================================================

class TestIteratedUpdate:

    def test_repeated_decay_converges(self, cx, backend, coder):
        """Applying decay N times: x -> 0.9 * x converges toward 0."""
        backend.unary_ops["decay_iter"] = lambda x: 0.9 * x
        real_sr = Semiring("real_iter", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = Sort("state_iter", real_sr)

        eq_decay = Equation("decay_iter", None, state_sort, state_sort, nonlinearity="decay_iter")
        prog = compile_program([eq_decay], backend=backend)

        x0 = np.array([10.0, 5.0, 1.0])
        x = x0.copy()
        for _ in range(10):
            x = prog("decay_iter", x)

        np.testing.assert_allclose(x, x0 * 0.9**10, rtol=1e-6)


# ===========================================================================
# Dynamic hyperparameters in update loops
# ===========================================================================

class TestHyperparamsInUpdate:
    """Learning rate and other params compose with update rules."""

    def test_rebind_learning_rate(self, cx, real_sr, hidden, backend, coder):
        """rebind_params changes the effective learning rate in a scaled update.

        scale(lr_vector, x) = lr_vector * x (element-wise multiply via semiring).
        rebind_params replaces ua.param.lr without recompiling primitives.
        """
        eq_scale = Equation("lr_scale", "i,i->i", hidden, hidden, real_sr)

        lr1 = encode_array(coder, np.array([0.1, 0.1, 0.1]))
        lr2 = encode_array(coder, np.array([0.01, 0.01, 0.01]))

        graph, *_ = assemble_graph(
            [eq_scale], backend,
            params={"lr": lr1},
        )

        grad = np.array([5.0, 10.0, 2.0])
        grad_enc = encode_array(coder, grad)

        # Apply the equation with the bound param as first arg (lr), then grad
        out1 = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.lr_scale"), var("ua.param.lr")), grad_enc)
        ))
        np.testing.assert_allclose(out1, 0.1 * grad)

        graph2 = rebind_params(graph, {"lr": lr2})
        out2 = decode_term(coder, assert_reduce_ok(
            cx, graph2,
            apply(apply(var("ua.equation.lr_scale"), var("ua.param.lr")), grad_enc)
        ))
        np.testing.assert_allclose(out2, 0.01 * grad)

        assert not np.allclose(out1, out2)

    def test_param_slots_scaled_update(self, cx, backend, coder):
        """param_slots for a temperature-scaled update: user-defined parametric op."""
        backend.unary_ops["scaled_decay"] = lambda x, rate: rate * x
        real_sr = Semiring("real_sd", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = Sort("state_sd", real_sr)

        eq = Equation("scaled_decay", None, state_sort, state_sort,
                      nonlinearity="scaled_decay", param_slots=("rate",))

        graph, *_ = assemble_graph(
            [eq], backend,
            params={"rate": Terms.float32(0.5)},
        )

        x = np.array([4.0, 6.0, 8.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.scaled_decay"), var("ua.param.rate")), x_enc)
        ))
        np.testing.assert_allclose(out, 0.5 * x)
