"""Update-rule tests that survive strict semiring validation.

The DSL originally exercised "update rules" (SGD, Bellman, max-min closure,
additive merges) by constructing fake `Semiring` instances whose `times` was
a custom binary op (e.g. `sgd_step`) — relying on the contraction engine to
treat the fake semiring's `times` as element-wise application. Those tests
were deleted because `Semiring.resolve()` now validates the semiring axioms
on instantiation; a fake semiring fails law checks and is rejected.

What remains here are the genuine semiring-based iteration patterns:
unfolding a contractive decay, and rebinding hyperparams in a scaled update.
"""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    Equation,
    assemble_graph, rebind_hyperparams,
    PathSpec, UnfoldSpec,
)


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
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


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
# Iteration via unfold (semiring-valid; uses real semiring + nonlinearity)
# ===========================================================================

class TestIteratedUpdate:

    def test_unfold_contractive_decay(self, cx, backend, coder):
        """Unfold a contractive map N times: x → 0.9 * x converges toward 0."""
        backend.unary_ops["decay"] = lambda x: 0.9 * x
        real_sr = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = Sort("state", real_sr)

        eq_decay = Equation("decay", None, state_sort, state_sort, nonlinearity="decay")

        graph, *_ = assemble_graph(
            [eq_decay], backend,
            specs=[UnfoldSpec("converge", "decay", 10, state_sort, state_sort)],
        )

        x0 = np.array([10.0, 5.0, 1.0])
        x0_enc = encode_array(coder, x0)

        out_term = assert_reduce_ok(
            cx, graph, apply(var("ua.unfold.converge"), x0_enc)
        )
        last = out_term.value[-1]
        final = decode_term(coder, last)
        np.testing.assert_allclose(final, x0 * 0.9**10, rtol=1e-6)


# ===========================================================================
# Dynamic hyperparameters in update loops
# ===========================================================================

class TestHyperparamsInUpdate:
    """Learning rate and other hyperparams compose with update rules."""

    def test_rebind_learning_rate(self, cx, real_sr, hidden, backend, coder):
        """rebind_hyperparams changes the effective learning rate in a scaled update.

        Path: scale(lr_vector, x) where lr_vector is a hyperparam.
        Two different lr_vectors produce different scaling.
        """
        eq_scale = Equation("lr_scale", "i,i->i", hidden, hidden, real_sr)

        lr1 = encode_array(coder, np.array([0.1, 0.1, 0.1]))
        lr2 = encode_array(coder, np.array([0.01, 0.01, 0.01]))

        graph, *_ = assemble_graph(
            [eq_scale], backend,
            hyperparams={"lr": lr1},
            specs=[PathSpec("scaled", ["lr_scale"], hidden, hidden,
                            {"lr_scale": [var("ua.param.lr")]})],
        )

        grad = np.array([5.0, 10.0, 2.0])
        grad_enc = encode_array(coder, grad)

        out1 = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.scaled"), grad_enc)
        ))
        np.testing.assert_allclose(out1, 0.1 * grad)

        graph2 = rebind_hyperparams(graph, {"lr": lr2})
        out2 = decode_term(coder, assert_reduce_ok(
            cx, graph2, apply(var("ua.path.scaled"), grad_enc)
        ))
        np.testing.assert_allclose(out2, 0.01 * grad)

        assert not np.allclose(out1, out2)

    def test_param_slots_scaled_update(self, cx, backend, coder):
        """param_slots for a temperature-scaled update: user-defined parametric op."""
        backend.unary_ops["scaled_decay"] = lambda x, rate: rate * x
        real_sr = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        state_sort = Sort("state", real_sr)

        eq = Equation("scaled_decay", None, state_sort, state_sort,
                      nonlinearity="scaled_decay", param_slots=("rate",))

        graph, *_ = assemble_graph(
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
