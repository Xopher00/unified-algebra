"""Feedforward tests: feedforward neural network expressed as a path of equations.

Demonstrates that a multi-layer feedforward network is a sequential composition
of linear morphisms and nonlinear pointwise equations, semiring-polymorphic.
"""

import numpy as np
import pytest

from hydra.core import Name

from unialg import (
    numpy_backend, semiring, sort, tensor_coder,
    equation, compile_program, PathSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()

@pytest.fixture
def real_sr():
    return semiring("real14ff", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def hidden(real_sr):
    return sort("h14ff", real_sr)

@pytest.fixture
def coder():
    return tensor_coder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    from hydra.dsl.python import Right
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


# ---------------------------------------------------------------------------
# Architecture: 3-layer feedforward (linear → relu → linear → relu → linear)
# ---------------------------------------------------------------------------

def build_ffn_equations(hidden, real_sr):
    """5 equations: 3 linear layers + 2 relu activations, wired with inputs=."""
    return [
        equation("ffn_linear1", "ij,j->i", hidden, hidden, real_sr),
        equation("ffn_relu1",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear1",)),
        equation("ffn_linear2", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu1",)),
        equation("ffn_relu2",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear2",)),
        equation("ffn_linear3", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu2",)),
    ]


class TestFeedforward:

    def test_graph_has_all_primitives(self, hidden, real_sr, backend):
        """compile_program produces primitives for all 5 equations and the path bound term."""
        eqs = build_ffn_equations(hidden, real_sr)
        prog = compile_program(
            eqs, backend=backend,
            specs=[PathSpec("ffn", ["ffn_linear1", "ffn_relu1",
                                    "ffn_linear2", "ffn_relu2",
                                    "ffn_linear3"], hidden, hidden)],
        )
        for name in ("ffn_linear1", "ffn_relu1", "ffn_linear2",
                     "ffn_relu2", "ffn_linear3"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives
        assert Name("ua.path.ffn") in prog.graph.bound_terms

    def test_path_produces_correct_output(self, hidden, real_sr, backend, coder):
        """Running the path produces W3 @ relu(W2 @ relu(W1 @ x))."""
        W1 = np.array([[1.0, -1.0], [-1.0,  1.0]])
        W2 = np.array([[0.5,  0.5], [ 0.5, -0.5]])
        W3 = np.array([[2.0,  0.0], [ 0.0,  2.0]])
        x  = np.array([3.0, 1.0])

        w1_enc = encode_array(coder, W1)
        w2_enc = encode_array(coder, W2)
        w3_enc = encode_array(coder, W3)

        eqs = build_ffn_equations(hidden, real_sr)
        prog = compile_program(
            eqs, backend=backend,
            specs=[PathSpec(
                "ffn",
                ["ffn_linear1", "ffn_relu1", "ffn_linear2", "ffn_relu2", "ffn_linear3"],
                hidden, hidden,
                params={"ffn_linear1": [w1_enc],
                        "ffn_linear2": [w2_enc],
                        "ffn_linear3": [w3_enc]},
            )],
        )

        out = prog("ffn", x)
        expected = W3 @ np.maximum(0, W2 @ np.maximum(0, W1 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_semiring_swap_tropical(self, backend):
        """Same 5-equation architecture with tropical semiring assembles without error."""
        trop = semiring("tropical14ff", plus="minimum", times="add",
                        zero=float("inf"), one=0.0)
        h = sort("h14ff_t", trop)
        eqs = [
            equation("tffn_linear1", "ij,j->i", h, h, trop),
            equation("tffn_relu1",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear1",)),
            equation("tffn_linear2", "ij,j->i", h, h, trop, inputs=("tffn_relu1",)),
            equation("tffn_relu2",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear2",)),
            equation("tffn_linear3", "ij,j->i", h, h, trop, inputs=("tffn_relu2",)),
        ]
        prog = compile_program(
            eqs, backend=backend,
            specs=[PathSpec("tffn", ["tffn_linear1", "tffn_relu1",
                                     "tffn_linear2", "tffn_relu2",
                                     "tffn_linear3"], h, h)],
        )
        assert Name("ua.path.tffn") in prog.graph.bound_terms
        for name in ("tffn_linear1", "tffn_relu1", "tffn_linear2",
                     "tffn_relu2", "tffn_linear3"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives
