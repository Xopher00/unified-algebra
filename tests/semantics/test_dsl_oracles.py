"""DSL oracle tests: programs verified against numpy ground truth.

Tests the full assemble → reduce_term path. One canonical execution model.

Tests 2 and 3 use the Python morphism API directly — branch (fan+merge) and
seq+ (residual) have no grammar yet, so we wire them via compile_program.
"""

import numpy as np

from unialg import NumpyBackend, Semiring, Sort, Equation, compile_program, parse_ua


class TestDSLOracles:
    def test_seq_relu_tanh(self):
        """Sequential composition via cell DSL: output = tanh(relu(x))."""
        backend = NumpyBackend()
        text = """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op dsl_relu : hidden -> hidden
  nonlinearity = relu

op dsl_tanh : hidden -> hidden
  nonlinearity = tanh

cell dsl_layer : hidden -> hidden = dsl_relu > dsl_tanh
"""
        prog = parse_ua(text, backend)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = prog("dsl_layer", x)
        np.testing.assert_allclose(result, np.tanh(np.maximum(0.0, x)), rtol=1e-6)

    def test_hadamard_merge(self):
        """Two-input einsum contraction: hadamard(relu(x), tanh(x)) = relu(x) * tanh(x)."""
        backend = NumpyBackend()
        real = Semiring("real2", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("h2", real)

        eq_relu = Equation("relu2", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("tanh2", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("hadamard2", "i,i->i", hidden, hidden, real)

        prog = compile_program([eq_relu, eq_tanh, eq_merge], backend=backend)

        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        relu_out = prog("relu2", x)
        tanh_out = prog("tanh2", x)
        result = prog("hadamard2", relu_out, tanh_out)

        np.testing.assert_allclose(result, np.maximum(0.0, x) * np.tanh(x), rtol=1e-6)

    def test_residual_skip(self):
        """Skip connection: relu(x) + x. Relu runs through reduce_term; add is the skip."""
        backend = NumpyBackend()
        real = Semiring("real3", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("h3", real)

        eq_relu = Equation("relu3", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq_relu], backend=backend)

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        relu_out = prog("relu3", x)
        result = relu_out + x  # skip connection: seq+ produces relu(x) ⊕ x

        np.testing.assert_allclose(result, np.maximum(0.0, x) + x, rtol=1e-6)
