"""Tests for native compile_program(...).run(...) with automatic boundary handling."""
import pytest
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp, softmax as scipy_softmax

from unialg import compile_program


class TestNativeCompileUnary:
    def test_tanh(self):
        prog = compile_program("load numpy\nlet f = tanh")
        result = prog.run(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, np.tanh([1.0, 2.0, 3.0]))

    def test_exp(self):
        prog = compile_program("load numpy\nlet f = exp")
        result = prog.run(np.array([0.0, 1.0, -1.0]))
        assert np.allclose(result, np.exp([0.0, 1.0, -1.0]))

    def test_sqrt(self):
        prog = compile_program("load numpy\nlet f = sqrt")
        result = prog.run(np.array([4.0, 9.0, 16.0]))
        assert np.allclose(result, np.sqrt([4.0, 9.0, 16.0]))

    def test_softmax_accepts_axis_argument(self):
        prog = compile_program("load numpy\nlet f = softmax")
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = prog.run(x, 0)
        assert np.allclose(result, scipy_softmax(x, axis=0))

    def test_softmax_accepts_inline_axis_literal(self):
        prog = compile_program("load numpy\nlet f = softmax(x, '-1')")
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = prog.run(x)
        assert np.allclose(result, scipy_softmax(x, axis=-1))


class TestNativeCompileBinary:
    def test_add(self):
        prog = compile_program("load numpy\nlet f = add")
        result = prog.run(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert np.allclose(result, [4.0, 6.0])

    def test_multiply(self):
        prog = compile_program("load numpy\nlet f = multiply")
        result = prog.run(np.array([2.0, 3.0]), np.array([4.0, 5.0]))
        assert np.allclose(result, [8.0, 15.0])

    def test_subtract(self):
        prog = compile_program("load numpy\nlet f = subtract")
        result = prog.run(np.array([10.0, 20.0]), np.array([3.0, 7.0]))
        assert np.allclose(result, [7.0, 13.0])


class TestNativeCompileAxisReductions:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("reduce.add", lambda x: np.sum(x, axis=0)),
            ("reduce.multiply", lambda x: np.prod(x, axis=0)),
            ("reduce.minimum", lambda x: np.min(x, axis=0)),
            ("reduce.maximum", lambda x: np.max(x, axis=0)),
            ("reduce.logaddexp", lambda x: scipy_logsumexp(x, axis=0)),
        ],
    )
    def test_numpy_reductions_accept_inline_axis_literal(self, name, expected):
        prog = compile_program(f"load numpy\nlet f = {name}(x, '0')")
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(prog.run(x), expected(x))


class TestNativeCompileComposition:
    def test_add_then_tanh(self):
        prog = compile_program("load numpy\nlet f = add >> tanh")
        result = prog.run(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert np.allclose(result, np.tanh([1.0, 1.0]))

    def test_chain_unary(self):
        prog = compile_program("load numpy\nlet f = exp >> log")
        result = prog.run(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_multiply_then_sqrt(self):
        prog = compile_program("load numpy\nlet f = multiply >> sqrt")
        result = prog.run(np.array([4.0, 9.0]), np.array([4.0, 9.0]))
        assert np.allclose(result, [4.0, 9.0])


class TestNativeCompileMorphismRefs:
    def test_let_ref(self):
        prog = compile_program("load numpy\nlet base = tanh\nlet final = base >> exp")
        result = prog.run(np.array([1.0]))
        assert np.allclose(result, np.exp(np.tanh([1.0])))


class TestStoreIdentity:
    def test_no_key_error(self):
        """Verify store identity: primitives and boundary use the same store."""
        prog = compile_program("load numpy\nlet f = add")
        result = prog.run(np.array([1.0]), np.array([2.0]))
        assert np.allclose(result, [3.0])

    def test_repeated_runs(self):
        """Store resets between runs — no stale handles."""
        prog = compile_program("load numpy\nlet f = tanh")
        for i in range(5):
            result = prog.run(np.array([float(i)]))
            assert np.allclose(result, np.tanh([float(i)]))


class TestNoBackendFallback:
    def test_identity_returns_unit(self):
        """Without a backend, run() decodes structurally (Unit → None)."""
        import hydra.dsl.meta.phantoms as P
        prog = compile_program("let f = id")
        result = prog.run(P.unit().value)
        assert result is None  # Unit decodes to None
