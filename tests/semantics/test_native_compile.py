"""Tests for native compile_program(...).run(...) with automatic boundary handling."""
import pytest
import numpy as np

from unialg import compile_program


class TestNativeCompileUnary:
    def test_tanh(self):
        prog = compile_program("load numpy\nroute f = tanh")
        result = prog.run(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, np.tanh([1.0, 2.0, 3.0]))

    def test_exp(self):
        prog = compile_program("load numpy\nroute f = exp")
        result = prog.run(np.array([0.0, 1.0, -1.0]))
        assert np.allclose(result, np.exp([0.0, 1.0, -1.0]))

    def test_sqrt(self):
        prog = compile_program("load numpy\nroute f = sqrt")
        result = prog.run(np.array([4.0, 9.0, 16.0]))
        assert np.allclose(result, np.sqrt([4.0, 9.0, 16.0]))


class TestNativeCompileBinary:
    def test_add(self):
        prog = compile_program("load numpy\nroute f = add")
        result = prog.run(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert np.allclose(result, [4.0, 6.0])

    def test_multiply(self):
        prog = compile_program("load numpy\nroute f = multiply")
        result = prog.run(np.array([2.0, 3.0]), np.array([4.0, 5.0]))
        assert np.allclose(result, [8.0, 15.0])

    def test_subtract(self):
        prog = compile_program("load numpy\nroute f = subtract")
        result = prog.run(np.array([10.0, 20.0]), np.array([3.0, 7.0]))
        assert np.allclose(result, [7.0, 13.0])


class TestNativeCompileComposition:
    def test_add_then_tanh(self):
        prog = compile_program("load numpy\nroute f = add >> tanh")
        result = prog.run(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert np.allclose(result, np.tanh([1.0, 1.0]))

    def test_chain_unary(self):
        prog = compile_program("load numpy\nroute f = exp >> log")
        result = prog.run(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_multiply_then_sqrt(self):
        prog = compile_program("load numpy\nroute f = multiply >> sqrt")
        result = prog.run(np.array([4.0, 9.0]), np.array([4.0, 9.0]))
        assert np.allclose(result, [4.0, 9.0])


class TestNativeCompileRouteRefs:
    def test_route_ref(self):
        prog = compile_program("load numpy\nroute base = tanh\nroute final = base >> exp")
        result = prog.run(np.array([1.0]))
        assert np.allclose(result, np.exp(np.tanh([1.0])))


class TestStoreIdentity:
    def test_no_key_error(self):
        """Verify store identity: primitives and boundary use the same store."""
        prog = compile_program("load numpy\nroute f = add")
        result = prog.run(np.array([1.0]), np.array([2.0]))
        assert np.allclose(result, [3.0])

    def test_repeated_runs(self):
        """Store resets between runs — no stale handles."""
        prog = compile_program("load numpy\nroute f = tanh")
        for i in range(5):
            result = prog.run(np.array([float(i)]))
            assert np.allclose(result, np.tanh([float(i)]))


class TestNoBackendFallback:
    def test_raw_hydra_term(self):
        """Without a backend, run() still accepts raw Hydra terms."""
        import hydra.dsl.meta.phantoms as P
        prog = compile_program("route f = id")
        result = prog.run(P.unit().value)
        assert result is not None
