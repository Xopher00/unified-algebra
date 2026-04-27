"""Unit tests for Program: rebind and entry_points."""

import numpy as np
import pytest

from unialg import (
    compile_program, Program,
    Semiring, Sort, Equation, NumpyBackend, tensor_coder,
    PathSpec, FanSpec, FoldSpec,
)
from unialg.assembly.compositions import PathComposition, FanComposition


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
def hidden(real_sr):
    return Sort("hidden", real_sr)

@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Test 4: rebind returns a fresh Program
# ---------------------------------------------------------------------------

class TestRebind:

    def test_rebind_returns_new_program(self, hidden, real_sr, backend, coder):
        """rebind() returns a new Program; original is not mutated."""
        import hydra.core as core
        from hydra.dsl.terms import var

        eq_relu = Equation("t4_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("t4_tanh", None, hidden, hidden, nonlinearity="tanh")

        # params creates a ua.param.dummy bound_term (not used in path,
        # just verifying the rebind machinery round-trips)
        dummy_val = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        prog = compile_program(
            [eq_relu, eq_tanh], backend=backend,
            specs=[PathSpec("t4_path", ["t4_relu", "t4_tanh"], hidden, hidden)],
            params={"dummy": dummy_val},
        )

        new_val = core.TermLiteral(value=core.LiteralFloat(value=2.0))
        prog2 = prog.rebind(dummy=new_val)

        # rebind returns a new Program instance
        assert isinstance(prog2, Program)
        assert prog2 is not prog

        # Both programs still produce correct output (path is unaffected by dummy rebind)
        x = np.array([-1.0, 0.5, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        np.testing.assert_allclose(prog("t4_path", x), expected, rtol=1e-6)
        np.testing.assert_allclose(prog2("t4_path", x), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: entry_points enumeration
# ---------------------------------------------------------------------------

class TestEntryPoints:

    def test_entry_points_include_path_and_equations(self, hidden, real_sr, backend, coder):
        """entry_points lists path + equations; not internal ua.prim names."""
        eq1 = Equation("t5_lin", "ij,j->i", hidden, hidden, real_sr)
        eq2 = Equation("t5_relu", None, hidden, hidden, nonlinearity="relu")
        W = coder.decode(None, np.eye(2)).value

        prog = compile_program(
            [eq1, eq2], backend=backend,
            specs=[PathSpec("t5_path", ["t5_lin", "t5_relu"], hidden, hidden,
                            params={"t5_lin": [W]})],
        )
        eps = prog.entry_points()
        assert "t5_path" in eps
        assert "t5_lin" in eps
        assert "t5_relu" in eps

    def test_entry_points_excludes_stdlib_primitives(self, hidden, real_sr, backend):
        """entry_points does not expose Hydra stdlib names."""
        eq = Equation("t5b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        for ep in prog.entry_points():
            assert not ep.startswith("hydra.")
