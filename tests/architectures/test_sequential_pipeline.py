"""Sequential pipeline tests: multi-step path expressed as a cell composition.

Demonstrates that a multi-layer feedforward network is a sequential composition
of linear morphisms and nonlinear pointwise equations, semiring-polymorphic.
"""

import numpy as np
import pytest

from hydra.core import Name

from unialg import NumpyBackend, Semiring, Sort, Equation, compile_program
from unialg.parser import NamedCell
import unialg.morphism as morphism
from conftest import encode_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr():
    return Semiring("real14ff", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def hidden(real_sr):
    return Sort("h14ff", real_sr)


# ---------------------------------------------------------------------------
# Architecture: 3-layer feedforward (linear -> relu -> linear -> relu -> linear)
# ---------------------------------------------------------------------------

def build_ffn_equations(hidden, real_sr):
    """5 equations: 3 linear layers + 2 relu activations."""
    return [
        Equation("ffn_linear1", "ij,j->i", hidden, hidden, real_sr),
        Equation("ffn_relu1",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear1",)),
        Equation("ffn_linear2", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu1",)),
        Equation("ffn_relu2",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear2",)),
        Equation("ffn_linear3", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu2",)),
    ]


def _ffn_cell(hidden, prefix="ffn"):
    """Build a seq cell composing 5 equations."""
    def _eq(name):
        return morphism.eq(name, domain=hidden, codomain=hidden)
    return NamedCell(
        name=prefix,
        cell=morphism.seq(
            morphism.seq(
                morphism.seq(
                    morphism.seq(_eq(f"{prefix}_linear1"), _eq(f"{prefix}_relu1")),
                    _eq(f"{prefix}_linear2"),
                ),
                _eq(f"{prefix}_relu2"),
            ),
            _eq(f"{prefix}_linear3"),
        ),
    )


class TestFeedforward:

    def test_graph_has_all_primitives(self, hidden, real_sr, backend):
        """compile_program produces primitives for all 5 equations and the cell morphism."""
        eqs = build_ffn_equations(hidden, real_sr)
        prog = compile_program(
            eqs, backend=backend,
            cells=[_ffn_cell(hidden)],
        )
        for name in ("ffn_linear1", "ffn_relu1", "ffn_linear2",
                     "ffn_relu2", "ffn_linear3"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives
        assert Name("ua.morphism.ffn") in prog.graph.primitives

    def test_path_produces_correct_output(self, hidden, real_sr, backend, coder):
        """Running a 3-step nonlinear cell produces tanh(relu(abs(x))).

        Uses only nonlinearity equations (no weight matrices needed).
        """
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        eq_abs  = Equation("ffn_abs",  None, hidden, hidden, nonlinearity="abs")
        eq_relu = Equation("ffn_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("ffn_tanh", None, hidden, hidden, nonlinearity="tanh")

        cell = NamedCell(
            name="ffn_nl",
            cell=morphism.seq(
                morphism.seq(
                    morphism.eq("ffn_abs",  domain=hidden, codomain=hidden),
                    morphism.eq("ffn_relu", domain=hidden, codomain=hidden),
                ),
                morphism.eq("ffn_tanh", domain=hidden, codomain=hidden),
            ),
        )

        prog = compile_program(
            [eq_abs, eq_relu, eq_tanh], backend=backend,
            cells=[cell],
        )

        out = prog("ffn_nl", x)
        expected = np.tanh(np.maximum(0, np.abs(x)))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_semiring_swap_tropical(self, backend):
        """Same 5-equation architecture with tropical semiring assembles without error."""
        trop = Semiring("tropical14ff", plus="minimum", times="add",
                        zero=float("inf"), one=0.0)
        h = Sort("h14ff_t", trop)

        def _eq(name):
            return morphism.eq(name, domain=h, codomain=h)

        eqs = [
            Equation("tffn_linear1", "ij,j->i", h, h, trop),
            Equation("tffn_relu1",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear1",)),
            Equation("tffn_linear2", "ij,j->i", h, h, trop, inputs=("tffn_relu1",)),
            Equation("tffn_relu2",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear2",)),
            Equation("tffn_linear3", "ij,j->i", h, h, trop, inputs=("tffn_relu2",)),
        ]
        cell = NamedCell(
            name="tffn",
            cell=morphism.seq(
                morphism.seq(
                    morphism.seq(
                        morphism.seq(_eq("tffn_linear1"), _eq("tffn_relu1")),
                        _eq("tffn_linear2"),
                    ),
                    _eq("tffn_relu2"),
                ),
                _eq("tffn_linear3"),
            ),
        )
        prog = compile_program(eqs, backend=backend, cells=[cell])
        assert Name("ua.morphism.tffn") in prog.graph.primitives
        for name in ("tffn_linear1", "tffn_relu1", "tffn_linear2",
                     "tffn_relu2", "tffn_linear3"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives
