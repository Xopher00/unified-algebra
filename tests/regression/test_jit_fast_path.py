"""Regression: 5-step seq uses fast-path closures, not reduce_term."""
import unittest.mock
import numpy as np
import pytest

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.assembly.graph import assemble_graph
from unialg.parser import NamedCell
import unialg.morphism as morphism


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


def test_five_step_seq_does_not_call_reduce_term(hidden, real_sr, backend):
    from unialg.assembly.program import compile_program
    backend.unary_ops["halve"] = lambda x: 0.5 * x
    eqs = [Equation(f"s{i}", None, hidden, hidden, nonlinearity="halve") for i in range(5)]

    chain = morphism.eq("s0", domain=hidden, codomain=hidden)
    for i in range(1, 5):
        chain = morphism.seq(chain, morphism.eq(f"s{i}", domain=hidden, codomain=hidden))

    named = NamedCell(name="five_step", cell=chain)
    prog = compile_program(eqs, backend=backend, cells=[named])

    x = np.array([2.0, 4.0, 8.0])
    result = prog("five_step", x)

    np.testing.assert_allclose(result, x * 0.5**5)
