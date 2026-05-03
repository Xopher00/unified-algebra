"""Tests for weight sharing via share_groups in compile_program.

Two ops placed in a share group receive the same bound param term.
The canonical slot is the first op in the group; all others alias to it.
"""

from __future__ import annotations

import numpy as np
import pytest
import hydra.core as core
from hydra.dsl.terms import apply, var
from hydra.dsl.python import Right

from unialg import NumpyBackend, Semiring, Sort, Equation, compile_program
from unialg.assembly.graph import assemble_graph
import hydra.dsl.terms as _hterms
from unialg.terms import tensor_coder
from conftest import encode_array, decode_term, assert_reduce_ok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("share_real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("share_hidden", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Test A: share_groups aliases params dict entries
# ---------------------------------------------------------------------------

class TestShareGroupsParamsDict:
    """compile_program wires share_groups into the params dict before assemble_graph."""

    def test_alias_appears_as_bound_term(self, hidden, real_sr, backend, coder):
        """Both canonical and alias ops get the same bound term in the graph."""
        # Two equations with param_slots so each takes a scalar param + an input.
        backend.unary_ops["scale_a"] = lambda x, w: w * x
        backend.unary_ops["scale_b"] = lambda x, w: w * x

        eq_a = Equation("share_a", None, hidden, hidden,
                        nonlinearity="scale_a", param_slots=("weight",))
        eq_b = Equation("share_b", None, hidden, hidden,
                        nonlinearity="scale_b", param_slots=("weight",))

        weight_term = _hterms.float32(2.5)

        # share_a is canonical; share_b aliases to it
        prog = compile_program(
            [eq_a, eq_b],
            backend=backend,
            params={"share_a": weight_term},
            share_groups={"wgroup": ["share_a", "share_b"]},
        )

        graph = prog.graph
        key_a = core.Name("ua.param.share_a")
        key_b = core.Name("ua.param.share_b")

        assert key_a in graph.bound_terms, "canonical bound term missing"
        assert key_b in graph.bound_terms, "alias bound term missing"
        assert graph.bound_terms[key_a] is graph.bound_terms[key_b], (
            "alias should point to the same term object as canonical"
        )

    def test_alias_absent_when_canonical_missing(self, hidden, real_sr, backend):
        """If canonical op has no param, alias is also None (not KeyError)."""
        backend.unary_ops["scale_c"] = lambda x, w: w * x
        backend.unary_ops["scale_d"] = lambda x, w: w * x

        eq_c = Equation("share_c", None, hidden, hidden,
                        nonlinearity="scale_c", param_slots=("weight",))
        eq_d = Equation("share_d", None, hidden, hidden,
                        nonlinearity="scale_d", param_slots=("weight",))

        # No params supplied — canonical is absent
        prog = compile_program(
            [eq_c, eq_d],
            backend=backend,
            params=None,
            share_groups={"wgroup2": ["share_c", "share_d"]},
        )
        # Bound terms should not contain the alias (value would be None)
        graph = prog.graph
        key_d = core.Name("ua.param.share_d")
        # The alias is None so assemble_graph skips it (None values are falsy)
        assert key_d not in graph.bound_terms


# ---------------------------------------------------------------------------
# Test B: live execution — tying is live, not just a dict entry
# ---------------------------------------------------------------------------

class TestShareGroupsLiveExecution:
    """Changing the shared tensor changes results for all tied ops."""

    def test_same_weight_same_result(self, hidden, real_sr, backend, coder, cx=None):
        """Both ops produce identical output when bound to the same weight tensor."""
        if cx is None:
            from hydra.context import Context
            from hydra.dsl.python import FrozenDict
            cx = Context(trace=(), messages=(), other=FrozenDict({}))

        backend.unary_ops["scale_e"] = lambda x, w: w * x
        backend.unary_ops["scale_f"] = lambda x, w: w * x

        eq_e = Equation("share_e", None, hidden, hidden,
                        nonlinearity="scale_e", param_slots=("weight",))
        eq_f = Equation("share_f", None, hidden, hidden,
                        nonlinearity="scale_f", param_slots=("weight",))

        weight_term = _hterms.float32(3.0)

        graph, _, _ = assemble_graph(
            [eq_e, eq_f],
            backend,
            params={"share_e": weight_term, "share_f": weight_term},
        )

        x = np.array([1.0, 2.0, 4.0])
        x_enc = encode_array(coder, x)

        # Both equations applied with their respective param vars
        out_e = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.share_e"), var("ua.param.share_e")), x_enc)
        ))
        out_f = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.share_f"), var("ua.param.share_f")), x_enc)
        ))

        np.testing.assert_allclose(out_e, 3.0 * x, rtol=1e-6)
        np.testing.assert_allclose(out_f, 3.0 * x, rtol=1e-6)
        np.testing.assert_allclose(out_e, out_f, rtol=1e-6)

    def test_shared_weight_via_compile_program(self, hidden, real_sr, backend, coder):
        """compile_program with share_groups aliases both ops to the same param term.

        Verifies that running with weight W produces the same result for both ops,
        and that rebinding the weight changes both results identically.
        """
        from hydra.context import Context
        from hydra.dsl.python import FrozenDict
        cx = Context(trace=(), messages=(), other=FrozenDict({}))

        backend.unary_ops["scale_g"] = lambda x, w: w * x
        backend.unary_ops["scale_h"] = lambda x, w: w * x

        eq_g = Equation("share_g", None, hidden, hidden,
                        nonlinearity="scale_g", param_slots=("weight",))
        eq_h = Equation("share_h", None, hidden, hidden,
                        nonlinearity="scale_h", param_slots=("weight",))

        weight_a = _hterms.float32(2.0)
        weight_b = _hterms.float32(5.0)

        x = np.array([1.0, 3.0, 7.0])
        x_enc = encode_array(coder, x)

        prog = compile_program(
            [eq_g, eq_h],
            backend=backend,
            params={"share_g": weight_a},
            share_groups={"wg": ["share_g", "share_h"]},
        )

        graph = prog.graph

        # Both ops produce result with weight=2.0
        out_g = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.share_g"), var("ua.param.share_g")), x_enc)
        ))
        out_h = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.share_h"), var("ua.param.share_h")), x_enc)
        ))

        np.testing.assert_allclose(out_g, 2.0 * x, rtol=1e-6)
        np.testing.assert_allclose(out_h, 2.0 * x, rtol=1e-6,
                                   err_msg="share_h should be aliased to share_g")

        # Rebind to weight_b — both ops should change
        prog2 = compile_program(
            [eq_g, eq_h],
            backend=backend,
            params={"share_g": weight_b},
            share_groups={"wg": ["share_g", "share_h"]},
        )
        graph2 = prog2.graph

        out_g2 = decode_term(coder, assert_reduce_ok(
            cx, graph2,
            apply(apply(var("ua.equation.share_g"), var("ua.param.share_g")), x_enc)
        ))
        out_h2 = decode_term(coder, assert_reduce_ok(
            cx, graph2,
            apply(apply(var("ua.equation.share_h"), var("ua.param.share_h")), x_enc)
        ))

        np.testing.assert_allclose(out_g2, 5.0 * x, rtol=1e-6)
        np.testing.assert_allclose(out_h2, 5.0 * x, rtol=1e-6,
                                   err_msg="alias share_h must change with canonical share_g")

        # Results differ from weight_a results — tying is live
        assert not np.allclose(out_g, out_g2), "result should change when weight changes"
        assert not np.allclose(out_h, out_h2), "alias result should change when canonical weight changes"
