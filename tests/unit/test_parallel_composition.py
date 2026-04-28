"""Parallel composition tests: bimap / monoidal product of two morphisms.

Tests both levels:
1. Direct ParallelComposition._compile() and to_lambda() (isolated)
2. End-to-end via reduce_term dispatch (full Hydra graph)
3. .ua DSL parse → assemble_graph → compiled_fns
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.python import Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    Equation,
    ParallelSpec,
)
from unialg.assembly.graph import assemble_graph
from unialg.assembly.compositions import ParallelComposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_pair(coder, a, b):
    from hydra.dsl.prims import pair as pair_coder
    pc = pair_coder(coder, coder)
    result = pc.decode(None, (a, b))
    assert isinstance(result, Right)
    return result.value


def decode_pair(coder, term):
    from hydra.dsl.prims import pair as pair_coder
    pc = pair_coder(coder, coder)
    result = pc.encode(None, None, term)
    assert isinstance(result, Right)
    return result.value


# ---------------------------------------------------------------------------
# Level 1: Direct ParallelComposition structure and _compile
# ---------------------------------------------------------------------------

class TestParallelCompositionStructure:

    def test_parallel_returns_name_and_lambda(self, hidden):
        name, term = ParallelComposition("bimap", left_name="f", right_name="g").to_lambda()
        assert name == Name("ua.parallel.bimap")
        assert isinstance(term.value, core.Lambda)

    def test_parallel_name_prefix(self, hidden):
        name, _ = ParallelComposition("my_bimap", left_name="a", right_name="b").to_lambda()
        assert name.value == "ua.parallel.my_bimap"

    def test_parallel_accesses_left_right_names(self):
        pc = ParallelComposition("test", left_name="left_op", right_name="right_op")
        assert pc.left_name == "left_op"
        assert pc.right_name == "right_op"
        assert pc.name == "test"

    def test_parallel_compile_routes_components_correctly(self, hidden, coder, backend):
        """_compile returns a fn that applies left to pair[0], right to pair[1]."""
        eq_relu = Equation("p_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("p_tanh", None, hidden, hidden, nonlinearity="tanh")

        spec = ParallelSpec("bimap_test", left_name="p_relu", right_name="p_tanh")
        graph, native_fns, compiled_fns = assemble_graph(
            [eq_relu.term, eq_tanh.term],
            backend=backend,
            specs=[spec],
        )

        bimap_fn = compiled_fns["bimap_test"]
        a = np.array([1.0, -1.0, 2.0, -2.0])
        b = np.array([-0.5, 0.0, 1.5])

        result = bimap_fn((a, b))
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2

        np.testing.assert_allclose(result[0], np.maximum(0.0, a), rtol=1e-6)
        np.testing.assert_allclose(result[1], np.tanh(b), rtol=1e-6)

    def test_parallel_left_only_applied_to_first(self, hidden, backend):
        """Left op must not touch the second component."""
        eq_relu = Equation("lonly_relu", None, hidden, hidden, nonlinearity="relu")
        eq_id = Equation("lonly_id", None, hidden, hidden, nonlinearity="relu")

        spec = ParallelSpec("lonly", left_name="lonly_relu", right_name="lonly_id")
        _, _, compiled_fns = assemble_graph(
            [eq_relu.term, eq_id.term], backend=backend, specs=[spec])

        fn = compiled_fns["lonly"]
        a = np.array([-3.0, 4.0])
        b = np.array([5.0, -6.0])

        out_a, out_b = fn((a, b))
        np.testing.assert_allclose(out_a, np.maximum(0.0, a))
        np.testing.assert_allclose(out_b, np.maximum(0.0, b))

    def test_parallel_from_term_round_trips(self):
        """ParallelComposition survives a to_lambda / from_term round trip."""
        pc = ParallelComposition("rt", left_name="lft", right_name="rgt")
        name, lam = pc.to_lambda()
        # The name should be stored in the kind prefix
        assert name.value == "ua.parallel.rt"
        # Re-wrap from the record term stored in the composition
        pc2 = ParallelComposition.from_term(pc._term)
        assert pc2.left_name == "lft"
        assert pc2.right_name == "rgt"
        assert pc2.name == "rt"


# ---------------------------------------------------------------------------
# Level 2: End-to-end via reduce_term
# ---------------------------------------------------------------------------

class TestParallelReduceTerm:

    def test_reduce_term_dispatches_parallel_primitive(self, hidden, coder, backend, cx):
        """reduce_term correctly dispatches ua.parallel.bimap on an encoded pair."""
        eq_relu = Equation("rt_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("rt_tanh", None, hidden, hidden, nonlinearity="tanh")

        spec = ParallelSpec("rt_bimap", left_name="rt_relu", right_name="rt_tanh")
        graph, _, _ = assemble_graph(
            [eq_relu.term, eq_tanh.term], backend=backend, specs=[spec])

        a = np.array([1.0, -1.0, 2.0])
        b = np.array([-0.5, 0.0, 1.5])
        pair_term = encode_pair(coder, a, b)

        call_term = apply(var("ua.parallel.rt_bimap"), pair_term)
        result = reduce_term(cx, graph, True, call_term)
        assert isinstance(result, Right), f"reduce_term returned Left: {result}"

        out = decode_pair(coder, result.value)
        np.testing.assert_allclose(out[0], np.maximum(0.0, a), rtol=1e-6)
        np.testing.assert_allclose(out[1], np.tanh(b), rtol=1e-6)

    def test_reduce_term_registered_under_correct_key(self, hidden, backend):
        """ua.parallel.<name> primitive is registered in graph.primitives."""
        eq_relu = Equation("reg_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("reg_tanh", None, hidden, hidden, nonlinearity="tanh")

        spec = ParallelSpec("reg_bimap", left_name="reg_relu", right_name="reg_tanh")
        graph, _, _ = assemble_graph(
            [eq_relu.term, eq_tanh.term], backend=backend, specs=[spec])

        assert Name("ua.parallel.reg_bimap") in graph.primitives


# ---------------------------------------------------------------------------
# Level 3: .ua DSL parse → assemble_graph → compiled_fns
# ---------------------------------------------------------------------------

class TestParallelDSL:

    def test_parallel_via_ua_parse(self, backend):
        """End-to-end: parse .ua program with 'parallel', run via compiled_fns."""
        from unialg.parser import parse_ua_spec

        src = """\
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec feat(real)

op dsl_relu : feat -> feat
  nonlinearity = relu

op dsl_tanh : feat -> feat
  nonlinearity = tanh

parallel dsl_bimap : feat -> feat = dsl_relu & dsl_tanh
"""
        ua_spec = parse_ua_spec(src)
        assert len(ua_spec.specs) == 1
        spec = ua_spec.specs[0]
        assert spec.name == "dsl_bimap"
        assert spec.left_name == "dsl_relu"
        assert spec.right_name == "dsl_tanh"

        eq_terms = [eq.term for eq in ua_spec.equations]
        graph, _, compiled_fns = assemble_graph(
            eq_terms, backend=backend, specs=ua_spec.specs)

        assert "dsl_bimap" in compiled_fns

        fn = compiled_fns["dsl_bimap"]
        a = np.array([-1.0, 0.0, 1.0])
        b = np.array([0.5, 1.0, 2.0])

        out_a, out_b = fn((a, b))
        np.testing.assert_allclose(out_a, np.maximum(0.0, a), rtol=1e-6)
        np.testing.assert_allclose(out_b, np.tanh(b), rtol=1e-6)

    def test_parallel_grammar_parses_correct_tuple(self):
        """Parser produces ('parallel', name, (left, right)) tuple."""
        from unialg.parser._grammar import _build_parser
        from hydra.parsers import run_parser

        parser = _build_parser()
        src = "parallel pair_op : dom_s -> cod_s = left_fn & right_fn\n"
        result = run_parser(parser, src)
        assert result is not None
        # run_parser returns ParseResultSuccess with .value = ParseSuccess(value=..., remainder=...)
        decls = result.value.value
        assert len(decls) == 1
        d = decls[0]
        assert d[0] == "parallel"
        assert d[1] == "pair_op"
        assert d[2] == ("dom_s", "cod_s")
        assert d[3] == ("left_fn", "right_fn")
