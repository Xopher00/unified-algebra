"""Fixpoint end-to-end tests: convergence and max-iter bounds via reduce_term."""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.dsl.prims import prim1, float32 as float32_coder
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    Equation,
    build_graph,
    FixpointSpec,
)
from unialg.assembly.compositions import FixpointComposition
from unialg.assembly._primitives import fixpoint_primitive
from unialg.algebra.sort import sort_wrap


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


def make_graph_with_stdlib(primitives=None, bound_terms=None, sorts=None):
    """Build a Hydra Graph with standard library + extra primitives/bound_terms."""
    from hydra.sources.libraries import standard_library
    all_prims = dict(standard_library())
    if primitives:
        all_prims.update(primitives)
    return build_graph(sorts or [], primitives=all_prims, bound_terms=bound_terms or {})


# ===========================================================================
# Fixpoint end-to-end
# ===========================================================================

class TestFixpointEndToEnd:
    """Fixpoint iteration via lower-level graph assembly and reduce_term.

    fixpoint_primitive uses fun(a, prims.float32()) to bridge the predicate.
    This means the predicate equation must be a prim1 with float32_coder() as
    its output coder — not the standard sort_coder used by resolve_equation.
    We therefore register the predicate primitive manually.
    """

    def _make_step_prim(self, name, sort_term, nl_name, backend):
        """Resolve a unary endomorphism equation into a Primitive."""
        eq = Equation(name, None, sort_term, sort_term, nonlinearity=nl_name)
        prim, *_ = eq.resolve(backend)
        return prim

    def _make_pred_prim(self, name, sort_term, fn, backend):
        """Build a predicate prim1 whose output is float32 (not tensor).

        fn: ndarray -> float

        This is required because fixpoint_primitive bridges the predicate
        through fun(a, prims.float32()), which decodes the result as float32.
        """
        in_coder = sort_wrap(sort_term).coder(backend)
        return prim1(Name(f"ua.equation.{name}"), fn, [], in_coder, float32_coder())

    def test_fixpoint_converges_to_zero(self, cx, backend, coder):
        """step(x) = 0.5 * x; pred(x) = max(abs(x)); epsilon=0.01.

        Starting from x0=[1.0, 2.0], after a few halvings abs values drop
        below epsilon. The fixpoint result is a pair: (final_state, iteration_count).
        """
        backend.unary_ops["halve"] = lambda x: 0.5 * x

        real_sr = Semiring("real_fp1", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = Sort("state_fp1", real_sr)

        step_prim = self._make_step_prim("fp1_step", s_sort, "halve", backend)
        pred_prim = self._make_pred_prim(
            "fp1_pred", s_sort,
            lambda x: float(np.max(np.abs(x))),
            backend,
        )

        epsilon = 0.01
        max_iter = 100
        fp_prim = fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = FixpointComposition(
            "converge1", "fp1_step", "fp1_pred", epsilon, max_iter
        ).to_lambda()

        graph = make_graph_with_stdlib(
            primitives={
                step_prim.name: step_prim,
                pred_prim.name: pred_prim,
                fp_prim.name: fp_prim,
            },
            bound_terms={fp_name: fp_term},
            sorts=[s_sort],
        )

        x0 = np.array([1.0, 2.0])
        x0_enc = encode_array(coder, x0)

        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.fixpoint.converge1"), x0_enc)
        )
        final_state_term, count_term = pair_term.value
        final_state = decode_term(coder, final_state_term)
        assert np.all(np.abs(final_state) < epsilon)
        raw_count = count_term.value.value
        if hasattr(raw_count, "value"):
            raw_count = raw_count.value
        assert raw_count < max_iter

    def test_fixpoint_hits_max_iter_when_no_convergence(self, cx, backend, coder):
        """step(x) = x + 1 never converges; fixpoint returns after exactly max_iter steps."""
        backend.unary_ops["increment"] = lambda x: x + 1.0

        real_sr = Semiring("real_fp2", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = Sort("state_fp2", real_sr)

        step_prim = self._make_step_prim("fp2_step", s_sort, "increment", backend)
        pred_prim = self._make_pred_prim("fp2_pred", s_sort, lambda x: 999.0, backend)

        epsilon = 0.01
        max_iter = 5
        fp_prim = fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = FixpointComposition(
            "no_converge", "fp2_step", "fp2_pred", epsilon, max_iter
        ).to_lambda()

        graph = make_graph_with_stdlib(
            primitives={
                step_prim.name: step_prim,
                pred_prim.name: pred_prim,
                fp_prim.name: fp_prim,
            },
            bound_terms={fp_name: fp_term},
            sorts=[s_sort],
        )

        x0 = np.array([0.0])
        x0_enc = encode_array(coder, x0)

        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.fixpoint.no_converge"), x0_enc)
        )
        final_state_term, count_term = pair_term.value
        raw_count = count_term.value.value
        if hasattr(raw_count, "value"):
            raw_count = raw_count.value
        assert raw_count == max_iter

    def test_fixpoint_single_element_convergence(self, cx, backend, coder):
        """Scalar fixpoint: halve a 1-element vector until abs(x) <= 0.001."""
        backend.unary_ops["halve2"] = lambda x: 0.5 * x

        real_sr = Semiring("real_fp3", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = Sort("state_fp3", real_sr)

        step_prim = self._make_step_prim("fp3_step", s_sort, "halve2", backend)
        pred_prim = self._make_pred_prim(
            "fp3_pred", s_sort,
            lambda x: float(np.abs(np.asarray(x)).max()),
            backend,
        )

        fp_prim = fixpoint_primitive(0.001, 50)
        fp_name, fp_term = FixpointComposition(
            "conv_scalar", "fp3_step", "fp3_pred", 0.001, 50
        ).to_lambda()

        graph = make_graph_with_stdlib(
            primitives={
                step_prim.name: step_prim,
                pred_prim.name: pred_prim,
                fp_prim.name: fp_prim,
            },
            bound_terms={fp_name: fp_term},
            sorts=[s_sort],
        )

        x0 = np.array([8.0])
        x0_enc = encode_array(coder, x0)

        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.fixpoint.conv_scalar"), x0_enc)
        )
        final_state_term, _ = pair_term.value
        final = decode_term(coder, final_state_term)
        assert float(np.abs(np.asarray(final)).max()) <= 0.001


# ===========================================================================
# Fixpoint via parser (DSL round-trip)
# ===========================================================================

class TestFixpointFromParser:
    """Fixpoint iteration via parse_ua — DSL round-trip tests.

    The predicate op must have a codomain sort that differs from the state sort
    (FixpointSpec.constraints enforces this). We declare a separate 'scalar'
    sort so the predicate can map hidden -> scalar.

    Program.__call__ for a compiled fixpoint entry point returns the raw result
    of the while_loop closure: (final_state_array, iteration_count_int).
    """

    def test_fixpoint_converges_via_dsl(self):
        """halve step + max-abs predicate converges below epsilon=0.01."""
        backend = NumpyBackend()
        backend.unary_ops["halve_dsl"] = lambda x: 0.5 * x
        backend.unary_ops["max_abs_dsl"] = lambda x: float(np.max(np.abs(x)))

        from unialg import parse_ua
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
spec scalar(real)

op halve_step : hidden -> hidden
  nonlinearity = halve_dsl

op convergence_check : hidden -> scalar
  nonlinearity = max_abs_dsl

fixpoint converge : hidden
  step = halve_step
  predicate = convergence_check
  epsilon = 0.01
  max_iter = 100
''', backend)
        x0 = np.array([1.0, 2.0])
        result = prog('converge', x0)
        # result is (final_state, iteration_count)
        final_state, iteration_count = result
        assert np.all(np.abs(final_state) < 0.01)
        assert iteration_count < 100

    def test_fixpoint_hits_max_iter_via_dsl(self):
        """increment step with always-large predicate hits max_iter exactly."""
        backend = NumpyBackend()
        backend.unary_ops["increment_dsl"] = lambda x: x + 1.0
        backend.unary_ops["always_large_dsl"] = lambda x: 999.0

        from unialg import parse_ua
        prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
spec scalar(real)

op increment_step : hidden -> hidden
  nonlinearity = increment_dsl

op never_converges : hidden -> scalar
  nonlinearity = always_large_dsl

fixpoint no_converge : hidden
  step = increment_step
  predicate = never_converges
  epsilon = 0.01
  max_iter = 5
''', backend)
        x0 = np.array([0.0])
        result = prog('no_converge', x0)
        final_state, iteration_count = result
        assert iteration_count == 5
        # After 5 increments starting from [0.0]: [5.0]
        np.testing.assert_allclose(final_state, np.array([5.0]))
