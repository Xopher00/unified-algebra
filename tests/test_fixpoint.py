"""Fixpoint iteration tests: convergence, max-iter bounds, and related features.

Covers fixpoint_primitive + fixpoint() as Hydra lambda terms,
validate_fixpoint() constraint enforcement, semiring residual field
extraction, and backend axis-aware operations.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.dsl.prims import prim1, float32 as float32_coder
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, Semiring, sort, tensor_coder, sort_coder,
    Equation, fixpoint,
    validate_spec, build_graph,
    FixpointSpec,
)
from unialg.backend import UnaryOp
from unialg.assembly.primitives import fixpoint_primitive
from unialg.utils import record_fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return sort("output", real_sr)


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
# Fixpoint validation
# ===========================================================================

class TestFixpointValidation:
    """validate_fixpoint enforces endomorphism and predicate domain constraints."""

    def test_validate_fixpoint_passes_for_valid_step_and_predicate(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint passes when step is endomorphism and predicate domain matches.

        The predicate codomain must differ from the state sort (it should be a scalar
        float32, not a tensor). output_sort stands in for a non-state codomain here;
        full float32 enforcement requires a prim1-level predicate (see TestFixpointEndToEnd).
        """
        step_eq = Equation("fp_step", None, hidden, hidden, nonlinearity="relu")
        pred_eq = Equation("fp_pred", None, hidden, output_sort, nonlinearity="abs")
        eq_by_name = {"fp_step": step_eq, "fp_pred": pred_eq}
        validate_spec(eq_by_name, FixpointSpec("_", "fp_step", "fp_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_endomorphism_predicate(
        self, hidden, real_sr
    ):
        """validate_fixpoint raises when predicate codomain == state sort (endomorphism)."""
        step_eq = Equation("endo_step", None, hidden, hidden, nonlinearity="relu")
        pred_eq = Equation("endo_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"endo_step": step_eq, "endo_pred": pred_eq}
        with pytest.raises(TypeError, match="scalar residual"):
            validate_spec(eq_by_name, FixpointSpec("_", "endo_step", "endo_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_non_endomorphism_step(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint raises when step maps hidden -> output (not endo)."""
        step_eq = Equation("bad_step", "i->j", hidden, output_sort, real_sr)
        pred_eq = Equation("bad_pred", None, hidden, output_sort, nonlinearity="abs")
        eq_by_name = {"bad_step": step_eq, "bad_pred": pred_eq}
        with pytest.raises(TypeError):
            validate_spec(eq_by_name, FixpointSpec("_", "bad_step", "bad_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_missing_predicate(self, hidden):
        """validate_fixpoint raises ValueError when predicate equation is not found."""
        step_eq = Equation("ms_step", None, hidden, hidden, nonlinearity="relu")
        eq_by_name = {"ms_step": step_eq}
        with pytest.raises(ValueError, match="predicate equation 'missing_pred' not found"):
            validate_spec(eq_by_name, FixpointSpec("_", "ms_step", "missing_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_missing_step(self, hidden):
        """validate_fixpoint raises ValueError when step equation is not found."""
        pred_eq = Equation("ms_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"ms_pred": pred_eq}
        with pytest.raises(ValueError, match="step equation 'missing_step' not found"):
            validate_spec(eq_by_name, FixpointSpec("_", "missing_step", "ms_pred", 0.0, 0, hidden))


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
        return eq.resolve(backend)

    def _make_pred_prim(self, name, sort_term, fn, backend):
        """Build a predicate prim1 whose output is float32 (not tensor).

        fn: ndarray -> float

        This is required because fixpoint_primitive bridges the predicate
        through fun(a, prims.float32()), which decodes the result as float32.
        """
        in_coder = sort_coder(sort_term, backend)
        return prim1(Name(f"ua.equation.{name}"), fn, [], in_coder, float32_coder())

    def test_fixpoint_converges_to_zero(self, cx, backend, coder):
        """step(x) = 0.5 * x; pred(x) = max(abs(x)); epsilon=0.01.

        Starting from x0=[1.0, 2.0], after a few halvings abs values drop
        below epsilon. The fixpoint result is a pair: (final_state, iteration_count).
        """
        backend.unary_ops["halve"] = UnaryOp(fn=lambda x: 0.5 * x)

        real_sr = Semiring("real_fp1", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp1", real_sr)

        step_prim = self._make_step_prim("fp1_step", s_sort, "halve", backend)
        pred_prim = self._make_pred_prim(
            "fp1_pred", s_sort,
            lambda x: float(np.max(np.abs(x))),
            backend,
        )

        epsilon = 0.01
        max_iter = 100
        fp_prim = fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = fixpoint(
            "converge1", "fp1_step", "fp1_pred", epsilon, max_iter
        )

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
        backend.unary_ops["increment"] = UnaryOp(fn=lambda x: x + 1.0)

        real_sr = Semiring("real_fp2", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp2", real_sr)

        step_prim = self._make_step_prim("fp2_step", s_sort, "increment", backend)
        pred_prim = self._make_pred_prim("fp2_pred", s_sort, lambda x: 999.0, backend)

        epsilon = 0.01
        max_iter = 5
        fp_prim = fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = fixpoint(
            "no_converge", "fp2_step", "fp2_pred", epsilon, max_iter
        )

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
        backend.unary_ops["halve2"] = UnaryOp(fn=lambda x: 0.5 * x)

        real_sr = Semiring("real_fp3", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp3", real_sr)

        step_prim = self._make_step_prim("fp3_step", s_sort, "halve2", backend)
        pred_prim = self._make_pred_prim(
            "fp3_pred", s_sort,
            lambda x: float(np.abs(np.asarray(x)).max()),
            backend,
        )

        fp_prim = fixpoint_primitive(0.001, 50)
        fp_name, fp_term = fixpoint(
            "conv_scalar", "fp3_step", "fp3_pred", 0.001, 50
        )

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
# Semiring residual field
# ===========================================================================

class TestSemiringResidualField:
    """Semiring() residual kwarg and resolve_semiring residual_elementwise extraction."""

    def test_semiring_with_residual_creates_record_with_residual_field(self):
        """Semiring(..., residual='divide') creates a TermRecord with a 'residual' field."""
        from unialg.utils import string_value
        sr = Semiring("real_res", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        fields = record_fields(sr.term)
        assert "residual" in fields
        assert string_value(fields["residual"]) == "divide"

    def test_resolve_semiring_extracts_residual_elementwise(self, backend):
        """Semiring.resolve extracts residual_elementwise as the divide callable."""
        sr = Semiring("real_res2", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        assert rsr.residual_name == "divide"
        assert rsr.residual_elementwise is not None
        result = rsr.residual_elementwise(
            np.array([6.0]), np.array([3.0])
        )
        np.testing.assert_allclose(result, [2.0])

    def test_resolve_semiring_without_residual_gives_none(self, backend):
        """Semiring.resolve with no residual gives residual_elementwise=None."""
        sr = Semiring("real_nores", plus="add", times="multiply", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        assert rsr.residual_name is None
        assert rsr.residual_elementwise is None

    def test_residual_operation_real_semiring_divide(self, backend):
        """Real semiring residual 'divide': residual_elementwise(a, c) = a / c."""
        sr = Semiring("real_div", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = sr.resolve(backend)
        a = np.array([8.0, 12.0, 25.0])
        c = np.array([2.0, 4.0, 5.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a / c)

    def test_residual_operation_tropical_semiring_subtract(self, backend):
        """Tropical semiring residual 'subtract': residual_elementwise(a, c) = a - c."""
        sr = Semiring("tropical_res", plus="minimum", times="add",
                      residual="subtract", zero=float("inf"), one=0.0)
        rsr = sr.resolve(backend)
        a = np.array([10.0, 5.0, 9.0])
        c = np.array([3.0, 1.0, 7.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a - c)


# ===========================================================================
# Backend axis-aware softmax and where
# ===========================================================================

class TestBackendAxisAwareOps:
    """numpy_backend softmax over last axis, custom axis-0 softmax, and where."""

    def test_softmax_normalizes_over_last_axis(self, backend):
        """numpy_backend 'softmax' normalizes each row of a 2D input (rows sum to 1)."""
        x = np.array([[1.0, 2.0, 3.0],
                       [0.0, 0.0, 0.0],
                       [-1.0, 0.0, 1.0]])
        softmax_fn = backend.unary("softmax")
        result = softmax_fn(x)
        assert result.shape == x.shape
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)
        np.testing.assert_array_less(result[0, 0], result[0, 1])
        np.testing.assert_array_less(result[0, 1], result[0, 2])

    def test_custom_axis0_softmax_registered_and_works(self, backend):
        """A custom axis-0 softmax via functools.partial works when registered."""
        import functools
        from scipy.special import softmax as scipy_softmax
        backend.unary_ops["softmax_axis0"] = UnaryOp(
            fn=functools.partial(scipy_softmax, axis=0)
        )
        x = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
        softmax_ax0 = backend.unary("softmax_axis0")
        result = softmax_ax0(x)
        assert result.shape == x.shape
        col_sums = result.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(2), atol=1e-6)

    def test_backend_where_exists_and_fills_masked_values(self, backend):
        """backend.where(mask, x, fill) returns filled values where mask is False."""
        assert backend.where is not None, "numpy_backend must expose 'where'"
        x = np.array([1.0, 2.0, 3.0, 4.0])
        fill = np.array([-999.0, -999.0, -999.0, -999.0])
        mask = np.array([True, False, True, False])
        result = backend.where(mask, x, fill)
        np.testing.assert_allclose(result, [1.0, -999.0, 3.0, -999.0])
