"""Phase 12 tests: product sorts, fixpoint iteration, and optic lens validation.

Three distinct feature groups are exercised:

1. Product sorts — typed tuples of sorts, with a round-trip TermCoder for
   tuple-of-arrays. Tests cover construction, element extraction, type identity,
   and the sort_coder dispatch to product_sort_coder.

2. Fixpoint iteration — _fixpoint_primitive + fixpoint() as a Hydra lambda term.
   validate_fixpoint() rejects non-endomorphisms and missing predicates; the
   end-to-end tests drive convergence toward zero and verify the hit-max_iter
   path when the step never converges.

   The predicate equation in end-to-end tests is assembled manually as a prim1
   with float32_coder() output. This is required because _fixpoint_primitive
   uses fun(a, prims.float32()), which bridges the predicate through Hydra's
   float32 coder. resolve_equation always uses sort_coder (binary tensor coder)
   as output, which is incompatible with float32 decoding.

3. Optic lens validation (residual_sort) — validate_lens() imposes richer
   constraints when a residual sort is supplied: forward codomain and backward
   domain must both be product sorts containing the residual. The end-to-end
   test wires a lens whose forward equation produces a product sort, assembles
   the graph, and reduces the forward path to confirm the pair is produced.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.dsl.prims import prim1, float32 as float32_coder
from hydra.reduction import reduce_term

from unified_algebra.backend import numpy_backend, UnaryOp
from unified_algebra.semiring import semiring
from unified_algebra.sort import (
    sort, tensor_coder, sort_coder,
    product_sort, is_product_sort, product_sort_elements,
    product_sort_coder, sort_type_from_term,
    PRODUCT_SORT_TYPE_NAME,
)
from unified_algebra.morphism import equation, resolve_equation
from unified_algebra.composition import lens, validate_lens, lens_path
from unified_algebra.recursion import fixpoint, _fixpoint_primitive
from unified_algebra.validation import validate_spec
from unified_algebra import FixpointSpec
from unified_algebra.graph import assemble_graph, build_graph
from unified_algebra import LensPathSpec
from unified_algebra.utils import record_fields


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
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return sort("residual", real_sr)


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
# Group 1: Product sorts
# ===========================================================================

class TestProductSorts:
    """product_sort, is_product_sort, product_sort_elements, sort_type_from_term."""

    def test_product_sort_creates_record_with_correct_type_name(self, hidden, output_sort):
        """product_sort([a, b]) is a TermRecord with type_name PRODUCT_SORT_TYPE_NAME."""
        ps = product_sort([hidden, output_sort])
        assert isinstance(ps, core.TermRecord)
        assert ps.value.type_name == PRODUCT_SORT_TYPE_NAME

    def test_is_product_sort_true_for_product(self, hidden, output_sort):
        """is_product_sort returns True for a product_sort term."""
        ps = product_sort([hidden, output_sort])
        assert is_product_sort(ps) is True

    def test_is_product_sort_false_for_plain_sort(self, hidden):
        """is_product_sort returns False for an ordinary sort term."""
        assert is_product_sort(hidden) is False

    def test_product_sort_elements_round_trips(self, hidden, output_sort, residual_sort):
        """product_sort_elements recovers the same sorts in declaration order."""
        ps = product_sort([hidden, output_sort, residual_sort])
        elements = product_sort_elements(ps)
        assert len(elements) == 3
        # Compare by TypeVariable name
        expected_types = [
            sort_type_from_term(hidden),
            sort_type_from_term(output_sort),
            sort_type_from_term(residual_sort),
        ]
        actual_types = [sort_type_from_term(e) for e in elements]
        assert actual_types == expected_types

    def test_product_sort_requires_at_least_two_elements(self, hidden):
        """product_sort([single]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            product_sort([hidden])

    def test_product_sort_empty_raises(self):
        """product_sort([]) raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            product_sort([])

    def test_sort_type_from_term_distinct_for_product_vs_components(self, hidden, output_sort):
        """sort_type_from_term produces distinct TypeVariable names for product sort vs components."""
        ps = product_sort([hidden, output_sort])
        ps_type = sort_type_from_term(ps)
        hidden_type = sort_type_from_term(hidden)
        output_type = sort_type_from_term(output_sort)
        assert ps_type != hidden_type
        assert ps_type != output_type
        # The product type name encodes both components
        assert "product" in ps_type.value.value

    def test_sort_coder_on_product_sort_encodes_decodes_tuple(self, hidden, output_sort, backend):
        """sort_coder on a product sort encodes/decodes a tuple of arrays correctly."""
        ps = product_sort([hidden, output_sort])
        prod_coder = sort_coder(ps, backend)

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])

        # Encode (Python tuple -> Hydra term)
        encoded = prod_coder.decode(None, (a, b))
        assert isinstance(encoded, Right)
        term = encoded.value

        # Decode (Hydra term -> Python tuple)
        decoded = prod_coder.encode(None, None, term)
        assert isinstance(decoded, Right)
        pair = decoded.value

        # pair is right-nested: (a, b) for 2 elements
        np.testing.assert_allclose(pair[0], a)
        np.testing.assert_allclose(pair[1], b)


# ===========================================================================
# Group 2: Lens with residual_sort / optic validation
# ===========================================================================

class TestOpticLensValidation:
    """validate_lens with residual_sort enforces product sort constraints."""

    def _make_eq_with_product_codomain(self, hidden, output_sort, residual_sort):
        """Make equations where fwd: hidden -> product(output, residual),
        bwd: product(output, residual) -> hidden."""
        prod_sort = product_sort([output_sort, residual_sort])
        # nonlinearity-only equations: structural test only, relu applied to sort
        eq_fwd = equation("optic_fwd", None, hidden, prod_sort, nonlinearity="relu")
        eq_bwd = equation("optic_bwd", None, prod_sort, hidden, nonlinearity="relu")
        return eq_fwd, eq_bwd, prod_sort

    def test_validate_lens_passes_with_residual_and_product_codomain(
        self, hidden, output_sort, residual_sort
    ):
        """validate_lens passes when residual_sort is set AND forward codomain is
        product_sort containing the residual."""
        eq_fwd, eq_bwd, prod_sort = self._make_eq_with_product_codomain(
            hidden, output_sort, residual_sort
        )
        l = lens("optic", "optic_fwd", "optic_bwd", residual_sort=residual_sort)
        eq_by_name = {"optic_fwd": eq_fwd, "optic_bwd": eq_bwd}
        # Should not raise
        validate_lens(eq_by_name, l)

    def test_validate_lens_raises_when_residual_set_but_codomain_not_product(
        self, hidden, output_sort, residual_sort
    ):
        """validate_lens raises when residual_sort is set but forward codomain is
        NOT a product sort."""
        # fwd: hidden -> output (plain sort, not a product)
        eq_fwd = equation("plain_fwd", None, hidden, output_sort, nonlinearity="relu")
        eq_bwd = equation("plain_bwd", None, output_sort, hidden, nonlinearity="relu")
        l = lens("bad_optic", "plain_fwd", "plain_bwd", residual_sort=residual_sort)
        eq_by_name = {"plain_fwd": eq_fwd, "plain_bwd": eq_bwd}
        with pytest.raises(TypeError, match="product sort"):
            validate_lens(eq_by_name, l)

    def test_validate_lens_raises_when_residual_not_in_forward_codomain(
        self, hidden, output_sort, residual_sort
    ):
        """validate_lens raises when residual_sort is set but the residual is not
        among the forward codomain's product elements."""
        # Product codomain contains only output + hidden, NOT residual
        wrong_prod = product_sort([output_sort, hidden])
        eq_fwd = equation("missing_fwd", None, hidden, wrong_prod, nonlinearity="relu")
        eq_bwd = equation("missing_bwd", None, wrong_prod, hidden, nonlinearity="relu")
        l = lens("missing_optic", "missing_fwd", "missing_bwd", residual_sort=residual_sort)
        eq_by_name = {"missing_fwd": eq_fwd, "missing_bwd": eq_bwd}
        with pytest.raises(TypeError, match="missing residual"):
            validate_lens(eq_by_name, l)

    def test_validate_lens_plain_still_works_with_residual_none(self, hidden, output_sort, real_sr):
        """A plain lens (no residual_sort) still validates normally."""
        eq_fwd = equation("plain2_fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("plain2_bwd", "j->i", output_sort, hidden, real_sr)
        l = lens("plain2", "plain2_fwd", "plain2_bwd")
        eq_by_name = {"plain2_fwd": eq_fwd, "plain2_bwd": eq_bwd}
        # Should not raise
        validate_lens(eq_by_name, l)

    def test_optic_lens_forward_produces_pair_end_to_end(self, cx, backend, coder):
        """End-to-end: a lens with a product codomain can be assembled and reduced.

        The forward equation uses a custom 'pair_relu' unary op that returns a
        tuple (relu(x), x), matching the product sort structure (h12, r12).
        The backward equation applies relu to its input — since product_sort_coder
        right-nests input tuples, we pass the first element of the pair back.

        We construct the graph with assemble_graph so that lens validation runs;
        the reduce_term call exercises the forward path of lens_path.
        """
        real_sr = semiring("real12", plus="add", times="multiply", zero=0.0, one=1.0)
        h_sort = sort("h12", real_sr)
        r_sort = sort("r12", real_sr)
        prod = product_sort([h_sort, r_sort])

        # Custom unary op: input ndarray -> tuple(relu(x), x)
        # Both components have the same shape, so the product coder can encode them.
        backend.unary_ops["pair_relu"] = UnaryOp(fn=lambda x: (np.maximum(0, x), x))

        eq_fwd = equation("optic2_fwd", None, h_sort, prod, nonlinearity="pair_relu")
        eq_bwd = equation("optic2_bwd", None, prod, h_sort, nonlinearity="relu")

        l = lens("optic2", "optic2_fwd", "optic2_bwd", residual_sort=r_sort)

        # assemble_graph validates the lens internally via validate_lens
        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("optic2_pipe", ["optic2"], h_sort, prod)],
        )

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        # Forward: the result is a pair term (product sort output)
        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.optic2_pipe.fwd"), x_enc)
        )
        # Decode both halves using the product sort coder
        prod_coder = sort_coder(prod, backend)
        decoded = prod_coder.encode(None, None, pair_term)
        assert isinstance(decoded, Right)
        first, second = decoded.value
        np.testing.assert_allclose(first, np.maximum(0, x))
        np.testing.assert_allclose(second, x)


# ===========================================================================
# Group 3: Fixpoint iteration
# ===========================================================================

class TestFixpointValidation:
    """validate_fixpoint enforces endomorphism and predicate domain constraints."""

    def test_validate_fixpoint_passes_for_valid_step_and_predicate(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint passes when step is endomorphism and predicate domain matches."""
        step_eq = equation("fp_step", None, hidden, hidden, nonlinearity="relu")
        pred_eq = equation("fp_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"fp_step": step_eq, "fp_pred": pred_eq}
        # Should not raise
        validate_spec(eq_by_name, FixpointSpec("_", "fp_step", "fp_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_non_endomorphism_step(
        self, hidden, output_sort, real_sr
    ):
        """validate_fixpoint raises when step maps hidden -> output (not endo)."""
        step_eq = equation("bad_step", "i->j", hidden, output_sort, real_sr)
        pred_eq = equation("bad_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"bad_step": step_eq, "bad_pred": pred_eq}
        with pytest.raises(TypeError):
            validate_spec(eq_by_name, FixpointSpec("_", "bad_step", "bad_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_missing_predicate(self, hidden):
        """validate_fixpoint raises ValueError when predicate equation is not found."""
        step_eq = equation("ms_step", None, hidden, hidden, nonlinearity="relu")
        eq_by_name = {"ms_step": step_eq}
        with pytest.raises(ValueError, match="predicate equation 'missing_pred' not found"):
            validate_spec(eq_by_name, FixpointSpec("_", "ms_step", "missing_pred", 0.0, 0, hidden))

    def test_validate_fixpoint_raises_for_missing_step(self, hidden):
        """validate_fixpoint raises ValueError when step equation is not found."""
        pred_eq = equation("ms_pred", None, hidden, hidden, nonlinearity="abs")
        eq_by_name = {"ms_pred": pred_eq}
        with pytest.raises(ValueError, match="step equation 'missing_step' not found"):
            validate_spec(eq_by_name, FixpointSpec("_", "missing_step", "ms_pred", 0.0, 0, hidden))


class TestFixpointEndToEnd:
    """Fixpoint iteration via lower-level graph assembly and reduce_term.

    _fixpoint_primitive uses fun(a, prims.float32()) to bridge the predicate.
    This means the predicate equation must be a prim1 with float32_coder() as
    its output coder — not the standard sort_coder used by resolve_equation.
    We therefore register the predicate primitive manually, following the same
    pattern as test_phase7.py for unfold_n.
    """

    def _make_step_prim(self, name, sort_term, nl_name, backend):
        """Resolve a unary endomorphism equation into a Primitive."""
        eq = equation(name, None, sort_term, sort_term, nonlinearity=nl_name)
        return resolve_equation(eq, backend)

    def _make_pred_prim(self, name, sort_term, fn, backend):
        """Build a predicate prim1 whose output is float32 (not tensor).

        fn: ndarray -> float

        This is required because _fixpoint_primitive bridges the predicate
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

        real_sr = semiring("real_fp1", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp1", real_sr)

        step_prim = self._make_step_prim("fp1_step", s_sort, "halve", backend)
        pred_prim = self._make_pred_prim(
            "fp1_pred", s_sort,
            lambda x: float(np.max(np.abs(x))),
            backend,
        )

        epsilon = 0.01
        max_iter = 100
        fp_prim = _fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = fixpoint(
            "converge1", "fp1_step", "fp1_pred"
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
        # pair_term is a TermPair; .value is a tuple (final_state_term, count_term)
        final_state_term, count_term = pair_term.value
        final_state = decode_term(coder, final_state_term)
        # All elements must be < epsilon in absolute value
        assert np.all(np.abs(final_state) < epsilon)
        # Must have converged before max_iter
        # count is a TermLiteral wrapping an integer — extract the raw int
        raw_count = count_term.value.value  # LiteralInteger(IntegerValueInt32(n)) chain
        if hasattr(raw_count, "value"):
            raw_count = raw_count.value  # unwrap IntegerValueInt32
        assert raw_count < max_iter

    def test_fixpoint_hits_max_iter_when_no_convergence(self, cx, backend, coder):
        """step(x) = x + 1 never converges; fixpoint returns after exactly max_iter steps."""
        backend.unary_ops["increment"] = UnaryOp(fn=lambda x: x + 1.0)

        real_sr = semiring("real_fp2", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp2", real_sr)

        step_prim = self._make_step_prim("fp2_step", s_sort, "increment", backend)
        # Predicate always returns 999.0, never <= epsilon
        pred_prim = self._make_pred_prim("fp2_pred", s_sort, lambda x: 999.0, backend)

        epsilon = 0.01
        max_iter = 5
        fp_prim = _fixpoint_primitive(epsilon, max_iter)

        fp_name, fp_term = fixpoint(
            "no_converge", "fp2_step", "fp2_pred"
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
        # When no convergence: returned count equals max_iter
        assert raw_count == max_iter

    def test_fixpoint_single_element_convergence(self, cx, backend, coder):
        """Scalar fixpoint: halve a 1-element vector until abs(x) <= 0.001."""
        backend.unary_ops["halve2"] = UnaryOp(fn=lambda x: 0.5 * x)

        real_sr = semiring("real_fp3", plus="add", times="multiply", zero=0.0, one=1.0)
        s_sort = sort("state_fp3", real_sr)

        step_prim = self._make_step_prim("fp3_step", s_sort, "halve2", backend)
        pred_prim = self._make_pred_prim(
            "fp3_pred", s_sort,
            lambda x: float(np.abs(np.asarray(x)).max()),
            backend,
        )

        fp_prim = _fixpoint_primitive(0.001, 50)
        fp_name, fp_term = fixpoint(
            "conv_scalar", "fp3_step", "fp3_pred"
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
# Group 4: Semiring residual field
# ===========================================================================

class TestSemiringResidualField:
    """semiring() residual kwarg and resolve_semiring residual_elementwise extraction."""

    def test_semiring_with_residual_creates_record_with_residual_field(self):
        """semiring(..., residual='divide') creates a TermRecord with a 'residual' field."""
        from unified_algebra.utils import record_fields, string_value
        sr = semiring("real_res", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        fields = record_fields(sr)
        assert "residual" in fields
        assert string_value(fields["residual"]) == "divide"

    def test_resolve_semiring_extracts_residual_elementwise(self, backend):
        """resolve_semiring extracts residual_elementwise as the divide callable."""
        from unified_algebra.semiring import resolve_semiring
        sr = semiring("real_res2", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = resolve_semiring(sr, backend)
        assert rsr.residual_name == "divide"
        assert rsr.residual_elementwise is not None
        # Sanity-check: divide(6, 3) == 2
        result = rsr.residual_elementwise(
            np.array([6.0]), np.array([3.0])
        )
        np.testing.assert_allclose(result, [2.0])

    def test_resolve_semiring_without_residual_gives_none(self, backend):
        """resolve_semiring with no residual gives residual_elementwise=None."""
        from unified_algebra.semiring import resolve_semiring
        sr = semiring("real_nores", plus="add", times="multiply", zero=0.0, one=1.0)
        rsr = resolve_semiring(sr, backend)
        assert rsr.residual_name is None
        assert rsr.residual_elementwise is None

    def test_residual_operation_real_semiring_divide(self, backend):
        """Real semiring residual 'divide': residual_elementwise(a, c) = a / c.

        The backend stores the raw np.divide callable, so argument order is
        positional: divide(a, c) = a / c.  The residual satisfies the adjoint
        condition a * b <= c <=> b <= c / a, so callers swap arguments as needed.
        """
        from unified_algebra.semiring import resolve_semiring
        sr = semiring("real_div", plus="add", times="multiply",
                      residual="divide", zero=0.0, one=1.0)
        rsr = resolve_semiring(sr, backend)
        a = np.array([8.0, 12.0, 25.0])
        c = np.array([2.0, 4.0, 5.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a / c)

    def test_residual_operation_tropical_semiring_subtract(self, backend):
        """Tropical semiring residual 'subtract': residual_elementwise(a, c) = a - c.

        The backend stores the raw np.subtract callable, so argument order is
        positional: subtract(a, c) = a - c.  The tropical residual satisfies
        a + b <= c <=> b <= c - a, so callers swap arguments as needed.
        """
        from unified_algebra.semiring import resolve_semiring
        sr = semiring("tropical_res", plus="minimum", times="add",
                      residual="subtract", zero=float("inf"), one=0.0)
        rsr = resolve_semiring(sr, backend)
        a = np.array([10.0, 5.0, 9.0])
        c = np.array([3.0, 1.0, 7.0])
        result = rsr.residual_elementwise(a, c)
        np.testing.assert_allclose(result, a - c)


# ===========================================================================
# Group 5: Backend axis-aware softmax and where
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
        # Monotonicity along each row
        np.testing.assert_array_less(result[0, 0], result[0, 1])
        np.testing.assert_array_less(result[0, 1], result[0, 2])

    def test_custom_axis0_softmax_registered_and_works(self, backend):
        """A custom axis-0 softmax via functools.partial works when registered."""
        import functools
        from scipy.special import softmax as scipy_softmax
        from unified_algebra.backend import UnaryOp
        backend.unary_ops["softmax_axis0"] = UnaryOp(
            fn=functools.partial(scipy_softmax, axis=0)
        )
        x = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
        softmax_ax0 = backend.unary("softmax_axis0")
        result = softmax_ax0(x)
        assert result.shape == x.shape
        # Columns should sum to 1 (axis-0 normalization)
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
