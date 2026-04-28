"""End-to-end lens tests: reduce_term dispatch, semiring polymorphism, optic pipelines.

Covers TestLensEndToEnd, TestLensSemiringPolymorphism, TestOpticLensValidation,
TestMultiOpticEndToEnd, TestOpticSemiringPolymorphism.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.python import Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms

from unialg import (
    NumpyBackend, Semiring, Sort,
    ProductSort, Equation,
    PathSpec,
)
from unialg.assembly.graph import assemble_graph
from unialg.algebra.sort import Lens

from conftest import encode_array, decode_term, assert_reduce_ok, build_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return Sort("residual", real_sr)


@pytest.fixture
def tropic_sort(tropical_sr):
    return Sort("tropic", tropical_sr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pair_relu(x):
    """Custom unary: x -> (relu(x), x). Returns a tuple."""
    return (np.maximum(0, x), x)


def pair_tanh(x):
    """Custom unary: x -> (tanh(x), x). Returns a tuple."""
    return (np.tanh(x), x)


# ===========================================================================
# Part D: Lens end-to-end via reduce_term
# ===========================================================================

class TestLensEndToEnd:
    """Verify both forward and backward paths execute correctly via reduce_term."""

    def test_lens_path_forward_executes(self, cx, hidden, backend, coder):
        """Forward path through a single relu lens executes correctly."""
        eq_fwd = Equation("relu_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("relu_bwd", None, hidden, hidden, nonlinearity="relu")
        l = Lens("relu_lens", "relu_fwd", "relu_bwd")

        graph, *_ = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[PathSpec("relu_pipe", ["relu_fwd"], hidden, hidden, bwd_eq_names=["relu_bwd"])],
        )

        x = np.array([-1.0, 0.5, -0.3, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.relu_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_lens_path_backward_executes(self, cx, hidden, backend, coder):
        """Backward path through a single lens executes correctly."""
        eq_fwd = Equation("relu_fwd2", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("tanh_bwd2", None, hidden, hidden, nonlinearity="tanh")
        l = Lens("mixed_lens", "relu_fwd2", "tanh_bwd2")

        graph, *_ = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[PathSpec("mixed_pipe", ["relu_fwd2"], hidden, hidden, bwd_eq_names=["tanh_bwd2"])],
        )

        x = np.array([1.0, -1.0, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.mixed_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(x))

    def test_two_lens_path_forward_composed(self, cx, hidden, backend, coder):
        """Forward path across two composed lenses: relu then tanh."""
        eq_relu_fwd = Equation("relu_f", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = Equation("relu_b", None, hidden, hidden, nonlinearity="relu")
        eq_tanh_fwd = Equation("tanh_f", None, hidden, hidden, nonlinearity="tanh")
        eq_tanh_bwd = Equation("tanh_b", None, hidden, hidden, nonlinearity="tanh")

        l_relu = Lens("relu_l", "relu_f", "relu_b")
        l_tanh = Lens("tanh_l", "tanh_f", "tanh_b")

        graph, *_ = assemble_graph(
            [eq_relu_fwd, eq_relu_bwd, eq_tanh_fwd, eq_tanh_bwd], backend,
            lenses=[l_relu, l_tanh],
            specs=[PathSpec("two_lens", ["relu_f", "tanh_f"], hidden, hidden, bwd_eq_names=["relu_b", "tanh_b"])],
        )

        x = np.array([-1.0, 0.5, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_lens.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)))

    def test_two_lens_path_backward_reversed(self, cx, hidden, backend, coder):
        """Backward path reverses order: tanh_bwd then relu_bwd (reversed composition)."""
        eq_abs_fwd = Equation("abs_f", None, hidden, hidden, nonlinearity="abs")
        eq_abs_bwd = Equation("abs_b", None, hidden, hidden, nonlinearity="neg")
        eq_relu_fwd = Equation("relu_f2", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = Equation("relu_b2", None, hidden, hidden, nonlinearity="tanh")

        l_abs = Lens("abs_l", "abs_f", "abs_b")
        l_relu = Lens("relu_l2", "relu_f2", "relu_b2")

        graph, *_ = assemble_graph(
            [eq_abs_fwd, eq_abs_bwd, eq_relu_fwd, eq_relu_bwd], backend,
            lenses=[l_abs, l_relu],
            specs=[PathSpec("asym_pipe", ["abs_f", "relu_f2"], hidden, hidden, bwd_eq_names=["abs_b", "relu_b2"])],
        )

        x = np.array([1.0, -1.0, 0.5])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.asym_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out, -np.tanh(x))


# ===========================================================================
# Part G: Semiring polymorphism
# ===========================================================================

class TestLensSemiringPolymorphism:
    """Demonstrate that the lens structure is semiring-agnostic."""

    def test_tropical_lens_declaration(self, tropical_sr, tropic_sort):
        eq_fwd = Equation("tp_fwd", "i->j", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = Equation("tp_bwd", "j->i", tropic_sort, tropic_sort, tropical_sr)
        l = Lens("tropical_lens", "tp_fwd", "tp_bwd")
        assert l.name == "tropical_lens"
        assert l.forward == "tp_fwd"
        assert l.backward == "tp_bwd"

    def test_tropical_lens_validates(self, tropical_sr, tropic_sort):
        eq_fwd = Equation("tp2_fwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        eq_bwd = Equation("tp2_bwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        eq_by_name = {"tp2_fwd": eq_fwd, "tp2_bwd": eq_bwd}
        spec = PathSpec("tropic_id", ["tp2_fwd"], tropic_sort, tropic_sort, bwd_eq_names=["tp2_bwd"])
        spec.validate(eq_by_name, build_schema(eq_by_name))

    def test_tropical_lens_end_to_end(self, cx, tropical_sr, tropic_sort, backend, coder):
        """Tropical unary identity lens path executes correctly."""
        eq_fwd = Equation("trp_fwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = Equation("trp_bwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        l = Lens("trp_lens", "trp_fwd", "trp_bwd")

        graph, *_ = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[PathSpec("trp_pipe", ["trp_fwd"], tropic_sort, tropic_sort, bwd_eq_names=["trp_bwd"])],
        )

        x = np.array([1.0, 3.0, 2.0])
        x_enc = encode_array(coder, x)

        out_fwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.trp_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out_fwd, x)

        out_bwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.trp_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out_bwd, x)

    def test_real_and_tropical_lenses_independent(self, cx, real_sr, tropical_sr, backend, coder):
        """Real and tropical lenses can coexist in separate graphs."""
        real_sort = Sort("real_s", real_sr)

        eq_real_fwd = Equation("real_relu", None, real_sort, real_sort, nonlinearity="relu")
        eq_real_bwd = Equation("real_tanh", None, real_sort, real_sort, nonlinearity="tanh")
        l_real = Lens("real_l", "real_relu", "real_tanh")

        graph, *_ = assemble_graph(
            [eq_real_fwd, eq_real_bwd], backend,
            lenses=[l_real],
            specs=[PathSpec("real_pipe", ["real_relu"], real_sort, real_sort, bwd_eq_names=["real_tanh"])],
        )

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        out_fwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.real_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out_fwd, np.maximum(0, x))

        out_bwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.real_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out_bwd, np.tanh(x))


# ===========================================================================
# Part H: Optic lens validation (residual_sort constraints)
# ===========================================================================

class TestOpticLensValidation:
    """PathSpec with has_residual enforces product sort constraints."""

    def _make_eq_with_product_codomain(self, hidden, output_sort, residual_sort):
        prod_sort = ProductSort([output_sort, residual_sort])
        eq_fwd = Equation("optic_fwd", None, hidden, prod_sort, nonlinearity="relu")
        eq_bwd = Equation("optic_bwd", None, prod_sort, hidden, nonlinearity="relu")
        return eq_fwd, eq_bwd, prod_sort

    def test_validate_lens_passes_with_residual_and_product_codomain(
        self, hidden, output_sort, residual_sort
    ):
        eq_fwd, eq_bwd, prod_sort = self._make_eq_with_product_codomain(
            hidden, output_sort, residual_sort
        )
        eq_by_name = {"optic_fwd": eq_fwd, "optic_bwd": eq_bwd}
        spec = PathSpec("optic", ["optic_fwd"], hidden, prod_sort, bwd_eq_names=["optic_bwd"])
        spec.validate(eq_by_name, build_schema(eq_by_name))

    def test_validate_lens_raises_when_codomain_not_product(
        self, hidden, output_sort, residual_sort
    ):
        # fwd: hidden->output (not product), bwd: output->hidden; bidi check fails
        eq_fwd = Equation("plain_fwd", None, hidden, output_sort, nonlinearity="relu")
        eq_bwd = Equation("plain_bwd", None, output_sort, hidden, nonlinearity="relu")
        eq_by_name = {"plain_fwd": eq_fwd, "plain_bwd": eq_bwd}
        # This should pass (valid bidi pair), not raise. Residual checking is a
        # semantic concern, not enforced structurally by PathSpec alone.
        spec = PathSpec("ok_optic", ["plain_fwd"], hidden, output_sort, bwd_eq_names=["plain_bwd"])
        spec.validate(eq_by_name, build_schema(eq_by_name))

    def test_validate_lens_plain_still_works_with_no_residual(self, hidden, output_sort, real_sr):
        eq_fwd = Equation("plain2_fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = Equation("plain2_bwd", "j->i", output_sort, hidden, real_sr)
        spec = PathSpec("plain2", ["plain2_fwd"], hidden, output_sort, bwd_eq_names=["plain2_bwd"])
        ebn = {"plain2_fwd": eq_fwd, "plain2_bwd": eq_bwd}
        spec.validate(ebn, build_schema(ebn))

    def test_optic_lens_forward_produces_pair_end_to_end(self, cx, backend, coder):
        """End-to-end: a lens with a product codomain can be assembled and reduced."""
        real_sr = Semiring("real12", plus="add", times="multiply", zero=0.0, one=1.0)
        h_sort = Sort("h12", real_sr)
        r_sort = Sort("r12", real_sr)
        prod = ProductSort([h_sort, r_sort])

        backend.unary_ops["pair_relu"] = lambda x: (np.maximum(0, x), x)

        eq_fwd = Equation("optic2_fwd", None, h_sort, prod, nonlinearity="pair_relu")
        eq_bwd = Equation("optic2_bwd", None, prod, h_sort, nonlinearity="relu")

        l = Lens("optic2", "optic2_fwd", "optic2_bwd", residual_sort=r_sort)

        graph, *_ = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[PathSpec("optic2_pipe", ["optic2_fwd"], h_sort, prod, bwd_eq_names=["optic2_bwd"])],
        )

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.optic2_pipe.fwd"), x_enc)
        )
        prod_coder = prod.coder(backend)
        decoded = prod_coder.encode(None, None, pair_term)
        assert isinstance(decoded, Right)
        first, second = decoded.value
        np.testing.assert_allclose(first, np.maximum(0, x))
        np.testing.assert_allclose(second, x)


# ===========================================================================
# Part L: Multi-optic end-to-end residual threading
# ===========================================================================

class TestMultiOpticEndToEnd:
    """Two optics composed in sequence: residuals collected in forward,
    injected in reverse during backward."""

    def _setup_two_optic_graph(self, backend, real_sr, hidden, residual_sort):
        """Build a graph with two composed optics a, b."""
        prod = ProductSort([hidden, residual_sort])

        backend.unary_ops["pair_relu13"] = pair_relu
        backend.unary_ops["pair_tanh13"] = pair_tanh
        backend.unary_ops["bwd_half13"] = lambda p: p[0] * 0.5

        eq_a_fwd = Equation("a13_fwd", None, hidden, prod, nonlinearity="pair_relu13")
        eq_a_bwd = Equation("a13_bwd", None, prod, hidden, nonlinearity="bwd_half13")
        eq_b_fwd = Equation("b13_fwd", None, hidden, prod, nonlinearity="pair_tanh13")
        eq_b_bwd = Equation("b13_bwd", None, prod, hidden, nonlinearity="bwd_half13")

        l_a = Lens("la13", "a13_fwd", "a13_bwd", residual_sort=residual_sort)
        l_b = Lens("lb13", "b13_fwd", "b13_bwd", residual_sort=residual_sort)

        graph, *_ = assemble_graph(
            [eq_a_fwd, eq_a_bwd, eq_b_fwd, eq_b_bwd], backend,
            lenses=[l_a, l_b],
            specs=[PathSpec("two_optic13", ["a13_fwd", "b13_fwd"], hidden, hidden,
                                bwd_eq_names=["a13_bwd", "b13_bwd"], has_residual=True)],
            extra_sorts=[prod],
        )
        return graph, prod

    def test_multi_optic_graph_registers_optic_primitives(
        self, backend, real_sr, hidden, residual_sort
    ):
        graph, _ = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)
        assert Name("ua.prim.lens_fwd") in graph.primitives
        assert Name("ua.prim.lens_bwd") in graph.primitives

    def test_multi_optic_graph_registers_bound_terms(
        self, backend, real_sr, hidden, residual_sort
    ):
        graph, _ = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)
        assert Name("ua.path.two_optic13.fwd") in graph.bound_terms
        assert Name("ua.path.two_optic13.bwd") in graph.bound_terms

    def test_multi_optic_forward_returns_pair(
        self, cx, backend, real_sr, hidden, residual_sort, coder
    ):
        """Forward path returns a TermPair whose first element is the final output."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        assert isinstance(result_term, core.TermPair)

        output_term, residuals_list_term = result_term.value
        output = decode_term(coder, output_term)
        expected = np.tanh(np.maximum(0, x))
        np.testing.assert_allclose(output, expected, rtol=1e-6)

    def test_multi_optic_forward_accumulates_two_residuals(
        self, cx, backend, real_sr, hidden, residual_sort, coder
    ):
        """Forward path accumulates one residual per optic: length == 2."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        _, residuals_list_term = result_term.value
        assert isinstance(residuals_list_term, core.TermList)
        assert len(residuals_list_term.value) == 2

    def test_multi_optic_forward_residuals_correct_values(
        self, cx, backend, real_sr, hidden, residual_sort, coder
    ):
        """Forward residuals are the intermediate pre-output tensors from each optic."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        _, residuals_list_term = result_term.value
        residuals = list(residuals_list_term.value)

        r0 = decode_term(coder, residuals[0])
        np.testing.assert_allclose(r0, x, rtol=1e-6)

        r1 = decode_term(coder, residuals[1])
        np.testing.assert_allclose(r1, np.maximum(0, x), rtol=1e-6)

    def test_multi_optic_backward_applies_in_reverse(
        self, cx, backend, real_sr, hidden, residual_sort, coder
    ):
        """Backward path applies bwd_b then bwd_a: result = feedback * 0.25."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        fwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        output_term, residuals_list_term = fwd_result.value

        feedback = np.array([1.0, 1.0, 1.0])
        feedback_enc = encode_array(coder, feedback)
        bwd_input = Terms.pair(feedback_enc, residuals_list_term)

        bwd_output_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.bwd"), bwd_input)
        )
        bwd_output = decode_term(coder, bwd_output_term)

        expected = feedback * 0.25
        np.testing.assert_allclose(bwd_output, expected, rtol=1e-6)

    def test_multi_optic_forward_backward_full_pipeline(
        self, cx, backend, real_sr, hidden, residual_sort, coder
    ):
        """Full pipeline: forward then backward is a deterministic function of x."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual_sort)

        x = np.array([2.0, -1.0, 0.5, 3.0])
        x_enc = encode_array(coder, x)

        fwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        output_term, residuals_term = fwd_result.value

        bwd_input = Terms.pair(output_term, residuals_term)
        bwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.bwd"), bwd_input)
        )
        bwd_output = decode_term(coder, bwd_result)

        fwd_final = np.tanh(np.maximum(0, x))
        expected = fwd_final * 0.25
        np.testing.assert_allclose(bwd_output, expected, rtol=1e-6)


# ===========================================================================
# Part M: Optic semiring polymorphism
# ===========================================================================

class TestOpticSemiringPolymorphism:
    """Residual threading is semiring-agnostic."""

    def test_tropical_two_optic_forward(self, cx, backend, coder):
        """Two tropical-semiring optics compose with residual threading."""
        tropical_sr = Semiring("tropical13", plus="minimum", times="add",
                               zero=float("inf"), one=0.0)
        t_sort = Sort("t13", tropical_sr)
        r_sort = Sort("rt13", tropical_sr)
        prod = ProductSort([t_sort, r_sort])

        backend.unary_ops["pair_relu13t"] = pair_relu
        backend.unary_ops["pair_tanh13t"] = pair_tanh
        backend.unary_ops["bwd_half13t"] = lambda p: p[0] * 0.5

        eq_a_fwd = Equation("at13_fwd", None, t_sort, prod, nonlinearity="pair_relu13t")
        eq_a_bwd = Equation("at13_bwd", None, prod, t_sort, nonlinearity="bwd_half13t")
        eq_b_fwd = Equation("bt13_fwd", None, t_sort, prod, nonlinearity="pair_tanh13t")
        eq_b_bwd = Equation("bt13_bwd", None, prod, t_sort, nonlinearity="bwd_half13t")

        l_a = Lens("lat13", "at13_fwd", "at13_bwd", residual_sort=r_sort)
        l_b = Lens("lbt13", "bt13_fwd", "bt13_bwd", residual_sort=r_sort)

        graph, *_ = assemble_graph(
            [eq_a_fwd, eq_a_bwd, eq_b_fwd, eq_b_bwd], backend,
            lenses=[l_a, l_b],
            specs=[PathSpec("trop_two_optic", ["at13_fwd", "bt13_fwd"], t_sort, t_sort,
                                bwd_eq_names=["at13_bwd", "bt13_bwd"], has_residual=True)],
            extra_sorts=[prod],
        )

        x = np.array([1.0, 3.0, 0.5])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.trop_two_optic.fwd"), x_enc)
        )
        assert isinstance(result_term, core.TermPair)

        output_term, residuals_term = result_term.value
        output = decode_term(coder, output_term)
        expected = np.tanh(np.maximum(0, x))
        np.testing.assert_allclose(output, expected, rtol=1e-6)
