"""Lens tests: bidirectional morphisms, optic validation, and residual threading.

Covers lens() declaration, validate_lens() sort enforcement, lens_path()
forward/backward composition, assemble_graph integration with lenses,
semiring polymorphism, optic lens validation with residual_sort,
and height-2 optics with runtime residual threading.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, semiring, sort, tensor_coder, sort_coder,
    product_sort, equation, resolve_equation, path, fan,
    build_graph, assemble_graph, lens, validate_lens, lens_path,
    FoldSpec, LensPathSpec,
)
from unialg.algebra.sort import product_sort_coder
from unialg.composition.lenses import LENS_TYPE_NAME
from unialg.backend import UnaryOp
from unialg.assembly.primitives import lens_fwd_primitive, lens_bwd_primitive
from unialg.utils import record_fields, string_value
from unialg.views import LensView


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
def tropical_sr():
    return semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return sort("residual", real_sr)


@pytest.fixture
def tropic_sort(tropical_sr):
    return sort("tropic", tropical_sr)


@pytest.fixture
def coder():
    return tensor_coder()


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


def pair_relu(x):
    """Custom unary: x -> (relu(x), x). Returns a tuple."""
    return (np.maximum(0, x), x)


def pair_tanh(x):
    """Custom unary: x -> (tanh(x), x). Returns a tuple."""
    return (np.tanh(x), x)


# ===========================================================================
# Part A: Lens declaration
# ===========================================================================

class TestLensDeclaration:
    """Verify lens() produces the correct Hydra record structure."""

    def test_lens_record_type_name(self):
        """lens() produces a record with the LENS_TYPE_NAME type."""
        l = lens("linear", "fwd", "bwd")
        assert l.value.type_name == LENS_TYPE_NAME

    def test_lens_record_name_field(self):
        """The 'name' field of the lens record matches the given name."""
        l = lens("my_lens", "forward_eq", "backward_eq")
        lf = LensView(l)
        assert lf.name == "my_lens"

    def test_lens_record_forward_field(self):
        """The 'forward' field stores the forward equation name."""
        l = lens("linear", "linear_fwd", "linear_bwd")
        lf = LensView(l)
        assert lf.forward == "linear_fwd"

    def test_lens_record_backward_field(self):
        """The 'backward' field stores the backward equation name."""
        l = lens("linear", "linear_fwd", "linear_bwd")
        lf = LensView(l)
        assert lf.backward == "linear_bwd"

    def test_lens_without_residual_stores_unit(self):
        """When no residual_sort is provided, the residualSort field is a unit term."""
        l = lens("linear", "fwd", "bwd")
        fields = record_fields(l)
        residual = fields["residualSort"]
        assert residual is not None

    def test_lens_with_residual_stores_sort(self, hidden):
        """lens() with residual_sort stores the sort term in the residualSort field."""
        import hydra.core as core
        l = lens("linear", "fwd", "bwd", residual_sort=hidden)
        fields = record_fields(l)
        residual = fields["residualSort"]
        from unialg.algebra import sort_type_from_term
        t = sort_type_from_term(residual)
        assert t.value.function == core.TypeVariable(core.Name("ua.sort.hidden"))


# ===========================================================================
# Part B: Lens validation
# ===========================================================================

class TestLensValidation:
    """Verify validate_lens() enforces the bidirectionality sort contract."""

    def _make_matching_pair(self, hidden, output_sort):
        """Build a valid fwd (hidden→output) + bwd (output→hidden) pair."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("bwd", "j->i", output_sort, hidden, real_sr)
        return eq_fwd, eq_bwd

    def test_valid_lens_passes(self, hidden, output_sort):
        eq_fwd, eq_bwd = self._make_matching_pair(hidden, output_sort)
        l = lens("enc", "fwd", "bwd")
        eq_by_name = {"fwd": eq_fwd, "bwd": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_valid_lens_same_sort_passes(self, hidden):
        eq_fwd = equation("fwd_id", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("bwd_id", None, hidden, hidden, nonlinearity="relu")
        l = lens("identity_lens", "fwd_id", "bwd_id")
        eq_by_name = {"fwd_id": eq_fwd, "bwd_id": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_invalid_lens_forward_not_found(self, hidden):
        eq_bwd = equation("bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("bad_lens", "nonexistent_fwd", "bwd")
        with pytest.raises(TypeError, match="forward equation 'nonexistent_fwd' not found"):
            validate_lens({"bwd": eq_bwd}, l)

    def test_invalid_lens_backward_not_found(self, hidden):
        eq_fwd = equation("fwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("bad_lens", "fwd", "nonexistent_bwd")
        with pytest.raises(TypeError, match="backward equation 'nonexistent_bwd' not found"):
            validate_lens({"fwd": eq_fwd}, l)

    def test_invalid_lens_domain_mismatch(self, hidden, output_sort):
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("bwd", "i->j", hidden, output_sort, real_sr)
        l = lens("bad", "fwd", "bwd")
        with pytest.raises(TypeError, match="forward domain.*!=.*backward codomain"):
            validate_lens({"fwd": eq_fwd, "bwd": eq_bwd}, l)

    def test_invalid_lens_codomain_mismatch(self, hidden, output_sort):
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("bad", "fwd", "bwd")
        with pytest.raises(TypeError, match="forward codomain.*!=.*backward domain"):
            validate_lens({"fwd": eq_fwd, "bwd": eq_bwd}, l)


# ===========================================================================
# Part C: Lens path structure
# ===========================================================================

class TestLensPath:
    """Verify lens_path() produces correctly wired forward and backward paths."""

    def _make_id_lens(self, name, hidden):
        """Make a self-inverse lens (identity-like, hidden→hidden in both dirs)."""
        eq_fwd = equation(f"{name}_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation(f"{name}_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens(name, f"{name}_fwd", f"{name}_bwd")
        return eq_fwd, eq_bwd, l

    def test_lens_path_single_returns_two_pairs(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        result = lens_path("pipe", ["a"], lens_by_name)
        (fwd_name, fwd_term), (bwd_name, bwd_term) = result
        assert fwd_name == Name("ua.path.pipe.fwd")
        assert bwd_name == Name("ua.path.pipe.bwd")

    def test_lens_path_forward_name(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        (fwd_name, _), _ = lens_path("mypipe", ["a"], lens_by_name)
        assert fwd_name == Name("ua.path.mypipe.fwd")

    def test_lens_path_backward_name(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        _, (bwd_name, _) = lens_path("mypipe", ["a"], lens_by_name)
        assert bwd_name == Name("ua.path.mypipe.bwd")

    def test_lens_path_terms_are_lambdas(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        (_, fwd_term), (_, bwd_term) = lens_path("pipe", ["a"], lens_by_name)
        assert isinstance(fwd_term.value, core.Lambda)
        assert isinstance(bwd_term.value, core.Lambda)

    def test_lens_path_composition_order_forward(self, hidden):
        """For lenses [a, b], forward body is: b_fwd(a_fwd(x)) — left-to-right."""
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        lens_by_name = {"a": l_a, "b": l_b}
        (_, fwd_term), _ = lens_path("pipe", ["a", "b"], lens_by_name)
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.b_fwd"

    def test_lens_path_composition_order_backward(self, hidden):
        """For lenses [a, b], backward body is: a_bwd(b_bwd(x)) — right-to-left."""
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        lens_by_name = {"a": l_a, "b": l_b}
        _, (_, bwd_term) = lens_path("pipe", ["a", "b"], lens_by_name)
        body = bwd_term.value.body
        assert body.value.function.value.value == "ua.equation.a_bwd"

    def test_lens_path_empty_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one lens"):
            lens_path("empty", [], {})


# ===========================================================================
# Part D: Lens end-to-end via reduce_term
# ===========================================================================

class TestLensEndToEnd:
    """Verify both forward and backward paths execute correctly via reduce_term."""

    def test_lens_path_forward_executes(self, cx, hidden, backend, coder):
        """Forward path through a single relu lens executes correctly."""
        eq_fwd = equation("relu_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("relu_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("relu_lens", "relu_fwd", "relu_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("relu_pipe", ["relu_lens"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.5, -0.3, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.relu_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_lens_path_backward_executes(self, cx, hidden, backend, coder):
        """Backward path through a single lens executes correctly."""
        eq_fwd = equation("relu_fwd2", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("tanh_bwd2", None, hidden, hidden, nonlinearity="tanh")
        l = lens("mixed_lens", "relu_fwd2", "tanh_bwd2")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("mixed_pipe", ["mixed_lens"], hidden, hidden)],
        )

        x = np.array([1.0, -1.0, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.mixed_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(x))

    def test_two_lens_path_forward_composed(self, cx, hidden, backend, coder):
        """Forward path across two composed lenses: relu then tanh."""
        eq_relu_fwd = equation("relu_f", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = equation("relu_b", None, hidden, hidden, nonlinearity="relu")
        eq_tanh_fwd = equation("tanh_f", None, hidden, hidden, nonlinearity="tanh")
        eq_tanh_bwd = equation("tanh_b", None, hidden, hidden, nonlinearity="tanh")

        l_relu = lens("relu_l", "relu_f", "relu_b")
        l_tanh = lens("tanh_l", "tanh_f", "tanh_b")

        graph = assemble_graph(
            [eq_relu_fwd, eq_relu_bwd, eq_tanh_fwd, eq_tanh_bwd], backend,
            lenses=[l_relu, l_tanh],
            specs=[LensPathSpec("two_lens", ["relu_l", "tanh_l"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.5, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_lens.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)))

    def test_two_lens_path_backward_reversed(self, cx, hidden, backend, coder):
        """Backward path reverses order: tanh_bwd then relu_bwd (reversed composition)."""
        eq_abs_fwd = equation("abs_f", None, hidden, hidden, nonlinearity="abs")
        eq_abs_bwd = equation("abs_b", None, hidden, hidden, nonlinearity="neg")
        eq_relu_fwd = equation("relu_f2", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = equation("relu_b2", None, hidden, hidden, nonlinearity="tanh")

        l_abs = lens("abs_l", "abs_f", "abs_b")
        l_relu = lens("relu_l2", "relu_f2", "relu_b2")

        graph = assemble_graph(
            [eq_abs_fwd, eq_abs_bwd, eq_relu_fwd, eq_relu_bwd], backend,
            lenses=[l_abs, l_relu],
            specs=[LensPathSpec("asym_pipe", ["abs_l", "relu_l2"], hidden, hidden)],
        )

        x = np.array([1.0, -1.0, 0.5])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.asym_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out, -np.tanh(x))


# ===========================================================================
# Part E: Full assemble_graph integration with lenses
# ===========================================================================

class TestAssembleGraphWithLenses:
    """Integration tests: lenses wired through assemble_graph."""

    def test_assemble_graph_registers_both_paths(self, hidden, backend):
        eq_fwd = equation("id_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("id_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("id_lens", "id_fwd", "id_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("id_pipe", ["id_lens"], hidden, hidden)],
        )

        assert Name("ua.path.id_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.id_pipe.bwd") in graph.bound_terms

    def test_assemble_graph_with_invalid_lens_raises(self, hidden, output_sort, backend):
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("enc_fwd", None, hidden, output_sort, real_sr, nonlinearity="relu")
        eq_bwd = equation("enc_bwd", None, hidden, output_sort, real_sr, nonlinearity="relu")
        l = lens("bad_enc", "enc_fwd", "enc_bwd")

        with pytest.raises(TypeError):
            assemble_graph([eq_fwd, eq_bwd], backend, lenses=[l])

    def test_assemble_graph_no_lens_paths_still_validates(self, hidden, backend):
        eq_fwd = equation("enc2_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("enc2_bwd", None, hidden, hidden, nonlinearity="tanh")
        l = lens("enc2", "enc2_fwd", "enc2_bwd")

        graph = assemble_graph([eq_fwd, eq_bwd], backend, lenses=[l])
        assert Name("ua.path.enc2.fwd") not in graph.bound_terms

    def test_multiple_lenses_in_one_graph(self, hidden, backend):
        eqs = [
            equation("relu_g_fwd", None, hidden, hidden, nonlinearity="relu"),
            equation("relu_g_bwd", None, hidden, hidden, nonlinearity="relu"),
            equation("tanh_g_fwd", None, hidden, hidden, nonlinearity="tanh"),
            equation("tanh_g_bwd", None, hidden, hidden, nonlinearity="tanh"),
        ]
        l_relu = lens("relu_g", "relu_g_fwd", "relu_g_bwd")
        l_tanh = lens("tanh_g", "tanh_g_fwd", "tanh_g_bwd")

        graph = assemble_graph(
            eqs, backend,
            lenses=[l_relu, l_tanh],
            specs=[
                LensPathSpec("relu_only", ["relu_g"], hidden, hidden),
                LensPathSpec("tanh_only", ["tanh_g"], hidden, hidden),
            ],
        )

        assert Name("ua.path.relu_only.fwd") in graph.bound_terms
        assert Name("ua.path.relu_only.bwd") in graph.bound_terms
        assert Name("ua.path.tanh_only.fwd") in graph.bound_terms
        assert Name("ua.path.tanh_only.bwd") in graph.bound_terms


# ===========================================================================
# Part F: Lens fold integration
# ===========================================================================

class TestLensFoldIntegration:
    """A lens's forward equation can serve as a fold step without interference."""

    def test_fold_with_lens_forward_equation(self, cx, backend, coder):
        """fold() using a lens's forward equation works normally."""
        add_sr = semiring("addsem", plus="add", times="add", zero=0.0, one=0.0)
        acc_sort = sort("acc", add_sr)
        eq_step_fwd = equation("acc_fwd", "i,i->i", acc_sort, acc_sort, add_sr)
        eq_step_bwd = equation("acc_bwd", "i,i->i", acc_sort, acc_sort, add_sr)
        l = lens("acc_lens", "acc_fwd", "acc_bwd")

        init = encode_array(coder, np.zeros(3))

        graph = assemble_graph(
            [eq_step_fwd, eq_step_bwd], backend,
            lenses=[l],
            specs=[FoldSpec("sum_fold", "acc_fwd", init, acc_sort, acc_sort)],
        )

        items = [
            encode_array(coder, np.array([1.0, 2.0, 3.0])),
            encode_array(coder, np.array([4.0, 5.0, 6.0])),
            encode_array(coder, np.array([7.0, 8.0, 9.0])),
        ]
        seq_term = Terms.list_(items)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fold.sum_fold"), seq_term)
        ))
        np.testing.assert_allclose(out, np.array([12.0, 15.0, 18.0]))

    def test_lens_coexists_with_fold_and_path(self, cx, hidden, backend, coder):
        """A graph can contain lenses, lens_paths, and folds simultaneously."""
        real_sr = record_fields(hidden)["semiring"]
        eq_relu_fwd = equation("rl_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = equation("rl_bwd", None, hidden, hidden, nonlinearity="relu")
        eq_step = equation("fold_step", "i,i->i", hidden, hidden, real_sr)
        l = lens("rl", "rl_fwd", "rl_bwd")

        init = encode_array(coder, np.zeros(4))
        graph = assemble_graph(
            [eq_relu_fwd, eq_relu_bwd, eq_step], backend,
            lenses=[l],
            specs=[
                LensPathSpec("rl_pipe", ["rl"], hidden, hidden),
                FoldSpec("rl_fold", "fold_step", init, hidden, hidden),
            ],
        )

        assert Name("ua.path.rl_pipe.fwd") in graph.bound_terms
        assert Name("ua.fold.rl_fold") in graph.bound_terms


# ===========================================================================
# Part G: Semiring polymorphism
# ===========================================================================

class TestLensSemiringPolymorphism:
    """Demonstrate that the lens structure is semiring-agnostic."""

    def test_tropical_lens_declaration(self, tropical_sr, tropic_sort):
        eq_fwd = equation("tp_fwd", "i->j", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = equation("tp_bwd", "j->i", tropic_sort, tropic_sort, tropical_sr)
        l = lens("tropical_lens", "tp_fwd", "tp_bwd")
        lf = LensView(l)
        assert lf.name == "tropical_lens"
        assert lf.forward == "tp_fwd"
        assert lf.backward == "tp_bwd"

    def test_tropical_lens_validates(self, tropical_sr, tropic_sort):
        eq_fwd = equation("tp2_fwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        eq_bwd = equation("tp2_bwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        l = lens("tropic_id", "tp2_fwd", "tp2_bwd")
        eq_by_name = {"tp2_fwd": eq_fwd, "tp2_bwd": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_tropical_lens_end_to_end(self, cx, tropical_sr, tropic_sort, backend, coder):
        """Tropical unary identity lens path executes correctly."""
        eq_fwd = equation("trp_fwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = equation("trp_bwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        l = lens("trp_lens", "trp_fwd", "trp_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("trp_pipe", ["trp_lens"], tropic_sort, tropic_sort)],
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
        real_sort = sort("real_s", real_sr)

        eq_real_fwd = equation("real_relu", None, real_sort, real_sort, nonlinearity="relu")
        eq_real_bwd = equation("real_tanh", None, real_sort, real_sort, nonlinearity="tanh")
        l_real = lens("real_l", "real_relu", "real_tanh")

        graph = assemble_graph(
            [eq_real_fwd, eq_real_bwd], backend,
            lenses=[l_real],
            specs=[LensPathSpec("real_pipe", ["real_l"], real_sort, real_sort)],
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
    """validate_lens with residual_sort enforces product sort constraints."""

    def _make_eq_with_product_codomain(self, hidden, output_sort, residual_sort):
        prod_sort = product_sort([output_sort, residual_sort])
        eq_fwd = equation("optic_fwd", None, hidden, prod_sort, nonlinearity="relu")
        eq_bwd = equation("optic_bwd", None, prod_sort, hidden, nonlinearity="relu")
        return eq_fwd, eq_bwd, prod_sort

    def test_validate_lens_passes_with_residual_and_product_codomain(
        self, hidden, output_sort, residual_sort
    ):
        eq_fwd, eq_bwd, prod_sort = self._make_eq_with_product_codomain(
            hidden, output_sort, residual_sort
        )
        l = lens("optic", "optic_fwd", "optic_bwd", residual_sort=residual_sort)
        eq_by_name = {"optic_fwd": eq_fwd, "optic_bwd": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_validate_lens_raises_when_residual_set_but_codomain_not_product(
        self, hidden, output_sort, residual_sort
    ):
        eq_fwd = equation("plain_fwd", None, hidden, output_sort, nonlinearity="relu")
        eq_bwd = equation("plain_bwd", None, output_sort, hidden, nonlinearity="relu")
        l = lens("bad_optic", "plain_fwd", "plain_bwd", residual_sort=residual_sort)
        eq_by_name = {"plain_fwd": eq_fwd, "plain_bwd": eq_bwd}
        with pytest.raises(TypeError, match="product sort"):
            validate_lens(eq_by_name, l)

    def test_validate_lens_raises_when_residual_not_in_forward_codomain(
        self, hidden, output_sort, residual_sort
    ):
        wrong_prod = product_sort([output_sort, hidden])
        eq_fwd = equation("missing_fwd", None, hidden, wrong_prod, nonlinearity="relu")
        eq_bwd = equation("missing_bwd", None, wrong_prod, hidden, nonlinearity="relu")
        l = lens("missing_optic", "missing_fwd", "missing_bwd", residual_sort=residual_sort)
        eq_by_name = {"missing_fwd": eq_fwd, "missing_bwd": eq_bwd}
        with pytest.raises(TypeError, match="missing residual"):
            validate_lens(eq_by_name, l)

    def test_validate_lens_plain_still_works_with_residual_none(self, hidden, output_sort, real_sr):
        eq_fwd = equation("plain2_fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("plain2_bwd", "j->i", output_sort, hidden, real_sr)
        l = lens("plain2", "plain2_fwd", "plain2_bwd")
        eq_by_name = {"plain2_fwd": eq_fwd, "plain2_bwd": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_optic_lens_forward_produces_pair_end_to_end(self, cx, backend, coder):
        """End-to-end: a lens with a product codomain can be assembled and reduced."""
        real_sr = semiring("real12", plus="add", times="multiply", zero=0.0, one=1.0)
        h_sort = sort("h12", real_sr)
        r_sort = sort("r12", real_sr)
        prod = product_sort([h_sort, r_sort])

        backend.unary_ops["pair_relu"] = UnaryOp(fn=lambda x: (np.maximum(0, x), x))

        eq_fwd = equation("optic2_fwd", None, h_sort, prod, nonlinearity="pair_relu")
        eq_bwd = equation("optic2_bwd", None, prod, h_sort, nonlinearity="relu")

        l = lens("optic2", "optic2_fwd", "optic2_bwd", residual_sort=r_sort)

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[LensPathSpec("optic2_pipe", ["optic2"], h_sort, prod)],
        )

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        pair_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.optic2_pipe.fwd"), x_enc)
        )
        prod_coder = sort_coder(prod, backend)
        decoded = prod_coder.encode(None, None, pair_term)
        assert isinstance(decoded, Right)
        first, second = decoded.value
        np.testing.assert_allclose(first, np.maximum(0, x))
        np.testing.assert_allclose(second, x)


# ===========================================================================
# Part I: Optic primitive structure
# ===========================================================================

class TestOpticPrimitives:
    """The raw lens_fwd_primitive and lens_bwd_primitive have correct names."""

    def test_lens_fwd_primitive_name(self):
        prim = lens_fwd_primitive
        assert prim.name == Name("ua.prim.lens_fwd")

    def test_lens_bwd_primitive_name(self):
        prim = lens_bwd_primitive
        assert prim.name == Name("ua.prim.lens_bwd")


# ===========================================================================
# Part J: Optic path structure (_lens_path_threaded)
# ===========================================================================

def _make_residual_lens_path(name, lens_names, fwd_names, bwd_names, residual_sort):
    """Helper: build lens records with residual and call lens_path."""
    lens_terms = {
        ln: lens(ln, forward=fwd, backward=bwd, residual_sort=residual_sort)
        for ln, fwd, bwd in zip(lens_names, fwd_names, bwd_names)
    }
    return lens_path(name, lens_names, lens_terms)


class TestOpticPathStructure:
    """lens_path with residual builds correctly named lambda terms."""

    def test_lens_path_threaded_returns_two_pairs(self, hidden, residual_sort):
        result = _make_residual_lens_path("enc", ["l1", "l2"], ["fwd1", "fwd2"], ["bwd1", "bwd2"], residual_sort)
        (fwd_name, fwd_term), (bwd_name, bwd_term) = result
        assert fwd_name == Name("ua.path.enc.fwd")
        assert bwd_name == Name("ua.path.enc.bwd")

    def test_lens_path_threaded_fwd_is_lambda(self, hidden, residual_sort):
        (_, fwd_term), _ = _make_residual_lens_path("enc2", ["l1", "l2"], ["fwd1", "fwd2"], ["bwd1", "bwd2"], residual_sort)
        assert isinstance(fwd_term.value, core.Lambda)

    def test_lens_path_threaded_bwd_is_lambda(self, hidden, residual_sort):
        _, (_, bwd_term) = _make_residual_lens_path("enc3", ["l1", "l2"], ["fwd1", "fwd2"], ["bwd1", "bwd2"], residual_sort)
        assert isinstance(bwd_term.value, core.Lambda)

    def test_lens_path_threaded_fwd_uses_lens_fwd_primitive(self, hidden, residual_sort):
        (_, fwd_term), _ = _make_residual_lens_path("enc4", ["l1", "l2"], ["fwd1", "fwd2"], ["bwd1", "bwd2"], residual_sort)
        outer_app = fwd_term.value.body
        inner_app = outer_app.value.function
        prim_var = inner_app.value.function
        assert prim_var.value.value == "ua.prim.lens_fwd"

    def test_lens_path_threaded_bwd_uses_lens_bwd_primitive(self, hidden, residual_sort):
        _, (_, bwd_term) = _make_residual_lens_path("enc5", ["l1", "l2"], ["fwd1", "fwd2"], ["bwd1", "bwd2"], residual_sort)
        outer_app = bwd_term.value.body
        inner_app = outer_app.value.function
        prim_var = inner_app.value.function
        assert prim_var.value.value == "ua.prim.lens_bwd"


# ===========================================================================
# Part K: Optic path routing
# ===========================================================================

class TestLensPathRouting:
    """lens_path routes to _lens_path_threaded only for multi-optic with residual."""

    def _make_optic_lens(self, backend, name, hidden, residual_sort):
        prod = product_sort([hidden, residual_sort])
        real_sr = list(hidden.value.fields)[1].term

        backend.unary_ops[f"pair_{name}"] = UnaryOp(fn=pair_relu)
        backend.unary_ops[f"bwd_{name}"] = UnaryOp(fn=lambda p: p[0] * 0.5)

        eq_fwd = equation(f"{name}_fwd", None, hidden, prod,
                          nonlinearity=f"pair_{name}")
        eq_bwd = equation(f"{name}_bwd", None, prod, hidden,
                          nonlinearity=f"bwd_{name}")
        l = lens(name, f"{name}_fwd", f"{name}_bwd", residual_sort=residual_sort)
        return eq_fwd, eq_bwd, l

    def test_single_optic_with_residual_uses_plain_path(self, hidden, residual_sort):
        """Single optic with residual_sort uses plain path (not optic_fwd/bwd)."""
        prod = product_sort([hidden, residual_sort])
        eq_fwd = equation("so_fwd", None, hidden, prod, nonlinearity="relu")
        eq_bwd = equation("so_bwd", None, prod, hidden, nonlinearity="relu")
        l = lens("so", "so_fwd", "so_bwd", residual_sort=residual_sort)
        lens_by_name = {"so": l}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "so_pipe", ["so"], lens_by_name
        )
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.so_fwd"

    def test_multi_optic_with_residual_uses__lens_path_threaded(self, hidden, residual_sort):
        """Multi-optic with residual_sort uses _lens_path_threaded (optic_fwd/bwd)."""
        prod = product_sort([hidden, residual_sort])
        eq_fwd_a = equation("mo_fwd_a", None, hidden, prod, nonlinearity="relu")
        eq_bwd_a = equation("mo_bwd_a", None, prod, hidden, nonlinearity="relu")
        eq_fwd_b = equation("mo_fwd_b", None, hidden, prod, nonlinearity="relu")
        eq_bwd_b = equation("mo_bwd_b", None, prod, hidden, nonlinearity="relu")
        l_a = lens("mo_a", "mo_fwd_a", "mo_bwd_a", residual_sort=residual_sort)
        l_b = lens("mo_b", "mo_fwd_b", "mo_bwd_b", residual_sort=residual_sort)
        lens_by_name = {"mo_a": l_a, "mo_b": l_b}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "mo_pipe", ["mo_a", "mo_b"], lens_by_name
        )
        body = fwd_term.value.body
        inner = body.value.function
        prim_ref = inner.value.function
        assert prim_ref.value.value == "ua.prim.lens_fwd"

    def test_plain_lens_without_residual_unaffected(self, hidden):
        """Plain lens (no residual_sort) still uses the original path composition."""
        eq_fwd = equation("pl_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("pl_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("pl", "pl_fwd", "pl_bwd")
        lens_by_name = {"pl": l}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "pl_pipe", ["pl"], lens_by_name
        )
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.pl_fwd"


# ===========================================================================
# Part L: Multi-optic end-to-end residual threading
# ===========================================================================

class TestMultiOpticEndToEnd:
    """Two optics composed in sequence: residuals collected in forward,
    injected in reverse during backward."""

    def _setup_two_optic_graph(self, backend, real_sr, hidden, residual_sort):
        """Build a graph with two composed optics a, b."""
        prod = product_sort([hidden, residual_sort])

        backend.unary_ops["pair_relu13"] = UnaryOp(fn=pair_relu)
        backend.unary_ops["pair_tanh13"] = UnaryOp(fn=pair_tanh)
        backend.unary_ops["bwd_half13"] = UnaryOp(fn=lambda p: p[0] * 0.5)

        eq_a_fwd = equation("a13_fwd", None, hidden, prod, nonlinearity="pair_relu13")
        eq_a_bwd = equation("a13_bwd", None, prod, hidden, nonlinearity="bwd_half13")
        eq_b_fwd = equation("b13_fwd", None, hidden, prod, nonlinearity="pair_tanh13")
        eq_b_bwd = equation("b13_bwd", None, prod, hidden, nonlinearity="bwd_half13")

        l_a = lens("la13", "a13_fwd", "a13_bwd", residual_sort=residual_sort)
        l_b = lens("lb13", "b13_fwd", "b13_bwd", residual_sort=residual_sort)

        graph = assemble_graph(
            [eq_a_fwd, eq_a_bwd, eq_b_fwd, eq_b_bwd], backend,
            lenses=[l_a, l_b],
            specs=[LensPathSpec("two_optic13", ["la13", "lb13"], hidden, hidden)],
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
        tropical_sr = semiring("tropical13", plus="minimum", times="add",
                               zero=float("inf"), one=0.0)
        t_sort = sort("t13", tropical_sr)
        r_sort = sort("rt13", tropical_sr)
        prod = product_sort([t_sort, r_sort])

        backend.unary_ops["pair_relu13t"] = UnaryOp(fn=pair_relu)
        backend.unary_ops["pair_tanh13t"] = UnaryOp(fn=pair_tanh)
        backend.unary_ops["bwd_half13t"] = UnaryOp(fn=lambda p: p[0] * 0.5)

        eq_a_fwd = equation("at13_fwd", None, t_sort, prod, nonlinearity="pair_relu13t")
        eq_a_bwd = equation("at13_bwd", None, prod, t_sort, nonlinearity="bwd_half13t")
        eq_b_fwd = equation("bt13_fwd", None, t_sort, prod, nonlinearity="pair_tanh13t")
        eq_b_bwd = equation("bt13_bwd", None, prod, t_sort, nonlinearity="bwd_half13t")

        l_a = lens("lat13", "at13_fwd", "at13_bwd", residual_sort=r_sort)
        l_b = lens("lbt13", "bt13_fwd", "bt13_bwd", residual_sort=r_sort)

        graph = assemble_graph(
            [eq_a_fwd, eq_a_bwd, eq_b_fwd, eq_b_bwd], backend,
            lenses=[l_a, l_b],
            specs=[LensPathSpec("trop_two_optic", ["lat13", "lbt13"], t_sort, t_sort)],
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
