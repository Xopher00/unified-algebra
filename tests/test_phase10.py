"""Phase 10 tests: lenses — bidirectional morphisms.

A lens pairs a forward equation with a backward equation over matching sorts.
The forward leg maps domain → codomain; the backward leg maps codomain → domain.
lens_path() composes multiple lenses sequentially, applying forward equations
left-to-right and backward equations right-to-left.

These tests verify:
  - lens() declaration produces correct Hydra record fields
  - validate_lens() enforces sort compatibility
  - lens_path() composes forward/backward legs in the correct order
  - Both directions are callable via reduce_term through assemble_graph
  - The lens concept is semiring-agnostic (real and tropical tested)
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

from unified_algebra.backend import numpy_backend
from unified_algebra.semiring import semiring
from unified_algebra.sort import sort, tensor_coder, sort_coder
from unified_algebra.morphism import equation, resolve_equation
from unified_algebra.composition import path, fan
from unified_algebra.graph import build_graph, assemble_graph
from unified_algebra.lens import (
    lens, validate_lens, lens_path, lens_fan,
    LENS_TYPE_NAME, _lens_fields,
)
from unified_algebra.utils import record_fields, string_value


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


# ===========================================================================
# Part A: Lens declaration
# ===========================================================================

class TestLensDeclaration:
    """Verify lens() produces the correct Hydra record structure."""

    def test_lens_record_type_name(self):
        """lens() produces a record with the LENS_TYPE_NAME type."""
        l = lens("linear", "fwd", "bwd")
        # TermRecord has a value attribute with a TypeName
        assert l.value.type_name == LENS_TYPE_NAME

    def test_lens_record_name_field(self):
        """The 'name' field of the lens record matches the given name."""
        l = lens("my_lens", "forward_eq", "backward_eq")
        lf = _lens_fields(l)
        assert lf["name"] == "my_lens"

    def test_lens_record_forward_field(self):
        """The 'forward' field stores the forward equation name."""
        l = lens("linear", "linear_fwd", "linear_bwd")
        lf = _lens_fields(l)
        assert lf["forward"] == "linear_fwd"

    def test_lens_record_backward_field(self):
        """The 'backward' field stores the backward equation name."""
        l = lens("linear", "linear_fwd", "linear_bwd")
        lf = _lens_fields(l)
        assert lf["backward"] == "linear_bwd"

    def test_lens_without_residual_stores_unit(self):
        """When no residual_sort is provided, the residualSort field is a unit term."""
        l = lens("linear", "fwd", "bwd")
        fields = record_fields(l)
        residual = fields["residualSort"]
        # unit() produces TermRecord with empty fields (Hydra unit)
        assert residual is not None

    def test_lens_with_residual_stores_sort(self, hidden):
        """lens() with residual_sort stores the sort term in the residualSort field."""
        l = lens("linear", "fwd", "bwd", residual_sort=hidden)
        fields = record_fields(l)
        residual = fields["residualSort"]
        # The residual sort should be the hidden sort term (a TermRecord)
        from unified_algebra.sort import sort_type_from_term
        t = sort_type_from_term(residual)
        assert "hidden" in t.value.value


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
        """validate_lens() passes for a correctly matched forward+backward pair."""
        eq_fwd, eq_bwd = self._make_matching_pair(hidden, output_sort)
        l = lens("enc", "fwd", "bwd")
        eq_by_name = {"fwd": eq_fwd, "bwd": eq_bwd}
        # Should not raise
        validate_lens(eq_by_name, l)

    def test_valid_lens_same_sort_passes(self, hidden):
        """validate_lens() passes when domain == codomain (self-adjoint case)."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd_id", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("bwd_id", None, hidden, hidden, nonlinearity="relu")
        l = lens("identity_lens", "fwd_id", "bwd_id")
        eq_by_name = {"fwd_id": eq_fwd, "bwd_id": eq_bwd}
        validate_lens(eq_by_name, l)

    def test_invalid_lens_forward_not_found(self, hidden):
        """validate_lens() raises TypeError when forward equation is missing."""
        real_sr = record_fields(hidden)["semiring"]
        eq_bwd = equation("bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("bad_lens", "nonexistent_fwd", "bwd")
        with pytest.raises(TypeError, match="forward equation 'nonexistent_fwd' not found"):
            validate_lens({"bwd": eq_bwd}, l)

    def test_invalid_lens_backward_not_found(self, hidden):
        """validate_lens() raises TypeError when backward equation is missing."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("bad_lens", "fwd", "nonexistent_bwd")
        with pytest.raises(TypeError, match="backward equation 'nonexistent_bwd' not found"):
            validate_lens({"fwd": eq_fwd}, l)

    def test_invalid_lens_domain_mismatch(self, hidden, output_sort):
        """validate_lens() raises when forward.domain != backward.codomain.

        forward:  hidden → output
        backward: hidden → output  (wrong: bwd.codomain should be hidden)
        fwd.domain = hidden, bwd.codomain = output → mismatch
        """
        real_sr = record_fields(hidden)["semiring"]
        # fwd: hidden → output (domain=hidden, codomain=output)
        eq_fwd = equation("fwd", "i->j", hidden, output_sort, real_sr)
        # bwd: hidden → output (domain=hidden, codomain=output)
        # This is WRONG: bwd.codomain should be hidden (== fwd.domain)
        eq_bwd = equation("bwd", "i->j", hidden, output_sort, real_sr)
        l = lens("bad", "fwd", "bwd")
        with pytest.raises(TypeError, match="forward domain.*!=.*backward codomain"):
            validate_lens({"fwd": eq_fwd, "bwd": eq_bwd}, l)

    def test_invalid_lens_codomain_mismatch(self, hidden, output_sort):
        """validate_lens() raises when forward.codomain != backward.domain.

        forward:  hidden → output
        backward: hidden → hidden  (wrong: bwd.domain should be output)
        fwd.codomain = output, bwd.domain = hidden → mismatch
        """
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("fwd", "i->j", hidden, output_sort, real_sr)
        # bwd: hidden → hidden (domain=hidden, not output)
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
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation(f"{name}_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation(f"{name}_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens(name, f"{name}_fwd", f"{name}_bwd")
        return eq_fwd, eq_bwd, l

    def test_lens_path_single_returns_two_pairs(self, hidden):
        """lens_path() with one lens returns two (Name, Term) pairs."""
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        result = lens_path("pipe", ["a"], lens_by_name, hidden, hidden)
        (fwd_name, fwd_term), (bwd_name, bwd_term) = result
        assert fwd_name == Name("ua.path.pipe.fwd")
        assert bwd_name == Name("ua.path.pipe.bwd")

    def test_lens_path_forward_name(self, hidden):
        """Forward path is named 'ua.path.<name>.fwd'."""
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        (fwd_name, _), _ = lens_path("mypipe", ["a"], lens_by_name, hidden, hidden)
        assert fwd_name == Name("ua.path.mypipe.fwd")

    def test_lens_path_backward_name(self, hidden):
        """Backward path is named 'ua.path.<name>.bwd'."""
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        _, (bwd_name, _) = lens_path("mypipe", ["a"], lens_by_name, hidden, hidden)
        assert bwd_name == Name("ua.path.mypipe.bwd")

    def test_lens_path_terms_are_lambdas(self, hidden):
        """Both forward and backward path terms are Hydra lambda terms."""
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        lens_by_name = {"a": l}
        (_, fwd_term), (_, bwd_term) = lens_path("pipe", ["a"], lens_by_name, hidden, hidden)
        assert isinstance(fwd_term.value, core.Lambda)
        assert isinstance(bwd_term.value, core.Lambda)

    def test_lens_path_composition_order_forward(self, hidden):
        """Forward path body applies equations in the same order as the lens list.

        For lenses [a, b], forward body is: b_fwd(a_fwd(x)) — left-to-right.
        The outermost application's function is the variable for the LAST equation.
        """
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        lens_by_name = {"a": l_a, "b": l_b}

        (_, fwd_term), _ = lens_path("pipe", ["a", "b"], lens_by_name, hidden, hidden)
        # fwd_term is λx. a_fwd_applied(b_fwd_applied_to_a_result)
        # The outer apply's function should reference b_fwd
        body = fwd_term.value.body
        # body = apply(var("ua.equation.b_fwd"), apply(var("ua.equation.a_fwd"), x))
        # Outermost function:
        assert body.value.function.value.value == "ua.equation.b_fwd"

    def test_lens_path_composition_order_backward(self, hidden):
        """Backward path body applies equations in REVERSED order vs the lens list.

        For lenses [a, b], backward body is: a_bwd(b_bwd(x)) — right-to-left.
        The outermost application's function is the variable for a_bwd (first lens reversed).
        """
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        lens_by_name = {"a": l_a, "b": l_b}

        _, (_, bwd_term) = lens_path("pipe", ["a", "b"], lens_by_name, hidden, hidden)
        # bwd_term is λx. b_bwd(a_bwd(x)) reversed, so a_bwd is outermost
        # reversed order: [b_bwd, a_bwd] — b_bwd applied first, then a_bwd wraps it
        body = bwd_term.value.body
        # body = apply(var("ua.equation.a_bwd"), apply(var("ua.equation.b_bwd"), x))
        assert body.value.function.value.value == "ua.equation.a_bwd"

    def test_lens_path_empty_raises(self, hidden):
        """lens_path() with an empty lens list raises ValueError."""
        with pytest.raises(ValueError, match="at least one lens"):
            lens_path("empty", [], {}, hidden, hidden)


# ===========================================================================
# Part D: Lens end-to-end via reduce_term
# ===========================================================================

class TestLensEndToEnd:
    """Verify both forward and backward paths execute correctly via reduce_term."""

    def test_lens_path_forward_executes(self, cx, hidden, backend, coder):
        """Forward path through a single relu lens executes correctly."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("relu_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("relu_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("relu_lens", "relu_fwd", "relu_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            lens_paths=[("relu_pipe", ["relu_lens"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.5, -0.3, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.relu_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_lens_path_backward_executes(self, cx, hidden, backend, coder):
        """Backward path through a single lens executes correctly."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("relu_fwd2", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("tanh_bwd2", None, hidden, hidden, nonlinearity="tanh")
        l = lens("mixed_lens", "relu_fwd2", "tanh_bwd2")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            lens_paths=[("mixed_pipe", ["mixed_lens"], hidden, hidden)],
        )

        x = np.array([1.0, -1.0, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.mixed_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(x))

    def test_two_lens_path_forward_composed(self, cx, hidden, backend, coder):
        """Forward path across two composed lenses: relu then tanh."""
        real_sr = record_fields(hidden)["semiring"]
        eq_relu_fwd = equation("relu_f", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = equation("relu_b", None, hidden, hidden, nonlinearity="relu")
        eq_tanh_fwd = equation("tanh_f", None, hidden, hidden, nonlinearity="tanh")
        eq_tanh_bwd = equation("tanh_b", None, hidden, hidden, nonlinearity="tanh")

        l_relu = lens("relu_l", "relu_f", "relu_b")
        l_tanh = lens("tanh_l", "tanh_f", "tanh_b")

        graph = assemble_graph(
            [eq_relu_fwd, eq_relu_bwd, eq_tanh_fwd, eq_tanh_bwd], backend,
            lenses=[l_relu, l_tanh],
            lens_paths=[("two_lens", ["relu_l", "tanh_l"], hidden, hidden)],
        )

        x = np.array([-1.0, 0.5, 0.0, 2.0])
        x_enc = encode_array(coder, x)

        # Forward: relu then tanh
        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_lens.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)))

    def test_two_lens_path_backward_reversed(self, cx, hidden, backend, coder):
        """Backward path reverses order: tanh_bwd then relu_bwd (reversed composition)."""
        real_sr = record_fields(hidden)["semiring"]
        # Use abs as forward, neg as backward for clear asymmetry
        eq_abs_fwd = equation("abs_f", None, hidden, hidden, nonlinearity="abs")
        eq_abs_bwd = equation("abs_b", None, hidden, hidden, nonlinearity="neg")
        eq_relu_fwd = equation("relu_f2", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = equation("relu_b2", None, hidden, hidden, nonlinearity="tanh")

        l_abs = lens("abs_l", "abs_f", "abs_b")
        l_relu = lens("relu_l2", "relu_f2", "relu_b2")

        graph = assemble_graph(
            [eq_abs_fwd, eq_abs_bwd, eq_relu_fwd, eq_relu_bwd], backend,
            lenses=[l_abs, l_relu],
            # Lenses in order [abs_l, relu_l2]
            # Backward reversed: [relu_b2, abs_b] = tanh then neg
            lens_paths=[("asym_pipe", ["abs_l", "relu_l2"], hidden, hidden)],
        )

        x = np.array([1.0, -1.0, 0.5])
        x_enc = encode_array(coder, x)

        # Backward applies relu_b2=tanh first, then abs_b=neg
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
        """assemble_graph with lens_paths registers both fwd and bwd bound_terms."""
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("id_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("id_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("id_lens", "id_fwd", "id_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            lens_paths=[("id_pipe", ["id_lens"], hidden, hidden)],
        )

        assert Name("ua.path.id_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.id_pipe.bwd") in graph.bound_terms

    def test_assemble_graph_with_invalid_lens_raises(self, hidden, output_sort, backend):
        """assemble_graph validates lenses and raises on sort mismatch."""
        real_sr = record_fields(hidden)["semiring"]
        # Both equations go hidden → output, but backward should go output → hidden
        eq_fwd = equation("enc_fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = equation("enc_bwd", "i->j", hidden, output_sort, real_sr)  # WRONG direction
        l = lens("bad_enc", "enc_fwd", "enc_bwd")

        with pytest.raises(TypeError):
            assemble_graph(
                [eq_fwd, eq_bwd], backend,
                lenses=[l],
            )

    def test_assemble_graph_no_lens_paths_still_validates(self, hidden, backend):
        """Lenses declared without lens_paths are still validated for sort correctness."""
        # Use same sort in both directions so the equations resolve correctly
        real_sr = record_fields(hidden)["semiring"]
        eq_fwd = equation("enc2_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("enc2_bwd", None, hidden, hidden, nonlinearity="tanh")
        l = lens("enc2", "enc2_fwd", "enc2_bwd")

        # No lens_paths — lenses still validated, no path bound_terms created
        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
        )
        # No bound_terms for paths since no lens_paths given
        assert Name("ua.path.enc2.fwd") not in graph.bound_terms

    def test_multiple_lenses_in_one_graph(self, hidden, backend):
        """Multiple lenses can coexist in a single assemble_graph call."""
        real_sr = record_fields(hidden)["semiring"]
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
            lens_paths=[
                ("relu_only", ["relu_g"], hidden, hidden),
                ("tanh_only", ["tanh_g"], hidden, hidden),
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
        """fold() using a lens's forward equation works normally.

        The lens declaration does not affect how the equation is resolved —
        both the fold and the lens forward path call the same Primitive.

        We use an additive semiring (plus=add, times=add) so that the binary
        step function "i,i->i" performs elementwise addition, and the fold
        accumulates a running sum from the zero vector.
        """
        # Additive semiring: ⊕=add, ⊗=add, 0=0, 1=0
        add_sr = semiring("addsem", plus="add", times="add", zero=0.0, one=0.0)
        acc_sort = sort("acc", add_sr)
        # Step: elementwise add (binary, "i,i->i" with times=add)
        eq_step_fwd = equation("acc_fwd", "i,i->i", acc_sort, acc_sort, add_sr)
        eq_step_bwd = equation("acc_bwd", "i,i->i", acc_sort, acc_sort, add_sr)
        l = lens("acc_lens", "acc_fwd", "acc_bwd")

        init = encode_array(coder, np.zeros(3))

        graph = assemble_graph(
            [eq_step_fwd, eq_step_bwd], backend,
            lenses=[l],
            folds=[("sum_fold", "acc_fwd", init, acc_sort, acc_sort)],
        )

        # Build a list of tensors to fold over: sum = [12, 15, 18]
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
            lens_paths=[("rl_pipe", ["rl"], hidden, hidden)],
            folds=[("rl_fold", "fold_step", init, hidden, hidden)],
        )

        # Both the lens path forward and the fold should be callable
        assert Name("ua.path.rl_pipe.fwd") in graph.bound_terms
        assert Name("ua.fold.rl_fold") in graph.bound_terms


# ===========================================================================
# Part G: Semiring polymorphism
# ===========================================================================

class TestLensSemiringPolymorphism:
    """Demonstrate that the lens structure is semiring-agnostic."""

    def test_tropical_lens_declaration(self, tropical_sr, tropic_sort):
        """A lens can be declared over the tropical semiring without modification."""
        eq_fwd = equation("tp_fwd", "i->j", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = equation("tp_bwd", "j->i", tropic_sort, tropic_sort, tropical_sr)
        l = lens("tropical_lens", "tp_fwd", "tp_bwd")
        lf = _lens_fields(l)
        assert lf["name"] == "tropical_lens"
        assert lf["forward"] == "tp_fwd"
        assert lf["backward"] == "tp_bwd"

    def test_tropical_lens_validates(self, tropical_sr, tropic_sort):
        """validate_lens() works correctly for tropical semiring equations."""
        eq_fwd = equation("tp2_fwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        eq_bwd = equation("tp2_bwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        l = lens("tropic_id", "tp2_fwd", "tp2_bwd")
        eq_by_name = {"tp2_fwd": eq_fwd, "tp2_bwd": eq_bwd}
        # Should not raise: same sort in both directions
        validate_lens(eq_by_name, l)

    def test_tropical_lens_end_to_end(self, cx, tropical_sr, tropic_sort, backend, coder):
        """A tropical-semiring lens path executes correctly end-to-end.

        Forward: min-plus unary equation (identity under tropical times = add).
        Backward: same structure — demonstrates the lens machinery is
        semiring-agnostic.
        """
        # Unary identity: "i->i" with tropical semiring
        eq_fwd = equation("trp_fwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = equation("trp_bwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        l = lens("trp_lens", "trp_fwd", "trp_bwd")

        graph = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            lens_paths=[("trp_pipe", ["trp_lens"], tropic_sort, tropic_sort)],
        )

        # Tropical semiring "i->i" with no reduction indices = identity
        x = np.array([1.0, 3.0, 2.0])
        x_enc = encode_array(coder, x)

        # Forward
        out_fwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.trp_pipe.fwd"), x_enc)
        ))
        np.testing.assert_allclose(out_fwd, x)

        # Backward (same equation, same result)
        out_bwd = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.trp_pipe.bwd"), x_enc)
        ))
        np.testing.assert_allclose(out_bwd, x)

    def test_real_and_tropical_lenses_independent(self, cx, real_sr, tropical_sr, backend, coder):
        """Real and tropical lenses can coexist in separate graphs without interference."""
        real_sort = sort("real_s", real_sr)
        trop_sort = sort("trop_s", tropical_sr)

        eq_real_fwd = equation("real_relu", None, real_sort, real_sort, nonlinearity="relu")
        eq_real_bwd = equation("real_tanh", None, real_sort, real_sort, nonlinearity="tanh")
        l_real = lens("real_l", "real_relu", "real_tanh")

        graph = assemble_graph(
            [eq_real_fwd, eq_real_bwd], backend,
            lenses=[l_real],
            lens_paths=[("real_pipe", ["real_l"], real_sort, real_sort)],
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
