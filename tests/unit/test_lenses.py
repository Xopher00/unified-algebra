"""Unit tests for lens declaration, structure, and path composition.

Covers Lens record fields, PathSpec._build_lens_pair() wiring,
LensFoldIntegration, OpticPrimitives, OpticPathStructure, and LensPathRouting.
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name

from unialg import (
    Sort, ProductSort, Equation,
    PathSpec, FoldSpec,
)
from unialg.assembly.graph import assemble_graph
from unialg.algebra.sort import Lens
from unialg.assembly.legacy._primitives import lens_fwd_primitive, lens_bwd_primitive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return Sort("residual", real_sr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from hydra.dsl.python import Right
from hydra.reduction import reduce_term
from unialg.terms import tensor_coder


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



# ===========================================================================
# Part A: Lens declaration
# ===========================================================================

class TestLensDeclaration:
    """Verify Lens() produces the correct Hydra record structure."""

    def test_lens_record_type_name(self):
        """Lens() produces a record with the Lens._type_name type."""
        l = Lens("linear", "fwd", "bwd")
        assert l.term.value.type_name == Lens._type_name

    def test_lens_record_name_field(self):
        """The 'name' field of the lens record matches the given name."""
        l = Lens("my_lens", "forward_eq", "backward_eq")
        assert l.name == "my_lens"

    def test_lens_record_forward_field(self):
        """The 'forward' field stores the forward equation name."""
        l = Lens("linear", "linear_fwd", "linear_bwd")
        assert l.forward == "linear_fwd"

    def test_lens_record_backward_field(self):
        """The 'backward' field stores the backward equation name."""
        l = Lens("linear", "linear_fwd", "linear_bwd")
        assert l.backward == "linear_bwd"

    def test_lens_without_residual_stores_unit(self):
        """When no residual_sort is provided, residual_sort property returns None."""
        l = Lens("linear", "fwd", "bwd")
        assert l.residual_sort is None

    def test_lens_with_residual_stores_sort(self, hidden):
        """Lens() with residual_sort stores the sort term in the residualSort field."""
        import hydra.core as core
        l = Lens("linear", "fwd", "bwd", residual_sort=hidden)
        t = Sort.from_term(l.residual_sort).type_
        assert t.value.function == core.TypeVariable(core.Name("ua.sort.hidden"))


# ===========================================================================
# Part C: Lens path structure
# ===========================================================================

class TestLensPath:
    """Verify PathSpec._build_lens_pair() produces correctly wired forward and backward paths."""

    def _make_id_lens(self, name, hidden):
        """Make a self-inverse lens (identity-like, hidden→hidden in both dirs)."""
        eq_fwd = Equation(f"{name}_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation(f"{name}_bwd", None, hidden, hidden, nonlinearity="relu")
        l = Lens(name, f"{name}_fwd", f"{name}_bwd")
        return eq_fwd, eq_bwd, l

    def test_lens_path_single_returns_two_pairs(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        result = PathSpec("pipe", ["a_fwd"], None, None, bwd_eq_names=["a_bwd"])._build_lens_pair()
        (fwd_name, fwd_term), (bwd_name, bwd_term) = result
        assert fwd_name == Name("ua.path.pipe.fwd")
        assert bwd_name == Name("ua.path.pipe.bwd")

    def test_lens_path_forward_name(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        (fwd_name, _), _ = PathSpec("mypipe", ["a_fwd"], None, None, bwd_eq_names=["a_bwd"])._build_lens_pair()
        assert fwd_name == Name("ua.path.mypipe.fwd")

    def test_lens_path_backward_name(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        _, (bwd_name, _) = PathSpec("mypipe", ["a_fwd"], None, None, bwd_eq_names=["a_bwd"])._build_lens_pair()
        assert bwd_name == Name("ua.path.mypipe.bwd")

    def test_lens_path_terms_are_lambdas(self, hidden):
        eq_fwd, eq_bwd, l = self._make_id_lens("a", hidden)
        (_, fwd_term), (_, bwd_term) = PathSpec("pipe", ["a_fwd"], None, None, bwd_eq_names=["a_bwd"])._build_lens_pair()
        assert isinstance(fwd_term.value, core.Lambda)
        assert isinstance(bwd_term.value, core.Lambda)

    def test_lens_path_composition_order_forward(self, hidden):
        """For lenses [a, b], forward body is: b_fwd(a_fwd(x)) — left-to-right."""
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        (_, fwd_term), _ = PathSpec("pipe", ["a_fwd", "b_fwd"], None, None, bwd_eq_names=["a_bwd", "b_bwd"])._build_lens_pair()
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.b_fwd"

    def test_lens_path_composition_order_backward(self, hidden):
        """For lenses [a, b], backward body is: a_bwd(b_bwd(x)) — right-to-left."""
        eq_a_fwd, eq_a_bwd, l_a = self._make_id_lens("a", hidden)
        eq_b_fwd, eq_b_bwd, l_b = self._make_id_lens("b", hidden)
        _, (_, bwd_term) = PathSpec("pipe", ["a_fwd", "b_fwd"], None, None, bwd_eq_names=["a_bwd", "b_bwd"])._build_lens_pair()
        body = bwd_term.value.body
        assert body.value.function.value.value == "ua.equation.a_bwd"

    def test_lens_path_empty_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one lens"):
            PathSpec("empty", [], None, None, bwd_eq_names=["x"])._build_lens_pair()


# ===========================================================================
# Part F: Lens fold integration
# ===========================================================================

class TestLensFoldIntegration:
    """A lens's forward equation can serve as a fold step without interference."""

    def test_lens_coexists_with_fold_and_path(self, cx, hidden, backend, coder):
        """A graph can contain lenses, lens_paths, and folds simultaneously."""
        real_sr = hidden.semiring
        eq_relu_fwd = Equation("rl_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_relu_bwd = Equation("rl_bwd", None, hidden, hidden, nonlinearity="relu")
        eq_step = Equation("fold_step", "i,i->i", hidden, hidden, real_sr)
        l = Lens("rl", "rl_fwd", "rl_bwd")

        init = encode_array(coder, np.zeros(4))
        graph, *_ = assemble_graph(
            [eq_relu_fwd, eq_relu_bwd, eq_step], backend,
            lenses=[l],
            specs=[
                PathSpec("rl_pipe", ["rl_fwd"], hidden, hidden, bwd_eq_names=["rl_bwd"]),
                FoldSpec("rl_fold", "fold_step", init, hidden, hidden),
            ],
        )

        assert Name("ua.path.rl_pipe.fwd") in graph.bound_terms
        assert Name("ua.fold.rl_fold") in graph.primitives


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
# Part J: Optic path structure (lens_path with has_residual)
# ===========================================================================

def _make_residual_lens_path(name, lens_names, fwd_names, bwd_names, residual_sort):
    """Helper: build lens records with residual and call lens_path."""
    lens_terms = {
        ln: Lens(ln, forward=fwd, backward=bwd, residual_sort=residual_sort)
        for ln, fwd, bwd in zip(lens_names, fwd_names, bwd_names)
    }
    return PathSpec(name, fwd_names, None, None, bwd_eq_names=bwd_names, has_residual=True)._build_lens_pair()


class TestOpticPathStructure:
    """lens_path with has_residual=True builds correctly named lambda terms."""

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
    """lens_path routes to threaded path only for multi-optic with has_residual=True."""

    def _make_optic_lens(self, backend, name, hidden, residual_sort):
        prod = ProductSort([hidden, residual_sort])
        real_sr = list(hidden.term.value.fields)[1].term

        backend.unary_ops[f"pair_{name}"] = pair_relu
        backend.unary_ops[f"bwd_{name}"] = lambda p: p[0] * 0.5

        eq_fwd = Equation(f"{name}_fwd", None, hidden, prod,
                          nonlinearity=f"pair_{name}")
        eq_bwd = Equation(f"{name}_bwd", None, prod, hidden,
                          nonlinearity=f"bwd_{name}")
        l = Lens(name, f"{name}_fwd", f"{name}_bwd", residual_sort=residual_sort)
        return eq_fwd, eq_bwd, l

    def test_single_optic_with_residual_uses_plain_path(self, hidden, residual_sort):
        """Single optic with has_residual=False uses plain path."""
        prod = ProductSort([hidden, residual_sort])
        eq_fwd = Equation("so_fwd", None, hidden, prod, nonlinearity="relu")
        eq_bwd = Equation("so_bwd", None, prod, hidden, nonlinearity="relu")
        l = Lens("so", "so_fwd", "so_bwd", residual_sort=residual_sort)

        (fwd_name, fwd_term), (bwd_name, bwd_term) = PathSpec(
            "so_pipe", ["so_fwd"], None, None, bwd_eq_names=["so_bwd"]
        )._build_lens_pair()
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.so_fwd"

    def test_multi_optic_with_residual_uses__lens_path_threaded(self, hidden, residual_sort):
        """Multi-optic with has_residual=True uses _lens_path_threaded (optic_fwd/bwd)."""
        prod = ProductSort([hidden, residual_sort])
        eq_fwd_a = Equation("mo_fwd_a", None, hidden, prod, nonlinearity="relu")
        eq_bwd_a = Equation("mo_bwd_a", None, prod, hidden, nonlinearity="relu")
        eq_fwd_b = Equation("mo_fwd_b", None, hidden, prod, nonlinearity="relu")
        eq_bwd_b = Equation("mo_bwd_b", None, prod, hidden, nonlinearity="relu")
        l_a = Lens("mo_a", "mo_fwd_a", "mo_bwd_a", residual_sort=residual_sort)
        l_b = Lens("mo_b", "mo_fwd_b", "mo_bwd_b", residual_sort=residual_sort)

        (fwd_name, fwd_term), (bwd_name, bwd_term) = PathSpec(
            "mo_pipe", ["mo_fwd_a", "mo_fwd_b"], None, None,
            bwd_eq_names=["mo_bwd_a", "mo_bwd_b"], has_residual=True
        )._build_lens_pair()
        body = fwd_term.value.body
        inner = body.value.function
        prim_ref = inner.value.function
        assert prim_ref.value.value == "ua.prim.lens_fwd"

    def test_plain_lens_without_residual_unaffected(self, hidden):
        """Plain lens (no has_residual) still uses the original path composition."""
        eq_fwd = Equation("pl_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("pl_bwd", None, hidden, hidden, nonlinearity="relu")
        l = Lens("pl", "pl_fwd", "pl_bwd")

        (fwd_name, fwd_term), (bwd_name, bwd_term) = PathSpec(
            "pl_pipe", ["pl_fwd"], None, None, bwd_eq_names=["pl_bwd"]
        )._build_lens_pair()
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.pl_fwd"
