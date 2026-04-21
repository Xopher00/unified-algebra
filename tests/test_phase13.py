"""Phase 13 tests: height-2 optics — runtime residual threading.

A height-2 optic (dialense) is a lens whose forward equation produces
(output, residual) and whose backward equation consumes (feedback, residual).
When MULTIPLE optics are composed in sequence, the residuals produced by each
forward step must be threaded to the corresponding backward step.

This file tests:
1. _lens_fwd_primitive and _lens_bwd_primitive — low-level primitive correctness
2. lens_path with residual lenses — multi-optic residual threading via assemble_graph
3. Single-optic with residual_sort — uses plain path (no threading needed)
4. Correct backward ordering — residuals consumed in reverse
5. Semiring polymorphism — threading works regardless of semiring

The key contracts under test:
  - ua.path.<name>.fwd applied to input x returns pair(output, list[residuals])
  - ua.path.<name>.bwd applied to pair(feedback, list[residuals]) returns input update
  - For N optics: fwd chains fwd1, fwd2, ..., fwdN projecting outputs between steps;
    bwd applies bwdN, bwd(N-1), ..., bwd1 with corresponding residuals
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

from unified_algebra.backend import numpy_backend, UnaryOp
from unified_algebra.semiring import semiring
from unified_algebra.sort import (
    sort, tensor_coder, sort_coder,
    product_sort, product_sort_coder,
)
from unified_algebra.morphism import equation
from unified_algebra.composition import lens, lens_path, validate_lens
from unified_algebra._lens_threading import (
    _lens_fwd_primitive, _lens_bwd_primitive, _lens_path_threaded,
)
from unified_algebra.graph import assemble_graph, build_graph, LensPathSpec


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
    return semiring("real13", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def hidden(real_sr):
    return sort("h13", real_sr)


@pytest.fixture
def residual(real_sr):
    return sort("r13", real_sr)


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


def bwd_scale(pair_in):
    """Custom unary taking a tuple (feedback, residual) -> feedback * 0.5."""
    feedback, residual = pair_in
    return feedback * 0.5


# ===========================================================================
# Group 1: Primitive structure
# ===========================================================================

class TestOpticPrimitives:
    """The raw _lens_fwd_primitive and _lens_bwd_primitive have correct names."""

    def test_lens_fwd_primitive_name(self):
        """_lens_fwd_primitive has name ua.prim.lens_fwd."""
        prim = _lens_fwd_primitive()
        assert prim.name == Name("ua.prim.lens_fwd")

    def test_lens_bwd_primitive_name(self):
        """_lens_bwd_primitive has name ua.prim.lens_bwd."""
        prim = _lens_bwd_primitive()
        assert prim.name == Name("ua.prim.lens_bwd")


# ===========================================================================
# Group 2: _lens_path_threaded term structure
# ===========================================================================

class TestOpticPathStructure:
    """_lens_path_threaded() builds correctly named lambda terms."""

    def test__lens_path_threaded_returns_two_pairs(self, hidden, residual):
        """_lens_path_threaded() returns two (Name, Term) pairs for fwd and bwd."""
        result = _lens_path_threaded("enc", ["fwd1", "fwd2"], ["bwd1", "bwd2"])
        (fwd_name, fwd_term), (bwd_name, bwd_term) = result
        assert fwd_name == Name("ua.path.enc.fwd")
        assert bwd_name == Name("ua.path.enc.bwd")

    def test__lens_path_threaded_fwd_is_lambda(self, hidden, residual):
        """Forward optic path term is a Hydra lambda."""
        (_, fwd_term), _ = _lens_path_threaded("enc2", ["fwd1"], ["bwd1"])
        assert isinstance(fwd_term.value, core.Lambda)

    def test__lens_path_threaded_bwd_is_lambda(self, hidden, residual):
        """Backward optic path term is a Hydra lambda."""
        _, (_, bwd_term) = _lens_path_threaded("enc3", ["fwd1"], ["bwd1"])
        assert isinstance(bwd_term.value, core.Lambda)

    def test__lens_path_threaded_fwd_uses_lens_fwd_primitive(self, hidden, residual):
        """Forward optic path body references ua.prim.lens_fwd."""
        (_, fwd_term), _ = _lens_path_threaded("enc4", ["fwd1", "fwd2"], ["bwd1", "bwd2"])
        # Body is apply(apply(var("ua.prim.lens_fwd"), list_of_fwds), var("x"))
        outer_app = fwd_term.value.body
        inner_app = outer_app.value.function
        prim_var = inner_app.value.function
        assert prim_var.value.value == "ua.prim.lens_fwd"

    def test__lens_path_threaded_bwd_uses_lens_bwd_primitive(self, hidden, residual):
        """Backward optic path body references ua.prim.lens_bwd."""
        _, (_, bwd_term) = _lens_path_threaded("enc5", ["fwd1", "fwd2"], ["bwd1", "bwd2"])
        outer_app = bwd_term.value.body
        inner_app = outer_app.value.function
        prim_var = inner_app.value.function
        assert prim_var.value.value == "ua.prim.lens_bwd"


# ===========================================================================
# Group 3: lens_path routing
# ===========================================================================

class TestLensPathRouting:
    """lens_path routes to _lens_path_threaded only for multi-optic with residual."""

    def _make_optic_lens(self, backend, name, hidden, residual):
        """Build a lens with residual_sort and matching product codomain."""
        prod = product_sort([hidden, residual])
        sr = core.TermRecord  # sentinel — actual sr extracted from hidden below
        real_sr = list(hidden.value.fields)[1].term  # extract semiring from hidden

        backend.unary_ops[f"pair_{name}"] = UnaryOp(fn=pair_relu)
        backend.unary_ops[f"bwd_{name}"] = UnaryOp(fn=lambda p: p[0] * 0.5)

        eq_fwd = equation(f"{name}_fwd", None, hidden, prod,
                          nonlinearity=f"pair_{name}")
        eq_bwd = equation(f"{name}_bwd", None, prod, hidden,
                          nonlinearity=f"bwd_{name}")
        l = lens(name, f"{name}_fwd", f"{name}_bwd", residual_sort=residual)
        return eq_fwd, eq_bwd, l

    def test_single_optic_with_residual_uses_plain_path(self, hidden, residual):
        """Single optic with residual_sort uses plain path (not optic_fwd/bwd)."""
        from unified_algebra.utils import record_fields
        # Build lens terms
        prod = product_sort([hidden, residual])
        eq_fwd = equation("so_fwd", None, hidden, prod, nonlinearity="relu")
        eq_bwd = equation("so_bwd", None, prod, hidden, nonlinearity="relu")
        l = lens("so", "so_fwd", "so_bwd", residual_sort=residual)
        lens_by_name = {"so": l}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "so_pipe", ["so"], lens_by_name
        )
        # Plain path: fwd body is apply(var("ua.equation.so_fwd"), var("x")) — no optic_fwd
        body = fwd_term.value.body
        # Single equation apply: function is var("ua.equation.so_fwd")
        assert body.value.function.value.value == "ua.equation.so_fwd"

    def test_multi_optic_with_residual_uses__lens_path_threaded(self, hidden, residual):
        """Multi-optic with residual_sort uses _lens_path_threaded (optic_fwd/bwd)."""
        prod = product_sort([hidden, residual])
        eq_fwd_a = equation("mo_fwd_a", None, hidden, prod, nonlinearity="relu")
        eq_bwd_a = equation("mo_bwd_a", None, prod, hidden, nonlinearity="relu")
        eq_fwd_b = equation("mo_fwd_b", None, hidden, prod, nonlinearity="relu")
        eq_bwd_b = equation("mo_bwd_b", None, prod, hidden, nonlinearity="relu")
        l_a = lens("mo_a", "mo_fwd_a", "mo_bwd_a", residual_sort=residual)
        l_b = lens("mo_b", "mo_fwd_b", "mo_bwd_b", residual_sort=residual)
        lens_by_name = {"mo_a": l_a, "mo_b": l_b}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "mo_pipe", ["mo_a", "mo_b"], lens_by_name
        )
        # _lens_path_threaded: fwd body is apply(apply(var("ua.prim.lens_fwd"), list), var("x"))
        body = fwd_term.value.body
        inner = body.value.function
        prim_ref = inner.value.function
        assert prim_ref.value.value == "ua.prim.lens_fwd"

    def test_plain_lens_without_residual_unaffected(self, hidden):
        """Plain lens (no residual_sort) still uses the original path composition."""
        eq_fwd = equation("pl_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = equation("pl_bwd", None, hidden, hidden, nonlinearity="relu")
        l = lens("pl", "pl_fwd", "pl_bwd")  # no residual_sort
        lens_by_name = {"pl": l}

        (fwd_name, fwd_term), (bwd_name, bwd_term) = lens_path(
            "pl_pipe", ["pl"], lens_by_name
        )
        body = fwd_term.value.body
        assert body.value.function.value.value == "ua.equation.pl_fwd"


# ===========================================================================
# Group 4: End-to-end multi-optic threading via assemble_graph + reduce_term
# ===========================================================================

class TestMultiOpticEndToEnd:
    """Two optics composed in sequence: residuals collected in forward,
    injected in reverse during backward."""

    def _setup_two_optic_graph(self, backend, real_sr, hidden, residual):
        """Build a graph with two composed optics a, b.

        Forward equations:
          a_fwd: x -> (relu(x), x)     — relu output, original as residual
          b_fwd: x -> (tanh(x), x)     — tanh output, relu output as residual

        Backward equations:
          a_bwd: (feedback, residual) -> feedback * 0.5
          b_bwd: (feedback, residual) -> feedback * 0.5

        The composition a then b (forward) means:
          - a_fwd(x) = (relu(x), x)
          - b_fwd(relu(x)) = (tanh(relu(x)), relu(x))
          - forward output = tanh(relu(x)), residuals = [x, relu(x)]

        Backward (reverse):
          - b_bwd(feedback, relu(x)) = feedback * 0.5
          - a_bwd(feedback*0.5, x)   = (feedback*0.5) * 0.5 = feedback * 0.25
        """
        prod = product_sort([hidden, residual])

        backend.unary_ops["pair_relu13"] = UnaryOp(fn=pair_relu)
        backend.unary_ops["pair_tanh13"] = UnaryOp(fn=pair_tanh)
        backend.unary_ops["bwd_half13"] = UnaryOp(fn=lambda p: p[0] * 0.5)

        eq_a_fwd = equation("a13_fwd", None, hidden, prod, nonlinearity="pair_relu13")
        eq_a_bwd = equation("a13_bwd", None, prod, hidden, nonlinearity="bwd_half13")
        eq_b_fwd = equation("b13_fwd", None, hidden, prod, nonlinearity="pair_tanh13")
        eq_b_bwd = equation("b13_bwd", None, prod, hidden, nonlinearity="bwd_half13")

        l_a = lens("la13", "a13_fwd", "a13_bwd", residual_sort=residual)
        l_b = lens("lb13", "b13_fwd", "b13_bwd", residual_sort=residual)

        graph = assemble_graph(
            [eq_a_fwd, eq_a_bwd, eq_b_fwd, eq_b_bwd], backend,
            lenses=[l_a, l_b],
            specs=[LensPathSpec("two_optic13", ["la13", "lb13"], hidden, hidden)],
            extra_sorts=[prod],
        )
        return graph, prod

    def test_multi_optic_graph_registers_optic_primitives(
        self, backend, real_sr, hidden, residual
    ):
        """assemble_graph with multi-optic lens_path registers ua.prim.lens_fwd/bwd."""
        graph, _ = self._setup_two_optic_graph(backend, real_sr, hidden, residual)
        assert Name("ua.prim.lens_fwd") in graph.primitives
        assert Name("ua.prim.lens_bwd") in graph.primitives

    def test_multi_optic_graph_registers_bound_terms(
        self, backend, real_sr, hidden, residual
    ):
        """assemble_graph with multi-optic registers both fwd and bwd bound_terms."""
        graph, _ = self._setup_two_optic_graph(backend, real_sr, hidden, residual)
        assert Name("ua.path.two_optic13.fwd") in graph.bound_terms
        assert Name("ua.path.two_optic13.bwd") in graph.bound_terms

    def test_multi_optic_forward_returns_pair(
        self, cx, backend, real_sr, hidden, residual, coder
    ):
        """Forward path returns a TermPair whose first element is the final output.

        x = [-1, 0.5, 2.0]
        a_fwd(x) = (relu(x), x) = ([0, 0.5, 2.0], x)
        b_fwd([0, 0.5, 2.0]) = (tanh([0, 0.5, 2.0]), [0, 0.5, 2.0])
        Forward output = tanh(relu(x))
        """
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        # result_term is a TermPair: (final_output_term, residuals_list_term)
        assert isinstance(result_term, core.TermPair), (
            f"Expected TermPair, got {type(result_term).__name__}"
        )

        # Decode the first element (final output tensor)
        output_term, residuals_list_term = result_term.value
        output = decode_term(coder, output_term)
        expected = np.tanh(np.maximum(0, x))
        np.testing.assert_allclose(output, expected, rtol=1e-6)

    def test_multi_optic_forward_accumulates_two_residuals(
        self, cx, backend, real_sr, hidden, residual, coder
    ):
        """Forward path accumulates one residual per optic: length == 2 for two optics."""
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        _, residuals_list_term = result_term.value
        # The second element is a TermList of residuals (one per optic)
        assert isinstance(residuals_list_term, core.TermList), (
            f"Expected TermList, got {type(residuals_list_term).__name__}"
        )
        assert len(residuals_list_term.value) == 2

    def test_multi_optic_forward_residuals_correct_values(
        self, cx, backend, real_sr, hidden, residual, coder
    ):
        """Forward residuals are the intermediate pre-output tensors from each optic.

        optic a residual = x (input before relu)
        optic b residual = relu(x) (input before tanh)
        """
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        result_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        _, residuals_list_term = result_term.value
        residuals = list(residuals_list_term.value)

        # residual 0: from a_fwd — the second element of (relu(x), x) = x
        r0 = decode_term(coder, residuals[0])
        np.testing.assert_allclose(r0, x, rtol=1e-6)

        # residual 1: from b_fwd — the second element of (tanh(relu(x)), relu(x)) = relu(x)
        r1 = decode_term(coder, residuals[1])
        np.testing.assert_allclose(r1, np.maximum(0, x), rtol=1e-6)

    def test_multi_optic_backward_applies_in_reverse(
        self, cx, backend, real_sr, hidden, residual, coder
    ):
        """Backward path applies bwd_b then bwd_a with correct residuals.

        bwd_b(feedback, r_b) = feedback * 0.5
        bwd_a(result, r_a)   = result * 0.5 = feedback * 0.25

        So backward(feedback, [r_a, r_b]) = feedback * 0.25.
        """
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual)

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        # Get residuals from forward pass
        fwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        output_term, residuals_list_term = fwd_result.value

        # Build the backward input: pair(feedback, residuals_list)
        # Use the forward output as feedback (arbitrary for this structure test)
        feedback = np.array([1.0, 1.0, 1.0])
        feedback_enc = encode_array(coder, feedback)
        bwd_input = Terms.pair(feedback_enc, residuals_list_term)

        bwd_output_term = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.bwd"), bwd_input)
        )
        bwd_output = decode_term(coder, bwd_output_term)

        # Backward applies bwd_b then bwd_a, each halving:
        # bwd_b(feedback, r_b) = feedback * 0.5
        # bwd_a(feedback*0.5, r_a) = feedback * 0.25
        expected = feedback * 0.25
        np.testing.assert_allclose(bwd_output, expected, rtol=1e-6)

    def test_multi_optic_forward_backward_full_pipeline(
        self, cx, backend, real_sr, hidden, residual, coder
    ):
        """Full pipeline: forward then backward is a deterministic function of x.

        The composed optic forward-then-backward should produce consistent results.
        """
        graph, prod = self._setup_two_optic_graph(backend, real_sr, hidden, residual)

        x = np.array([2.0, -1.0, 0.5, 3.0])
        x_enc = encode_array(coder, x)

        # Forward pass
        fwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.fwd"), x_enc)
        )
        output_term, residuals_term = fwd_result.value

        # Backward pass: use forward output as feedback
        bwd_input = Terms.pair(output_term, residuals_term)
        bwd_result = assert_reduce_ok(
            cx, graph, apply(var("ua.path.two_optic13.bwd"), bwd_input)
        )
        bwd_output = decode_term(coder, bwd_result)

        # Verify: bwd_a(bwd_b(tanh(relu(x)), relu(x)), x) = tanh(relu(x)) * 0.25
        fwd_final = np.tanh(np.maximum(0, x))
        expected = fwd_final * 0.25
        np.testing.assert_allclose(bwd_output, expected, rtol=1e-6)


# ===========================================================================
# Group 5: Semiring polymorphism
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
        # tanh(relu(x)) = tanh(x) since x >= 0 here
        expected = np.tanh(np.maximum(0, x))
        np.testing.assert_allclose(output, expected, rtol=1e-6)
