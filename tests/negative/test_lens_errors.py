"""Negative tests for lens validation errors and assemble_graph integration failures.

Covers TestLensValidation (sort contract violations) and
TestAssembleGraphWithLenses (integration-level error paths).
"""

import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.terms import apply, var

from unialg import (
    NumpyBackend, Semiring, Sort,
    ProductSort, Equation,
    FoldSpec, PathSpec,
)
from unialg.assembly.graph import build_graph, assemble_graph
from unialg.assembly.legacy.compositions import PathComposition, FanComposition
from unialg.algebra.sort import Lens
from unialg.assembly._primitives import lens_fwd_primitive, lens_bwd_primitive
from conftest import build_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def residual_sort(real_sr):
    return Sort("residual", real_sr)


# ===========================================================================
# Part B: Lens validation via PathSpec
# ===========================================================================

class TestLensValidation:
    """Verify PathSpec.validate() enforces the bidirectionality sort contract."""

    def _make_matching_pair(self, hidden, output_sort):
        """Build a valid fwd (hidden→output) + bwd (output→hidden) pair."""
        real_sr = hidden.semiring
        eq_fwd = Equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = Equation("bwd", "j->i", output_sort, hidden, real_sr)
        return eq_fwd, eq_bwd

    def test_valid_lens_passes(self, hidden, output_sort):
        eq_fwd, eq_bwd = self._make_matching_pair(hidden, output_sort)
        eq_by_name = {"fwd": eq_fwd, "bwd": eq_bwd}
        spec = PathSpec("test", ["fwd"], hidden, output_sort, bwd_eq_names=["bwd"])
        spec.validate(eq_by_name, build_schema(eq_by_name))

    def test_valid_lens_same_sort_passes(self, hidden):
        eq_fwd = Equation("fwd_id", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("bwd_id", None, hidden, hidden, nonlinearity="relu")
        eq_by_name = {"fwd_id": eq_fwd, "bwd_id": eq_bwd}
        spec = PathSpec("test", ["fwd_id"], hidden, hidden, bwd_eq_names=["bwd_id"])
        spec.validate(eq_by_name, build_schema(eq_by_name))

    def test_invalid_lens_forward_not_found(self, hidden):
        eq_bwd = Equation("bwd", None, hidden, hidden, nonlinearity="relu")
        spec = PathSpec("bad_lens", ["nonexistent_fwd"], hidden, hidden, bwd_eq_names=["bwd"])
        with pytest.raises((TypeError, ValueError), match="nonexistent_fwd"):
            ebn = {"bwd": eq_bwd}
            spec.validate(ebn, build_schema(ebn))

    def test_invalid_lens_backward_not_found(self, hidden):
        eq_fwd = Equation("fwd", None, hidden, hidden, nonlinearity="relu")
        spec = PathSpec("bad_lens", ["fwd"], hidden, hidden, bwd_eq_names=["nonexistent_bwd"])
        with pytest.raises((TypeError, ValueError), match="nonexistent_bwd"):
            ebn = {"fwd": eq_fwd}
            spec.validate(ebn, build_schema(ebn))

    def test_invalid_lens_domain_mismatch(self, hidden, output_sort):
        real_sr = hidden.semiring
        # both fwd and bwd go hidden->output, so fwd.domain != bwd.codomain
        eq_fwd = Equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = Equation("bwd", "i->j", hidden, output_sort, real_sr)
        spec = PathSpec("bad", ["fwd"], hidden, output_sort, bwd_eq_names=["bwd"])
        with pytest.raises(TypeError):
            ebn = {"fwd": eq_fwd, "bwd": eq_bwd}
            spec.validate(ebn, build_schema(ebn))

    def test_invalid_lens_codomain_mismatch(self, hidden, output_sort):
        real_sr = hidden.semiring
        # fwd: hidden->output, bwd: hidden->hidden; fwd.codomain != bwd.domain
        eq_fwd = Equation("fwd", "i->j", hidden, output_sort, real_sr)
        eq_bwd = Equation("bwd", None, hidden, hidden, nonlinearity="relu")
        spec = PathSpec("bad", ["fwd"], hidden, output_sort, bwd_eq_names=["bwd"])
        with pytest.raises(TypeError):
            ebn = {"fwd": eq_fwd, "bwd": eq_bwd}
            spec.validate(ebn, build_schema(ebn))


# ===========================================================================
# Part E: Full assemble_graph integration with lenses
# ===========================================================================

class TestAssembleGraphWithLenses:
    """Integration tests: lenses wired through assemble_graph."""

    def test_assemble_graph_registers_both_paths(self, hidden, backend):
        eq_fwd = Equation("id_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("id_bwd", None, hidden, hidden, nonlinearity="relu")
        l = Lens("id_lens", "id_fwd", "id_bwd")

        graph, *_ = assemble_graph(
            [eq_fwd, eq_bwd], backend,
            lenses=[l],
            specs=[PathSpec("id_pipe", ["id_fwd"], hidden, hidden, bwd_eq_names=["id_bwd"])],
        )

        assert Name("ua.path.id_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.id_pipe.bwd") in graph.bound_terms

    def test_assemble_graph_with_invalid_lens_raises(self, hidden, output_sort, backend):
        real_sr = hidden.semiring
        # Both fwd and bwd go hidden->output: bidi sort contract violated
        eq_fwd = Equation("enc_fwd", None, hidden, output_sort, real_sr, nonlinearity="relu")
        eq_bwd = Equation("enc_bwd", None, hidden, output_sort, real_sr, nonlinearity="relu")
        l = Lens("bad_enc", "enc_fwd", "enc_bwd")

        with pytest.raises(TypeError):
            assemble_graph(
                [eq_fwd, eq_bwd], backend,
                lenses=[l],
                specs=[PathSpec("bad_enc_pipe", ["enc_fwd"], hidden, output_sort, bwd_eq_names=["enc_bwd"])],
            )

    def test_assemble_graph_no_lens_paths_still_validates(self, hidden, backend):
        eq_fwd = Equation("enc2_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("enc2_bwd", None, hidden, hidden, nonlinearity="tanh")
        l = Lens("enc2", "enc2_fwd", "enc2_bwd")

        graph, *_ = assemble_graph([eq_fwd, eq_bwd], backend, lenses=[l])
        assert Name("ua.path.enc2.fwd") not in graph.bound_terms

    def test_multiple_lenses_in_one_graph(self, hidden, backend):
        eqs = [
            Equation("relu_g_fwd", None, hidden, hidden, nonlinearity="relu"),
            Equation("relu_g_bwd", None, hidden, hidden, nonlinearity="relu"),
            Equation("tanh_g_fwd", None, hidden, hidden, nonlinearity="tanh"),
            Equation("tanh_g_bwd", None, hidden, hidden, nonlinearity="tanh"),
        ]
        l_relu = Lens("relu_g", "relu_g_fwd", "relu_g_bwd")
        l_tanh = Lens("tanh_g", "tanh_g_fwd", "tanh_g_bwd")

        graph, *_ = assemble_graph(
            eqs, backend,
            lenses=[l_relu, l_tanh],
            specs=[
                PathSpec("relu_only", ["relu_g_fwd"], hidden, hidden, bwd_eq_names=["relu_g_bwd"]),
                PathSpec("tanh_only", ["tanh_g_fwd"], hidden, hidden, bwd_eq_names=["tanh_g_bwd"]),
            ],
        )

        assert Name("ua.path.relu_only.fwd") in graph.bound_terms
        assert Name("ua.path.relu_only.bwd") in graph.bound_terms
        assert Name("ua.path.tanh_only.fwd") in graph.bound_terms
        assert Name("ua.path.tanh_only.bwd") in graph.bound_terms
