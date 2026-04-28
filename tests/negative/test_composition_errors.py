"""Composition negative tests: validation errors for path and fan."""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.terms import apply, var

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    PathSpec, FanSpec, ParallelSpec,
)
from unialg.assembly.graph import build_graph, assemble_graph
from unialg.assembly.compositions import PathComposition, FanComposition
from conftest import encode_array, decode_term, assert_reduce_ok, build_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def tropical_sort(tropical_sr):
    return Sort("tropic", tropical_sr)


# ---------------------------------------------------------------------------
# Path: validation
# ---------------------------------------------------------------------------

class TestPathValidation:

    def test_valid_path(self, real_sr, hidden):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")
        ebn = {"a": eq_a, "b": eq_b}
        PathSpec("_", ["a", "b"], hidden, hidden).validate(ebn, build_schema(ebn))

    def test_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="'a' domain mismatch"):
            ebn = {"a": eq_a}
            PathSpec("_", ["a"], output_sort, hidden).validate(ebn, build_schema(ebn, [output_sort]))

    def test_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="'a' codomain mismatch"):
            ebn = {"a": eq_a}
            PathSpec("_", ["a"], hidden, output_sort).validate(ebn, build_schema(ebn, [output_sort]))

    def test_junction_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="relu")
        with pytest.raises(TypeError, match="Attempted to unify schema names"):
            ebn = {"a": eq_a, "b": eq_b}
            PathSpec("_", ["a", "b"], hidden, hidden).validate(ebn, build_schema(ebn))

    def test_cross_semiring_path(self, real_sr, hidden, tropical_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, tropical_sort, tropical_sort, nonlinearity="relu")
        with pytest.raises(TypeError, match="Attempted to unify schema names"):
            ebn = {"a": eq_a, "b": eq_b}
            PathSpec("_", ["a", "b"], hidden, tropical_sort).validate(ebn, build_schema(ebn))


# ---------------------------------------------------------------------------
# Fan: validation
# ---------------------------------------------------------------------------

class TestFanValidation:

    def test_valid_fan(self, real_sr, hidden):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")
        eq_m = Equation("m", "ij,ij->ij", hidden, hidden, real_sr)
        ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
        FanSpec("_", ["a", "b"], ["m"], hidden, hidden).validate(ebn, build_schema(ebn))

    def test_branch_domain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, output_sort, hidden, nonlinearity="tanh")
        eq_m = Equation("m", "ij,ij->ij", hidden, hidden, real_sr)
        with pytest.raises(TypeError, match="'b' domain mismatch"):
            ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
            FanSpec("_", ["a", "b"], ["m"], hidden, hidden).validate(ebn, build_schema(ebn))

    def test_merge_codomain_mismatch(self, real_sr, hidden, output_sort):
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_m = Equation("m", None, hidden, output_sort, nonlinearity="relu")
        with pytest.raises(TypeError, match="'m' codomain mismatch"):
            ebn = {"a": eq_a, "m": eq_m}
            FanSpec("_", ["a"], ["m"], hidden, hidden).validate(ebn, build_schema(ebn))


# ---------------------------------------------------------------------------
# Negative integration
# ---------------------------------------------------------------------------

class TestNegativeIntegration:

    def test_assemble_rejects_invalid_path(self, real_sr, hidden, output_sort, backend):
        """assemble_graph raises TypeError when path has sort junction mismatch."""
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, hidden, nonlinearity="tanh")

        with pytest.raises(TypeError):
            assemble_graph(
                [eq_a, eq_b], backend,
                specs=[PathSpec("bad", ["a", "b"], hidden, hidden)],
            )

    def test_assemble_rejects_invalid_fan(self, real_sr, hidden, output_sort, backend):
        """assemble_graph raises TypeError when fan branch codomain mismatches merge domain."""
        # Pipeline order [a, m, b] passes linear validation (all hidden→hidden junctions)
        # but fan validation catches branch 'b' codomain (output_sort) != merge domain (hidden)
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, output_sort, nonlinearity="tanh")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'b' codomain != merge domain"):
            assemble_graph(
                [eq_a, eq_m, eq_b], backend,
                specs=[FanSpec("bad", ["a", "b"], ["m"], hidden, hidden)],
            )


# ---------------------------------------------------------------------------
# Branch codomain validation
# ---------------------------------------------------------------------------

class TestBranchCodomainValidation:

    def test_branch_codomain_must_match_merge_domain(self, real_sr, hidden, output_sort):
        """validate_fan catches when branch codomains don't match merge's domain."""
        # Branch 'a' outputs to output_sort, but merge expects hidden as input
        eq_a = Equation("a", None, hidden, output_sort, nonlinearity="relu")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'a' codomain != merge domain"):
            ebn = {"a": eq_a, "m": eq_m}
            FanSpec("_", ["a"], ["m"], hidden, hidden).validate(ebn, build_schema(ebn))

    def test_mixed_branch_codomains_rejected(self, real_sr, hidden, output_sort):
        """Two branches with different codomains — first mismatch is caught."""
        eq_a = Equation("a", None, hidden, hidden, nonlinearity="relu")
        eq_b = Equation("b", None, hidden, output_sort, nonlinearity="tanh")
        eq_m = Equation("m", "i,i->i", hidden, hidden, real_sr)

        with pytest.raises(TypeError, match="Branch 'b' codomain != merge domain"):
            ebn = {"a": eq_a, "b": eq_b, "m": eq_m}
            FanSpec("_", ["a", "b"], ["m"], hidden, hidden).validate(ebn, build_schema(ebn))


# ---------------------------------------------------------------------------
# Parallel sort validation
# ---------------------------------------------------------------------------

class TestParallelValidation:

    def test_parallel_domain_mismatch_raises(self, real_sr, hidden, output_sort):
        """ParallelSpec.constraints catches right op domain != declared domain sort."""
        eq_left = Equation("left", None, hidden, hidden, nonlinearity="relu")
        eq_right = Equation("right", None, output_sort, output_sort, nonlinearity="tanh")

        ebn = {"left": eq_left, "right": eq_right}
        with pytest.raises(TypeError, match="right op domain != declared domain sort"):
            ParallelSpec("p", "left", "right", hidden, hidden).validate(ebn, build_schema(ebn))

    def test_parallel_codomain_mismatch_raises(self, real_sr, hidden, output_sort):
        """ParallelSpec.constraints catches right op codomain != declared codomain sort."""
        eq_left = Equation("left", None, hidden, hidden, nonlinearity="relu")
        eq_right = Equation("right", None, hidden, output_sort, nonlinearity="tanh")

        ebn = {"left": eq_left, "right": eq_right}
        with pytest.raises(TypeError, match="right op codomain != declared codomain sort"):
            ParallelSpec("p", "left", "right", hidden, hidden).validate(ebn, build_schema(ebn))

    def test_parallel_matching_sorts_pass(self, real_sr, hidden):
        """ParallelSpec.constraints passes when both ops match declared sorts."""
        eq_left = Equation("left", None, hidden, hidden, nonlinearity="relu")
        eq_right = Equation("right", None, hidden, hidden, nonlinearity="tanh")

        ebn = {"left": eq_left, "right": eq_right}
        ParallelSpec("p", "left", "right", hidden, hidden).validate(ebn, build_schema(ebn))
