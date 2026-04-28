"""Batched sort unit tests: sort construction, einsum rewriting, junctions, graph registration.

Key invariant: declaration is logical (unbatched einsum stored in the
equation record); resolution is physical (einsum rewritten at resolve time
when the domain sort is batched).
"""

import numpy as np
import pytest

import hydra.core as core

from unialg import (
    Semiring, Sort,
    Equation,
)
from unialg.assembly.graph import validate_pipeline, build_graph
from unialg.algebra.equation import _prepend_batch_dim
from unialg.assembly._equation_resolution import resolve_equation


# ===========================================================================
# Part A: Sort construction and type encoding
# ===========================================================================

class TestBatchedSortConstruction:
    """Verify that the batched flag is stored and reflected in the TypeVariable."""

    def test_batched_sort_identity(self, real_sr):
        """Batched and unbatched sorts with the same name produce distinct types."""
        import hydra.core as core
        unbatched = Sort("hidden", real_sr, batched=False)
        batched = Sort("hidden", real_sr, batched=True)

        t_unbatched = Sort.from_term(unbatched).type_
        t_batched = Sort.from_term(batched).type_

        assert t_unbatched != t_batched
        # Unbatched: TypeApplication(ua.sort.hidden, ua.semiring.real)
        assert t_unbatched.value.function == core.TypeVariable(core.Name("ua.sort.hidden"))
        assert t_unbatched.value.argument == core.TypeVariable(core.Name("ua.semiring.real"))
        # Batched: TypeApplication(ua.batched, TypeApplication(ua.sort.hidden, ua.semiring.real))
        assert t_batched.value.function == core.TypeVariable(core.Name("ua.batched"))
        assert t_batched.value.argument == t_unbatched

    def test_batched_type_from_term(self, real_sr):
        """sort_type_from_term with batched=True wraps in ua.batched TypeApplication."""
        tropical_sr = Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        s = Sort("output", tropical_sr, batched=True)
        t = Sort.from_term(s).type_
        assert t.value.function == core.TypeVariable(core.Name("ua.batched"))
        inner = t.value.argument
        assert inner.value.function == core.TypeVariable(core.Name("ua.sort.output"))
        assert inner.value.argument == core.TypeVariable(core.Name("ua.semiring.tropical"))

    def test_unbatched_type_from_term(self, real_sr):
        """sort_type_from_term with batched=False has no batched wrapper."""
        tropical_sr = Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        s = Sort("output", tropical_sr, batched=False)
        t = Sort.from_term(s).type_
        assert t.value.function == core.TypeVariable(core.Name("ua.sort.output"))
        assert t.value.argument == core.TypeVariable(core.Name("ua.semiring.tropical"))

    def test_is_batched_helper_true(self, real_sr):
        """is_batched returns True for batched=True sorts."""
        s = Sort("hidden", real_sr, batched=True)
        assert Sort.from_term(s).batched is True

    def test_is_batched_helper_false(self, real_sr):
        """is_batched returns False for batched=False sorts."""
        s = Sort("hidden", real_sr, batched=False)
        assert Sort.from_term(s).batched is False

    def test_is_batched_helper_default(self, real_sr):
        """is_batched returns False when batched flag is absent (old records)."""
        # Create a sort without the batched field by constructing the record manually
        import hydra.dsl.terms as T
        old_style = T.record(Sort._type_name, [
            T.field("name", T.string("legacy")),
            T.field("semiring", real_sr),
        ])
        assert Sort.from_term(old_style).batched is False

    def test_sort_type_from_term_batched_structure(self, real_sr):
        """Sort.from_term for a batched sort produces the expected TypeApplication structure."""
        s = Sort("encoder", real_sr, batched=True)
        t = Sort.from_term(s).type_
        assert t.value.function == core.TypeVariable(core.Name("ua.batched"))
        inner = t.value.argument
        assert inner.value.function == core.TypeVariable(core.Name("ua.sort.encoder"))
        assert inner.value.argument == core.TypeVariable(core.Name("ua.semiring.real"))


# ===========================================================================
# Part B: _prepend_batch_dim helper
# ===========================================================================

class TestPrependBatchDim:
    """Unit tests for the einsum batch-prepend helper."""

    def test_vector_matvec(self):
        """ij,j->i becomes bij,bj->bi (standard matmul with batch)."""
        result = _prepend_batch_dim("ij,j->i")
        # 'b' is the first unused letter; i and j are used
        assert result == "bij,bj->bi"

    def test_matmul_chain(self):
        """ij,jk->ik: batch char must not be i, j, or k."""
        result = _prepend_batch_dim("ij,jk->ik")
        lhs, rhs = result.split("->")
        parts = lhs.split(",")
        batch_char = parts[0][0]
        assert batch_char not in {"i", "j", "k"}
        assert all(p.startswith(batch_char) for p in parts)
        assert rhs.startswith(batch_char)

    def test_single_input(self):
        """i->i (identity): becomes bi->bi."""
        result = _prepend_batch_dim("i->i")
        assert result == "bi->bi"

    def test_empty_string_passthrough(self):
        """Empty string passes through unchanged."""
        assert _prepend_batch_dim("") == ""

    def test_batch_char_avoids_used(self):
        """When 'b' is already in use, the next available char is chosen."""
        # 'b' is used as an index, so the batch char should be 'c' (next in sequence)
        result = _prepend_batch_dim("bj,j->b")
        batch_char = result[0]
        assert batch_char == "c"

    def test_output_structure(self):
        """The output always has the same number of comma-separated inputs as the original."""
        original = "ij,jk,kl->il"
        result = _prepend_batch_dim(original)
        orig_inputs = original.split("->")[0].split(",")
        result_inputs = result.split("->")[0].split(",")
        assert len(result_inputs) == len(orig_inputs)


# ===========================================================================
# Part C: resolve_equation with batched sorts
# ===========================================================================

class TestBatchedEquationResolution:
    """Verify that resolve_equation produces correct primitives for batched sorts."""

    def test_batched_pointwise_resolves(self, real_sr, backend):
        """Pointwise equation on batched sort resolves without error.

        Pointwise ops are elementwise and need no einsum rewriting.
        """
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name == core.Name("ua.equation.relu_b")

    def test_batched_unary_einsum_resolves(self, real_sr, backend):
        """Unary einsum on batched sort resolves (the einsum gets prepended)."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("bn_scale", "i->i", hidden_b, hidden_b, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name == core.Name("ua.equation.bn_scale")

    def test_batched_binary_einsum_resolves(self, real_sr, backend):
        """Binary einsum on batched sort resolves — becomes a 2-input prim2."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("linear_b", "ij,j->i", hidden_b, hidden_b, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name == core.Name("ua.equation.linear_b")

    def test_unbatched_still_works(self, real_sr, backend):
        """Sort() with default batched=False is unchanged from pre-Phase9 behaviour."""
        hidden = Sort("hidden", real_sr)  # batched=False by default
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, *_ = resolve_equation(eq, backend)
        assert prim.name == core.Name("ua.equation.linear")


# ===========================================================================
# Part D: Sort junctions with batched sorts
# ===========================================================================

class TestBatchedSortJunctions:
    """validate_pipeline must treat batched and unbatched sorts as distinct types."""

    def test_batched_to_batched_ok(self, real_sr):
        """Batched codomain → batched domain: junction passes."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq1 = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq2 = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh", inputs=("relu_b",))
        # Should not raise
        validate_pipeline([eq1, eq2])

    def test_unbatched_to_unbatched_ok(self, real_sr):
        """Unbatched codomain → unbatched domain: junction passes."""
        hidden = Sort("hidden", real_sr, batched=False)
        eq1 = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq2 = Equation("tanh", None, hidden, hidden, nonlinearity="tanh", inputs=("relu",))
        validate_pipeline([eq1, eq2])

    def test_batched_to_unbatched_fails(self, real_sr):
        """Batched codomain → unbatched domain: junction raises TypeError."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        hidden = Sort("hidden", real_sr, batched=False)
        eq_batched = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_unbatched = Equation("tanh", None, hidden, hidden, nonlinearity="tanh", inputs=("relu_b",))
        with pytest.raises(TypeError):
            validate_pipeline([eq_batched, eq_unbatched])

    def test_unbatched_to_batched_fails(self, real_sr):
        """Unbatched codomain → batched domain: junction raises TypeError."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        hidden = Sort("hidden", real_sr, batched=False)
        eq_unbatched = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_batched = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh", inputs=("relu",))
        with pytest.raises(TypeError):
            validate_pipeline([eq_unbatched, eq_batched])


# ===========================================================================
# Part G: build_graph registers batched sort type correctly
# ===========================================================================

class TestBatchedGraphRegistration:
    """Sorts are registered in schema_types as component names."""

    def test_batched_sort_in_schema(self, real_sr):
        """build_graph registers component names (ua.sort.X, ua.semiring.Y, ua.batched)."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        graph = build_graph([hidden_b])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types
        assert core.Name("ua.batched") in graph.schema_types

    def test_unbatched_sort_in_schema(self, real_sr):
        """build_graph registers component names for unbatched sort."""
        hidden = Sort("hidden", real_sr, batched=False)
        graph = build_graph([hidden])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types

    def test_both_sorts_in_schema(self, real_sr):
        """Both batched and unbatched variants share component name entries."""
        hidden = Sort("hidden", real_sr, batched=False)
        hidden_b = Sort("hidden", real_sr, batched=True)
        graph = build_graph([hidden, hidden_b])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types
        assert core.Name("ua.batched") in graph.schema_types
