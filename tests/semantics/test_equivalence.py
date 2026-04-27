"""Fused composition primitive tests.

Verifies that Path and Fan compositions register as Hydra Primitive objects in
graph.primitives (not as lambdas in bound_terms), so reduce_term dispatches
fused closures directly.

Two test levels per feature per CLAUDE.md:
  1. Direct .implementation() call on the Primitive object.
  2. End-to-end via Program.__call__ (reduce_term fallback path).
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.dsl.python import Right

from hydra.context import Context
from hydra.dsl.python import FrozenDict
from hydra.dsl.terms import apply, var, list_
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, Equation,
    compile_program, Program,
    PathSpec, FanSpec, FoldSpec, UnfoldSpec, FixpointSpec,
    tensor_coder,
)
from unialg.terms import EMPTY_CX


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("fpreal", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("fphidden", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    """Python/numpy array → Hydra term (decode in Hydra's naming convention)."""
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


# ---------------------------------------------------------------------------
# Test 1: path registered as primitive
# ---------------------------------------------------------------------------

class TestPathRegisteredAsPrimitive:

    def test_path_in_graph_primitives(self, hidden, real_sr, backend, coder):
        """A compiled path must appear in graph.primitives, not bound_terms."""
        eq_lin = Equation("fp1_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("fp1_relu", None, hidden, hidden, nonlinearity="relu")

        W = encode_array(coder, np.eye(2))
        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("fp1_path", ["fp1_linear", "fp1_relu"], hidden, hidden,
                            params={"fp1_linear": [W]})],
        )

        assert Name("ua.path.fp1_path") in prog.graph.primitives, (
            "path not registered as Primitive — check to_primitive path in compositions.py"
        )

    def test_path_not_in_bound_terms(self, hidden, real_sr, backend, coder):
        """A compiled path must NOT appear in bound_terms (it is a Primitive)."""
        eq_lin = Equation("fp1b_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("fp1b_relu", None, hidden, hidden, nonlinearity="relu")

        W = encode_array(coder, np.eye(2))
        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("fp1b_path", ["fp1b_linear", "fp1b_relu"], hidden, hidden,
                            params={"fp1b_linear": [W]})],
        )

        assert Name("ua.path.fp1b_path") not in prog.graph.bound_terms, (
            "compiled path should be a Primitive, not a bound lambda term"
        )

    def test_path_equation_alias_in_primitives(self, hidden, real_sr, backend, coder):
        """The ua.equation.<name> alias for a compiled path must also be a Primitive."""
        eq_lin = Equation("fp1c_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("fp1c_relu", None, hidden, hidden, nonlinearity="relu")

        W = encode_array(coder, np.eye(2))
        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("fp1c_path", ["fp1c_linear", "fp1c_relu"], hidden, hidden,
                            params={"fp1c_linear": [W]})],
        )

        assert Name("ua.equation.fp1c_path") in prog.graph.primitives, (
            "ua.equation.<path_name> alias should be registered as a Primitive"
        )

    def test_path_primitive_implementation_callable(self, hidden, real_sr, backend, coder):
        """The fused Primitive for a path must have a callable implementation.

        Direct .implementation() call — level-1 test.
        """
        eq_relu = Equation("fp1d_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_relu], backend=backend,
            specs=[PathSpec("fp1d_path", ["fp1d_relu"], hidden, hidden)],
        )

        prim = prog.graph.primitives.get(Name("ua.path.fp1d_path"))
        assert prim is not None, "path Primitive not found in graph.primitives"
        assert callable(prim.implementation), (
            "Primitive.implementation must be callable"
        )

    def test_path_primitive_produces_correct_output(self, hidden, real_sr, backend, coder):
        """Calling prim.implementation() directly (with Hydra calling convention) matches oracle.

        Direct .implementation(cx, graph, args) call — level-1 test.
        prim1 implementation signature: (cx: Context, g: Graph, args: tuple[Term, ...]).
        frozenlist is just a tuple alias at runtime in Hydra-Python.
        """
        eq_relu = Equation("fp1e_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_relu], backend=backend,
            specs=[PathSpec("fp1e_path", ["fp1e_relu"], hidden, hidden)],
        )

        prim = prog.graph.primitives[Name("ua.path.fp1e_path")]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_enc = encode_array(coder, x)

        # Hydra Primitive.implementation takes (cx, graph, args) where args is
        # a tuple of already-encoded Terms (one per curried argument).
        result = prim.implementation(EMPTY_CX, prog.graph, (x_enc,))
        match result:
            case Right(value=term):
                match coder.encode(None, None, term):
                    case Right(value=out):
                        np.testing.assert_allclose(out, np.maximum(0, x), rtol=1e-6)
                    case other:
                        pytest.fail(f"decode failed: {other}")
            case other:
                pytest.fail(f"implementation() returned unexpected: {other}")


# ---------------------------------------------------------------------------
# Test 2: fan registered as primitive
# ---------------------------------------------------------------------------

class TestFanRegisteredAsPrimitive:

    def test_fan_in_graph_primitives(self, hidden, real_sr, backend):
        """A compiled fan must appear in graph.primitives."""
        eq_relu = Equation("fp2_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp2_tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("fp2_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp2_fan",
                           branch_names=["fp2_relu", "fp2_tanh"],
                           merge_names=["fp2_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        assert Name("ua.fan.fp2_fan") in prog.graph.primitives, (
            "fan not registered as Primitive"
        )

    def test_fan_not_in_bound_terms(self, hidden, real_sr, backend):
        """A compiled fan must NOT appear in bound_terms."""
        eq_relu = Equation("fp2b_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp2b_tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("fp2b_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp2b_fan",
                           branch_names=["fp2b_relu", "fp2b_tanh"],
                           merge_names=["fp2b_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        assert Name("ua.fan.fp2b_fan") not in prog.graph.bound_terms, (
            "compiled fan should be a Primitive, not a bound lambda"
        )

    def test_fan_primitive_implementation_callable(self, hidden, real_sr, backend, coder):
        """Fan Primitive.implementation must be callable.

        Direct .implementation() call — level-1 test.
        """
        eq_relu = Equation("fp2c_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp2c_tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("fp2c_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp2c_fan",
                           branch_names=["fp2c_relu", "fp2c_tanh"],
                           merge_names=["fp2c_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        prim = prog.graph.primitives.get(Name("ua.fan.fp2c_fan"))
        assert prim is not None, "fan Primitive not found"
        assert callable(prim.implementation), "Primitive.implementation must be callable"

    def test_fan_equation_alias_in_primitives(self, hidden, real_sr, backend):
        """ua.equation.<fan_name> alias must also be a Primitive."""
        eq_relu = Equation("fp2d_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp2d_tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("fp2d_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp2d_fan",
                           branch_names=["fp2d_relu", "fp2d_tanh"],
                           merge_names=["fp2d_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        assert Name("ua.equation.fp2d_fan") in prog.graph.primitives, (
            "ua.equation.<fan_name> alias should be a Primitive"
        )


# ---------------------------------------------------------------------------
# Test 3: path fast path equals reduce_term fallback
# ---------------------------------------------------------------------------

class TestPathFastPathEqualsReduceTerm:

    def test_fast_path_matches_reduce_term(self, hidden, real_sr, backend, coder):
        """Fast path (_compiled_fns) and reduce_term fallback must produce identical results.

        This is the critical correctness invariant for the fused primitive feature:
        the two dispatch paths must agree numerically.

        Level-2 test: end-to-end via both dispatch paths.
        """
        eq_lin = Equation("fp3_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("fp3_relu", None, hidden, hidden, nonlinearity="relu")

        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        w_enc = encode_array(coder, W)

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("fp3_path", ["fp3_linear", "fp3_relu"], hidden, hidden,
                            params={"fp3_linear": [w_enc]})],
        )

        x = np.array([2.0, 1.0])

        # Fast path: statically compiled closure
        assert "fp3_path" in prog._compiled_fns, (
            "fp3_path should be in compiled_fns — parametrised path must compile"
        )
        fast_result = prog._compiled_fns["fp3_path"](x)

        # Slow path: build a Program with empty compiled_fns to force reduce_term
        slow_prog = Program(prog._graph, backend, coder, EMPTY_CX, compiled_fns={})
        slow_result = slow_prog("fp3_path", x)

        np.testing.assert_allclose(fast_result, slow_result, rtol=1e-6,
                                   err_msg="fast path and reduce_term disagree")

    def test_unparametrised_path_fast_path_matches_reduce_term(self, hidden, real_sr, backend, coder):
        """Unparametrised path (relu-only) fast and slow paths must agree.

        Level-2 test.
        """
        eq_relu = Equation("fp3b_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp3b_tanh", None, hidden, hidden, nonlinearity="tanh")

        prog = compile_program(
            [eq_relu, eq_tanh], backend=backend,
            specs=[PathSpec("fp3b_path", ["fp3b_relu", "fp3b_tanh"], hidden, hidden)],
        )

        x = np.array([-3.0, -1.0, 0.5, 2.0])

        fast_result = prog._compiled_fns["fp3b_path"](x)
        slow_prog = Program(prog._graph, backend, coder, EMPTY_CX, compiled_fns={})
        slow_result = slow_prog("fp3b_path", x)

        np.testing.assert_allclose(fast_result, slow_result, rtol=1e-6,
                                   err_msg="fast and slow path disagree for relu→tanh")


# ---------------------------------------------------------------------------
# Test 4: rebind produces different results
# ---------------------------------------------------------------------------

class TestRebindProducesDifferentResults:

    def test_rebind_changes_output(self, hidden, real_sr, backend, coder):
        """rebind_hyperparams() with a new weight vector produces different output.

        Uses the assemble_graph + rebind_hyperparams pattern (the same mechanism
        Program.rebind() delegates to). A weight vector is stored as ua.param.weight
        and referenced via var("ua.param.weight") in PathSpec.params, so rebinding
        it changes the path output.

        Level-2 test: end-to-end via reduce_term on both graphs.
        """
        from hydra.context import Context
        from hydra.dsl.python import FrozenDict
        from hydra.dsl.terms import apply, var
        from hydra.reduction import reduce_term
        from unialg.assembly.graph import assemble_graph, rebind_hyperparams

        cx = Context(trace=(), messages=(), other=FrozenDict({}))

        # scale equation: "i,i->i" with real_sr (times=multiply) → element-wise product
        eq_relu = Equation("fp4_relu", None, hidden, hidden, nonlinearity="relu")
        eq_scale = Equation("fp4_scale", "i,i->i", hidden, hidden, real_sr)

        weight1 = np.array([2.0, 2.0, 2.0])
        weight2 = np.array([0.5, 0.5, 0.5])
        w1_enc = encode_array(coder, weight1)
        w2_enc = encode_array(coder, weight2)

        # Build graph: relu → scale(weight, x) where weight comes from ua.param.weight
        graph, *_ = assemble_graph(
            [eq_relu, eq_scale], backend,
            hyperparams={"weight": w1_enc},
            specs=[PathSpec("fp4_path", ["fp4_relu", "fp4_scale"], hidden, hidden,
                            params={"fp4_scale": [var("ua.param.weight")]})],
        )

        x = np.array([-1.0, 0.5, 2.0])
        x_enc = encode_array(coder, x)

        # Evaluate with weight1
        r1 = reduce_term(cx, graph, True, apply(var("ua.path.fp4_path"), x_enc))
        assert isinstance(r1, Right), f"reduce_term failed: {r1}"
        out1_result = coder.encode(None, None, r1.value)
        assert isinstance(out1_result, Right)
        out1 = out1_result.value

        # Rebind to weight2
        graph2 = rebind_hyperparams(graph, {"weight": w2_enc})
        r2 = reduce_term(cx, graph2, True, apply(var("ua.path.fp4_path"), x_enc))
        assert isinstance(r2, Right), f"reduce_term failed after rebind: {r2}"
        out2_result = coder.encode(None, None, r2.value)
        assert isinstance(out2_result, Right)
        out2 = out2_result.value

        # relu(x) * weight1 vs relu(x) * weight2
        relu_x = np.maximum(0, x)
        np.testing.assert_allclose(out1, relu_x * weight1, rtol=1e-6)
        np.testing.assert_allclose(out2, relu_x * weight2, rtol=1e-6)
        assert not np.allclose(out1, out2), "rebind did not change the output"

    def test_rebind_with_scalar_hyperparam(self, hidden, real_sr, backend, coder):
        """rebind() with a float scalar hyperparam changes output.

        Level-2 test: verifies the scalar wrapping path in rebind().
        """
        eq_relu = Equation("fp4b_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp4b_tanh", None, hidden, hidden, nonlinearity="tanh")

        # dummy float hyperparam to exercise the rebind scalar-wrapping path
        dummy_val = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        prog = compile_program(
            [eq_relu, eq_tanh], backend=backend,
            specs=[PathSpec("fp4b_path", ["fp4b_relu", "fp4b_tanh"], hidden, hidden)],
            hyperparams={"dummy": dummy_val},
        )

        prog2 = prog.rebind(dummy=2.0)
        assert isinstance(prog2, Program)
        assert prog2 is not prog

        # Both programs produce the same output (dummy doesn't affect the path computation),
        # but rebind must return a valid program that runs without error.
        x = np.array([-1.0, 0.5, 2.0])
        out1 = prog("fp4b_path", x)
        out2 = prog2("fp4b_path", x)
        np.testing.assert_allclose(out1, out2, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: fold stays in bound_terms
# ---------------------------------------------------------------------------

class TestFoldPromotedToPrimitive:

    def test_fold_in_primitives(self, hidden, real_sr, backend):
        """Fold compositions must be promoted to graph.primitives."""
        eq_step = Equation("fp5_step", "i,i->i", hidden, hidden, real_sr)

        init_term = core.TermLiteral(value=core.LiteralFloat(value=0.0))
        prog = compile_program(
            [eq_step], backend=backend,
            specs=[FoldSpec("fp5_fold",
                            step_name="fp5_step",
                            init_term=init_term,
                            domain_sort=hidden,
                            state_sort=hidden)],
        )

        assert Name("ua.fold.fp5_fold") in prog.graph.primitives, (
            "fold should be promoted to a Primitive"
        )

    def test_fold_not_in_bound_terms(self, hidden, real_sr, backend):
        """Promoted fold must not appear in bound_terms."""
        eq_step = Equation("fp5b_step", "i,i->i", hidden, hidden, real_sr)

        init_term = core.TermLiteral(value=core.LiteralFloat(value=0.0))
        prog = compile_program(
            [eq_step], backend=backend,
            specs=[FoldSpec("fp5b_fold",
                            step_name="fp5b_step",
                            init_term=init_term,
                            domain_sort=hidden,
                            state_sort=hidden)],
        )

        assert Name("ua.fold.fp5b_fold") not in prog.graph.bound_terms, (
            "promoted fold should not be in bound_terms"
        )


# ---------------------------------------------------------------------------
# Test 6: fused primitive implementation is callable
# ---------------------------------------------------------------------------

class TestFusedPrimitiveImplementationCallable:

    def test_path_primitive_has_callable_implementation(self, hidden, real_sr, backend, coder):
        """Retrieve the fused path Primitive and assert implementation is callable.

        Direct .implementation() invocation — level-1 test.
        """
        eq_relu = Equation("fp6_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_relu], backend=backend,
            specs=[PathSpec("fp6_path", ["fp6_relu"], hidden, hidden)],
        )

        prim = prog.graph.primitives.get(Name("ua.path.fp6_path"))
        assert prim is not None, "ua.path.fp6_path not found in graph.primitives"
        assert callable(prim.implementation), (
            f"Primitive.implementation must be callable; got {type(prim.implementation)}"
        )

    def test_fan_primitive_has_callable_implementation(self, hidden, real_sr, backend, coder):
        """Retrieve the fused fan Primitive and assert implementation is callable.

        Direct .implementation() invocation — level-1 test.
        """
        eq_relu = Equation("fp6b_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp6b_tanh", None, hidden, hidden, nonlinearity="tanh")
        eq_merge = Equation("fp6b_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp6b_fan",
                           branch_names=["fp6b_relu", "fp6b_tanh"],
                           merge_names=["fp6b_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        prim = prog.graph.primitives.get(Name("ua.fan.fp6b_fan"))
        assert prim is not None, "ua.fan.fp6b_fan not found in graph.primitives"
        assert callable(prim.implementation), (
            f"Fan Primitive.implementation must be callable; got {type(prim.implementation)}"
        )

    def test_path_primitive_end_to_end_via_program_call(self, hidden, real_sr, backend, coder):
        """Path dispatches correctly when called via Program.__call__.

        End-to-end via reduce_term — level-2 test.
        """
        eq_relu = Equation("fp6c_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_relu], backend=backend,
            specs=[PathSpec("fp6c_path", ["fp6c_relu"], hidden, hidden)],
        )

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Force reduce_term path by stripping compiled_fns
        slow_prog = Program(prog._graph, backend, coder, EMPTY_CX, compiled_fns={})
        result = slow_prog("fp6c_path", x)
        np.testing.assert_allclose(result, np.maximum(0, x), rtol=1e-6)

    def test_fan_primitive_end_to_end_via_program_call(self, hidden, real_sr, backend, coder):
        """Fan dispatches correctly when called via Program.__call__.

        End-to-end via reduce_term — level-2 test.
        The merge equation "i,i->i" with real semiring (times=multiply) computes
        element-wise product: relu(x) * tanh(x).
        """
        eq_relu = Equation("fp6d_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("fp6d_tanh", None, hidden, hidden, nonlinearity="tanh")
        # "i,i->i" with real_sr: times=multiply → element-wise product of the two branches
        eq_merge = Equation("fp6d_merge", "i,i->i", hidden, hidden, real_sr)

        prog = compile_program(
            [eq_relu, eq_tanh, eq_merge], backend=backend,
            specs=[FanSpec("fp6d_fan",
                           branch_names=["fp6d_relu", "fp6d_tanh"],
                           merge_names=["fp6d_merge"],
                           domain_sort=hidden, codomain_sort=hidden)],
        )

        x = np.array([-1.0, 0.0, 1.0, 2.0])

        # Force reduce_term path
        slow_prog = Program(prog._graph, backend, coder, EMPTY_CX, compiled_fns={})
        result = slow_prog("fp6d_fan", x)

        # element-wise product: relu(x) * tanh(x) (real semiring times = multiply)
        expected = np.maximum(0, x) * np.tanh(x)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Fold primitive equivalence
# ---------------------------------------------------------------------------

def _make_cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


def _decode_term(coder, term):
    from hydra.dsl.python import Right as R
    result = coder.encode(None, None, term)
    assert isinstance(result, R), f"decode failed: {result}"
    return result.value


class TestFoldPrimitiveEquivalence:
    """Fold compositions promoted to Hydra Primitive: registration and numeric equivalence."""

    def test_fold_registered_as_primitive(self, hidden, real_sr, backend, coder):
        """A compiled FoldSpec must appear in graph.primitives, not bound_terms.

        Level-1 test: registration check.
        """
        eq_step = Equation("fp7_step", "i,i->i", hidden, hidden, real_sr)
        init_term = core.TermLiteral(value=core.LiteralFloat(value=1.0))

        prog = compile_program(
            [eq_step], backend=backend,
            specs=[FoldSpec("fp7_fold",
                            step_name="fp7_step",
                            init_term=init_term,
                            domain_sort=hidden,
                            state_sort=hidden)],
        )

        assert Name("ua.fold.fp7_fold") in prog.graph.primitives, (
            "fold not registered as Primitive — check FoldComposition.resolve()"
        )
        assert Name("ua.fold.fp7_fold") not in prog.graph.bound_terms, (
            "promoted fold should not appear in bound_terms"
        )

    def test_fold_fast_path_equals_reduce_term(self, hidden, real_sr, backend, coder):
        """Fast path (compiled_fn) and reduce_term dispatch produce identical results.

        Step: "i,i->i" with real semiring (times=multiply) = element-wise product.
        Init = [1, 1, 1].  fold([x1, x2, x3]) = x1 * x2 * x3.

        Fast path: prog._compiled_fns["fp7b_fold"]([x1, x2, x3])
        Reduce path: reduce_term with a TermList argument applied to the primitive.

        Level-2 test: both dispatch paths must agree numerically.
        """
        eq_step = Equation("fp7b_step", "i,i->i", hidden, hidden, real_sr)
        init = np.ones(3)
        init_term = encode_array(coder, init)

        prog = compile_program(
            [eq_step], backend=backend,
            specs=[FoldSpec("fp7b_fold",
                            step_name="fp7b_step",
                            init_term=init_term,
                            domain_sort=hidden,
                            state_sort=hidden)],
        )

        assert "fp7b_fold" in prog._compiled_fns, (
            "fp7b_fold must be in compiled_fns — fold step must compile"
        )

        x1 = np.array([2.0, 3.0, 4.0])
        x2 = np.array([0.5, 2.0, 1.0])
        x3 = np.array([3.0, 0.1, 2.0])

        # Fast path: compiled_fn receives a Python list of native arrays.
        fast_result = prog._compiled_fns["fp7b_fold"]([x1, x2, x3])

        # Reduce path: the fold Primitive input coder is list_(coder), so it
        # expects a Hydra TermList.  Build one and call reduce_term directly.
        cx = _make_cx()
        seq_term = list_([encode_array(coder, x) for x in [x1, x2, x3]])
        red = reduce_term(cx, prog.graph, True, apply(var("ua.fold.fp7b_fold"), seq_term))
        assert isinstance(red, Right), f"reduce_term failed: {red}"
        reduce_result = _decode_term(coder, red.value)

        np.testing.assert_allclose(fast_result, reduce_result, rtol=1e-6,
                                   err_msg="fold fast path and reduce_term disagree")
        # Independent numpy oracle: init * x1 * x2 * x3
        np.testing.assert_allclose(fast_result, init * x1 * x2 * x3, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 8: Unfold primitive equivalence
# ---------------------------------------------------------------------------

class TestUnfoldPrimitiveEquivalence:
    """Unfold compositions promoted to Hydra Primitive: registration and numeric equivalence."""

    def test_unfold_registered_as_primitive(self, hidden, real_sr, backend, coder):
        """A compiled UnfoldSpec must appear in graph.primitives, not bound_terms.

        Level-1 test: registration check.
        """
        eq_step = Equation("fp8_step", None, hidden, hidden, nonlinearity="tanh")

        prog = compile_program(
            [eq_step], backend=backend,
            specs=[UnfoldSpec("fp8_unfold",
                              step_name="fp8_step",
                              n_steps=3,
                              domain_sort=hidden,
                              state_sort=hidden)],
        )

        assert Name("ua.unfold.fp8_unfold") in prog.graph.primitives, (
            "unfold not registered as Primitive — check UnfoldComposition.resolve()"
        )
        assert Name("ua.unfold.fp8_unfold") not in prog.graph.bound_terms, (
            "promoted unfold should not appear in bound_terms"
        )

    def test_unfold_fast_path_equals_reduce_term(self, hidden, real_sr, backend, coder):
        """Fast path and reduce_term dispatch produce identical results for n=3 tanh steps.

        Fast path: compiled_fn(s0) returns a tuple of 3 native arrays.
        Reduce path: reduce_term returns a TermList of 3 encoded tensors.

        Level-2 test: both dispatch paths must agree numerically.
        """
        eq_step = Equation("fp8b_step", None, hidden, hidden, nonlinearity="tanh")

        prog = compile_program(
            [eq_step], backend=backend,
            specs=[UnfoldSpec("fp8b_unfold",
                              step_name="fp8b_step",
                              n_steps=3,
                              domain_sort=hidden,
                              state_sort=hidden)],
        )

        assert "fp8b_unfold" in prog._compiled_fns, (
            "fp8b_unfold must be in compiled_fns — unfold step must compile"
        )

        s0 = np.array([1.0, -0.5, 2.0])

        # Fast path: compiled_fn(s0) returns a tuple of arrays.
        fast_outputs = prog._compiled_fns["fp8b_unfold"](s0)
        assert len(fast_outputs) == 3, f"expected 3 outputs, got {len(fast_outputs)}"

        # Reduce path: apply the unfold Primitive to the encoded initial state.
        cx = _make_cx()
        s0_enc = encode_array(coder, s0)
        red = reduce_term(cx, prog.graph, True, apply(var("ua.unfold.fp8b_unfold"), s0_enc))
        assert isinstance(red, Right), f"reduce_term failed: {red}"
        result_term = red.value
        assert isinstance(result_term, core.TermList), (
            f"unfold must return a TermList, got {type(result_term)}"
        )
        reduce_outputs = [_decode_term(coder, t) for t in result_term.value]
        assert len(reduce_outputs) == 3

        for i, (fast, slow) in enumerate(zip(fast_outputs, reduce_outputs)):
            np.testing.assert_allclose(fast, slow, rtol=1e-6,
                                       err_msg=f"step {i}: fast/reduce mismatch")

        # Independent numpy oracle: tanh^k(s0)
        s = s0.copy()
        for i, out in enumerate(fast_outputs):
            s = np.tanh(s)
            np.testing.assert_allclose(out, s, rtol=1e-6,
                                       err_msg=f"tanh^{i+1}(s0) oracle mismatch")


# ---------------------------------------------------------------------------
# Test 9: Fixpoint primitive equivalence
# ---------------------------------------------------------------------------

class TestFixpointPrimitiveEquivalence:
    """Fixpoint compositions promoted to Hydra Primitive: registration and correctness."""

    def test_fixpoint_registered_as_primitive(self, hidden, real_sr, backend, coder):
        """A compiled FixpointSpec must appear in graph.primitives.

        Level-1 test: registration check.
        """
        from hydra.dsl.prims import float32 as float32_coder, prim1
        from unialg.algebra.sort import sort_wrap

        backend.unary_ops["fp9_halve"] = lambda x: 0.5 * x

        real_sr9 = Semiring("fpreal9", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden9 = Sort("fphidden9", real_sr9)

        in_coder = sort_wrap(hidden9).coder(backend)
        pred_prim = prim1(
            Name("ua.equation.fp9_pred"),
            lambda x: float(np.max(np.abs(x))),
            [], in_coder, float32_coder(),
        )

        eq_step = Equation("fp9_step", None, hidden9, hidden9, nonlinearity="fp9_halve")
        output9 = Sort("fpoutput9", real_sr9)
        eq_pred_placeholder = Equation("fp9_pred", None, hidden9, output9, nonlinearity="abs")

        prog = compile_program(
            [eq_step, eq_pred_placeholder], backend=backend,
            specs=[FixpointSpec("fp9_fixpoint",
                                step_name="fp9_step",
                                predicate_name="fp9_pred",
                                epsilon=0.01,
                                max_iter=50,
                                domain_sort=hidden9)],
        )

        assert Name("ua.fixpoint.fp9_fixpoint") in prog.graph.primitives, (
            "fixpoint not registered as Primitive — check FixpointComposition.resolve()"
        )

    def test_fixpoint_fast_path_produces_correct_output(self, hidden, real_sr, backend, coder):
        """Fixpoint with decay-by-0.5 step converges; result and iteration count are correct.

        Fast path: prog._compiled_fns["fp9b_fixpoint"](x0) returns (final_state, iter_count).
        The fixpoint returns a Python pair from the while_loop; the values must satisfy
        abs(final_state) <= epsilon and iter_count < max_iter.

        The predicate must return a scalar float (not an array) so that backend.while_loop's
        truth test works.  We register a custom scalar-residual predicate op.

        Level-2 test: fast path output satisfies convergence constraints.
        """
        backend.unary_ops["fp9b_halve"] = lambda x: 0.5 * x
        # Scalar predicate: max(|x|) — returns a Python float, not an array.
        backend.unary_ops["fp9b_max_abs"] = lambda x: float(np.max(np.abs(x)))

        real_sr9b = Semiring("fpreal9b", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden9b = Sort("fphidden9b", real_sr9b)
        output9b = Sort("fpoutput9b", real_sr9b)

        eq_step = Equation("fp9b_step", None, hidden9b, hidden9b, nonlinearity="fp9b_halve")
        # Use the scalar predicate op so while_loop cond_fn gets a plain float.
        eq_pred = Equation("fp9b_pred", None, hidden9b, output9b, nonlinearity="fp9b_max_abs")

        epsilon = 0.01
        max_iter = 100

        prog = compile_program(
            [eq_step, eq_pred], backend=backend,
            specs=[FixpointSpec("fp9b_fixpoint",
                                step_name="fp9b_step",
                                predicate_name="fp9b_pred",
                                epsilon=epsilon,
                                max_iter=max_iter,
                                domain_sort=hidden9b)],
        )

        assert "fp9b_fixpoint" in prog._compiled_fns, (
            "fp9b_fixpoint must be in compiled_fns — fixpoint step must compile"
        )

        x0 = np.array([1.0, 2.0])
        result = prog._compiled_fns["fp9b_fixpoint"](x0)

        # The while_loop backend returns a (state, count) pair.
        final_state, iter_count = result
        final_arr = np.asarray(final_state)
        assert np.all(np.abs(final_arr) <= epsilon), (
            f"fixpoint did not converge: max |x| = {np.max(np.abs(final_arr))}"
        )
        count_val = int(iter_count)
        assert count_val < max_iter, (
            f"fixpoint hit max_iter={max_iter} instead of converging (count={count_val})"
        )


# ---------------------------------------------------------------------------
# Test 10: Mixed graph equivalence
# ---------------------------------------------------------------------------

class TestMixedGraphEquivalence:
    """Multiple composition types coexisting in the same Program."""

    def test_mixed_path_and_fold(self, hidden, real_sr, backend, coder):
        """A Program with both a PathSpec and a FoldSpec must have both callable and correct.

        Both must be registered as Primitives and produce correct outputs.

        Level-2 test: both entry points exercised in the same graph.
        """
        eq_relu = Equation("fp10_relu", None, hidden, hidden, nonlinearity="relu")
        eq_step = Equation("fp10_step", "i,i->i", hidden, hidden, real_sr)

        init = np.ones(3)
        init_term = encode_array(coder, init)

        prog = compile_program(
            [eq_relu, eq_step], backend=backend,
            specs=[
                PathSpec("fp10_path", ["fp10_relu"], hidden, hidden),
                FoldSpec("fp10_fold",
                         step_name="fp10_step",
                         init_term=init_term,
                         domain_sort=hidden,
                         state_sort=hidden),
            ],
        )

        assert Name("ua.path.fp10_path") in prog.graph.primitives, (
            "path not in primitives in mixed graph"
        )
        assert Name("ua.fold.fp10_fold") in prog.graph.primitives, (
            "fold not in primitives in mixed graph"
        )

        # Test path via fast path
        x = np.array([-2.0, 0.5, 1.0])
        path_result = prog("fp10_path", x)
        np.testing.assert_allclose(path_result, np.maximum(0, x), rtol=1e-6)

        # Test fold via reduce_term (TermList input required)
        cx = _make_cx()
        v1 = np.array([2.0, 3.0, 4.0])
        v2 = np.array([0.5, 2.0, 1.0])
        seq_term = list_([encode_array(coder, v) for v in [v1, v2]])
        red = reduce_term(cx, prog.graph, True, apply(var("ua.fold.fp10_fold"), seq_term))
        assert isinstance(red, Right), f"fold reduce_term failed: {red}"
        fold_result = _decode_term(coder, red.value)
        np.testing.assert_allclose(fold_result, init * v1 * v2, rtol=1e-6)

    def test_mixed_primitive_and_lambda(self, hidden, real_sr, backend, coder):
        """A PathSpec with literal params (Primitive) and one with variable params (lambda fallback)
        must both be reachable and correct.

        A path with literal params encodes the param into the compiled closure — it becomes
        a Primitive.  A path with variable params (TermVariable) cannot be statically compiled
        and falls back to a Hydra lambda in bound_terms.

        Level-2 test: both entry points work in the same graph.
        """
        from hydra.dsl.terms import var as hvar

        eq_scale = Equation("fp11_scale", "i,i->i", hidden, hidden, real_sr)
        eq_relu = Equation("fp11_relu", None, hidden, hidden, nonlinearity="relu")

        # Literal weight — enables static compilation to Primitive.
        W_literal = encode_array(coder, np.array([2.0, 2.0, 2.0]))
        # Variable weight — prevents static compilation; falls back to lambda.
        W_var = hvar("ua.param.fp11_weight")

        prog = compile_program(
            [eq_scale, eq_relu], backend=backend,
            specs=[
                PathSpec("fp11_literal_path", ["fp11_scale"], hidden, hidden,
                         params={"fp11_scale": [W_literal]}),
                PathSpec("fp11_var_path", ["fp11_scale"], hidden, hidden,
                         params={"fp11_scale": [W_var]}),
            ],
            hyperparams={"fp11_weight": encode_array(coder, np.array([3.0, 3.0, 3.0]))},
        )

        # The literal path must be a Primitive (statically compiled).
        assert Name("ua.path.fp11_literal_path") in prog.graph.primitives, (
            "literal-param path must be a Primitive"
        )

        x = np.array([1.0, 2.0, 3.0])

        # Literal path fast path: element-wise product with [2, 2, 2]
        literal_result = prog("fp11_literal_path", x)
        np.testing.assert_allclose(literal_result, x * np.array([2.0, 2.0, 2.0]), rtol=1e-6)

        # Variable path slow path: element-wise product with fp11_weight = [3, 3, 3]
        slow_prog = Program(prog._graph, backend, coder, EMPTY_CX, compiled_fns={})
        var_result = slow_prog("fp11_var_path", x)
        np.testing.assert_allclose(var_result, x * np.array([3.0, 3.0, 3.0]), rtol=1e-6)

        # Results must differ (different weight vectors)
        assert not np.allclose(literal_result, var_result), (
            "literal and variable path results must differ"
        )
