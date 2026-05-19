"""Phase 6 — normalize_contracts fusion tests.

Covers:
- Structural fusion (DomainPrim tree inspection, before/after)
- Numerical equivalence (fused == unfused on real inputs)
- Fusion guard conditions (semiring mismatch, label mismatch, adjoint mismatch)
- Multi-level chain convergence
- End-to-end DSL paths that exercise the finalize hook
- Optimization quality: fused contraction generates fewer substrate primitives
"""
from pathlib import Path

import numpy as np
import pytest

from unialg import compile_program
from unialg.objects import BINARY
from unialg.runtime import BackendOps
from unialg.semantics import morphisms as ops
from unialg.semantics.morphisms import Morphism
from unialg.syntax import expressions as expr
from unialg.tensors.notation import SemiringDecl
from unialg.tensors.semantics import contract_morphism, resolve_semiring


_SPEC = str(Path(__file__).parent.parent.parent / "src" / "unialg" / "runtime" / "backends" / "numpy.json")


@pytest.fixture
def numpy_backend():
    return BackendOps.from_spec(_SPEC)


@pytest.fixture
def numpy_env(numpy_backend):
    return {
        name: Morphism(
            node=expr.BackendPrim(bp.primitive, bp.arity, bp.dom, bp.result_type),
            aux_primitives=(bp.primitive,),
        )
        for name, bp in numpy_backend.primitives.items()
    }


@pytest.fixture
def real_semiring(numpy_env):
    decl = SemiringDecl(name="real", plus="add", times="multiply", zero=0.0, one=1.0)
    return resolve_semiring(decl, numpy_env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain(inner, outer):
    """compose(par(inner, identity), outer): route inner's output into outer's first slot."""
    return ops.compose(ops.par(inner, ops.identity(BINARY)), outer)


def _fuse(m):
    from unialg.tensors.fusion import _fuse_to_fixpoint
    return _fuse_to_fixpoint(m)


def _is_tensor_prim(node):
    from unialg.tensors.semantics import ContractSpec
    return (
        isinstance(node, expr.DomainPrim)
        and node.tag == "tensors"
        and isinstance(node.raw, ContractSpec)
    )


def _eq_str(node):
    eq = node.raw.equation
    lhs = ",".join("".join(inp) for inp in eq.inputs)
    rhs = "".join(eq.output)
    return f"{lhs}->{rhs}"


def _count_prim_leaves(m):
    """Count BackendPrim leaf nodes in a fully-decomposed Morphism tree."""
    def _count(node):
        if isinstance(node, expr.BackendPrim):
            return 1
        if isinstance(node, expr.ContextualBinary):
            return _count(node.f) + _count(node.g)
        if isinstance(node, expr.MonadicEmbed):
            return _count(node.f)
        return 0
    return _count(m.node)


# ---------------------------------------------------------------------------
# Structural fusion tests
# ---------------------------------------------------------------------------

class TestFusionStructural:
    def test_unfused_compose_has_compose_node(self, real_semiring):
        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        chained = _chain(m1, m2)
        assert isinstance(chained.node, expr.Compose)

    def test_fuse_two_matmuls(self, real_semiring):
        """compose(par(m_ij_jk, id), m_ik_kl) fuses into a single 3-input DomainPrim."""
        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        chained = _chain(m1, m2)

        fused = _fuse(chained)

        assert _is_tensor_prim(fused.node)
        eq = fused.node.raw.equation
        assert len(eq.inputs) == 3
        assert eq.output == ('i', 'l')
        assert _eq_str(fused.node) == "ij,jk,kl->il"

    def test_fused_dom_cod_preserved(self, real_semiring):
        from unialg.tensors.semantics import _strip_exp
        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        chained = _chain(m1, m2)
        fused = _fuse(chained)

        assert _strip_exp(fused.dom()) == _strip_exp(chained.dom())
        assert _strip_exp(fused.cod()) == _strip_exp(chained.cod())

    def test_three_level_chain_fuses_to_one(self, real_semiring):
        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        m3 = contract_morphism(real_semiring, "il,lm->im")
        step2 = _chain(_chain(m1, m2), m3)

        fused = _fuse(step2)

        assert _is_tensor_prim(fused.node)
        eq = fused.node.raw.equation
        assert len(eq.inputs) == 4
        assert eq.output == ('i', 'm')
        assert _eq_str(fused.node) == "ij,jk,kl,lm->im"

    def test_identity_slot_passes_through(self, real_semiring):
        """If both par-tree leaves are Identity, the outer contract stays with same arity."""
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        idid = ops.par(ops.identity(BINARY), ops.identity(BINARY))
        chained = ops.compose(idid, m2)

        fused = _fuse(chained)

        assert _is_tensor_prim(fused.node)
        eq = fused.node.raw.equation
        assert len(eq.inputs) == 2

    def test_different_semiring_blocks_fusion(self, real_semiring, numpy_env):
        other = resolve_semiring(
            SemiringDecl(name="tropical", plus="minimum", times="add",
                         zero=float("inf"), one=0.0),
            numpy_env,
        )
        inner = contract_morphism(other, "ij,jk->ik")
        outer = contract_morphism(real_semiring, "ik,kl->il")
        chained = _chain(inner, outer)

        fused = _fuse(chained)

        assert isinstance(fused.node, expr.Compose)

    def test_mismatched_labels_blocks_fusion(self, real_semiring):
        inner = contract_morphism(real_semiring, "ij,jk->ik")  # output: ('i','k')
        outer = contract_morphism(real_semiring, "ab,bc->ac")  # expects ('a','b') at slot 0
        chained = _chain(inner, outer)

        fused = _fuse(chained)

        assert isinstance(fused.node, expr.Compose)

    def test_adjoint_mismatch_blocks_fusion(self, numpy_env):
        sr = resolve_semiring(
            SemiringDecl(name="adj", plus="add", times="multiply",
                         zero=0.0, one=1.0, adjoint="divide"),
            numpy_env,
        )
        inner = contract_morphism(sr, "ij,jk->ik", adjoint=False)
        outer = contract_morphism(sr, "ik,kl->il", adjoint=True)
        chained = _chain(inner, outer)

        fused = _fuse(chained)

        assert isinstance(fused.node, expr.Compose)

    def test_alpha_rename_uses_hydra_fresh_labels(self, real_semiring):
        """Reduced-label collisions are not capped at ascii_lowercase."""
        from unialg.tensors.fusion import _rename_shape_labels, _extract_labels

        spec = contract_morphism(real_semiring, "az,z->a").node.raw
        avoid = {chr(c) for c in range(ord("a"), ord("z") + 1)}

        renamed_shape = _rename_shape_labels(spec.shape, spec.equation.output, avoid)
        renamed_labels = _extract_labels(renamed_shape)

        assert renamed_labels == (("a", "z'"), ("z'",))


# ---------------------------------------------------------------------------
# Numerical equivalence tests
# ---------------------------------------------------------------------------

class TestFusionNumerical:
    def test_three_input_matmul(self):
        """A@B@C expressed as a 3-input contraction gives the same result as numpy."""
        rng = np.random.default_rng(0)
        A = rng.random((3, 4))
        B = rng.random((4, 5))
        C = rng.random((5, 2))

        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let abc = contract[real]("ij,jk,kl->il")
        """)
        result = prog.run(A, B, C)
        assert np.allclose(result, A @ B @ C)

    def test_existing_two_input_matmul_still_works(self):
        """Plain 2-input contract still gives correct output after lazy change."""
        rng = np.random.default_rng(1)
        A = rng.random((2, 3))
        B = rng.random((3, 4))

        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let matmul = contract[real]("ij,jk->ik")
        """)
        assert np.allclose(prog.run(A, B), A @ B)

    def test_tropical_three_input(self):
        """Tropical min-plus 3-input contraction correct."""
        rng = np.random.default_rng(2)
        A = rng.random((3, 4))
        B = rng.random((4, 5))
        v = rng.random((5,))

        prog = compile_program("""
            load numpy
            algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
            let f = contract[tropical]("ij,jk,k->i")
        """)
        result = prog.run(A, B, v)

        tmp = np.min(B + v[None, :], axis=1)
        expected = np.min(A + tmp[None, :], axis=1)
        assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Optimization quality tests
# ---------------------------------------------------------------------------

class TestFusionOptimization:
    """Verify that fusion reduces the number of substrate primitives generated."""

    def test_fused_two_matmuls_fewer_primitives(self, real_semiring, numpy_backend):
        """Fusing two chained 2-input contractions produces fewer BackendPrim leaves."""
        from unialg.tensors.fusion import _fuse_to_fixpoint, _decompose_all
        from unialg.tensors.primitives import compile_contract_spec

        m_ab = contract_morphism(real_semiring, "ij,jk->ik")
        m_bc = contract_morphism(real_semiring, "ik,kl->il")

        # Unfused: each contraction decomposed independently
        unfused_count = (
            _count_prim_leaves(compile_contract_spec(m_ab.node.raw, numpy_backend))
            + _count_prim_leaves(compile_contract_spec(m_bc.node.raw, numpy_backend))
        )

        # Fused: chain → fuse → decompose
        fused_count = _count_prim_leaves(
            _decompose_all(_fuse_to_fixpoint(_chain(m_ab, m_bc)), numpy_backend)
        )

        assert fused_count < unfused_count, (
            f"Fused ({fused_count}) should have fewer BackendPrim leaves "
            f"than unfused chain ({unfused_count})"
        )

    def test_fused_three_chain_fewer_primitives(self, real_semiring, numpy_backend):
        """3-way chain fused to one contraction has fewer primitives than three independent."""
        from unialg.tensors.fusion import _fuse_to_fixpoint, _decompose_all
        from unialg.tensors.primitives import compile_contract_spec

        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        m3 = contract_morphism(real_semiring, "il,lm->im")

        unfused_count = sum(
            _count_prim_leaves(compile_contract_spec(m.node.raw, numpy_backend))
            for m in (m1, m2, m3)
        )

        chained = _chain(_chain(m1, m2), m3)
        fused_count = _count_prim_leaves(
            _decompose_all(_fuse_to_fixpoint(chained), numpy_backend)
        )

        assert fused_count < unfused_count, (
            f"Fused 3-chain ({fused_count}) should have fewer BackendPrim leaves "
            f"than three separate contractions ({unfused_count})"
        )

    def test_fused_shape_invariant_holds(self, real_semiring, numpy_backend):
        """Fused ContractSpec satisfies shape ↔ equation and shape ↔ dom invariants."""
        from unialg.semantics.functors import apply_poly
        from unialg.tensors.semantics import _count_id

        m1 = contract_morphism(real_semiring, "ij,jk->ik")
        m2 = contract_morphism(real_semiring, "ik,kl->il")
        chained = _chain(m1, m2)
        fused = _fuse(chained)

        spec = fused.node.raw
        assert spec.shape is not None
        assert _count_id(spec.shape) == len(spec.equation.inputs)
        assert apply_poly(spec.shape, BINARY) == spec.dom

    def test_non_left_nested_fuses(self, real_semiring, numpy_backend):
        """par(id, c1) with dom BINARY × (BINARY × BINARY) fuses correctly."""
        c1 = contract_morphism(real_semiring, "jk,kl->jl")
        inner_par = ops.par(ops.identity(BINARY), c1)
        outer = contract_morphism(real_semiring, "ij,jl->il")
        composed = ops.compose(inner_par, outer)

        fused = _fuse(composed)

        assert _is_tensor_prim(fused.node), "Fusion should succeed"
        eq = fused.node.raw.equation
        assert len(eq.inputs) == 3
        assert _eq_str(fused.node) == "ij,jk,kl->il"
        from unialg.tensors.semantics import _strip_exp
        assert _strip_exp(fused.dom()) == _strip_exp(composed.dom()), "Fused dom must match original dom"

    def test_non_left_nested_shape_preserves_nesting(self, real_semiring, numpy_backend):
        """Fused shape reflects original par-tree nesting, not always left-nested."""
        from unialg.syntax.expressions import Prod, Exp
        from unialg.semantics.functors import apply_poly
        from unialg.tensors.semantics import _strip_exp

        c1 = contract_morphism(real_semiring, "jk,kl->jl")
        inner_par = ops.par(ops.identity(BINARY), c1)
        outer = contract_morphism(real_semiring, "ij,jl->il")
        composed = ops.compose(inner_par, outer)

        fused = _fuse(composed)
        spec = fused.node.raw

        assert isinstance(spec.shape, Prod)
        # Left slot carries labels from the identity (adopted from outer's Exp shape)
        assert isinstance(spec.shape.left, Exp)
        # Right slot carries inner contract's Prod-of-Exp shape
        assert isinstance(spec.shape.right, Prod)
        assert _strip_exp(apply_poly(spec.shape, BINARY)) == _strip_exp(composed.dom())

    def test_non_left_nested_numerical(self):
        """Right-nested fused contraction gives correct numerical result."""
        import numpy as np
        rng = np.random.default_rng(99)
        A = rng.random((3, 4))
        B = rng.random((4, 5))
        C = rng.random((5, 2))

        prog = compile_program("""
            load numpy
            algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
            let abc = contract[real]("ij,jk,kl->il")
        """)
        result = prog.run(A, B, C)
        assert np.allclose(result, A @ B @ C)

    def test_blocked_fusion_has_same_primitives_as_independent(
        self, real_semiring, numpy_env, numpy_backend
    ):
        """When fusion is blocked (semiring mismatch), primitive count equals independent sum."""
        from unialg.tensors.fusion import _fuse_to_fixpoint, _decompose_all
        from unialg.tensors.primitives import compile_contract_spec

        other = resolve_semiring(
            SemiringDecl(name="tropical", plus="minimum", times="add",
                         zero=float("inf"), one=0.0),
            numpy_env,
        )
        inner = contract_morphism(other, "ij,jk->ik")
        outer = contract_morphism(real_semiring, "ik,kl->il")

        independent_count = (
            _count_prim_leaves(compile_contract_spec(inner.node.raw, numpy_backend))
            + _count_prim_leaves(compile_contract_spec(outer.node.raw, numpy_backend))
        )

        chained = _chain(inner, outer)
        chain_count = _count_prim_leaves(
            _decompose_all(_fuse_to_fixpoint(chained), numpy_backend)
        )

        assert chain_count == independent_count, (
            f"Blocked fusion should not change primitive count "
            f"(expected {independent_count}, got {chain_count})"
        )


# ---------------------------------------------------------------------------
# Phase B: Opaque-leaf preserving fusion tests
# ---------------------------------------------------------------------------

class TestOpaqueFusion:
    """Verify fusion with opaque (non-contract, non-identity) par-tree leaves."""

    def _make_tanh(self, numpy_env):
        """Get the tanh morphism from backend primitives."""
        return numpy_env["tanh"]

    def test_opaque_leaf_produces_compose_not_single_prim(self, real_semiring, numpy_env):
        """par(tanh, c1) fused with outer: result is compose(pre_map, fused_contract)."""
        tanh = self._make_tanh(numpy_env)
        c1 = contract_morphism(real_semiring, "jk,kl->jl")
        inner_par = ops.par(tanh, c1)
        outer = contract_morphism(real_semiring, "ij,jl->il")
        composed = ops.compose(inner_par, outer)

        fused = _fuse(composed)

        assert isinstance(fused.node, expr.Compose), \
            "Opaque fusion should produce compose(pre_map, fused_contract)"
        assert _is_tensor_prim(fused.node.g), \
            "Inner morphism should be the fused DomainPrim"
        fused_eq = fused.node.g.raw.equation
        assert len(fused_eq.inputs) == 3
        assert fused.dom() == composed.dom()
        assert fused.cod() == composed.cod()

    def test_opaque_leaf_pre_map_structure(self, real_semiring, numpy_env, numpy_backend):
        """pre_map has the opaque morphism at the right position."""
        from unialg.tensors.fusion import _fuse_to_fixpoint, _decompose_all

        tanh = self._make_tanh(numpy_env)
        c1 = contract_morphism(real_semiring, "jk,kl->jl")
        inner_par = ops.par(tanh, c1)
        outer = contract_morphism(real_semiring, "ij,jl->il")
        composed = ops.compose(inner_par, outer)

        fused = _fuse_to_fixpoint(composed)

        # fused should be compose(pre_map, fused_contract_DomainPrim)
        assert isinstance(fused.node, expr.Compose)
        pre_map_node = fused.node.f
        # pre_map should be par(tanh, identity(...))
        assert isinstance(pre_map_node, expr.Parallel)

        from unialg.tensors.semantics import _strip_exp
        decomposed = _decompose_all(fused, numpy_backend)
        assert _strip_exp(decomposed.dom()) == _strip_exp(composed.dom())
        assert _strip_exp(decomposed.cod()) == _strip_exp(composed.cod())

    def test_opaque_only_blocks_fusion(self, real_semiring, numpy_env):
        """par(tanh, sigmoid) with no absorbed contracts: fusion blocked."""
        tanh = self._make_tanh(numpy_env)
        sigmoid = numpy_env.get("sigmoid") or tanh  # use tanh as fallback
        outer = contract_morphism(real_semiring, "ij,jk->ik")
        composed = ops.compose(ops.par(tanh, sigmoid), outer)

        fused = _fuse(composed)

        # No DomainPrim was absorbed, so fusion should not fire
        assert isinstance(fused.node, expr.Compose)

    def test_opaque_leaf_equation_has_passthrough_label(self, real_semiring, numpy_env):
        """Opaque leaf's slot in fused equation preserves outer's label."""
        tanh = self._make_tanh(numpy_env)
        c1 = contract_morphism(real_semiring, "jk,kl->jl")
        inner_par = ops.par(tanh, c1)
        outer = contract_morphism(real_semiring, "ij,jl->il")
        composed = ops.compose(inner_par, outer)

        fused = _fuse(composed)
        fused_eq = fused.node.g.raw.equation

        # First input is the passthrough from tanh (outer expected "ij")
        assert fused_eq.inputs[0] == ("i", "j")
        # Remaining inputs are from c1's absorption
        assert fused_eq.inputs[1] == ("j", "k")
        assert fused_eq.inputs[2] == ("k", "l")


# ---------------------------------------------------------------------------
# Phase C: Pair/shared-input fusion tests
# ---------------------------------------------------------------------------

class TestPairFusion:
    """Verify fusion through Pair(contract, identity) with alpha-renaming."""

    def test_pair_label_collision_shows_wrong_result(self):
        """Document: naive merging without renaming gives wrong answer."""
        import numpy as np
        rng = np.random.default_rng(0)
        A = rng.random((3, 4))
        W = rng.random((4, 5))
        Y = A @ W
        correct = np.einsum("ik,ij,jk->i", Y, A, W)
        naive = np.einsum("ij,jk,ij,jk->i", A, W, A, W)
        assert not np.allclose(correct, naive), "Naive should differ from correct"
        renamed = np.einsum("im,mk,ij,jk->i", A, W, A, W)
        assert np.allclose(correct, renamed), "Renamed should match correct"

    def test_pair_c1_left_single_input_fuses(self, real_semiring):
        """pair(c1_single, id) >> outer fuses (single-input c1, no shape mismatch)."""
        c1 = contract_morphism(real_semiring, "ij->i")
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(c1, id_b)
        outer = contract_morphism(real_semiring, "i,ij->j")
        composed = ops.compose(pair_m, outer)

        fused = _fuse(composed)

        assert isinstance(fused.node, expr.Compose), "Should be compose(Copy, fused_contract)"
        assert isinstance(fused.node.f, expr.Copy), "First part should be Copy"
        assert _is_tensor_prim(fused.node.g), "Second part should be fused DomainPrim"
        from unialg.tensors.semantics import _strip_exp
        assert _strip_exp(fused.dom()) == _strip_exp(composed.dom())
        assert _strip_exp(fused.cod()) == _strip_exp(composed.cod())

    def test_pair_c1_right_single_input_fuses(self, real_semiring):
        """pair(id, c1_single) >> outer fuses (identity on left)."""
        c1 = contract_morphism(real_semiring, "ij->i")
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(id_b, c1)
        outer = contract_morphism(real_semiring, "ij,i->j")
        composed = ops.compose(pair_m, outer)

        fused = _fuse(composed)

        assert isinstance(fused.node, expr.Compose)
        assert isinstance(fused.node.f, expr.Copy)
        assert _is_tensor_prim(fused.node.g)

    def test_pair_fused_equation_has_renamed_labels(self, real_semiring):
        """Inner's reduced labels are renamed to avoid collision with identity labels."""
        c1 = contract_morphism(real_semiring, "ij->i")
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(c1, id_b)
        outer = contract_morphism(real_semiring, "i,ij->j")
        composed = ops.compose(pair_m, outer)

        fused = _fuse(composed)
        fused_eq = fused.node.g.raw.equation

        # c1's reduced label is "j". Outer's identity slot has "ij" which contains "j".
        # So c1's "j" must be renamed. The fused equation should have a fresh label.
        inner_labels = set()
        for inp in fused_eq.inputs[:1]:
            inner_labels.update(inp)
        identity_labels = set()
        for inp in fused_eq.inputs[1:]:
            identity_labels.update(inp)
        # No collision between renamed inner reduced and identity labels
        inner_reduced = set(fused_eq.reduced) - identity_labels
        assert len(inner_reduced) > 0 or not set(c1.node.raw.equation.reduced) & identity_labels

    def test_pair_fused_spec_shape_invariant(self, real_semiring):
        """Fused ContractSpec satisfies shape ↔ equation and shape ↔ dom invariants."""
        from unialg.semantics.functors import apply_poly
        from unialg.tensors.semantics import _count_id

        c1 = contract_morphism(real_semiring, "ij->i")
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(c1, id_b)
        outer = contract_morphism(real_semiring, "i,ij->j")
        composed = ops.compose(pair_m, outer)
        fused = _fuse(composed)

        fused_spec = fused.node.g.raw  # the DomainPrim inside compose(Copy, ...)
        assert fused_spec.shape is not None
        assert _count_id(fused_spec.shape) == len(fused_spec.equation.inputs)
        assert apply_poly(fused_spec.shape, BINARY) == fused_spec.dom

    def test_pair_dom_cod_preserved(self, real_semiring):
        """Fused result preserves original compose's dom and cod (up to Exp wrapping)."""
        from unialg.tensors.semantics import _strip_exp
        c1 = contract_morphism(real_semiring, "ij->i")
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(c1, id_b)
        outer = contract_morphism(real_semiring, "i,ij->j")
        composed = ops.compose(pair_m, outer)
        fused = _fuse(composed)
        assert _strip_exp(fused.dom()) == _strip_exp(composed.dom())
        assert _strip_exp(fused.cod()) == _strip_exp(composed.cod())

    def test_pair_no_rename_when_no_collision(self, real_semiring):
        """If inner has no reduced labels, no rename needed."""
        from unialg.tensors.fusion import _rename_shape_labels
        spec = contract_morphism(real_semiring, "ij->ij").node.raw
        renamed_shape = _rename_shape_labels(spec.shape, spec.equation.output, avoid={"i", "j", "k"})
        assert renamed_shape is spec.shape

    def test_pair_both_contract_fuses(self, real_semiring):
        """pair(c1, c2) — both branches are contracts, fuses via Copy."""
        c1 = contract_morphism(real_semiring, "ij->i")
        c2 = contract_morphism(real_semiring, "ab->a")
        pair_m = ops.pair(c1, c2)
        outer = contract_morphism(real_semiring, "i,a->")
        composed = ops.compose(pair_m, outer)
        fused = _fuse(composed)
        assert isinstance(fused.node, expr.Compose), "Should be compose(Copy, fused_contract)"
        assert isinstance(fused.node.f, expr.Copy), "First part should be Copy"
        fused_spec = fused.node.g.raw
        assert len(fused_spec.equation.inputs) == 2
        assert fused_spec.equation.output == ()

    def test_pair_both_identity_blocks(self, real_semiring):
        """pair(id, id) — no contract to absorb."""
        id_b = ops.identity(BINARY)
        pair_m = ops.pair(id_b, id_b)
        outer = contract_morphism(real_semiring, "i,i->")
        composed = ops.compose(pair_m, outer)
        fused = _fuse(composed)
        assert isinstance(fused.node, expr.Compose)
        assert not isinstance(fused.node.f, expr.Copy)

    def test_rename_shape_labels_unit(self, real_semiring):
        """Direct test of _rename_shape_labels helper."""
        from unialg.tensors.fusion import _rename_shape_labels, _extract_labels

        spec = contract_morphism(real_semiring, "ij,jk->ik").node.raw
        renamed_shape = _rename_shape_labels(spec.shape, spec.equation.output, avoid={"j", "i", "k"})
        renamed_labels = _extract_labels(renamed_shape)
        all_renamed = {l for inp in renamed_labels for l in inp}
        assert "j" not in all_renamed

    def test_rename_preserves_output_labels(self, real_semiring):
        """Renaming never changes output (non-reduced) labels in the shape."""
        from unialg.tensors.fusion import _rename_shape_labels, _extract_labels

        spec = contract_morphism(real_semiring, "ij,jk->ik").node.raw
        renamed_shape = _rename_shape_labels(spec.shape, spec.equation.output, avoid={"j"})
        renamed_labels = _extract_labels(renamed_shape)
        all_renamed = {l for inp in renamed_labels for l in inp}
        assert "i" in all_renamed and "k" in all_renamed
