"""Tensor contraction morphisms — tensor helper layer.

Responsibility: produce a typed ``Morphism`` for a semiring contraction.  The
result is assembled from the semiring's own sub-morphisms (``sr.plus``,
``sr.times`` and reduce morphisms).  Identity values such as ``sr.zero`` and
``sr.one`` are floats for semantic law checking, not scalar runtime morphisms.
No Hydra term reduction and no Python-native numeric evaluation happen here.

Layer position: tensors/ — receives a parsed index equation and a Semiring,
then produces a Morphism.  If tensor lowering grows beyond named backend
primitive composition, keep that code in a tensor-specific helper unless it
becomes general structure-layer machinery.

─────────────────────────────────────────────────────────────────────────────
Core equation
─────────────────────────────────────────────────────────────────────────────

    C[i,k] = ⊕_j  A[i,j] ⊗ B[j,k]

⊕ = sr.plus   (carrier × carrier → carrier)
⊗ = sr.times  (carrier × carrier → carrier)
ε = sr.zero   (float additive identity, reserved for semantic law checks)

This is NOT numpy.einsum.  numpy.einsum is hardwired to + and ×.  Here ⊕ and
⊗ are arbitrary user-defined morphisms.  Change the semiring, change the
meaning; keep the equation structure.

─────────────────────────────────────────────────────────────────────────────
Two operation roles — elementwise vs. reduction
─────────────────────────────────────────────────────────────────────────────

Every semiring operation appears in two distinct roles in a contraction:

  elementwise  — apply ⊗ at every position of two co-shaped tensors
                 Input: two tensors A, B with the same index shape
                 Output: one tensor with A[idx] ⊗ B[idx] at each index

  reduction    — fold ⊕ over a specific index axis, collapsing it away
                 Input: a tensor T with contracted axes included
                 Output: ⊕ of all slices T[..., j, ...] for j in J

The tensor lowering/helper layer is responsible for lifting the
scalar morphisms sr.times and sr.plus into these tensor-level operations (via
expand_dims + transpose alignment, then elementwise application, then axis
fold).  This module only needs to track which indexes are free (appear
in the output) and which are contracted (summed away).

─────────────────────────────────────────────────────────────────────────────
Batching via functor application
─────────────────────────────────────────────────────────────────────────────

Batching ("bij,bjk->bik") is NOT a special case.  The batch index b is a free
index that appears in all operands and the output.  It is handled by treating
the unbatched contraction as a morphism and applying it functorially over the
batch index type:

    batched = poly_fmap(Functor_B, contract_morphism(sr, "ij,jk->ik"))

where Functor_B is the function-space functor (B → -).  This lifts the scalar
contraction to run independently for each batch element without duplicating the
contraction logic.

The structural equivalent in the old code was CompiledEinsum.prepend_batch_var,
which mutated the compiled equation.  The new approach uses the existing functor
machinery instead, and composition of functor actions handles multi-level
batching cleanly. This depends on a settled encoding for finite batch index
types; do not introduce a parallel batching syntax just for contractions.

─────────────────────────────────────────────────────────────────────────────
Fusion at the semantics level
─────────────────────────────────────────────────────────────────────────────

For multi-step contractions such as "ab,bc,cd->ad", the naive form is:

    step1: C_ac = ⊕_b A_ab ⊗ B_bc
    step2: D_ad = ⊕_c C_ac ⊗ E_cd

By semiring distributivity (⊗ distributes over ⊕), this fuses into:

    fused: D_ad = ⊕_b ⊕_c A_ab ⊗ B_bc ⊗ E_cd

Fusion eliminates the materialized intermediate C_ac.  Because fusion follows
from the semiring axioms, it is algebraically safe to perform here, at the
Morphism level, before any lowering.  The tensor lowering/helper layer then sees a single
assembled computation rather than a chain.

─────────────────────────────────────────────────────────────────────────────
Expected public surface (not yet implemented)
─────────────────────────────────────────────────────────────────────────────

    contract_morphism(sr: Semiring, equation: str, *, adjoint: bool = False) -> Morphism

        Parses ``equation`` into a CompiledEquation (a pure dataclass carrying
        input_vars, output_vars, reduced_vars — ported from the old
        algebra/contraction.compile_einsum, which had no backend coupling).

        Builds and returns a Morphism whose:
          dom = product of tensor types for each input operand
          cod = tensor_type(output_index, carrier)

        The adjoint flag selects which op pair drives the contraction:

          adjoint=False (standard):
            elementwise step → sr.times
            fold step        → sr.plus_reduce   (identity value: sr.zero)

          adjoint=True:
            elementwise step → sr.adjoint   (right residual of ⊗)
            fold step        → sr.times_reduce  (identity value: sr.one)

        Raises if adjoint=True and sr.adjoint is None.

        The user sets adjoint at call time — they do not define a separate
        operation.  The same (sr, equation) pair produces two distinct
        Morphisms depending on the flag.

        Batching is handled by the caller via poly_fmap, not by this function.
        Fusion of a chain of contract_morphism calls may be applied at this
        layer using semiring distributivity before the result is passed to the
        tensor lowering/helper layer.

─────────────────────────────────────────────────────────────────────────────
Open question before implementing
─────────────────────────────────────────────────────────────────────────────

Index-type encoding: what Hydra Type represents the index set I, J, K?
Options: abstract nominal type (Name("I")), Hydra TypeVariable, Nat-indexed.
Nominal is the likely right choice to start — sufficient for composition
type-checking without requiring size arithmetic.  Settle this before
implementing tensor_type().

─────────────────────────────────────────────────────────────────────────────
Dependencies (when implemented)
─────────────────────────────────────────────────────────────────────────────

- unialg.tensors.semirings.Semiring
- unialg.semantics.morphisms.Morphism
- unialg.semantics.functors.poly_fmap (for batching)
- unialg.objects.ExpType (for a tensor_type helper, once designed)
- runtime backend primitives for concrete tensor kernels, if needed
"""

# Nothing is implemented here yet.
# Sign off on this skeleton before adding any logic.
