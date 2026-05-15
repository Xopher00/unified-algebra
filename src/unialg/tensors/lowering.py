"""Tensor contraction lowering — structure layer.

Responsibility: lift scalar semiring morphisms to tensor-level operations and
lower an assembled contraction Morphism to Hydra terms.  Also owns the
memory-aware blocking optimization.  No semantic meaning lives here.

Layer position: structure/ — downstream of semantics/contractions.py, upstream
of run() / realize.py.

─────────────────────────────────────────────────────────────────────────────
What this layer does
─────────────────────────────────────────────────────────────────────────────

The semantics layer produces a Morphism assembled from scalar sr.plus and
sr.times sub-morphisms, with a CompiledEquation describing the index structure.
This layer lowers that into concrete tensor operations in three steps:

1. Alignment
   ──────────
   Each input tensor is unsqueezed and permuted so all operands share the
   combined dim order: output_vars ++ reduced_vars.  After alignment, every
   operand has a size-1 dimension wherever it does not participate in a
   given index variable — broadcasting takes care of the rest.

   This corresponds to _align_tensor / _align_factors in the old
   algebra/contraction.py.  The backend ops needed: expand_dims, transpose.

2. Elementwise product
   ────────────────────
   Apply the product-op elementwise across all aligned operands.

   The product-op is selected by the adjoint flag at the semantics layer
   (contract_morphism) before this layer is reached:
     standard mode  → sr.times    (elementwise ⊗)
     adjoint mode   → sr.adjoint  (elementwise ⊘, right residual of ⊗)

   This layer receives the already-selected Morphism from op_env() and lifts
   it to operate element-by-element on co-shaped tensors.  It does not
   inspect the adjoint flag itself.

3. Axis reduction
   ───────────────
   Apply the fold-op over the reduced_dims axes, collapsing them away.

   The fold-op is also selected by the semantics layer:
     standard mode  → sr.plus   (fold ⊕, seed: sr.zero)
     adjoint mode   → sr.times  (fold ⊗, seed: sr.one)

   This layer receives the already-selected Morphism and seed from op_env()
   and performs the axis fold.  No knowledge of the adjoint flag is needed here.

─────────────────────────────────────────────────────────────────────────────
Blocked contraction (memory optimization)
─────────────────────────────────────────────────────────────────────────────

For large contractions the intermediate tensor (output_vars ++ reduced_vars)
may exceed available memory.  The old code handled this by chunking the first
contracted dimension into blocks and accumulating partial results using
sr.plus elementwise (plus_elementwise in Semiring.Resolved, as opposed to the
fold-based plus_reduce).

This optimization is valid because sr.plus is associative.  It belongs here,
not in semantics/, because it is a memory management concern, not a change in
algebraic meaning.

Blocked vs full contraction produces numerically identical results when the
semiring's ⊕ is exactly associative.  For approximate semirings (smooth-max
etc.) this may introduce floating-point order sensitivity — document this when
implementing.

The old implementation: _auto_block_size (memory budget heuristic),
_slice_args (slice one operand along a dim), and the accumulation loop in
semiring_contract.  All portable; no backend coupling beyond expand_dims,
transpose, and scalar tensor creation.

─────────────────────────────────────────────────────────────────────────────
What this layer does NOT do
─────────────────────────────────────────────────────────────────────────────

- Does not own fusion.  Fusion of chained contractions is algebraic
  (semiring distributivity) and belongs in semantics/contractions.py, where
  it can operate on Morphisms before any lowering.
- Does not register a single "einsum" primitive and delegate to numpy.einsum.
  That would bypass the semiring entirely.
- Does not evaluate numeric values in Python outside of the Hydra impl closure.

─────────────────────────────────────────────────────────────────────────────
Expected contents (not yet implemented)
─────────────────────────────────────────────────────────────────────────────

    lower_contraction(morphism: Morphism, equation: CompiledEquation) -> Morphism
        Applies the three-step lowering (align, times_elementwise, plus_reduce)
        using backend primitives registered via BackendPrimitive machinery.

    register_tensor_structural_ops(ops: BackendOps) -> None
        Register expand_dims and transpose as backend primitives so the
        alignment step can go through run() → realize.py → Hydra reduction.

    blocked_contract(...) -> Morphism
        Memory-aware variant: chunks the first contracted dim, accumulates
        partial results via sr.plus elementwise.  Numerically equivalent to
        lower_contraction for associative semirings.

─────────────────────────────────────────────────────────────────────────────
Dependencies (when implemented)
─────────────────────────────────────────────────────────────────────────────

- unialg.structure.backend.register_backend_primitive, BackendOps
- unialg.structure.codecs.TERM_CODER_REGISTRY
- unialg.semantics.semirings.Semiring (for op_env() dict)
- semantics/contractions.py CompiledEquation (pure dataclass, no execution)
- hydra.rewriting.rewrite_term  (only if term-level structural rewrites needed)
- hydra.dsl.meta.phantoms (for term construction)

Do not implement until semantics/contractions.py skeleton is signed off and
the index-type encoding (nominal vs. TypeVariable) is settled.
"""

# Nothing is implemented here yet.
# Sign off on this skeleton before adding any logic.
