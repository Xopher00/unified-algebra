# Tensor Extension — Next Steps

## Current state: Phase 6.7 — fusion unified via Optic.par (2026-05-19)

402 tests passing, 4 skipped. All import boundaries clean.

Fusion pass (`tensors/fusion.py`) handles:
- Parallel-tree fusion with polynomial-shape-correct nesting (Phase A)
- Opaque-leaf preserving fusion via `compose(pre_map, fused_contract)` (Phase B)
- **All Pair cases** via `Optic.par` — Pair(c,id), Pair(id,c), Pair(c1,c2) (Phase C unified)
- Multi-input identity slots via `_count_slots` (handles `Identity(BINARY × BINARY)`)
- Mutual alpha-renaming of reduced labels lifted into the walk (sibling collision handling)
- Diagonal/trace semantics: repeated labels, iterative extraction, cross-backend `_call_diagonal` (Phase D)

`ContractSpec` carries a `shape: PolyExpr` field. `apply_poly(shape, BINARY)` computes exact nested ProductType for dom. `compile_contract_spec` builds alignment and fold trees guided by the polynomial shape.

### Fusion architecture

The bespoke `_absorb_par_tree` + `_try_fuse_pair` + four helpers were replaced by a single `_par_to_optic` catamorphism that emits `Optic` instances. `Optic.par` (`semantics/optics.py`) is the parallel/product combinator on polynomial-functor optics — the optic-level analogue of `ops.par` on morphisms. Pair factoring (`Pair(f,g) = compose(Copy, par(f,g))`) is handled inline. Net −63 production LOC.

## Phase 7: Custom semiring end-to-end (next)

Smooth-tropical from legacy example 05. Exercises the full pipeline with a non-standard semiring.

## Deferred fusion extensions

### Pair nesting (not yet tested end-to-end)
- **Pair(Pair(c1, id), c2)** — multi-level Pair trees. The `_par_to_optic` walk handles recursive Pair structurally (each Pair factors through Copy, inner Pairs recurse). Needs end-to-end test coverage to confirm label alignment and shape correctness for deeply nested cases.

### Opaque fusion — optimization metric
- Add a test verifying `compose(pre_map, fused_contract)` has fewer BackendPrim leaves than the unfused chain (if applicable — depends on whether the opaque leaf introduces more primitives than it saves).

### Backend fast-paths
- Route real-semiring fused contraction to native `einsum` instead of align/fold/reduce decomposition. Requires detecting the special case in `compile_contract_spec` and calling `numpy.einsum` (or equivalent) directly.

### Contraction order optimization
- Cost-model-driven choice of which contractions to fuse and in what order. Currently fusion is greedy (fixpoint iteration).

## Transformer/encoder DSL follow-ups

- Axis-specialized softmax (axis=-1 baked into generated primitive)
- ReLU or GELU coverage (not in backend JSON specs)
- Layer normalization (axis-aware mean/variance/sqrt)
- Attention masking (mask-aware softmax or pre-softmax addition)
- Routing carried parameters through larger composed blocks

## Deferred items

- Fix lexer/parser comment robustness (`RecursionError` on long comments)
- Block-wise splitting for memory-safe broadcast
- Semiring law checking using zero/one identity values
