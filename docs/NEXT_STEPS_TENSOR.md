# Tensor Extension — Next Steps

## Current state: Phase 6.5 complete (2026-05-18)

392 tests passing, 4 skipped. All import boundaries clean.

Fusion pass (`tensors/fusion.py`) handles:
- Parallel-tree fusion with polynomial-shape-correct nesting (Phase A)
- Opaque-leaf preserving fusion via `compose(pre_map, fused_contract)` (Phase B)
- Pair(contract, identity) fusion via `compose(Copy, fused_contract)` with Hydra-based alpha-renaming (Phase C1)
- Diagonal/trace semantics: repeated labels, iterative extraction, cross-backend `_call_diagonal` (Phase D)

`ContractSpec` carries a `shape: PolyExpr` field. `apply_poly(shape, BINARY)` computes exact nested ProductType for dom. `compile_contract_spec` builds alignment and fold trees guided by the polynomial shape.

## Phase 7: Custom semiring end-to-end (next)

Smooth-tropical from legacy example 05. Exercises the full pipeline with a non-standard semiring.

## Deferred fusion extensions

### Pair fusion — general cases
- **Pair(c1, c2)**: both branches are contractions. Requires mutual alpha-renaming (both branches' reduced labels may collide with each other). The equation model would need to track which renamed labels map to which original labels across both branches.
- **Multi-input Pair(c1, identity)**: requires the outer contract to be constructed with non-left-nested shape matching `pair.cod()`. Currently only single-input inner contractions compose naturally (both produce `BINARY × BINARY`).
- **Pair nesting**: `Pair(Pair(c1, id), c2)` — multi-level Pair trees.

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
