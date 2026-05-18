# Tensor Extension — Next Steps

## Immediate: Phase 5 lowering design (BLOCKS everything)

The lowerer converts `ContractSpec` → executable substrate composition:
```
par(align_0, align_1) >> sr.times.node >> reduce_prim
```

### Open design questions (settle before writing code)

**1. LoweringContext — what carries the runtime state?**

The lowerer needs: RuntimeStore, arg/result coders, expand_dims/transpose callables, axis-aware reduce callables. None of these exist in the current `protocol.lower(spec, ctx)` signature — ctx is currently `None`.

Proposal: `main.py` constructs a `LoweringContext` dataclass from the loaded `BackendOps` and passes it as `ctx`. The lowerer receives callables, never imports a backend.

**2. Structural ops (expand_dims, transpose) — source?**

NOT in JSON backend specs. Options:
- (A) Resolve from backend module name at lowering time in `main.py` (`importlib.import_module("numpy").expand_dims`)
- (B) Add to JSON specs as `"kind": "structural"` entries
- (C) The BackendOps instance exposes a `structural_ops` dict populated at load time

Option A is simplest. Option B is most consistent with existing spec pattern.

**3. Axis-aware reduce — how to parameterize?**

`sr.plus_reduce` wraps `numpy.sum` (all-axis). Contraction needs `numpy.sum(arr, axis=(2,))`. Options:
- (A) `LoweringContext` carries a factory: `make_axis_reduce(reduce_morphism, axes) -> Callable`
- (B) The lowerer creates a new Hydra Primitive whose impl wraps the reduce callable with baked axes — requires access to the underlying Python fn from the BackendPrimitive

**4. `_lower_domain_prims` recursion**

Currently walks `ContextualBinary` only. Needs to also recurse through `PolyFmap`, `BackendPrim(args=...)`, `MonadicLift`, `AlgExpr` before Phase 6. Not blocking for simple contractions but will be needed.

### Lowered primitive pattern

Each generated Hydra Primitive follows existing `BackendPrimitive.impl` signature:
```python
def impl(ctx, graph, args, *, store, fn, ...):
    key = arg_coder.encode(ctx, graph, args[0])
    arr = store.get(key)
    result = fn(arr)  # expand_dims+transpose, or axis-reduce
    return result_coder.decode(ctx, store.put(result))
```

TypeScheme: `BINARY → BINARY`. Registered via aux_primitives.

## Phase 6: End-to-end runtime tests

After Phase 5:
1. Real semiring matmul: `"ij,jk->ik"` with add/multiply, verify against `numpy.matmul`
2. Tropical shortest-path: `"ij,j->i"` with min/add, verify against manual
3. Three-operand: `"ij,jk,kl->il"`

## Phase 7: Fusion (optional, deferred)

`normalize_contracts` — semantic-level rewrite fusing adjacent contractions.

## Phase 8: Custom semiring end-to-end (deferred)

Smooth-tropical from legacy example 05.

## Deferred items

- Block-wise splitting for memory-safe broadcast
- Backend fast-paths (real semiring → native einsum)
- Semiring law checking using zero/one identity values
- Diagonal/trace semantics (repeated labels in single operand)
- `_lower_domain_prims` full recursion through all MorphismExpr variants
