# Tensor Extension — Next Steps

## Current state: Phase 5 implemented

Tensor contractions compile inside the tensor extension before returning to
core semantics. `ContractSpec` is internal compile data only; it is not returned
as `Prim(ContractSpec)` and there is no `main.py` domain-lowering traversal.

The compiler builds ordinary substrate composition:
```
par_all(align_0, align_1, ...)
>> fold_product(sr.times, n_inputs)
>> axis_reduce(sr.plus_reduce, reduced_axes)
```

For adjoint mode:
```
par_all(align_0, align_1, ...)
>> fold_product(sr.adjoint, n_inputs)
>> axis_reduce(sr.times_reduce, reduced_axes)
```

Implementation anchors:
- `main.py` is orchestration only. It loads backend env and passes the loaded `BackendOps` as opaque `domain_context`.
- `semantics/construct.py` delegates domain-tagged expression nodes to the registered domain protocol and includes `_domain_context` in the extension env.
- `tensors/semantics.py::construct_expr` resolves the semiring and calls `contract_morphism(..., context=...)`.
- `tensors/primitives.py` consumes tensor compile data and returns a composed `Morphism`.
- `structure/realize.py` remains the only lowering path from known `MorphismExpr` nodes to Hydra terms.

### Backend structural metadata

Backend JSON specs carry structural sections:
```json
"structural": {
  "expand_dims": "numpy.expand_dims",
  "transpose": "numpy.transpose"
}
```

These entries are not ordinary user morphisms. They are generic backend metadata
resolved by `BackendOps.from_spec()` and used by tensor-generated primitive
closures. No numpy/torch/jax/cupy-specific contraction behavior belongs in
`runtime/backend.py`.

### Generated primitive pattern

Each generated tensor primitive follows the existing backend primitive impl
shape and is attached through `Morphism.aux_primitives`:
```python
def impl(ctx, graph, args, *, store, fn, ...):
    key = arg_coder.encode(ctx, graph, args[0])
    arr = store.get(key)
    result = fn(arr)  # expand_dims+transpose, or axis-reduce
    return result_coder.decode(ctx, store.put(result))
```

TypeScheme: `BINARY → BINARY`. Registered via aux_primitives.

## Verified

- Real semiring matmul: `"ij,jk->ik"` with add/multiply, verified against `numpy.matmul`
- Tropical/min-plus matvec: `"ij,j->i"` with minimum/add, verified against manual NumPy
- Existing backend behavior still works, including `load numpy; let f = add >> tanh`
- Full test suite: `342 passed, 6 skipped`

## Next: Phase 6 fusion (optional, deferred)

`normalize_contracts` — semantic-level rewrite fusing adjacent contractions.

## Transformer/encoder DSL follow-ups

The notebook encoder smoke test now composes attention and an FFN residual in
DSL syntax, but it intentionally stays within currently registered primitives.
The remaining work for a production-faithful Transformer layer is:

- Axis-specialized softmax. `softmax` is available as a unary backend primitive,
  but the current DSL path does not yet bake an axis like `axis=-1`. Tensor
  contractions have internal axis-aware generated reductions, but that machinery
  is not yet exposed as a general axis-aware activation primitive.
- ReLU or GELU coverage. `relu` is not currently registered in the backend JSON
  specs. The notebook uses `tanh` as the FFN activation because it is available.
- Layer normalization. Proper layer norm needs axis-aware mean/variance/sqrt
  composition or a generated backend primitive with baked axis/epsilon metadata.
  Do not hide this in Python-only notebook helpers; it should be represented by
  backend metadata and constructed through the DSL/runtime boundary.
- Attention masking. Masked attention needs either mask-aware softmax or explicit
  DSL support for adding a mask before softmax.
- Routing carried parameters through larger composed blocks. Attention and FFN
  residual blocks can each be composed in DSL syntax today, and functors can act
  on both. A single full encoder morphism also needs to carry FFN weights through
  the attention stage and then assemble the next payload using projections. The
  current projection/parallel typing path can leave unresolved type variables in
  that shape, so the notebook keeps attention and FFN as two executable DSL
  morphisms and checks their combined NumPy parity at the notebook level.

## Phase 7: Custom semiring end-to-end (deferred)

Smooth-tropical from legacy example 05.

## Deferred items

- Fix lexer/parser comment robustness. The lexer is intended to support `#`
  line comments via `_comment()` and `_skip()` in `syntax/_lex.py`, and short
  comments work. Longer comment lines can currently trigger a `RecursionError`
  in the Hydra parser-combinator stack during tokenization. Repro:
  `tokenize("# " + "x" * 200 + "\nlet a = id\n")`. This should be fixed
  before relying on richly commented DSL examples or notebooks.
- Block-wise splitting for memory-safe broadcast
- Backend fast-paths (real semiring → native einsum)
- Semiring law checking using zero/one identity values
- Diagonal/trace semantics (repeated labels in single operand)
