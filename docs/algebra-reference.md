# algebra/ — Developer Reference

## Overview

`algebra/` is the declaration layer of unified-algebra. It defines the four core
domain objects — `Sort`, `Semiring`, `Equation`, and the contraction engine — as
pure data structures with no execution logic. Everything in this module produces
Hydra terms or resolved Python objects that the assembly layer lowers to callables.
No backend calls happen here.

## Architecture

| File | Role |
|------|------|
| `sort.py` | `Sort`, `ProductSort`, `Lens`, `sort_wrap` — typed sort declarations |
| `semiring.py` | `Semiring`, `Semiring.Resolved` — semiring declarations and backend resolution |
| `equation.py` | `Equation` — einsum op declaration with sort, rank, and axis metadata |
| `contraction.py` | `CompiledEinsum`, `semiring_contract`, `contract_and_apply`, `contract_merge`, `CONTRACTION_REGISTRY` |

Data flow:

```
Sort + Semiring + Equation
        │
        ▼
  Semiring.resolve(backend)     ← algebra/semiring.py
        │
        ▼
  Semiring.Resolved             ← carries contraction_fn, plus_elementwise, etc.
        │
        ▼
  compile_einsum(einsum_str)    ← algebra/contraction.py
        │
        ▼
  CompiledEinsum                ← axis index maps, input/output var lists
        │
        ▼
  semiring_contract(...)        ← executes via backend
        │
        ▼
  backend array
```

`algebra/` imports from `terms.py` only. It never imports from `assembly/`, `morphism/`, or `parser/`.

## Key abstractions

### `Sort`

A named tensor type with an optional semiring and optional named axes.

```python
@dataclass
class Sort:
    name: str
    semiring: core.Term | None      # Hydra semiring term
    axes: list[SortAxis] | None     # named axes with optional sizes
```

Constructed via `Sort({"name": "Feature", "semiring": sr_term})` or from the DSL.
`sort_wrap(s)` converts a `Sort | ProductSort | core.Type` into a Hydra term
suitable for embedding inside other Hydra records.

### `ProductSort`

A product of two or more sorts representing tuple types.

```python
class ProductSort:
    sorts: list[Sort | ProductSort]
```

Used as domain or codomain of morphisms and lenses. `ProductSort([A, B])` represents
`A × B`. Nested products are allowed.

### `Lens`

A sort-level record pairing a residual sort with a focus sort. Lives in `sort.py`.
`Lens.residual_sort` is currently set but not wired into optic threading by assembly.

### `Semiring`

A named semiring declaration carrying plus, times, zero, one, and optional residual
operation names.

```python
class Semiring:
    name: str
    plus: str        # backend op name for ⊕
    times: str       # backend op name for ⊗
    zero: float | int
    one: float | int
    residual: str | None
```

#### `Semiring.Resolved`

The runtime-ready form produced by `sr.resolve(backend)`.

```python
@dataclass
class Semiring.Resolved:
    contraction_fn: Callable        # (contraction_shape, backend, params) → array
    plus_elementwise: Callable      # (a, b) → a ⊕ b
    times_reduce: Callable          # (tensor, axis) → reduced tensor
    residual_elementwise: Callable | None
```

Key method: `with_adjoint(op_name="") -> Semiring.Resolved` — returns a new
`Resolved` with `contraction_fn` swapped to use `residual_elementwise`. Raises
`ValueError` if `residual_elementwise` is None.

### `Equation`

A tensor op declaration: an einsum string, domain/codomain sorts, optional axis names
and dimension constraints, and a semiring reference.

```python
class Equation:
    name: str
    einsum: str                      # e.g. "bi,oi->bo"
    semiring: core.Term
    domain_sort: Sort | None
    codomain_sort: Sort | None
    adjoint: bool
    nonlinearity: str | None
    inputs: list[str] | None         # explicit input axis names
```

`Equation.make(name, einsum, semiring, ...)` is the primary constructor.
`eq.to_term()` serialises to a Hydra term for passing to `compile_program`.

### `CompiledEinsum` and `semiring_contract`

`compile_einsum(einsum_str)` parses an einsum string into a `CompiledEinsum` that
carries input variable lists, output variable list, and axis-to-index maps.

`semiring_contract(compiled, tensor_args, sr_resolved, backend, params=(), block_size=None)`
executes the contraction against a backend. `block_size` enables chunked evaluation
for memory-bounded contexts.

`CONTRACTION_REGISTRY` is a mutable dict mapping strategy names to callables. Register
a custom strategy with `CONTRACTION_REGISTRY["my_strategy"] = fn`.

## Public API reference

### `Sort(fields)`

Construct a sort. `fields` is a dict with `"name"` (required) and optional
`"semiring"` (Hydra term) and `"axes"` (list of `SortAxis`).

### `ProductSort(sorts)`

Construct a product sort from a list of `Sort | ProductSort`.

### `sort_wrap(s) -> core.Term`

Convert a `Sort | ProductSort | core.Type` to a Hydra term for embedding.

### `Semiring(fields)`

Construct a semiring declaration. `fields` dict: `"name"`, `"plus"`, `"times"`,
`"zero"`, `"one"`, optional `"residual"`.

### `Semiring.resolve(backend) -> Semiring.Resolved`

Resolve op names against a backend. Validates semiring laws by default.
Pass `check_laws=False` for approximate semirings.

### `Semiring.Resolved.with_adjoint(op_name="") -> Semiring.Resolved`

Return a new `Resolved` with contraction swapped for the adjoint form.
Raises `ValueError` if `residual_elementwise` is None.

### `Equation.make(name, einsum, semiring, *, domain_sort=None, codomain_sort=None, adjoint=False, nonlinearity=None, inputs=None) -> Equation`

Primary constructor. `semiring` must be a Hydra term (from `Semiring.to_term()`).

### `Equation.to_term() -> core.Term`

Serialise to a Hydra term for `compile_program(equations=[...])`.

### `compile_einsum(einsum_str) -> CompiledEinsum`

Parse an einsum string. Raises `ValueError` on malformed input.

### `semiring_contract(compiled, tensor_args, sr, backend, params=(), block_size=None)`

Execute a contraction. `sr` must be a `Semiring.Resolved`. Returns a backend array.

### `CONTRACTION_REGISTRY`

Dict of named contraction strategy overrides. Register hooks here before compiling.

## How to use

```python
from unialg.algebra import Sort, Semiring, Equation
from unialg.backend import NumpyBackend
import numpy as np

backend = NumpyBackend()

# 1. Declare a semiring
sr = Semiring({"name": "real", "plus": "add", "times": "multiply", "zero": 0.0, "one": 1.0})

# 2. Declare sorts
feature = Sort({"name": "Feature"})
label   = Sort({"name": "Label"})

# 3. Declare an equation
eq = Equation.make(
    name="dense",
    einsum="bi,oi->bo",
    semiring=sr.to_term(),
    domain_sort=feature,
    codomain_sort=label,
)

# 4. Resolve semiring against backend
sr_res = sr.resolve(backend)

# 5. Compile the einsum
from unialg.algebra.contraction import compile_einsum, semiring_contract
compiled = compile_einsum("bi,oi->bo")

# 6. Contract
W = np.random.randn(16, 32).astype("float32")
x = np.random.randn(4, 32).astype("float32")
result = semiring_contract(compiled, [x, W], sr_res, backend)

# 7. Or compile the whole program via assembly
from unialg.assembly import compile_program
program = compile_program([eq.to_term()], backend=backend, semirings={"real": sr})
result = program("dense", W, x)

# 8. Adjoint form
sr_adj = sr_res.with_adjoint("dense")
```

## Common patterns and gotchas

**`sort_wrap` is idempotent.** Calling it on an already-wrapped term is safe; it
returns the term unchanged.

**`Semiring.resolve` caches nothing.** Call it once and reuse `Semiring.Resolved`;
resolving repeatedly against the same backend is wasteful.

**`compile_einsum` is cheap.** It only parses strings — no backend calls. Cache
`CompiledEinsum` if calling `semiring_contract` in a hot loop.

**`Equation.inputs` controls axis name validation.** When set, assembly validates
that the einsum input axes match the declared names. Omit for unvalidated equations.

**`with_adjoint` requires `residual=` on the semiring.** If the semiring declaration
has no `residual` op, `with_adjoint` raises immediately. Declare `residual` in the
`.ua` source or in `Semiring({"residual": "..."})`.

**`ProductSort` constraints.** Assembly validates product sort arity at graph
edges — a 2-element product is required for lenses, unit for `delete`, etc. Errors
surface as `TypeError` from `compile_program`.

**Axis dimension validation is opt-in.** Sizes must be set on `SortAxis` objects;
unsized axes skip dimension checking and only validate axis name compatibility.

**`CONTRACTION_REGISTRY` is global mutable state.** Tests that register hooks must
clean up in `finally` blocks to avoid cross-test contamination.

**`algebra/` never calls the backend.** Resolution (`Semiring.resolve`) and
contraction (`semiring_contract`) take a backend argument — the algebra layer
holds no backend reference of its own. This is by design: the algebra layer is
declaration-only.
