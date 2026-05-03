# Assembly Layer Reference

## Overview

`assembly/` is the post-resolution realization layer. It takes resolved semantic objects — `Equation` lists, `TypedMorphism`/`NamedCell` lists, a `Backend`, optional `Semiring` maps and parameter bindings — and produces a `Program`: a compiled, callable wrapper around a Hydra `Graph`. Every tensor computation runs through `reduce_term` on that graph; the assembly layer's job is to register all backend callables as Hydra primitives or `bound_terms` before the graph is frozen.

## Architecture

### File roles

| File | Role |
|---|---|
| `__init__.py` | Public exports: `Program`, `compile_program`, `assemble_graph`, `rebind_params`, `register_defines` |
| `program.py` | `Program` class and `compile_program()` factory; entry-point lookup, `__call__`, `rebind`, `type_check` |
| `graph.py` | `assemble_graph()` and `build_graph()`: orchestrates equation resolution, cell registration, and Hydra `Graph` construction; `rebind_params()` |
| `_equation_resolution.py` | Compiles `Equation` objects to Hydra `Primitive` instances; handles einsum, nonlinearity, skip connections, adjoint flag, and list-packing for high-arity ops |
| `_morphism_compile.py` | Compiles `TypedMorphism`/`NamedCell` terms into native callables or `CompiledLens`; registers them as primitives or `bound_terms` |
| `_define_lowering.py` | Compiles `define` declarations (expression ASTs) to unary/binary backend callables scoped to a `_ScopedBackend` |
| `_validation.py` | Topological sort of the equation DAG; checks sort, rank, axis name, and dimension compatibility at all inter-equation junctions |

### Data flow

```
[Equation list]    [NamedCell list]    [Backend]    [params dict]
      |                   |                |               |
      v                   |           resolve semirings    |
_resolve_equations()      |                               |
  compile_equation()      |                               |
  resolve_equation()      |                               |
  validate_pipeline()     |                               |
      |                   v                               |
 primitives dict    register_cells()                      |
 native_fns dict      compile_morphism()                  |
 list_packed_info     lenses -> 2 primitives              |
      |               fallback -> bound_terms             |
      +-------------------+-------------------------------+
                          |
                    build_graph()
             elements_to_graph(parent, schema, bindings)
                          |
                     hydra.Graph
                          |
                  compile_program()
        wraps graph + tensor_coder + EMPTY_CX
                          |
                       Program
```

Parameters are stored as `bound_terms` under the `ua.param.<name>` key. On `rebind()`, either `rebind_params()` performs substitution in-place, or `compile_program()` is called again with merged params (when `_build_args` is present).

## Key abstractions

### `Program`

A frozen, callable wrapper around a Hydra `Graph`. Do not construct directly; use `compile_program()`.

Internal fields: `_graph` (the `hydra.graph.Graph`), `_backend`, `_coder` (`tensor_coder(backend)`), `_cx` (`empty_context()`), `_build_args` (original `compile_program()` kwargs), `_list_packed_info` (maps `Name` → `(n_params, n_inputs)` for high-arity ops).

`Program.__call__(entry_point, *args)` pipeline:
1. Resolve entry-point short name via `_resolve_full_name()` (adds `ua.X.` prefix)
2. Encode each array with `coder.decode(None, arg)` (Hydra convention: "decode" converts domain objects into Hydra terms)
3. Build a Hydra application term; list-pack args when `_list_packed_info` says to
4. Call `reduce_term(cx, graph, True, term)`
5. Decode result with `coder.encode(None, None, reduced)` back to a backend array

### `compile_program()`

```python
def compile_program(
    equations: list,
    *,
    backend,
    params: dict | None = None,
    extra_sorts: list | None = None,
    semirings: dict | None = None,
    cells: list | None = None,
    share_groups: dict | None = None,
) -> Program
```

Single entry point for DSL output and hand-written Python alike. `share_groups` implements Para-style weight tying: `{"group": ["op_a", "op_b"]}` makes `op_b`'s parameter slot alias `op_a`'s. The canonical op is `op_names[0]`.

### `Semiring.Resolved`

Consumed in `_resolve_equations()` via `sr.resolve(backend)`. Carries `contraction_fn`, `plus_elementwise`, `times_reduce`, and optionally `residual_elementwise`. The `adjoint` flag on an `Equation` requires `residual_elementwise` to be non-None. Call `sr.with_adjoint(op_name)` to get a resolved semiring with swapped contraction_fn.

### `_ScopedBackend`

A thin overlay over a `Backend` that adds scoped unary/binary ops without mutating the base. Created by `register_defines()`. Overrides `unary()`, `elementwise()`, and `reduce()` to read from shallow-copied op tables; forwards all other attribute access to the wrapped backend via `__getattr__`.

### `CompiledLens`

```python
@dataclass(frozen=True, slots=True)
class CompiledLens:
    forward: Callable
    backward: Callable
    residual_sort: object | None = None
```

Produced by `compile_morphism()` when the term is a lens record (`_LENS_TYPE`). Lenses are registered as two Hydra primitives: `ua.morphism.<name>.forward` and `ua.morphism.<name>.backward`. Lens-sequencing (`_LENS_SEQ_TYPE`) composes two `CompiledLens` values and produces a `ProductSort` residual when both have residuals.

### Entry-prefix naming contract

Every callable entry point carries a structured `ua.X.<name>` key. Recognised prefixes:

```python
_ENTRY_PREFIXES = (
    "ua.path.",
    "ua.fan.",
    "ua.fold.",
    "ua.unfold.",
    "ua.fixpoint.",
    "ua.parallel.",
    "ua.equation.",
    "ua.morphism.",
)
```

`Program.entry_points()` strips these to return short names. `__call__()` and `type_check()` accept short names and re-resolve them. Names ending in `.__merge__` are excluded (internal fan merge primitives). Parameters live under `ua.param.<name>` and do not appear in `entry_points()`.

## Public API reference

### `compile_program(equations, *, backend, ...) -> Program`

| Parameter | Type | Description |
|---|---|---|
| `equations` | `list[core.Term]` | Hydra terms encoding `Equation` declarations |
| `backend` | `Backend` | Provides tensor ops |
| `params` | `dict[str, core.Term] \| None` | Initial parameter bindings, keyed by op name |
| `extra_sorts` | `list \| None` | Additional sort objects to register in the schema |
| `semirings` | `dict[str, Semiring] \| None` | Named semirings to resolve |
| `cells` | `list[NamedCell] \| None` | Typed morphism cells to compile and register |
| `share_groups` | `dict[str, list[str]] \| None` | Weight tying groups |

Raises `ValueError` on duplicate equation names, unknown ops, and bad arity. Raises `TypeError` on sort/rank/axis mismatches from `validate_pipeline()`.

### `Program.__call__(entry_point, *args)`

Invoke a named entry point on backend arrays. `entry_point` is a short name (prefix stripped). Returns a backend array.

```python
result = program("dense", weights, inputs)
```

### `Program.rebind(**params) -> Program`

Return a new `Program` with named parameters substituted. Accepts `float`, `int`, or Hydra `Term`. Does not mutate the original. When `_build_args` is present (always true from `compile_program()`), triggers full recompilation.

```python
p2 = program.rebind(W=new_weights)
```

### `Program.entry_points() -> list[str]`

Returns sorted list of short entry-point names. Lens paths appear as `"name.fwd"` and `"name.bwd"`.

### `Program.type_check(entry_point) -> core.Type`

Returns the Hydra `Type` of the named entry point.

### `assemble_graph(eq_terms, backend, ...) -> tuple[Graph, dict, dict]`

Exported for testing and advanced use. Returns `(graph, native_fns, list_packed_info)`.

### `rebind_params(graph, updates) -> Graph`

Cheap in-place substitution via Hydra `substitute_in_term`. `updates` is `dict[str, core.Term]` keyed by bare op name (no `ua.param.` prefix). Does not recompile primitives.

### `register_defines(defines, backend) -> Backend`

```python
register_defines(defines: list[tuple], backend) -> Backend
```

Compile `define` declarations and return a new `_ScopedBackend` that overlays `backend` with the new ops. The original backend is not mutated. If `defines` is empty, returns `backend` unchanged. Each entry: `(arity, name, params, expr_ast)`. `arity` is `'unary'` or `'binary'`. `expr_ast` is a `DefineExpr` from `unialg._define_ast`. Must be called before `compile_program()`.

## How to compile and run programs

```python
from unialg.assembly import compile_program, register_defines
from unialg.backend import NumpyBackend
import numpy as np

# 1. Create a backend
backend = NumpyBackend()

# 2. Optionally register define ops (scoped, does not mutate backend)
backend = register_defines(spec.defines, backend)

# 3. Compile from a parsed UASpec
from unialg.parser import parse_ua_spec
spec = parse_ua_spec("""
    semiring real (plus=add, times=multiply, zero=0, one=1)
    op dense[real]: feature -> label = "bi,oi->bo"
""")
program = compile_program(
    spec.equations,
    backend=backend,
    semirings=spec.semirings,
)

# 4. Inspect entry points
print(program.entry_points())  # ['dense']

# 5. Run
W = np.random.randn(out_dim, in_dim).astype("float32")
x = np.random.randn(batch, in_dim).astype("float32")
y = program("dense", W, x)

# 6. Rebind parameters (full recompilation)
program2 = program.rebind(dense=W_new)
```

For direct Python use without the DSL:

```python
from unialg.algebra import Equation, Semiring, Sort

sr = Semiring({"name": "real", "plus": "add", "times": "multiply", "zero": 0.0, "one": 1.0})
eq = Equation.make(name="dense", einsum="bi,oi->bo", semiring=sr.to_term())
program = compile_program([eq.to_term()], backend=backend, semirings={"real": sr})
```

## Common patterns and gotchas

**`rebind()` triggers full recompilation.** When `_build_args` is present (always via `compile_program()`), `rebind()` calls `compile_program()` again. For cheap weight substitution when no fused primitives exist, call `rebind_params(program._graph, updates)` directly.

**`register_defines()` returns a new backend; capture the return value.** The original backend is not mutated. `parse_ua()` does this automatically; hand-written callers must do `backend = register_defines(defines, backend)`.

**Entry-prefix ambiguity.** `_resolve_full_name()` tries all eight prefixes in order and returns the first match. Use unique names across morphisms and equations.

**List-packing for high-arity ops.** When `n_params + n_inputs > 3`, the primitive receives a `TermList`. Code that bypasses `Program` and applies a term directly must replicate this packing.

**Lens `residual_sort` is populated but not consumed.** `CompiledLens.residual_sort` is set from the Hydra lens record but the assembly layer does not yet wire it into optic threading. Forward passes do not collect residuals; backward passes do not consume them in sequence.

**`validate_pipeline()` runs at compile time, not call time.** Sort, rank, axis-name, and dimension checks run inside `_resolve_equations()` before the graph is built. Errors surface as `TypeError` from `compile_program()`.
