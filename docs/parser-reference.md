# parser/ — Developer Reference

## Overview

`parser/` transforms `.ua` source text into resolved Python objects ready for
assembly. It operates in two strictly separated passes: the grammar pass
(`_grammar.py`) turns text into raw `Decl` nodes, and the resolution pass
(`_resolver.py`) validates and converts those nodes into typed algebra objects
(`UASpec`). The two public entry points are `parse_ua_spec` (parse only) and
`parse_ua` (parse + compile).

## Architecture

| File | Role |
|------|------|
| `_grammar.py` | Hydra parser combinators; builds `list[Decl]` from source text; `_source_location` error helper |
| `_resolver.py` | Resolves `list[Decl]` into `UASpec`; validates semiring/sort/equation/functor references |
| `_decl_ast.py` | `Decl` node dataclasses: `ImportDecl`, `AlgebraDecl`, `OpDecl`, etc. |
| `_pratt.py` | Generic Pratt parser used for define-expr, poly-expr, and cell-expr sub-grammars |
| `_cell_ast.py` | `CellExpr` constructors for morphism cell expressions |
| `_resolve_cells.py` | Resolves cell declarations into `NamedCell` objects; synthesises adjoint equations |
| `_parse.py` | `parse_ua_spec` and `parse_ua` — public entry points |
| `__init__.py` | Re-exports: `parse_ua_spec`, `parse_ua`, `UASpec`, `NamedCell` |

Data flow:

```
.ua source text
      │
      ▼
  _grammar._build_parser()       Hydra parser combinators
      │
      ▼  list[Decl]
  _resolver._resolve_spec()      validates + resolves references
      │
      ▼
   UASpec                        typed Python object
      │
      ├─ equations  → compile_program(equations, ...)
      ├─ semirings  → compile_program(semirings=...)
      ├─ cells      → compile_program(cells=...)
      └─ defines    → register_defines(defines, backend)
```

Three Pratt sub-parsers handle operator expressions within declarations:

| Sub-parser | Used for | Key operators |
|------------|----------|---------------|
| `_define_expr` | `define` body expressions | function calls, arithmetic ops |
| `_poly_expr` | functor body `F = ...` | `+`, `*`, `->` |
| `_cell_expr` | cell body morphisms | `;`, `&`, `~`, `>`, `<`, `*[R]` |

## Key abstractions

### `UASpec`

The result of `parse_ua_spec`. All fields are lists; missing declarations yield
empty lists (never `None`).

```python
@dataclass
class UASpec:
    equations:    list[core.Term]          # Equation.to_term() outputs
    semirings:    dict[str, Semiring]      # name → Semiring object
    cells:        list[NamedCell]          # compiled cell morphisms
    defines:      list[tuple]              # (arity, name, params, DefineExpr)
    share_groups: dict[str, list[str]]     # weight-tying groups
    backend_name: str | None              # from 'import <backend>'
    functors:     list[Functor]            # polynomial functor declarations
    sorts:        list[Sort]              # declared sorts (extra_sorts)
```

### Decl node types

| Class | DSL keyword | Key fields |
|-------|-------------|------------|
| `ImportDecl` | `import` | `name` |
| `AlgebraDecl` | `semiring` | `name`, `kw_args` |
| `SpecDecl` | `sort` | `name`, `attrs` |
| `OpDecl` | `op` | `name`, `sig`, `kw_args`, `attrs` |
| `ShareDecl` | `share` | `group`, `names` |
| `DefineDecl` | `define unary/binary` | `arity`, `name`, `params`, `body` |
| `FunctorDecl` | `functor` | `name`, `body` (PolyExpr), `category` |
| `CellDecl` | `cell` | `name`, `sig`, `expr` (CellExpr) |

### `DefineExpr`

The IR for define-declaration body expressions. A `TermInject` union with three variants:

| Constructor | `kind` | Meaning |
|-------------|--------|---------|
| `def_lit(value)` | `"lit"` | Literal constant |
| `def_var(name)` | `"var"` | Parameter reference |
| `def_call(fn, args)` | `"call"` | Function application (1 or 2 args) |

`def_call` enforces arity at construction: 1-arg → unary, 2-arg → binary. Raises
`ValueError` for 0 or 3+ args. Lives at `unialg._define_ast`, not in `parser/`.

### `NamedCell`

A named `TypedMorphism` produced by resolving a `CellDecl`.

```python
@dataclass
class NamedCell:
    name: str
    morphism: TypedMorphism
```

Passed to `compile_program(cells=[...])`.

### Pratt parser

`_pratt.py` provides a generic `PrattParser` and `parse_pratt(tokens, nud, led, rbp)`
function. Binding powers follow standard Pratt conventions: higher number = tighter
binding. The three sub-parsers (`_define_expr`, `_poly_expr`, `_cell_expr`) are
created inside `_build_parser()` and share the same Hydra parser combinator context.

## Public API reference

### `parse_ua_spec(text: str) -> UASpec`

Parse `.ua` source into a `UASpec` without compiling. Raises `SyntaxError` on
parse failure or unconsumed input. Does not call any backend or assembly code.

```python
from unialg.parser import parse_ua_spec
spec = parse_ua_spec("""
    semiring real (plus=add, times=multiply, zero=0, one=1)
    op dense[real]: feature -> label = "bi,oi->bo"
""")
print(spec.semirings)   # {"real": Semiring(...)}
print(spec.equations)   # [core.Term(...)]
```

### `parse_ua(text: str, backend=None) -> Program`

Parse and compile `.ua` source to a `Program`. If `backend=None`, uses the backend
named by `import <name>` in the source. A `backend=` kwarg overrides the source
import. Raises `ValueError` if no backend is available.

```python
from unialg.parser import parse_ua
from unialg.backend import NumpyBackend
import numpy as np

program = parse_ua("""
    semiring real (plus=add, times=multiply, zero=0, one=1)
    op dense[real]: feature -> label = "bi,oi->bo"
""", backend=NumpyBackend())

W = np.random.randn(16, 32).astype("float32")
x = np.random.randn(4, 32).astype("float32")
result = program("dense", W, x)
```

### `UASpec` fields

See the Key abstractions section above. All list/dict fields are always present;
check `.equations`, `.semirings`, etc. directly without `None` guards.

### Re-exported names

`from unialg.parser import parse_ua_spec, parse_ua, UASpec, NamedCell`

## How to parse

**Parse only (no backend):**
```python
spec = parse_ua_spec(source)
program = compile_program(spec.equations, backend=backend, semirings=spec.semirings)
```

**Parse and compile in one call:**
```python
program = parse_ua(source, backend=backend)
```

**With define ops:**
```python
from unialg.assembly import register_defines
spec = parse_ua_spec(source)
backend = register_defines(spec.defines, backend)
program = compile_program(spec.equations, backend=backend, semirings=spec.semirings)
```

**With cells (typed morphisms):**
```python
spec = parse_ua_spec(source)
program = compile_program(
    spec.equations,
    backend=backend,
    semirings=spec.semirings,
    cells=spec.cells,
)
```

**With share groups (weight tying):**
```python
program = compile_program(
    spec.equations,
    backend=backend,
    semirings=spec.semirings,
    share_groups=spec.share_groups,
)
```

**With functor declarations:**
```python
spec = parse_ua_spec(source)
# spec.functors contains Functor objects for algebra_hom use
```

**Error handling:**
```python
try:
    spec = parse_ua_spec(bad_source)
except SyntaxError as e:
    print(e.lineno, e.offset, e.text)   # line, col, source line
```

## Common patterns and gotchas

**Declaration order is enforced.** Semirings must be declared before ops that
reference them. Sorts must be declared before equations that use them. The resolver
raises `ValueError` on forward references.

**`parse_ua_spec` is cheap; `parse_ua` is not.** `parse_ua_spec` only parses and
resolves. `parse_ua` additionally calls `compile_program`, which resolves backends,
builds a Hydra graph, and freezes primitives. Separate them when you need the spec
for introspection before committing to a backend.

**Product sort syntax.** `A * B` in a sort position means `ProductSort([A, B])`.
The `*` operator is only valid in sort signatures, not in cell expressions (where
`*[R]` is lens residual annotation).

**`inputs` is the only list-valued op attribute.** `op dense[real](inputs=[b,i]):
...` declares named input axes. All other attributes are scalar.

**Adjoint synthesis.** The `'` suffix on a cell name (`cell foo': ...`) triggers
`_ensure_adjoint_eq` in `_resolve_cells.py`, which synthesises an adjoint equation
and appends it to the equation list. This is done explicitly by the resolver, not
as a side effect.

**`~` vs `lens(f,g)*[R]`.** In cell expressions, `~f` is a fixpoint cell.
`lens(fwd, bwd)*[R]` constructs a lens with residual sort `R`.

**`>[F]` and `<[F]` are catamorphism and anamorphism.** `>` folds, `<` unfolds.
Both take a functor name in brackets.

**`?` modifier is parsed but unimplemented.** The grammar accepts `op foo?[sr]`
but the resolver does not yet wire this into any semantic.

**`_source_location` is in `_grammar.py`.** It is an internal helper; do not import
it from `__init__`.
