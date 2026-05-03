# morphism/ — Developer Reference

## Overview

`morphism/` owns the typed morphism layer of unified-algebra. It defines what
a morphism *is* (a Hydra term paired with domain and codomain sorts), provides
smart constructors for the common categorical shapes, and supplies the
algebra-hom machinery for building catamorphisms over polynomial functors.

The module sits between two layers:

- **Above it (assembly/):** `assembly/` takes `TypedMorphism` objects and
  compiles them into Hydra graphs for execution. `morphism/` never imports from
  `assembly/` or `parser/`.
- **Below it (algebra/):** `morphism/` imports `Sort`, `ProductSort`, and
  `Semiring` from `algebra/` but adds no resolution or compilation logic of
  its own.

In the DSL, a morphism, an equation, and an op are the same construct seen from
different angles. `morphism/` provides the Python-level representation.

## Architecture

| File | Role |
|------|------|
| `_typed_morphism.py` | Core `TypedMorphism` class; boundary helpers `unwrap()`, `product()`, `unit()`, `same_sort()` |
| `morphism.py` | Leaf constructors (`eq`, `lit`, `iden`, `copy`, `delete`) and composition combinators (`seq`, `par`) |
| `algebra_hom.py` | `algebra_hom()` — induces a catamorphism over a `Functor`; `summand_domain()` helper |
| `functor.py` | `Functor` and `PolyExpr` — polynomial endofunctor declarations |
| `lens.py` | `lens()` and `lens_seq()` — lens morphism constructors |

Dependency arrows within the module:

```
morphism.py  ──►  algebra_hom.py  ──►  functor.py
     │                   │                  │
     └───────────────────┴──────────────────┴──► _typed_morphism.py
                                                        │
                                                   algebra/ (Sort, ProductSort)
```

`morphism/` imports from `algebra/` and `terms.py` only. No imports from `assembly/` or `parser/`.

## Key abstractions

### `TypedMorphism`

A Hydra function term annotated with its domain and codomain boundaries.

```python
class TypedMorphism:
    term: hydra.core.Term          # The underlying Hydra term
    domain_sort: SortLike          # As passed to the constructor
    codomain_sort: SortLike        # As passed to the constructor
    _function_type: core.Type      # Normalized Hydra TypeFunction
```

`SortLike = Sort | ProductSort | core.Type`. The constructor calls `TypedMorphism.unwrap()` on `term`, so passing a `_RecordView` subclass works as well as a raw Hydra term.

Key static / class methods:

| Method | Purpose |
|--------|---------|
| `TypedMorphism.unwrap(value)` | Strip record-view wrappers to a raw Hydra term |
| `TypedMorphism.require(value, label)` | Assert `value` is a `TypedMorphism` |
| `TypedMorphism.same_sort(actual, expected, label)` | Assert boundary types match |
| `TypedMorphism.product(*sorts)` | Build a Hydra pair/unit type from one or more `SortLike` values |
| `TypedMorphism.unit()` | Hydra unit type |
| `TypedMorphism.split_product2(value, label)` | Assert `value` is a `TypePair`; return `(first, second)` |

### `Functor` and `PolyExpr`

A `Functor` is a named polynomial endofunctor `F : C → C` with a category-of-discourse tag.

```python
class Functor(_RecordView):
    name: str           # declared name
    body: PolyExpr      # polynomial expression for F
    category: str       # "set" (default) or "poset"
```

A `PolyExpr` wraps a `TermInject` of union type `ua.functor.PolyExpr`. Its `.kind` property returns the variant tag.

| `kind` | Meaning | Accessible fields |
|--------|---------|-------------------|
| `"zero"` | Initial object `0` | — |
| `"one"` | Terminal object `1` | — |
| `"id"` | Identity `X` (recursion variable) | — |
| `"const"` | Constant functor `S` | `.sort` |
| `"sum"` | Coproduct `F + G` | `.left`, `.right` |
| `"prod"` | Product `F * G` | `.left`, `.right` |
| `"exp"` | Exponential `A → F` | `.base_sort`, `.body` |

`category="poset"` means the body must be `id_()` and the induced algebra hom is a Tarski least-fixpoint iteration. Call `functor.validate()` explicitly before passing to assembly (the assembly layer does this).

### `algebra_hom()`

Induces a catamorphism over a declared `Functor`.

Supported functor shapes:
- `F = 1 + Const × X` (List): `cata(init, cons)` where `cons.domain = product(B, C)`, `init.domain = unit()`
- `F = 1 + X` (Maybe): `cata(nothing, just)`

Other shapes raise `NotImplementedError`.

### `lens()` and `lens_seq()`

A lens morphism packs forward and backward components into a Hydra record (`ua.morphism.Lens`). The forward component's codomain must be `R × A` (residual first, focus second). The backward component's domain must be `R × B`.

`lens_seq(l1, l2)` composes two lenses sequentially. Both must be `TypedMorphism`. When both carry a `residual_sort`, the composed residual is `ProductSort([l1.residual, l2.residual])`.

## Public API reference

All symbols below are importable from `unialg.morphism`.

### `TypedMorphism(term, domain, codomain)`

Construct a typed morphism. `term` may be any Hydra term or `_RecordView` subclass.

### `eq(name, *, domain, codomain)`

Reference to a declared `Equation` by name. Builds `Terms.var("ua.equation.<name>")`.

### `lit(value_term, sort)`

0-ary constant morphism `1 → A`.

### `iden(sort)`

Identity morphism `id_A : A → A`.

### `copy(sort)`

Comonoid copy `Δ_A : A → A × A` as `λx. (x, x)`.

### `delete(sort)`

Comonoid delete `!_A : A → 1` as `λ_. ()`.

### `seq(f, g)`

Sequential composition `f ; g`. Requires `f.codomain_type == g.domain_type`.

### `par(f, g)`

Monoidal product `f ⊗ g` via `hydra.lib.pairs.bimap`.

### `algebra_hom(functor, direction, morphisms) -> TypedMorphism`

```python
algebra_hom(
    functor: Functor,
    direction: str,           # "algebra" | "coalgebra"
    morphisms: list[TypedMorphism],
) -> TypedMorphism
```

### `lens(forward, backward, residual_sort=None) -> TypedMorphism`

### `lens_seq(l1, l2) -> TypedMorphism`

### PolyExpr constructors

| Constructor | Signature |
|-------------|-----------|
| `zero()` | `() -> PolyExpr` |
| `one()` | `() -> PolyExpr` |
| `id_()` | `() -> PolyExpr` |
| `const(sort)` | `(Sort | ProductSort) -> PolyExpr` |
| `sum_(left, right)` | `(PolyExpr, PolyExpr) -> PolyExpr` |
| `prod(left, right)` | `(PolyExpr, PolyExpr) -> PolyExpr` |
| `exp(base_sort, body)` | `(Sort | ProductSort, PolyExpr) -> PolyExpr` |

## How to build and compose morphisms

```python
from unialg.algebra import Sort, ProductSort
from unialg.morphism import eq, iden, seq, par, copy, lens, lens_seq
import hydra.dsl.terms as Terms

# Define sorts
feature = Sort({"name": "Feature", "semiring": semiring_term})
label   = Sort({"name": "Label",   "semiring": semiring_term})

# Leaf morphisms
linear = eq("linear", domain=feature, codomain=label)

# Sequential composition
pipeline = seq(linear, activation_morphism)

# Monoidal product
parallel = seq(copy(feature), par(linear, iden(feature)))

# Build a lens
residual     = Sort({"name": "Residual", "semiring": semiring_term})
fwd_codomain = ProductSort([residual, label])
forward  = eq("my_fwd", domain=feature,      codomain=fwd_codomain)
backward = eq("my_bwd", domain=fwd_codomain, codomain=feature)
l = lens(forward, backward, residual_sort=residual)

# Pass to assembly
from unialg.assembly import compile_program
program = compile_program(equations=[...], cells=[cell], backend=backend)
```

## Common patterns and gotchas

**Lens residual ordering is strict.** `forward.codomain` must be a 2-element product with residual as the **first** element and focus as the second. `lens()` validates this at construction time with `split_product2`.

**`algebra_hom` summand ordering.** For `F = 1 + Const × X`, `morphisms` must align with `functor.summands()`. The `one`-branch morphism is detected by its `unit()` domain, so ordering between `init` and `cons` is handled automatically.

**`algebra_hom` for List: `Const` must precede `Id` in the product.** `prod(const(B), id_())` is supported. `prod(id_(), const(B))` raises `NotImplementedError` — reorder it.

**`category="poset"` is not validated at construction.** `Functor.validate()` must be called explicitly. A poset functor with a non-`id_()` body passes construction silently.

**`algebra_hom` handles only `List` and `Maybe` today.** Tree, stream, and Mealy shapes raise `NotImplementedError`.

**`_LENS_TYPE` and `_LENS_SEQ_TYPE` are semi-private.** They are exported from `morphism/__init__.py` with underscore prefixes and consumed by `assembly/_morphism_compile.py` for dispatch. Do not use them outside assembly.
