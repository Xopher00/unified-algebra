# unialg Layer Reference

A developer guide to the module structure, layer responsibilities, and key abstractions.

## Overview

unialg compiles typed algebraic morphisms into executable Hydra terms. A morphism
is a typed arrow in a bicartesian closed category — it can be composed sequentially,
run in parallel, split over sums, mapped through polynomial functors, or folded over
recursive structures. The compilation pipeline runs through five layers, each with
exclusive ownership of one concern.

## Layer Architecture

```
objects.py                 ← type constructors and monad descriptors
syntax/expressions.py      ← expression syntax (no types, no Hydra)
         ↓
semantics/morphisms.py     ← typed morphism handles and combinators
semantics/functors.py      ← polynomial functor object action
semantics/optics.py        ← functor optics and recursion schemes
         ↓
structure/terms.py         ← Hydra term vocabulary
structure/realize.py       ← DSL-to-Hydra translation
structure/recursion.py     ← fixed-point optic bridge
         ↓
runtime/                   ← native backend boundary and Python codecs
         ↓
main.py                    ← compile/run orchestration boundary
         ↓
Hydra                      ← external graph reduction engine
```

Dependency policy follows ownership. `semantics/` depends only on foundational
objects and syntax, never on structure/runtime/orchestration. `structure/`
depends on syntax and semantics because it realizes validated semantic nodes,
but it does not depend on runtime or `main.py`. `runtime/` owns native boundary
machinery and must not depend on syntax, semantics, structure, or `main.py`.

## Files and Responsibilities

### `objects.py` — type-level ground

Thin constructors for Hydra `Type` values and monad descriptors.

| Symbol | Role |
|--------|------|
| `ProductType(l, r)` | Builds `TypePair` — the product object `l × r` |
| `SumType(l, r)` | Builds `TypeEither` — the coproduct object `l + r` |
| `ExpType(a, b)` | Builds `TypeFunction` — the exponential object `a → b` |
| `VoidType()` | Builds `TypeVoid` — the initial object `0` |
| `ListType(a)` | Builds `TypeList` — the list type constructor |
| `MaybeType(a)` | Builds `TypeMaybe` — the maybe type constructor |
| `Monad` | Descriptor for a monad: `type_ctor`, `bind_name`, `pure_name` |
| `MAYBE`, `LIST` | Pre-built monad descriptors for Hydra's standard monads |

No Hydra terms. No imports from other unialg modules.

---

### `syntax/expressions.py` — expression syntax ADTs

Two frozen-dataclass hierarchies: `MorphismExpr` (arrow expressions) and `PolyExpr`
(functor shape expressions). These are pure data — no type checking, no realization.

**`MorphismExpr` nodes:**

| Node | Meaning |
|------|---------|
| `Identity(space)` | `id_A : A → A` |
| `Copy(space)` | `Δ_A : A → A × A` |
| `Delete(space)` | `!_A : A → 1` |
| `First(ab)` | `π₁ : A × B → A` |
| `Second(ab)` | `π₂ : A × B → B` |
| `Left(ab)` | `ι₁ : A → A + B` |
| `Right(ab)` | `ι₂ : B → A + B` |
| `Absurd(cod)` | `absurd : 0 → A` |
| `Assoc(dom, cod)` | Associativity iso for `×` or `+` |
| `Symmetry(dom, cod)` | Symmetry iso for `×` or `+` |
| `DistributeLeft(dom, cod)` | `distl : A × (B + C) → (A × B) + (A × C)` |
| `DistributeRight(dom, cod)` | `distr : (A + B) × C → (A × C) + (B × C)` |
| `MonadicEmbed(f, monad)` | `η ∘ f` — lift a plain morphism into a monad |
| `ContextualBinary` | Base for `Compose`, `SharedCompose`, `Parallel`, `Coparallel`, `Pair`, `Case` |
| `PolyFmap(body, f, ...)` | Deferred functor action `F(f)` |
| `SelfRef(name, dom, cod)` | Self-reference inside a fixpoint equation |
| `AlgExpr(name, body, dom, cod)` | Base for deferred recursive schemes |
| `Cata(name, body, dom, cod)` | Deferred catamorphism |
| `Ana(name, body, dom, cod)` | Deferred anamorphism |
| `Prim(raw, dom, cod)` | Escape hatch: pre-built Hydra term with explicit types |

**`PolyExpr` nodes:** `Zero`, `One`, `Id`, `Const(space)`, `Sum(l, r)`, `Prod(l, r)`,
`Exp(base: PolyExpr, body)`, `PolyCompose(l, r)`, `List(body)`, `Maybe(body)`.

`pretty(expr)` renders any expression as a human-readable string.

---

### `semantics/morphisms.py` — typed morphism handles

`Morphism` is the central type. It wraps a `MorphismExpr` with:

- `param: Type` — a contextual parameter prefix (`TypeUnit()` = no parameter)
- `monad: Monad | None` — effect context for lax/effectful morphisms
- `aux_primitives: tuple` — Hydra primitives accumulated from subexpressions

Four modes:

| Mode | Raw term type |
|------|--------------|
| plain | `A → B` |
| parametric | `P × A → B` |
| lax | `A → T(B)` |
| lax-parametric | `P × A → T(B)` |

`dom()` and `cod()` return the *visible* domain and codomain, stripping the param
prefix and monad wrapper from the raw type.

**Combinators** all return a new `Morphism` with resolved `param` and `monad`:

```python
compose(f, g)              # f ; g  — requires f.cod() == g.dom()
par(f, g)                  # f × g  — A×C → B×D
copar(f, g)                # f + g  — A+C → B+D
pair(f, g)                 # ⟨f,g⟩  — A → B×C, requires f.dom() == g.dom()
case(f, g)                 # [f,g]  — A+B → C, requires f.cod() == g.cod()
distribute_left(a, b, c)   # distl  — A×(B+C) → (A×B)+(A×C)
distribute_right(a, b, c)  # distr  — (A+B)×C → (A×C)+(B×C)
merge(a)                   # ∇     — A+A → A, codiagonal
```

`shared_context=True` on any combinator shares a matching non-unit param between
children instead of combining them into `g_param × f_param`. Recursive morphisms
use this so the self-reference and algebra carry the same param, not a duplication.

`MorphismError.check(a, b, label)` raises `MorphismError` when `a != b`.

---

### `semantics/functors.py` — polynomial functor semantics

`Functor(name, body: PolyExpr)` is a named polynomial endofunctor.

Key methods:

| Method | Meaning |
|--------|---------|
| `apply(space)` | Computes `F(space)` — the object action |
| `unapply(fa)` | Solves `F(A) = fa` and returns `A` (via Hydra unification) |
| `compose(inner)` | Returns functor composition `self ∘ inner` |
| `map(h)` | Returns `poly_fmap(self, h)` — the semantic arrow action |
| `x_arity()` | Counts occurrences of `Id` in the body |
| `summands()` | Flattens top-level `Sum` nodes left-to-right |
| `consts()` | Collects all constant and exponential-base types |

`poly_fmap(functor, h)` returns a `Morphism` wrapping a `PolyFmap` deferred node.
No Hydra terms are built here — `realize.py` handles that when the term is needed.

Free-standing constructors (`zero()`, `one()`, `id_()`, `const(s)`, `sum_(f, g)`,
`prod(f, g)`, `exp(base: PolyExpr, body)`) return `PolyExpr` values for building functor bodies.

Dispatch dicts `_COMPOSE_POLY` and `_APPLY_POLY` encode the same 10 node types in
the same order — composition at expression level, application at type level. This
parallel structure is the functor algebra's combinator table.

---

### `semantics/optics.py` — polynomial functor optics and recursion

`Optic(functor, forward, backward, carrier=None)` encodes any polynomial functor
optic:

- `forward : S → F(A)` decomposes the source
- `backward : F(B) → T` reconstructs the target
- `focus` (derived) — `A`, extracted from `forward.cod()` via `functor.unapply`
- `replacement` (derived) — `B`, extracted from `backward.dom()` via `functor.unapply`

Three optic families, distinguished by `functor.body`:

| Family | Functor body | Focus type |
|--------|-------------|------------|
| Lens | `Id × Const(R)` | `A` from `A × R` |
| Prism | `Id + Const(R)` | `A` from `A + R` |
| Traversal | Arbitrary `PolyExpr` | `A` from `F(A)` |

**Optic actions:**

```python
optic.act(h)            # S → T — full optic action on morphism h : A → B
optic.act_forward(h)    # S → F(B) — decompose, then lift h through F
optic.act_backward(h)   # F(A) → T — lift h through F, then reconstruct
optic.compose(inner)    # Optic composition: focus through outer then inner
```

**Recursion schemes** (requires `carrier` to be set on the optic):

```python
cata(fp, alg)           # Catamorphism: fold carrier type using algebra alg
ana(fp, coalg)          # Anamorphism: unfold using coalgebra coalg
hylo(fp, coalg, alg)    # Hylomorphism: compose ana then cata
```

`cata` and `ana` return `Morphism` values wrapping `Cata`/`Ana` deferred nodes.
No Hydra primitives are built here; realization is deferred to `structure/realize.py`.

---

---

## `tensors/` — tensor extension

### `tensors/notation.py` — surface notation types

| Type | Role |
|------|------|
| `Equation(inputs, output)` | Einstein-like index equation; `inputs` is a list of index tuples, `output` is an index tuple |
| `SemiringDecl(name, carrier, plus, times, zero, one, ...)` | Surface declaration of a semiring before resolution |
| `AlignmentPlan` | Alignment metadata for contraction inputs |
| `ContractExpr` | Parsed surface contraction expression |

### `tensors/semirings.py` — resolved semiring

`Semiring` dataclass: `name`, `carrier: Type`, `plus/times: Morphism`, `zero/one`, optional `adjoint`, `plus_reduce/times_reduce/adjoint_reduce`. `op_env(*, adjoint=False)` returns `{"product": ..., "fold": ..., "seed": ...}` selecting the correct operation mode.

### `tensors/semantics.py` — semiring resolution and contract morphisms

`resolve_semiring(decl: SemiringDecl, op_morphisms) → Semiring` validates binary op types and assembles a `Semiring`.

`ContractSpec` — internal spec: `semiring`, `equation`, `adjoint`, `shape`, `_domain_tag`. `.dom` and `.cod` emit `ExpType`-wrapped types; `_strip_exp` helper removes `ExpType` wrappers for substrate compatibility.

`contract_morphism(semiring, equation, ...)` — builds a lazy `DomainPrim` pending finalize; `construct_expr` resolves a `ContractExpr` into a composed substrate `Morphism`.

---

### `structure/terms.py` — Hydra term vocabulary

The single file that owns how to build a Hydra term for a semantic concept. All other
structure-layer code uses this module as its term vocabulary.

**Lambda and application helpers:**

```python
prim2(name, a, b)              # Apply a binary Hydra primitive
term_lambda(name, body_fn)     # Build a lambda from a name and body function
lam2(n1, n2, body_fn)          # Curried two-argument lambda
```

**Monad-polymorphic effects** (take `monad: MonadDescriptor | None`; `None` = plain):

```python
bind(monad, value, name, body_fn)     # Monadic bind
pure(monad, value)                    # Monadic pure / unit
apply_effect(monad, ff, fx)           # Applicative apply
map_effect(monad, f, fx)              # Functor map over effect
lift2_effect(monad, f, left, right)   # Lift a binary function into effect
pure_unit(monad)                      # Constant-unit morphism, plain or effectful
pure_identity(monad)                  # Identity morphism, plain or effectful
product_action(monad, lf, rf)         # Parallel product morphism, plain or effectful
pair_effects(monad, left, right)      # Pair two effectful values
case_effects(monad, bl, br)           # Build sum eliminator over effects
list_effects(monad, item_action)      # Build list traversal with effects
maybe_effects(monad, item_action)     # Build maybe traversal with effects
```

**Projection and injection:**

```python
pair_first()          # π₁ as a term
pair_second()         # π₂ as a term
pair_swap()           # A × B → B × A
either_swap()         # A + B → B + A
pairs_bimap(l, r)     # bimap over a pair
eithers_bimap(l, r)   # bimap over either
eithers_either(l, r)  # case elimination
left_injection()      # ι₁ as a term function
right_injection()     # ι₂ as a term function
absurd()              # Term-level eliminator for Void
```

**List and maybe constructors:**

```python
lists_cons(head, tail)        # head : tail
lists_empty()                 # []
lists_foldr(f, z, xs)         # right fold
lists_map(f)                  # partially applied map
lists_uncons(xs)              # List → Maybe(A × List)
maybes_maybe(d, f, x)         # maybe eliminator
maybes_nothing()              # Nothing
maybes_just()                 # Just constructor as a term function
```

**Term optimization:**

```python
optimize_term(term)    # Peephole simplifier: first(pair(a,b)) → a, etc.
```

---

### `structure/realize.py` — DSL-to-Hydra translation

`realize(node: MorphismExpr, _prims: list | None = None) -> Term` translates a
morphism expression to a raw Hydra term. The `_prims` accumulator collects newly
created `Primitive` objects for `Cata`/`Ana` nodes; the caller adds them to the
graph before reduction.

`realize_term(node, _prims=None) -> TTerm` is the typed-handle wrapper.

**The deferred-node pattern for recursion:**

`Cata` and `Ana` nodes are created pure-semantically (no Hydra imports) in
`semantics/optics.py`. When `realize` encounters one:

1. A fresh `Primitive` is constructed with a generated name.
2. The body expression (which contains `SelfRef` nodes) is realized recursively,
   with `SelfRef` resolving to the same primitive name.
3. The primitive's implementation calls `realize` on the body at reduction time,
   so the self-reference closes the loop.
4. The primitive is appended to `_prims` so `main.run()` or `CompiledProgram.run()`
   can register it.

`poly_action_term(body, h, monad)` builds the Hydra term for the functor action
`F(h)`. This is the structural realization of `PolyFmap` nodes.

---

### `structure/recursion.py` — fixed-point optic bridge

Thin adapter between user-supplied roll/unroll functions and the `Optic` type.

```python
recursive_carrier(*, functor, carrier, unroll, roll) -> Optic
```

Wraps user-supplied Python callables `unroll : carrier → F(carrier)` and
`roll : F(carrier) → carrier` as `expr.Prim` morphisms and returns a carrier
`Optic` suitable for `cata`/`ana`. The callables receive a single `TTerm` argument.

`fixed_point_optic` is retained as a lower-level shim for direct raw-term injection.

---

### `runtime/` — native boundary support

Runtime owns the value boundary around native backends:

- `runtime/backend.py` loads backend specs and registers Hydra primitives.
- `runtime/codecs.py` owns Hydra term coders and structural term decoding.
- `runtime/boundary.py` adapts Python/native values to backend handles, stores
  native values, packs program arguments, and delegates boundary I/O.

Runtime modules do not build semantic expression trees.

---

### `main.py` — orchestration and execution boundary

```python
compile_program(src: str, *, env=None, graph=None, route=None) -> CompiledProgram
```
Parses source, constructs one semantic program tree, realizes the selected final
route/map, and returns a runnable program.

```python
compile_morphism(morphism: Morphism, graph=None, backend=None) -> CompiledProgram
```
Compiles an already-constructed semantic morphism.

```python
lower(morphism: Morphism, graph, _extra_prims=None) -> Term
```
Realizes a morphism to a raw Hydra term without evaluating it.

```python
run(morphism: Morphism, argument, ctx, graph)
```
Applies a morphism to a Hydra argument value and reduces it to a result. Registers
any `aux_primitives` (including those collected during realization of `Cata`/`Ana`
nodes) into a temporary graph copy before calling `R.reduce_term`.

`CompiledProgram.run(*args)` is the source-program path. It uses
`runtime/boundary.py` for argument packing and input/output decoding when a
backend is loaded, and returns decoded Python values.

---

## Key Abstractions

### `Type` (Hydra core)

The ground truth for all type-level computations. Equality is structural Hydra type
equality. `show_type(t)` renders it for display. All `dom()`, `cod()`, and functor
`apply()` calls return `Type` values.

### `Morphism`

The primary user-facing value. Wraps a `MorphismExpr` node with `param`, `monad`,
and `aux_primitives`. Use `dom()` / `cod()` to read the visible signature. Build
morphisms with the combinators in `semantics/morphisms.py` and `semantics/functors.py`.

### `Functor`

Describes the shape of a container type: `body: PolyExpr` specifies which positions
are recursive (`Id`) vs. constant. `apply(A)` computes the concrete type at element
type `A`. `unapply(FA)` inverts this. `map(h)` constructs the lifted morphism.

### `Optic`

Unifies Lens, Prism, and Traversal under one data type. The `functor` field specifies
the container shape; `forward` and `backward` are the decompose/reconstruct morphisms.
Set `carrier` to use the optic as a fixed point for `cata`/`ana`/`hylo`.

### `MonadDescriptor` (protocol in `terms.py`)

Used internally by monad-polymorphic term builders. Any object with `bind_name` and
`pure_name` fields (both `Name`) satisfies this protocol. `objects.MAYBE` and
`objects.LIST` are the two standard instances.

### The deferred-node pattern

`PolyFmap`, `SelfRef`, `Cata`, and `Ana` are syntax nodes that carry their semantic
content but defer Hydra construction. This keeps the semantics layer Hydra-free while
allowing `realize.py` to choose the correct lowering strategy per node type. The
`_prims` accumulator threads through `realize` so that recursive nodes can register
their `Primitive` objects before the outer reduction runs.

---

## How to Build and Run a Morphism

```python
from unialg import (
    Morphism, Functor, Optic, recursive_carrier,
    identity, compose, par, copar, pair, case, fst, snd,
    cata, ana, hylo,
    id_, const, sum_, prod,
    ProductType, SumType,
    MAYBE, LIST,
)
from hydra.core import Type
from unialg import compile_program, run

# --- 1. Define types ---
A = ...   # some Hydra Type
B = ...

# --- 2. Build morphisms ---
f = identity(A)
g = some_morphism(A, B)
h = compose(f, g)           # f ; g  (requires f.cod() == g.dom())

# --- 3. Lift through a functor ---
F = Functor("F", sum_(id_(), const(A)))   # F(X) = X + A
Fh = F.map(h)                             # F(h) : F(A) → F(B)

# --- 4. Catamorphism ---
carrier_type = ...   # the μF type
unroll = lambda x: ...   # TTerm → TTerm
roll   = lambda t: ...

fp = recursive_carrier(
    functor=F,
    carrier=carrier_type,
    unroll=unroll,
    roll=roll,
)

alg = some_algebra(F.apply(B), B)         # F(B) → B
fold = cata(fp, alg)                      # μF → B

# --- 5. Run a raw morphism ---
import hydra.dsl.std as Std
result = run(fold, hydra_argument, Std.context(), Std.graph())

# Or compile and run from source:
program = compile_program("""
route f = id
""")
python_value = program.run()
```

---

## Common Patterns and Gotchas

**`shared_context=True` in recursive schemes.** `cata` and `ana` use
`compose(..., shared_context=True)` internally so the self-reference and algebra
morphism share one copy of the parameter instead of producing `P × P`. Never use
`compose(ana_result, cata_result)` directly — use `hylo(fp, coalg, alg)` to get the
correct shared context.

**`_prims` accumulator.** When calling `lower()` or `run()` on a morphism that
contains `Cata`/`Ana` nodes, the `aux_primitives` on the morphism may not include
all necessary primitives until `realize` runs. `run()` handles this automatically.
If you call `lower()` directly and then reduce manually, pass `extra_prims=[]` and
register those alongside `morphism.aux_primitives`.

**`unapply` round-trip check.** `Functor.unapply(fa)` raises `TypeError` if Hydra's
unifier cannot solve `F(A) = fa` or if the recovered `A` does not round-trip back to
`fa`. This is intentional — an optic built on an incompatible functor/type pair is
rejected at `Optic.__post_init__` time rather than at reduction time.

**`param` ordering.** When two morphisms with distinct non-unit params are composed
with `shared_context=False` (the default), the combined raw domain is
`g_param × f_param` — the *second* child's param is placed first. This matches the
order the contextual realizer in `realize.py` expects when splitting the input.

**Lax promotion.** Composing a plain morphism with a lax one automatically wraps the
plain morphism in `MonadicEmbed`. You do not need to do this manually; the
`_contextual_binary` helper handles it.
