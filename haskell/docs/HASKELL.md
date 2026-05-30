# UniAlg Haskell branch

## Purpose

A mathematician writes a program in ordinary Haskell.
GHC type-checks it.
The Haskell runtime evaluates the expressions, which silently build Hydra IR.
Hydra's Python coder renders that IR to Python source.
The generated Python is semantically correct and numerically verified against
canonical ML libraries (NumPy, TensorFlow).

The goal is to let someone who knows category theory write a neural architecture
the same way they would write a proof — using the abstractions they already know
— and have a working ML implementation come out the other end.

---

## How it differs from the Python branch

Both branches share the same pipeline shape:

```
surface expression → typed interpretation → algebraic construction → executable assembly → backend realization
```

And the same backend-agnosticism: JSON spec files map logical op names to
backend-specific paths; lowering rewrites those names before codegen.

The difference is where each pipeline stage lives:

| Stage | Python branch | Haskell branch |
|---|---|---|
| Surface language | Custom string DSL + parser | Haskell itself |
| Type checking | `semantics/construct.py`, `typeops.py` | GHC |
| Morphism representation | `MorphismExpr` ADT | `TArr a b` — functions over `TTerm` |
| Functor structure | `PolyExpr` ADT | `Data.Functor.*` composed at the type level |
| Recursion schemes | `optics.py` `Optic` class | `cataT`/`anaT`/`hyloT` with `ImplicitParams` |
| Code generation | `realize.py` → Hydra term → `runtime/boundary.py` | `TTerm` is already Hydra IR; Hydra Python coder runs directly |

In the Python branch, expressions are data that gets walked.
In the Haskell branch, expressions are functions that, when called, build Hydra IR.
There is no separate AST — the "expression tree" is constructed lazily as
the Haskell evaluation proceeds.

---

## The `arr` constraint

`TArr a b` implements `Control.Category`, `Control.Arrow`, and
`Control.ArrowChoice` so that programs can be written in standard Haskell
arrow notation: `>>>`, `&&&`, `|||`, etc.

However, `Arrow` requires:

```haskell
arr :: (a -> b) -> f a b
```

`TArr` cannot implement this correctly. A Haskell function `a -> b` is opaque
— there is no way to inspect it and emit the equivalent Python source.
`arr` is therefore left as a runtime error:

```haskell
arr _ = error "TArr: arr cannot inspect Haskell functions to generate code"
```

**Consequence:** any combinator from a standard library that calls `arr`
internally will fail at code-generation time.
This is why the early experiment using `Control.Arrow` and `Control.Category`
library combinators produced static one-liners — those combinators bottomed
out in `arr` before the structure could be captured.

**The fix:** write all combinators as explicit `TArr` wrappers in
`UniAlg.Semantics.Category`. They use the same operator symbols (`>>>`,
`&&&`, `|||`, `***`, `+++`) so the user-facing syntax is identical to
standard arrow notation, but the implementations build `TTerm` nodes
directly and never call `arr`.

---

## `TFunctor` vs standard `Functor`

Standard `Functor` maps over real Haskell values:

```haskell
fmap :: (a -> b) -> f a -> f b
```

Here the "values" are symbolic `TTerm` nodes that need to be wired into
Hydra IR. `TFunctor` adds three TTerm-level operations on top of `Functor`:

| Method | Purpose |
|---|---|
| `tfmap` | Insert the recursive self-call `TTerm` into one functor layer |
| `applyAlg` | Peel a `TTerm` functor node into real Haskell constructors so algebra functions can pattern-match |
| `foldToTerm` | Reassemble a real Haskell functor value back into a `TTerm` (used by `anaT`) |

Every instance also retains a standard `Functor` constraint so that the
plain Haskell recursion schemes (`cata`, `ana`, `hylo`) still work on real
values — for building `Fix`-structured inputs to pass to `cataT`.

The polynomial functor atoms (`Identity`, `Const`, `Product`, `Sum`) are the
same types as `Data.Functor.*`; `TFunctor` just adds codegen semantics to them.

---

## Self-reference and `withSelf`

A recursive Python function must call itself by name:

```python
def fold_rnn(w, s0, x):
    ...
    return fold_rnn(w, s0, next_x)   # self-call
```

At code-generation time that name is not yet bound in any Haskell scope.
`withSelf` solves this using GHC's `ImplicitParams` extension:

```haskell
withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k
```

`?self` is injected once at the definition site and is automatically
threaded through every recursive call site without being mentioned explicitly.
Each recursive step emits `Terms.apply (unTTerm ?self) (unTTerm arg)`, which
generates the correct Python self-application.

When a definition has shared outer parameters (e.g. weights `w`, initial
state `s0`), those are prepended to `?self` as a partial application:

```haskell
-- ?self = var "fold_rnn" `apply` var "w" `apply` var "s0"
-- so every recursive step emits: fold_rnn(w, s0, next_x)
```

`recModule` and `recDef` handle this automatically. `withSelf` only needs to
be called directly when constructing modules outside those builders.

---

## Why `Control.Arrow` library combinators failed

Tracing through the failure:

1. User writes `f >>> returnA` using a standard library combinator.
2. `returnA` is defined as `arr id` in `Control.Arrow`.
3. `arr id` hits `TArr`'s `arr _ = error "..."` at evaluation time.
4. The composition terminates before any structure is captured.
5. Result: a single opaque application term — the "static one-liner".

The same failure occurs with any library combinator that routes through `arr`,
including `loop`, `app`, and most of the `Arrows` module utilities.

The solution is not to call those combinators. The `UniAlg.Semantics.Category`
module re-implements the operators that matter (`>>>`, `&&&`, `***`, `|||`,
`+++`, `copy`, `delete`, `symmetry`, `assoc`, `merge`, `distributeLeft`,
`distributeRight`) as `TArr` wrappers that bypass `arr` entirely.

---

## Van Laarhoven optics vs the Python `Optic` class

The Python branch has a named `Optic` class with explicit `.par()` and
`.compose()` methods.

The Haskell branch uses the standard van Laarhoven / profunctor encoding:

```haskell
type Lens  s t a b = forall f. Functor f      => (a -> f b) -> s -> f t
type Prism s t a b = forall p. Choice p       => p a b -> p s t
```

This means:
- Optics compose with `(.)` — no separate combinator needed.
- `view`, `set`, `over`, `preview`, `review` work without modification.
- The `Endo` typeclass unifies plain Haskell modifiers and `TArr` morphisms
  so that `over lens myTArr someValue` works at the `TTerm` level.

The tradeoff is that van Laarhoven optics are harder to extend with new
combinators than a named class, but they come with the full Haskell lens
ecosystem at no cost.

---

## Backend JSON specs

Both branches use the same JSON format:

```json
{
  "backend": "numpy",
  "ops": {
    "matmul":                   { "path": "numpy.matmul",        "arity": 2 },
    "reduce.add":               { "path": "numpy.sum",           "arity": 2 },
    "structural.expand_dims":   { "path": "numpy.expand_dims",   "arity": 2 },
    "structural.transpose":     { "path": "numpy.transpose",     "arity": 2 }
  }
}
```

At DSL time, morphisms reference ops symbolically as `unialg.backend.matmul`.
The lowering pass (`UniAlg.Pipeline.Lowering`) rewrites these names to
`numpy.matmul` before Hydra generates Python source.

External module stubs (`UniAlg.Pipeline.Externals`) declare eta-expanded
definitions for every backend op at the correct arity so Hydra's type system
can resolve them during code generation. These stubs appear in the `universe`
list but are not emitted as output.

---

## Known open issues

**`Transformer.hs` executable** — the transformer example in `test/Transformer.hs`
does not compile. It exercises optics and `cataT` together in a way that
triggers type ambiguities. This is the primary known issue and is the next
thing to address.

**`arr` at the boundary** — if a user accidentally uses a standard library
combinator that calls `arr`, the error is a runtime panic rather than a
compile error. There is no way to make this a type error given Haskell's
`Arrow` class definition. The current mitigation is documentation; a future
option is to provide a custom `Arrow`-like class that omits `arr`.

**Duplicate export warning** — `UniAlg.hs` re-exports `reify` from both
`UniAlg.Semantics.Category` and `UniAlg.Semantics.Arrows`. GHC emits
`-Wduplicate-exports`. One of the re-exports should be removed.

---

## Module map

```
src/UniAlg.hs                     Top-level re-export surface
src/UniAlg/
  Semantics/
    Arrows.hs                     TArr — the core morphism type
    Category.hs                   Structural morphisms and operator aliases
    Functors.hs                   TFunctor class and polynomial functor aliases
    Recursion.hs                  cataT / anaT / hyloT / withSelf
    Optics.hs                     Van Laarhoven optics over TTerm values
  Domain/
    Tensors.hs                    Einstein notation, semirings, tensor contraction
  Pipeline/
    Backend.hs                    Backend JSON loading and op resolution
    Lowering.hs                   Hydra IR rewriting (symbolic → backend names)
    Codegen.hs                    recModule / writePythonWithBackend / evalPython
    Externals.hs                  Universe-only backend stub declarations
  Core/
    Reduce.hs                     Hydra IR simplification (beta, pair, either)
```
