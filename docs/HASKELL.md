# UniAlg Haskell branch

## Purpose

A mathematician writes a program in ordinary Haskell.
GHC type-checks it.
The Haskell runtime evaluates the expressions, which silently build Hydra IR.
Hydra's Python coder renders that IR to Python source.
The generated Python is semantically correct and numerically verified against
NumPy, TensorFlow, and PyTorch.

The goal is to let someone who knows category theory write a neural architecture
the same way they write a proof, using familiar abstractions, and have a working
ML implementation come out the other end.

---

## How it differs from the Python branch

Both branches share the same pipeline shape:

```
surface expression → typed interpretation → algebraic construction → executable assembly → backend realization
```

And the same backend-agnosticism: JSON spec files map logical op names to
backend-specific paths; lowering rewrites those names before codegen.

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

## The `Shape` class

The `Shape` class (`UniAlg.Shape.Encode`) adds two code-generation operations
on top of a standard `Functor`:

| Method | Purpose |
|---|---|
| `matchLayer` | Pattern-match a `TTerm` against functor shape `f`, extracting real Haskell constructors so algebra functions can branch on them |
| `buildLayer` | Reassemble a real Haskell functor value back into a `TTerm` (used by `anaT` to encode the coalgebra's return value) |

Every polynomial functor atom (`Identity`, `Const`, `Product`, `Sum`, `Exp`)
has a `Shape` instance, and instances compose automatically for any combination.
This means `(Shape f, Shape g) => Shape (Sum f g)` is already provided; there is
nothing extra to write when combining atoms.

The plain Haskell recursion schemes (`cata`, `ana`, `hylo` in
`UniAlg.Scheme.Internal`) still operate on `Fix`-structured real values for
building test inputs in `arch.py`; the `Shape` constraint is only required for
the code-generating variants (`cataT`, `anaT`, `hyloT`).

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

`withSelf` binds `?self` once at the definition site; GHC threads it through
every recursive call site automatically, with no explicit mention needed.
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

1. User writes `f >>> returnA` using a standard library combinator.
2. `returnA` is defined as `arr id` in `Control.Arrow`.
3. `arr id` hits `TArr`'s `arr _ = error "..."` at evaluation time.
4. The composition terminates before any structure is captured.
5. Result: a single opaque application term — the "static one-liner".

The same failure occurs with any library combinator that routes through `arr`,
including `loop`, `app`, and most of the `Arrows` module utilities.

The solution is not to call those combinators. `UniAlg.Term`
re-implements the operators that matter (`>>>`, `&&&`, `***`, `|||`,
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
src/UniAlg.hs                      Top-level re-export surface
src/UniAlg/
  Architecture.hs                  cataModule / anaModule / hyloModule; Elim / CoElim
  Scheme.hs                        Re-export surface for recursion schemes
  Scheme/Internal.hs               cataT / anaT / hyloT / withSelf / Fix
  Term.hs                          TArr, reify, structural morphisms (pair, either, …)
  Term/Internal.hs                 Low-level TTerm construction helpers
  Shape.hs                         Polynomial functor atoms and derived type aliases
  Shape/Encode.hs                  Shape class (matchLayer / buildLayer) and instances
  Tensor.hs                        Semiring, contraction, Einstein notation
  Optics.hs                        Van Laarhoven optics over TTerm values
  Backend.hs                       Backend loading re-export surface
  Backend/Spec.hs                  BackendOp, BackendBinding, LoadedBackend
  Backend/Lowering.hs              Hydra IR rewriting: symbolic names → backend paths
  Backend/Externals.hs             Universe-only backend stub declarations
  Core/BackendSpec.hs              JSON deserialisation types for backend specs
  Core/Ops.hs                      Op resolution (symbolic key → TTerm)
  Core/Ops/Generate.hs             TTerm builders for unary / binary / ternary ops
  Core/Reduce.hs                   Hydra IR simplification (beta, pair, either)
  Codegen.hs                       writePythonWithBackend / loadBackendAndWritePython
```

---

## Writing a new architecture

An architecture is a Haskell module that produces one or more `SeedEntry` values
and exports `backendSeeds :: [(String, SeedEntry)]`.

### 1. Choose the polynomial functor

Express the recursive shape as a type alias over the atoms in `UniAlg.Shape`:

```haskell
-- List spine  F(X) = 1 + (A × X)
type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

-- Binary tree  F(X) = A + (X × X)
type TreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

-- Moore machine  F(X) = O × (I → X)
type MooreF o i = Product (Const (TTerm o)) (Exp (TTerm i))
```

Record the same shape at value level in `seedPolyF` using `Grammar.PolyF` atoms:

```haskell
KConst :+: (Hole :*: Hole)   -- binary tree A + (X × X)
```

### 2. Choose the recursion direction

| Direction | Builder | Use when |
|-----------|---------|----------|
| Catamorphism | `cataModule` | Input is a structure to consume bottom-up |
| Anamorphism  | `anaModule`  | Output is produced top-down from a seed |
| Hylomorphism | `hyloModule` | Coalgebra decomposes input; algebra reassembles |

### 3. Write the algebra or coalgebra body

**Catamorphism** — the `Elim f Tensor (TTerm Tensor)` argument expands to a
curried tuple with one branch per functor constructor:

```haskell
cataModule @(TreeF Tensor)
  "seed.tree" "fold_tree"
  [Namespace "torch"] ["w"] $ \[w] ->
    ( \leaf        -> contraction real "hi,i->h" w leaf   -- InL: leaf
    , \left right  -> add left right                       -- InR: pair of children
    )
```

**Anamorphism** — the coalgebra returns a plain Haskell pair or `Either`
mirroring the functor; `anaModule` encodes it back to `TTerm`:

```haskell
anaModule @(MooreF Tensor Tensor)
  "seed.moore" "moore_step"
  [Namespace "torch"] ["w"] $ \[w] ->
    \s -> ( decode w s                   -- Const: output
          , \inp -> transition w s inp   -- Exp:   next state
          )
```

**Hylomorphism** — return a `(coalg, alg)` pair.  The coalgebra uses the
right adjoint of the algebra's forward operation (e.g. `subtract` when the
algebra uses `add`):

```haskell
hyloModule @(EdgeF Tensor)
  "seed.edge" "edge_conv"
  [Namespace "torch"] ["w"] $ \[w] ->
    ( \p    -> subtract (second p) (first p)   -- coalg: decompose via subtract
    , \case
        InL d          -> relu (contraction real "ij,j->i" w d)
        InR (Pair l r) -> maximum l r           -- alg: fold differences
    )
```

### 4. Use backend-declared op aliases only

Op names must come from `backends/*.json`.  Using an alias not declared in those
files will fail at lowering time.  Common aliases:

| Alias | Semiring |
|-------|----------|
| `add`, `multiply` | real |
| `maximum`, `add` | tropical (ops are swapped: `maximum` is ⊕, `add` is ⊗) |
| `subtract` | right adjoint of `add` |
| `relu` | torch and tensorflow only |
| `tanh`, `sigmoid` | all backends |

Use `Seed.contraction` to apply a tensor contraction parameterised on a semiring:

```haskell
real     = Semiring "add"     "multiply" (Just "subtract")
tropical = Semiring "maximum" "add"      Nothing
```

### 5. Register in cabal and catalogue

Add to `library explore` in `unialg.cabal`:

```cabal
hs-source-dirs: ... explore/archs/my_arch
exposed-modules: ... MyArch
```

Then regenerate the catalogue:

```bash
runghc explore/gen-catalogue.hs
```

Never edit `Catalogue.hs` by hand.
