# Writing Recursive Architectures

Use this path when a term is defined by a recursion scheme: catamorphism,
anamorphism, or hylomorphism. For flat code generation without recursive shape,
use [Generating code](generating-code.md).

A recursive architecture is a Haskell module that produces one or more
`SeedEntry` values and exports `backendSeeds :: [(String, SeedEntry)]`.
The examples below use `Tensor`, but the recursion builders operate on
`TTerm` structure. The same pattern applies to other domains when the terms and
backend ops represent that domain.

## Where does new code live?

Architecture modules belong under `explore/archs/<name>/`. The `src/UniAlg/`
tree is the stable library and should not be modified when writing new
architectures. The `explore/` layer imports from `src/` but not vice versa.

Most architecture modules start with these imports:

```haskell
import UniAlg
import Grammar (PolyF(..))
import Seed (SeedEntry(..), ArchClass(..), contraction, adjointContraction)
```

Import `adjointContraction` only when the module uses adjoint contractions.

## 1. Choose the polynomial functor

Express the recursive shape as a type alias over the atoms in `UniAlg.Shape`:

```haskell
-- List spine  F(X) = 1 + (A × X)
type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

-- Binary tree  F(X) = A + (X × X)
type TreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

-- Moore machine  F(X) = O × (I → X)
type MooreF o i = Product (Const (TTerm o)) (Exp (TTerm i))
```

The `@(TreeF Tensor)` syntax requires `{-# LANGUAGE TypeApplications #-}` at
the top of the module. All arch modules in `explore/archs/` already enable it.

If a module uses DSL names that also exist in Prelude, hide the Prelude names:

```haskell
import Prelude hiding (fst, snd, maximum, tanh)
```

Record the same shape at value level in `seedPolyF` using `Grammar.PolyF` atoms:

```haskell
KConst :+: (Hole :*: Hole)   -- binary tree A + (X × X)
```

## 2. Choose the recursion direction

| Direction | Builder | Use when |
|-----------|---------|----------|
| Catamorphism | `cataModule` | Input is a structure to consume bottom-up |
| Anamorphism  | `anaModule`  | Output is produced top-down from a seed |
| Hylomorphism | `hyloModule` | Coalgebra decomposes input; algebra reassembles |

### Recursive self-reference

A generated recursive function must call itself by name:

```python
def fold_rnn(w, s0, x):
    ...
    return fold_rnn(w, s0, next_x)   # self-call
```

At code-generation time that name is not yet bound in any Haskell scope.
`withSelf` solves this using GHC's `ImplicitParams` extension. The public module
builders handle it automatically:

```haskell
cataModule ...
anaModule ...
hyloModule ...
```

`withSelf` only needs to be called directly when constructing a recursive module
outside those three builders.

## 3. Write the algebra or coalgebra body

The weight list pattern -- `[w, wRec, s0] ->` -- is a partial match on a plain
Haskell list. The list length must exactly match the names declared two lines
above; if they differ the program raises a runtime error. GHC does not catch the
mismatch at compile time.

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

Elim expands one branch per functor constructor. For SeqF:

```haskell
Elim (SeqF Tensor) Tensor (TTerm Tensor)
≡  (TTerm Tensor, TTerm Tensor -> TTerm Tensor -> TTerm Tensor)
     ↑ empty case              ↑ cons case: element → state → state
```

For a Sum functor the result is always a 2-tuple; for Product it is a curried
function. Read the Elim Haddock in `src/UniAlg/Architecture.hs` for the full
expansion rules.

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

The Namespace list is stored as `moduleTermDependencies` and helps Hydra resolve
module dependencies before code generation. Backend-specific Python imports come
from the lowered backend external stubs, not from this list. Practical rule:
set `Namespace` to a backend that defines any structural ops the generated
module calls directly, such as `torch` when the module uses `torch.nn` calls.
For modules that only use backend-declared algebraic ops such as `add`,
`multiply`, `tanh`, or `contraction`, the choice does not affect correctness;
a seed may declare `[Namespace "numpy"]` and still generate a torch file that
imports `torch`.

**Hylomorphism** — return a `(coalg, alg)` pair.  The coalgebra uses the
right adjoint of the algebra's forward operation (e.g. `divide` when the
algebra uses `multiply`):

This example is illustrative. `EdgeF`, `seed.edge`, and `edge_conv` are names
for a possible hylomorphic architecture, not a module currently present under
`explore/archs/`.

```haskell
hyloModule @(EdgeF Tensor)
  "seed.edge" "edge_conv"
  [Namespace "torch"] ["w"] $ \[w] ->
    ( \p    -> divide (snd p) (fst p)          -- coalg: decompose via divide
    , \case
        InL d          -> tanh (contraction real "ij,j->i" w d)
        InR (Pair l r) -> maximum l r           -- alg: fold differences
    )
```

`InL`, `InR`, and `Pair` are constructors from `UniAlg.Shape` (re-exported via
`UniAlg`). `fst` and `snd` are the projection morphisms from `UniAlg.Term`, not
the Prelude functions; they produce `TTerm` nodes rather than unwrapping a
Haskell pair. Hide `Prelude.fst` and `Prelude.snd` when using them unqualified.

The `\case` syntax requires `{-# LANGUAGE LambdaCase #-}` at the top of the
module. All arch modules in `explore/archs/` that use `\case` enable it.

## 4. Choose a semiring and use backend-declared op aliases

A `Semiring` parameterises how tensor contractions are compiled.  It has three
fields:

```haskell
data Semiring = Semiring
  { semiringPlus    :: String        -- reduction op  (the ⊕ of the semiring)
  , semiringTimes   :: String        -- element-wise product op  (the ⊗)
  , semiringAdjoint :: Maybe String  -- adjoint of ⊗, needed for backward passes
  }
```

A contraction `"ij,j->i"` over a semiring compiles to:
`reduce_plus (times w x)` — element-wise product along the contracted index,
then reduction.  The op names must match keys declared in `backends/*.json`.

Common semirings:

```haskell
real     = Semiring "add"     "multiply" (Just "divide")
tropical = Semiring "maximum" "add"      (Just "subtract")
```

These values are not imported from `UniAlg`; define the semirings your module
uses locally.

In the **real** semiring ⊕ = `add`, ⊗ = `multiply` — the standard dot product.
In the **tropical** semiring the roles swap: ⊕ = `maximum` (or `minimum`), ⊗ =
`add`.  A contraction over tropical becomes `max(w + x)`
rather than `sum(w * x)`, which is the correct lowering for max-plus networks
and shortest-path architectures.

The `semiringAdjoint` field is the right-adjoint of ⊗ — `divide` for real,
`subtract` for tropical.  It is used by `adjointContraction` and hylomorphism
coalgebras that decompose by inverting the algebra's forward operation.  Set it
to `Nothing` if no backward decomposition is needed.

Op names must come from `backends/*.json`; using an undeclared alias fails at
lowering time. Common aliases by backend availability:

| Role | Aliases | Available |
|------|---------|-----------|
| Elementwise binary | `add`, `subtract`, `multiply`, `divide`, `minimum`, `maximum` | all backends |
| Reductions | `reduce.add`, `reduce.multiply`, `reduce.minimum`, `reduce.maximum` | all backends |
| Unary ops | `tanh` | all backends |
| Unary ops | `sigmoid` | numpy, tensorflow, torch, jax |

`Semiring` values use the base binary aliases for `semiringPlus`,
`semiringTimes`, and `semiringAdjoint`; `contraction` adds the `reduce.` prefix
for reductions internally.

Use `Seed.contraction` for the forward direction and `Seed.adjointContraction`
for the adjoint (requires `semiringAdjoint /= Nothing`):

```haskell
-- forward: sum(w * x)  — standard matrix-vector multiply
lin     mat vec = contraction        real "ij,j->i" mat vec

-- adjoint: prod(w / x)  — used in hylo coalgebras that invert the algebra
linAdj  mat vec = adjointContraction real "ij,j->i" mat vec
```

### Einstein notation

Equation strings follow NumPy einsum convention with one restriction: exactly
two inputs are supported (weight matrix and input vector/tensor).

```haskell
"ij,j->i"    matrix-vector multiply  (contracts j)
"hi,i->h"    same, different labels
"ij,jk->ik"  matrix-matrix multiply  (contracts j)
```

Rules:
- Labels are single lowercase ASCII letters.
- Indices appearing on both input sides are contracted (summed over in the real
  semiring; see Semiring for other semirings).
- Indices appearing only on the left of `->` are reduced (batch dimensions).
- Indices on the right of `->` are kept.
- Equation parsing is done at Haskell evaluation time via `parseEquation`; a
  malformed string produces a `Left` error that is re-thrown as a runtime error
  by `contraction`/`adjointContraction`.

## 5. Register in cabal and catalogue

In `unialg.cabal`, edit the `library explore` stanza. Add the new architecture
directory to `hs-source-dirs` and the Haskell module name to `exposed-modules`:

```cabal
library explore
  hs-source-dirs:
    explore/archs/my_arch
    explore/support/haskell

  exposed-modules:
    Catalogue
    MyArch
```

Then regenerate the catalogue:

```bash
runghc explore/gen-catalogue.hs
```

Never edit `Catalogue.hs` by hand.

## 6. Generate recursive modules

Load a backend and write Python for all seeds in a module:

```haskell
{-# LANGUAGE OverloadedStrings #-}

import qualified Moore
import Seed (SeedEntry(..))
import UniAlg

main :: IO ()
main = do
  let selected =
        [ seedModule entry
        | (backend, entry) <- Moore.backendSeeds
        , backend == "torch"
        ]

  result <- loadBackendAndWritePythonRec
    "backends"        -- backend directory
    "torch"           -- backend name
    "generated/torch" -- output directory
    selected          -- universe modules
    selected          -- modules to emit

  case result of
    Left msg -> fail msg
    Right n  -> putStrLn ("wrote " <> show n <> " files")
```

`backendSeeds` is a plain list, so select the entries for the backend you want.
`seedModule entry` uses the `seedModule` record field from `SeedEntry`, which is
why the example imports `SeedEntry(..)`. Use `loadBackendAndWritePythonRec` for
modules built with `cataModule`, `anaModule`, or `hyloModule`.
