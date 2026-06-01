# unialg — Haskell branch

unialg is a typed DSL for expressing neural architectures as algebraic
morphisms, polynomial functors, and recursion schemes. It lowers those
programs to [Hydra](https://github.com/CategoricalData/hydra) IR and from
there to executable Python for NumPy, TensorFlow, and PyTorch backends.

In this branch the surface language is Haskell itself. GHC type-checks the
program. When it runs, Hydra IR is built as a side-effect of evaluation.
Hydra's Python coder renders that IR to source, then verified numerically
against NumPy, TensorFlow, and PyTorch.

```haskell
-- Fold over a list spine  F(X) = 1 + (A × X)
type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

seqRnn :: SeedEntry
seqRnn = SeedEntry "seqCata" CataArch (KUnit :+: (KConst :*: Hole)) $
  cataModule @(SeqF Tensor)
    "seed.seq" "fold_seq"
    [Namespace "torch"] ["wIn", "wRec", "s0"] $ \[wIn, wRec, s0] ->
      ( s0                          -- empty list: return initial state
      , \a s -> add (contraction real "hi,i->h" wIn a)
                    (contraction real "hj,j->h" wRec s)
      )
```

## Quick start

Requires GHC 9.10+ (via [ghcup](https://www.haskell.org/ghcup/)) and cabal 3.14+.

```bash
cabal build lib:unialg lib:explore
cabal test explore-test --test-show-details=direct
```

For the Python differential tests:

```bash
uv sync
uv run pytest explore/archs/ -v
```

## How it works

```
Haskell source
  → GHC type-checker
  → Hydra IR (built during evaluation)
  → Hydra Python coder
  → backend realization (NumPy / TensorFlow / PyTorch)
```

There is no separate AST. `TArr a b` is a function `TTerm a -> TTerm b`;
composing two `TArr` values with `>>>` calls the functions and assembles
Hydra IR on the fly. The re-implemented operators in `UniAlg.Term` (`>>>`,
`&&&`, `|||`) bypass `Control.Arrow.arr`, which cannot be implemented for
`TTerm` because Haskell functions are opaque at generation time.

Self-reference uses GHC's `ImplicitParams`: `withSelf` binds `?self` once,
and every recursive step emits `apply ?self arg` — the correct Python
self-call, threaded automatically.

## Non-recursive modules

Plain functions over `TTerm` values generate Python directly, no recursion
scheme required:

```haskell
import UniAlg

gate :: TTerm Tensor -> TTerm Tensor
gate x = multiply (tanh x) (sigmoid x)

main :: IO ()
main = generatePython
  "generated/torch"
  "backends/torch.json"
  "activations"
  [ ("gate", reify gate) ]
```

`generatePython` accepts any `[(String, TTerm a)]` list and emits a single
Python module. Op names must be declared in `backends/*.json`.

## Writing an architecture

See [`docs/HASKELL.md`](docs/HASKELL.md) for the full guide: functor atoms,
direction choice, algebra/coalgebra patterns, backend alias rules, and
cabal/catalogue registration.

## Research context

The project is grounded in categorical deep learning (Gavranović,
*Fundamental Components of Deep Learning*, arXiv 2402.15332), Para
constructions, optics, polynomial functors, and recursion schemes. The goal
is to make the classification of neural architectures explicit: each
architecture is a coordinate on independent axes (functor shape, semiring,
activation, direction) that the DSL enforces rather than leaving implicit.

- Gavranović, *Fundamental Components of Deep Learning* (arXiv 2402.15332)
- Fong & Spivak, *An Invitation to Applied Category Theory*
- Hehner's *Unified Algebra*, the namesake inspiration for the notation
