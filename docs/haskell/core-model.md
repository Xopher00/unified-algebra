# Core Model

## `TArr` morphisms

`TArr a b` implements `Control.Category`, `Control.Arrow`, and
`Control.ArrowChoice` so that programs can be written in standard Haskell
arrow notation: `>>>`, `&&&`, `|||`, etc.

Use the operators exported by `UniAlg.Term`: `>>>`, `&&&`, `|||`, `***`, `+++`,
`copy`, `delete`, `symmetry`, `assoc`, `merge`, `distributeLeft`, and
`distributeRight`. These operators build `TTerm` nodes directly and preserve
the structure needed for backend code generation.

Do not use standard-library combinators that construct morphisms through
`Control.Arrow.arr`. A plain Haskell function `a -> b` is opaque to the code
generator, so only explicit `TArr` combinators can be rendered as backend code.

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
`UniAlg.Scheme.Internal`) operate on `Fix`-structured real values and are used
in Haskell-side tests; the `Shape` constraint is only required for the
code-generating variants (`cataT`, `anaT`, `hyloT`).

## Mapping through a functor layer

`fmap` maps over the parameter positions of a functor value. In a recursion
scheme those positions are recursive children; outside recursion they are just
the slots occupied by the functor parameter.

`Shape` lets the DSL expose one encoded `TTerm` layer as a Haskell functor
value, apply `fmap`, then encode the layer back to a `TTerm`.

```haskell
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

import UniAlg
import UniAlg.Shape.Encode (Shape(..))

mapLayer :: forall f a b. Shape f
         => (TTerm a -> TTerm b)
         -> TTerm a
         -> TTerm b
mapLayer h =
  matchLayer @f (buildLayer @f . fmap h)
```

Read this left to right:

- `matchLayer @f` decodes one outer layer of shape `f`.
- `fmap h` applies `h` only to the positions where `f` contains its parameter.
- `buildLayer @f` encodes the updated layer back into a `TTerm`.

For `Product Identity Identity`, `fmap h` applies `h` to both sides. For
`Product (Const c) Identity`, it leaves the `Const c` side alone and applies
`h` to the `Identity` side. For `Sum f g`, it maps whichever branch is present.

```haskell
mapBoth :: TTerm Tensor -> TTerm Tensor
mapBoth =
  mapLayer @(Product Identity Identity) neg
```

This is one-layer mapping. It does not traverse a recursive structure by
itself. Use `cataModule`, `anaModule`, or `hyloModule` when the same pattern
must repeat through a recursive shape.

## Optics

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

Use optics when a plain Haskell modifier or a `TArr` morphism should act inside
a larger structure. The `Endo` instance keeps those updates at the `TTerm`
level so they remain visible to code generation.
