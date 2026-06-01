{-|
Polynomial functor atoms and derived type aliases.

These are the building blocks for the functor @f@ in every recursion-scheme
module.  The atoms correspond directly to the 'Grammar.PolyF' grammar:

@
PolyF atom   Haskell type            Role
-----------  ----------------------  -----------------------------------
KUnit        Const ()                unit leaf — no data carried
KConst       Const (TTerm a)         fixed-value leaf (weights, inputs)
Hole         Identity                recursive position (the carrier)
a :+: b      Sum f g                 coproduct / branching
a :*: b      Product f g             product / pairing
ExpF a       Exp (TTerm i)           exponential / reader (indexed by i)
@

The 'Shape' class (from "UniAlg.Shape.Encode") must be satisfied for any @f@
used with 'cataModule', 'anaModule', or 'hyloModule'.  All atoms and their
compositions satisfy it automatically via the instances in
"UniAlg.Shape.Encode".

The derived aliases ('MaybeF', 'ListF', 'RoseF', 'TreeF') are provided as
convenience; concrete architecture modules typically define their own aliases
following the same pattern.
-}
module UniAlg.Shape
  ( Identity(..)
  , Const(..)
  , Product(..)
  , Sum(..)
  , Exp(..)
  , MaybeF
  , ListF
  , RoseF
  , TreeF
  ) where

import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))

import Hydra.Phantoms (TTerm)

import UniAlg.Shape.Encode (Exp(..))


-- | @F(X) = 1 + X@ — the Maybe functor: a leaf or a single recursive child.
type MaybeF = Sum (Const ())

-- | @F(X) = 1 + (A × X)@ — the list functor: empty list or head-plus-tail.
-- @a@ is the element type carried at each cons cell.
type ListF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

-- | @F(X) = f X × [X]@ — rose tree layer: one typed node and a variable-length
-- list of recursive children.  @f@ provides the node label.
type RoseF f = Product f []

-- | @F(X) = 1 + (f X × [X])@ — labelled rose tree: empty leaf or labelled node.
type TreeF f = Sum (Const ()) (RoseF f)
