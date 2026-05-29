{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

{-|
The core morphism type.

'TArr' @a b@ is a morphism @a → b@ represented as a Haskell function
@'TTerm' a -> 'TTerm' b@.  When called, it builds Hydra IR rather than
executing Python.

'TArr' implements 'Control.Category.Category', 'Control.Arrow.Arrow', and
'Control.Arrow.ArrowChoice'.  The operator aliases (@'>>>'@, @'&&&'@,
@'***'@, @'|||'@, @'+++'@) are re-exported from "UniAlg.Semantics.Category"
with the same fixities and semantics as their 'Control.Arrow' counterparts —
existing intuitions transfer directly.  They shadow the base imports by
design; do not import both together.

=== The @arr@ constraint

'arr' is intentionally left as a runtime error.  Haskell functions are
opaque — there is no way to inspect a @TTerm a -> TTerm b@ closure and emit
the corresponding Python source.  Any combinator that calls @arr@ internally
will throw at code-generation time.  Use the explicit 'TArr' constructors
exported from "UniAlg.Semantics.Category" instead.
-}
module UniAlg.Semantics.Arrows
  ( TArr(..)
  , reify
  , reify2
  ) where

import Prelude hiding (id, (.))

import Control.Category (Category(..))
import Control.Arrow    (Arrow(..), ArrowChoice(..))

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms (unaryFunction, var, (~>))
import qualified Hydra.Dsl.Meta.Lib.Pairs   as Pairs
import qualified Hydra.Dsl.Meta.Lib.Eithers as Eithers


-- | A morphism @a → b@ in the UniAlg DSL.
--
-- Wraps a Haskell function @'TTerm' a -> 'TTerm' b@.  Calling 'runTArr'
-- applies the morphism to a symbolic input and returns the symbolic output
-- — it does not execute any Python.  The resulting 'TTerm' is later handed
-- to Hydra's Python coder to produce source.
newtype TArr a b = TArr { runTArr :: TTerm a -> TTerm b }


instance Category TArr where
  id                = TArr (\x -> x)
  TArr f . TArr g   = TArr (\x -> f (g x))


-- | 'arr' is intentionally broken: Haskell functions are opaque and cannot
-- be reified into Hydra IR.  Use the named 'TArr' combinators instead.
instance Arrow TArr where
  arr _             = error "TArr: arr cannot inspect Haskell functions to generate code"
  first  (TArr f)   = TArr (\p -> tPair (f (tFst p)) (tSnd p))
  second (TArr f)   = TArr (\p -> tPair (tFst p) (f (tSnd p)))
  TArr f *** TArr g = TArr (\p -> tPair (f (tFst p)) (g (tSnd p)))
  TArr f &&& TArr g = TArr (\x -> tPair (f x) (g x))


instance ArrowChoice TArr where
  left  (TArr f)    = TArr (\e -> tEither (\l -> tLeft  (f l)) tRight e)
  right (TArr f)    = TArr (\e -> tEither tLeft (\r -> tRight (f r)) e)
  TArr f +++ TArr g = TArr (\e -> tEither (\l -> tLeft  (f l)) (\r -> tRight (g r)) e)
  TArr f ||| TArr g = TArr (\e -> tEither f g e)


-- | Lift a Haskell function over 'TTerm's into a 'TTerm' lambda.
--
-- Used to pass an algebra or coalgebra body to Hydra codegen as a first-class
-- term.  The resulting @'TTerm' (a -> b)@ can be applied with @('@@')@.
reify :: (TTerm a -> TTerm b) -> TTerm (a -> b)
reify = unaryFunction

-- | Lift a binary Haskell function over 'TTerm's into a curried 'TTerm' lambda.
reify2 :: (TTerm a -> TTerm b -> TTerm c) -> TTerm (a -> b -> c)
reify2 f = "x" ~> "y" ~> f (var "x") (var "y")


-- ── Private helpers ──────────────────────────────────────────────────────────

tFst :: TTerm (a, b) -> TTerm a
tFst = Pairs.first

tSnd :: TTerm (a, b) -> TTerm b
tSnd = Pairs.second

tPair :: TTerm a -> TTerm b -> TTerm (a, b)
tPair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

tEither :: (TTerm a -> TTerm c) -> (TTerm b -> TTerm c) -> TTerm (Either a b) -> TTerm c
tEither f g e = Eithers.either_ (unaryFunction f) (unaryFunction g) e

tLeft :: TTerm a -> TTerm (Either a b)
tLeft x = TTerm (Terms.left (unTTerm x))

tRight :: TTerm b -> TTerm (Either a b)
tRight x = TTerm (Terms.right (unTTerm x))
