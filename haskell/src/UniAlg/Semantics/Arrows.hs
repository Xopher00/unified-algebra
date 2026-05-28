{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

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


newtype TArr a b = TArr { runTArr :: TTerm a -> TTerm b }


instance Category TArr where
  id                = TArr (\x -> x)
  TArr f . TArr g   = TArr (\x -> f (g x))


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


-- | Reify a Haskell function over TTerms into a TTerm lambda.
reify :: (TTerm a -> TTerm b) -> TTerm (a -> b)
reify = unaryFunction

-- | Reify a binary Haskell function over TTerms into a TTerm lambda.
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
