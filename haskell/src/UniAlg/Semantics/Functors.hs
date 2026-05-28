{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module UniAlg.Semantics.Functors
  ( -- * Polynomial functor atoms (re-exported from Data.Functor.*)
    Identity(..)
  , Const(..)
  , Product(..)
  , Sum(..)

    -- * TTerm-level functor class
  , TFunctor(..)

    -- * Functor aliases — general
  , MaybeF
  , ListF
  , RoseF
  , TreeF

    -- * Functor aliases — neural architectures
  , SeqF
  , RTreeF
  , StreamF
  ) where

import Data.Coerce            ( coerce )

import Data.Functor.Identity  ( Identity(..) )
import Data.Functor.Const     ( Const(..) )
import Data.Functor.Product   ( Product(..) )
import Data.Functor.Sum       ( Sum(..) )

import Hydra.Kernel           ( Name(..), Term(..) )
import Hydra.Phantoms         ( TTerm(..) )
import qualified Hydra.Dsl.Terms as Terms

import Hydra.Sources.Libraries
  ( _eithers_either
  , _lists_map
  , _pairs_first
  , _pairs_second
  )


-- ── TFunctor — TTerm-level analogue of Functor ───────────────────────────────

class Functor f => TFunctor f where
  -- | Internal: map a self-call TTerm over the functor layer (used by cataBody).
  tfmap :: TTerm a -> TTerm a -> TTerm a

  -- | Peel apart a TTerm functor layer into real Haskell constructors,
  -- pass them to the algebra, and return the result.
  -- This is what lets algebra functions use native Haskell pattern matching.
  applyAlg :: (f (TTerm a) -> TTerm a) -> TTerm a -> TTerm a

  -- | Reassemble a real Haskell functor value back into a TTerm.
  -- Inverse of applyAlg; needed for anaBody.
  foldToTerm :: f (TTerm a) -> TTerm a


instance TFunctor Identity where
  tfmap recCall x    = tApp recCall x
  applyAlg alg x     = alg (Identity x)
  foldToTerm (Identity x) = x


-- Base case: unit constant carries no information — ignore the incoming TTerm.
instance TFunctor (Const ()) where
  tfmap _ x            = x
  applyAlg alg _       = alg (Const ())
  foldToTerm (Const ()) = TTerm Terms.unit

-- TTerm-valued constant: the incoming TTerm IS the constant, phantom-coerce safely.
instance TFunctor (Const (TTerm k)) where
  tfmap _ x            = x
  applyAlg alg x       = alg (Const (coerce x))
  foldToTerm (Const k) = coerce k


instance (TFunctor f, TFunctor g) => TFunctor (Sum f g) where
  tfmap recCall x =
    tEither
      (tLam "l" (tLeft  (tfmap @f recCall (tVar "l"))))
      (tLam "r" (tRight (tfmap @g recCall (tVar "r"))))
      x

  applyAlg alg x =
    tEither
      (tLam "l" (applyAlg @f (alg . InL) (tVar "l")))
      (tLam "r" (applyAlg @g (alg . InR) (tVar "r")))
      x

  foldToTerm (InL l) = tLeft  (foldToTerm @f l)
  foldToTerm (InR r) = tRight (foldToTerm @g r)


instance (TFunctor f, TFunctor g) => TFunctor (Product f g) where
  tfmap recCall x =
    tPair
      (tfmap @f recCall (tFst x))
      (tfmap @g recCall (tSnd x))

  applyAlg alg x =
    applyAlg @f
      (\fl -> applyAlg @g (\gr -> alg (Pair fl gr)) (tSnd x))
      (tFst x)

  foldToTerm (Pair l r) = tPair (foldToTerm @f l) (foldToTerm @g r)


instance TFunctor [] where
  tfmap recCall xs =
    tApp (tApp (TTerm (TermVariable _lists_map)) recCall) xs

  applyAlg _ _ =
    error "TFunctor []: applyAlg not supported; list structure is opaque at codegen time"

  foldToTerm _ =
    error "TFunctor []: foldToTerm not supported"


-- ── Private type-erased helpers ──────────────────────────────────────────────

tApp :: TTerm a -> TTerm a -> TTerm a
tApp f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))

tLam :: String -> TTerm a -> TTerm a
tLam p body = TTerm (Terms.lambda p (unTTerm body))

tVar :: String -> TTerm a
tVar = TTerm . Terms.var

tPair :: TTerm a -> TTerm a -> TTerm a
tPair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

tFst :: TTerm a -> TTerm a
tFst x = TTerm (Terms.apply (TermVariable _pairs_first) (unTTerm x))

tSnd :: TTerm a -> TTerm a
tSnd x = TTerm (Terms.apply (TermVariable _pairs_second) (unTTerm x))

tEither :: TTerm a -> TTerm a -> TTerm a -> TTerm a
tEither f g x = TTerm
  (Terms.apply
    (Terms.apply
      (Terms.apply (TermVariable _eithers_either) (unTTerm f))
      (unTTerm g))
    (unTTerm x))

tLeft :: TTerm a -> TTerm a
tLeft x = TTerm (Terms.left (unTTerm x))

tRight :: TTerm a -> TTerm a
tRight x = TTerm (Terms.right (unTTerm x))


-- ── Functor aliases — general ─────────────────────────────────────────────────

type MaybeF   = Sum (Const ())
type ListF  a = Sum (Const ()) (Product (Const a) Identity)
type RoseF  f = Product f []
type TreeF  f = Sum (Const ()) (RoseF f)


-- ── Functor aliases — neural architectures ───────────────────────────────────

-- F(X) = 1 + (a × X)   sequence / RNN / transformer layer
type SeqF a = Sum (Const ()) (Product (Const a) Identity)

-- F(X) = a + (X × X)   binary tree: leaves carry a, nodes recurse twice
type RTreeF a = Sum (Const a) (Product Identity Identity)

-- F(X) = o × X         stream / unfolding RNN: emit o, continue with X
type StreamF o = Product (Const o) Identity
