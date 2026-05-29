{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

{-|
Polynomial functor building blocks and the 'TFunctor' class.

Polynomial functors are built from 'Identity', 'Const', 'Product', and 'Sum'
using the standard @Data.Functor.*@ types.  'TFunctor' augments these with a
TTerm-level map so that 'cataT', 'anaT', and 'hyloT' can drive code generation
through them.

=== Why not use standard @Functor@?

@Functor@ maps over real Haskell values.  Here the "values" are symbolic
'TTerm' nodes — they need to be wired into Hydra IR, not evaluated.
'TFunctor' provides:

* 'tfmap' — insert a recursive self-call 'TTerm' into the functor layer.
* 'applyAlg' — peel apart a functor-shaped 'TTerm' into real Haskell
  constructors so that algebra functions can use native Haskell pattern matching.
* 'foldToTerm' — reassemble a real Haskell functor value back into a 'TTerm'
  (the inverse direction, used by 'anaT').

=== Predefined aliases

General aliases: 'MaybeF', 'ListF', 'RoseF', 'TreeF'.

Neural-architecture aliases, chosen so that the functor shape makes the
recursion scheme obvious.  The type parameter names the *underlying* element
type (@Tensor@, not @'TTerm' Tensor@), so you write @\@('SeqF' Tensor)@:

* 'SeqF' @a@ — @F(X) = 1 + ('TTerm' a × X)@ — sequence / RNN / transformer layer.
* 'RTreeF' @a@ — @F(X) = 'TTerm' a + (X × X)@ — binary tree with data at leaves.
* 'StreamF' @o@ — @F(X) = 'TTerm' o × X@ — infinite stream / unfolding RNN.

=== Writing a bare @\@MyF@ type application

The predefined aliases keep their element-type parameter so they stay general.
If you want to write just @\@MyF@ (no argument), declare a saturated synonym in
your own module:

@
type MySeq = SeqF Tensor
-- then: recModule \@MySeq ...
@
-}
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

-- | 'TTerm'-level counterpart of 'Functor', used to drive 'cataT', 'anaT',
-- and 'hyloT'.
--
-- Every instance @f@ must also be a plain 'Functor' so that the standard
-- recursion schemes ('cata', 'ana', 'hylo') still work on real Haskell values.
class Functor f => TFunctor f where
  -- | Map the recursive self-call 'TTerm' over one functor layer.
  --
  -- The first argument is the self-call (e.g. @var \"f\"@); the second is
  -- the current node.  Used internally by 'cataT'.
  tfmap :: TTerm a -> TTerm a -> TTerm a

  -- | Peel one functor layer off a 'TTerm' node into real Haskell
  -- constructors, pass them to the algebra, and return the result.
  --
  -- This is what allows algebra functions to use native Haskell pattern
  -- matching (@\case@) instead of operating on raw 'TTerm' values.
  applyAlg :: (f (TTerm s) -> TTerm r) -> TTerm s -> TTerm r

  -- | Reassemble a real Haskell functor value back into a 'TTerm'.
  --
  -- Inverse of 'applyAlg'; used by 'anaT' to fold a coalgebra result back
  -- into the Hydra IR.
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
      (coerce x)

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

-- | @F(X) = 1 + X@
type MaybeF   = Sum (Const ())
-- | @F(X) = 1 + (a × X)@
type ListF  a = Sum (Const ()) (Product (Const (TTerm a)) Identity)
-- | @F(X) = f(X) × [X]@  — rose tree functor
type RoseF  f = Product f []
-- | @F(X) = 1 + f(X) × [X]@  — general tree
type TreeF  f = Sum (Const ()) (RoseF f)


-- ── Functor aliases — neural architectures ───────────────────────────────────

-- | @F(X) = 1 + (a × X)@  — sequence, RNN layer, or transformer block.
-- The base case @InL (Const ())@ is the empty sequence; the recursive case
-- @InR (Pair (Const layer) (Identity rest))@ carries one layer ('TTerm' @a@)
-- and the tail.
type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

-- | @F(X) = a + (X × X)@  — binary tree with data at leaves.
-- @InL (Const a)@ is a leaf carrying @'TTerm' a@; @InR (Pair l r)@ is an
-- internal node with two recursive subtrees.
type RTreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

-- | @F(X) = o × X@  — stream or unfolding RNN.
-- Each step emits an output (@'TTerm' o@) and continues with the next state.
-- The anamorphism 'anaT' over this functor generates a corecursive Python
-- function.
type StreamF o = Product (Const (TTerm o)) Identity
