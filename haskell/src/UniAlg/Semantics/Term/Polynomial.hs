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
module UniAlg.Semantics.Term.Polynomial
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

    -- * Exponential / Reader functor
  , Exp(..)

  ) where

import Data.Coerce            ( coerce )

import Data.Functor.Identity  ( Identity(..) )
import Data.Functor.Const     ( Const(..) )
import Data.Functor.Product   ( Product(..) )
import Data.Functor.Sum       ( Sum(..) )

import Hydra.Kernel           ( Term(..) )
import Hydra.Phantoms         ( TTerm(..) )
import qualified Hydra.Dsl.Terms as Terms

import Hydra.Sources.Libraries
  ( _lists_map
  )

import UniAlg.Semantics.Term.Builders


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


-- | Reader / exponential functor @(I→−)@.
--
-- @Exp (TTerm i) a@ wraps a Haskell function from @'TTerm' i@ to @a@.
-- The 'TFunctor' instance treats the held 'TTerm' as a symbolic function-typed
-- term: 'applyAlg' wraps it so algebras can call 'runExp' with a 'TTerm' input,
-- and 'foldToTerm' emits a lambda.
--
-- Primary use: 'MooreF'.
newtype Exp r a = Exp { runExp :: r -> a }

instance Functor (Exp r) where
  fmap f (Exp g) = Exp (f . g)

-- The held TTerm is treated as a function-typed term (kind @i → s@).
-- 'applyAlg' wraps it so the algebra receives a Haskell @Exp (TTerm i) (TTerm s)@
-- and can call 'runExp' with concrete 'TTerm i' inputs.
-- 'foldToTerm' emits a fresh lambda binding @"inp"@.
instance TFunctor (Exp (TTerm i)) where
  tfmap recCall x =
    tLam "inp" (tApp recCall (tApp x (tVar "inp")))

  applyAlg alg x =
    alg (Exp (\inp -> coerce (tApp (coerce x :: TTerm i) inp)))

  foldToTerm (Exp g) =
    tLam "inp" (g (tVar "inp"))


-- ── Functor aliases — general ─────────────────────────────────────────────────

-- | @F(X) = 1 + X@
type MaybeF   = Sum (Const ())
-- | @F(X) = 1 + (a × X)@
type ListF  a = Sum (Const ()) (Product (Const (TTerm a)) Identity)
-- | @F(X) = f(X) × [X]@  — rose tree functor
type RoseF  f = Product f []
-- | @F(X) = 1 + f(X) × [X]@  — general tree
type TreeF  f = Sum (Const ()) (RoseF f)
