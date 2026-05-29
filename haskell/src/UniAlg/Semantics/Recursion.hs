{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImplicitParams      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

{-|
Recursion schemes at two levels.

=== Haskell level

'Fix', 'cata', 'ana', and 'hylo' are the standard recursion schemes over
real Haskell values.  Use these when building 'Fix'-structured inputs to
pass to the TTerm-level schemes.

=== TTerm level (code generation)

'cataT', 'anaT', and 'hyloT' operate on symbolic 'TTerm' values and emit
Hydra IR.  They have the same type signatures as their Haskell counterparts
except that the algebra and coalgebra work with @f ('TTerm' a) -> 'TTerm' a@.

The key difference is self-reference.  A recursive Python function must call
itself by name, but at code-generation time the name is not yet bound.
'withSelf' injects it as an implicit parameter @?self :: 'TTerm' a@ so that
each recursive step can emit the correct self-application.

=== Typical usage

@
-- Simple catamorphism (no outer parameters):
withSelf (var \"f\") $ cataT @MyF myAlg

-- With shared outer parameters (e.g. weights w, initial state s0):
recModule ns \"f\" deps [\"w\", \"s0\"] $
  cataT @MyF myAlg
-- recModule handles withSelf and the partial self-application automatically.
@
-}
module UniAlg.Semantics.Recursion
  ( Fix(..)
  , cata
  , ana
  , hylo
  , withSelf
  , cataT
  , anaT
  , hyloT
  ) where

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms
import UniAlg.Semantics.Functors (TFunctor(..))


-- ── Fixed points and Haskell-level recursion schemes ─────────────────────────

-- | Least fixed point of a functor.  Used to build recursive data structures
-- that are then consumed by 'cata' or 'cataT'.
newtype Fix f = Fix
  { unFix :: f (Fix f)
  }


-- | Catamorphism (fold) over a 'Fix'-structured value.
cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg =
  alg . fmap (cata alg) . unFix


-- | Anamorphism (unfold) producing a 'Fix'-structured value.
ana :: Functor f => (a -> f a) -> a -> Fix f
ana coalg =
  Fix . fmap (ana coalg) . coalg


-- | Hylomorphism: unfold then fold, fusing coalgebra and algebra in one pass.
hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
hylo alg coalg =
  alg . fmap (hylo alg coalg) . coalg


-- ── TTerm-level recursion schemes for code generation ────────────────────────

-- | Bind the recursive self-reference before running a TTerm recursion scheme.
--
-- @?self@ is an implicit parameter carrying the 'TTerm' that represents the
-- function being defined.  For a plain recursive definition use
-- @var \"name\"@.  When outer parameters are shared (e.g. weights @w@), use
-- a partial application: @var \"name\" \`tApply\` var \"w\"@.
--
-- 'recModule' and 'recDef' handle 'withSelf' automatically; call it directly
-- only when building a module outside those helpers.
withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k


-- | TTerm-level catamorphism.  Algebra type @f ('TTerm' a) -> 'TTerm' a@
-- mirrors the Haskell 'cata' signature.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
-- Use @TypeApplications@ to select the functor: @cataT \@('SeqF' Layer) alg@.
cataT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
cataT alg x =
  applyAlg @f (\layer -> alg (fmap step layer)) x
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


-- | TTerm-level anamorphism.  Dispatches at runtime via 'applyAlg' — the
-- dual of 'cataT'.  The coalgebra must produce a runtime @F@-shaped term so
-- that 'applyAlg' can branch it with @eithers.either@ at Python runtime.
--
-- Coalgebra type @'TTerm' a -> 'TTerm' a@ matches 'hyloT'.
-- Defined as @'hyloT' coalg ('foldToTerm' \@f)@.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
anaT :: forall f a. (TFunctor f, ?self :: TTerm a)
     => (TTerm a -> TTerm a)    -- ^ coalgebra
     -> TTerm a -> TTerm a
anaT coalg = hyloT @f coalg (foldToTerm @f)


-- | TTerm-level hylomorphism.  Applies @coalg@ to the seed, then dispatches
-- on the functor shape at Python runtime via 'applyAlg' (the same mechanism
-- 'cataT' uses), and tears down with @alg@.
--
-- @coalg = id@ collapses exactly to 'cataT'.  A non-trivial @coalg@ encodes
-- the coalgebraic (generation) side of an architecture: it transforms the seed
-- before the fold step, so both algebra and coalgebra are declared together in
-- one 'recModule' call.
--
-- Requires @?self@ to be in scope — use 'withSelf' or 'recModule'.
hyloT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (TTerm a -> TTerm a)        -- ^ coalgebra: TTerm-level seed transform
      -> (f (TTerm a) -> TTerm a)    -- ^ algebra
      -> TTerm a -> TTerm a
hyloT coalg alg x =
  applyAlg @f (\layer -> alg (fmap step layer)) (coalg x)
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))
