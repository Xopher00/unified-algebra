{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImplicitParams      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

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

newtype Fix f = Fix
  { unFix :: f (Fix f)
  }


cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg =
  alg . fmap (cata alg) . unFix


ana :: Functor f => (a -> f a) -> a -> Fix f
ana coalg =
  Fix . fmap (ana coalg) . coalg


hylo :: Functor f => (f b -> b) -> (a -> f a) -> a -> b
hylo alg coalg =
  alg . fmap (hylo alg coalg) . coalg


-- ── TTerm-level recursion schemes for code generation ────────────────────────
-- Algebra type f (TTerm a) -> TTerm a is identical to the Haskell schemes.
-- The self-reference is implicit: establish it once at the definition site.
--
-- Simple recursion:       withSelf (var "f") $ cataT @F alg
-- With outer args (e.g. weights): withSelf (var "f" `tApp` var "w") $ cataT @F alg
--
-- tApp: TTerm (a -> b) -> TTerm a -> TTerm b
-- tApp f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))

withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k


cataT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
cataT alg x =
  applyAlg @f (\layer -> alg (fmap step layer)) x
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


anaT :: forall f a. (TFunctor f, ?self :: TTerm a)
     => (TTerm a -> f (TTerm a))
     -> TTerm a -> TTerm a
anaT coalg x =
  foldToTerm @f (fmap step (coalg x))
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))


hyloT :: forall f a. (TFunctor f, ?self :: TTerm a)
      => (TTerm a -> f (TTerm a))
      -> (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
hyloT coalg alg x =
  alg (fmap step (coalg x))
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))
