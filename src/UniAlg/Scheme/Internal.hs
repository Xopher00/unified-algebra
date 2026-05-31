{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module UniAlg.Scheme.Internal
  ( Fix(..)
  , cata
  , ana
  , hylo
  , withSelf
  , cataT
  , anaT
  , hyloT
  ) where

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Shape.Encode
  ( Shape(..)
  )


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

withSelf :: TTerm a -> ((?self :: TTerm a) => r) -> r
withSelf s k = let ?self = s in k

cataT :: forall f a. (Shape f, ?self :: TTerm a)
      => (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
cataT alg x =
  matchLayer @f (\layer -> alg (fmap step layer)) x
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))

anaT :: forall f a. (Shape f, ?self :: TTerm a)
     => (TTerm a -> TTerm a)
     -> TTerm a -> TTerm a
anaT coalg = hyloT @f coalg (buildLayer @f)

hyloT :: forall f a. (Shape f, ?self :: TTerm a)
      => (TTerm a -> TTerm a)
      -> (f (TTerm a) -> TTerm a)
      -> TTerm a -> TTerm a
hyloT coalg alg x =
  matchLayer @f (\layer -> alg (fmap step layer)) (coalg x)
  where
    step arg = TTerm (Terms.apply (unTTerm ?self) (unTTerm arg))
