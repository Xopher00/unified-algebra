{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module UniAlg.Shape.Encode
  ( Shape(..)
  , Exp(..)
  ) where

import Data.Coerce (coerce)
import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Term.Internal
  ( tApp
  , tEither
  , tFst
  , tLam
  , tLeft
  , tPair
  , tRight
  , tSnd
  , tVar
  )


class Functor f => Shape f where
  matchLayer :: (f (TTerm a) -> TTerm r) -> TTerm a -> TTerm r
  buildLayer :: f (TTerm a) -> TTerm a

instance Shape Identity where
  matchLayer alg x = alg (Identity x)
  buildLayer (Identity x) = x

instance Shape (Const ()) where
  matchLayer alg _ = alg (Const ())
  buildLayer (Const ()) = TTerm Terms.unit

instance Shape (Const (TTerm k)) where
  matchLayer alg x = alg (Const (coerce x))
  buildLayer (Const k) = coerce k

instance (Shape f, Shape g) => Shape (Sum f g) where
  matchLayer alg x =
    tEither
      (tLam "l" (matchLayer @f (alg . InL) (tVar "l")))
      (tLam "r" (matchLayer @g (alg . InR) (tVar "r")))
      (coerce x)

  buildLayer (InL l) = tLeft (buildLayer @f l)
  buildLayer (InR r) = tRight (buildLayer @g r)

instance (Shape f, Shape g) => Shape (Product f g) where
  matchLayer alg x =
    matchLayer @f
      (\fl -> matchLayer @g (\gr -> alg (Pair fl gr)) (tSnd x))
      (tFst x)

  buildLayer (Pair l r) = tPair (buildLayer @f l) (buildLayer @g r)

newtype Exp r a = Exp { runExp :: r -> a }

instance Functor (Exp r) where
  fmap f (Exp g) = Exp (f . g)

instance Shape (Exp (TTerm i)) where
  matchLayer alg x =
    alg (Exp (\inp -> coerce (tApp (coerce x :: TTerm i) inp)))

  buildLayer (Exp g) =
    tLam "inp" (g (tVar "inp"))
