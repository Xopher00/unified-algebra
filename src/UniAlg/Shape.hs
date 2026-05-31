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


type MaybeF = Sum (Const ())

type ListF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

type RoseF f = Product f []

type TreeF f = Sum (Const ()) (RoseF f)
