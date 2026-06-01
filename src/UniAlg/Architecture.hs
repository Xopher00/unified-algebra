{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module UniAlg.Architecture
  ( Elim
  , CoElim
  , hyloDef
  , hyloModule
  , cataModule
  , anaModule
  ) where

import Data.Coerce (coerce)
import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))
import qualified Data.Kind as Kind

import Hydra.Phantoms (TTerm(..), unTTerm)
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms
  ( var
  , (~>)
  )
import Hydra.Kernel
  ( Definition(..)
  , FunctionType(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , TermDefinition(..)
  , Type(..)
  , TypeScheme(..)
  )

import UniAlg.Core.Reduce (reduceTerm)
import UniAlg.Shape.Encode
  ( Exp(..)
  , Shape(..)
  )
import UniAlg.Scheme.Internal
  ( hyloT
  , withSelf
  )
import UniAlg.Term.Internal
  ( tApply
  , tLam
  , tLeft
  , tPair
  , tRight
  , tVar
  )


type family Elim (f :: Kind.Type -> Kind.Type) (a :: Kind.Type) (r :: Kind.Type) :: Kind.Type where
  Elim (Const ())        a r = r
  Elim (Const (TTerm k)) a r = TTerm k -> r
  Elim Identity          a r = TTerm a -> r
  Elim (Sum f g)         a r = (Elim f a r, Elim g a r)
  Elim (Product f g)     a r = Elim f a (Elim g a r)
  Elim (Exp (TTerm i))   a r = (TTerm i -> TTerm a) -> r

class Shape f => TElim f where
  elimToAlg :: Elim f a r -> f (TTerm a) -> r

instance TElim (Const ()) where
  elimToAlg r (Const ()) = r

instance TElim (Const (TTerm k)) where
  elimToAlg f (Const k) = f k

instance TElim Identity where
  elimToAlg f (Identity x) = f x

instance (TElim f, TElim g) => TElim (Sum f g) where
  elimToAlg (fl, fr) = \case
    InL l -> elimToAlg @f fl l
    InR r -> elimToAlg @g fr r

instance (TElim f, TElim g) => TElim (Product f g) where
  elimToAlg h (Pair fl gr) = elimToAlg @g (elimToAlg @f h fl) gr

instance TElim (Exp (TTerm i)) where
  elimToAlg h (Exp g) = h g

type family CoElim (f :: Kind.Type -> Kind.Type) (a :: Kind.Type) :: Kind.Type where
  CoElim (Const ())        a = ()
  CoElim (Const (TTerm k)) a = TTerm k
  CoElim Identity          a = TTerm a
  CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
  CoElim (Product f g)     a = (CoElim f a, CoElim g a)
  CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a
  CoElim (Const (TTerm i -> TTerm o)) a = TTerm i -> TTerm o

class Shape f => TCoElim f where
  coElimToTerm :: CoElim f a -> TTerm a

instance TCoElim (Const ()) where
  coElimToTerm () = TTerm Terms.unit

instance TCoElim (Const (TTerm k)) where
  coElimToTerm k = coerce k

instance TCoElim Identity where
  coElimToTerm x = x

instance (TCoElim f, TCoElim g) => TCoElim (Sum f g) where
  coElimToTerm (Left fv) = tLeft (coElimToTerm @f fv)
  coElimToTerm (Right gv) = tRight (coElimToTerm @g gv)

instance (TCoElim f, TCoElim g) => TCoElim (Product f g) where
  coElimToTerm (fv, gv) = tPair (coElimToTerm @f fv) (coElimToTerm @g gv)

instance TCoElim (Exp (TTerm i)) where
  coElimToTerm fn = tLam "inp" (fn (tVar "inp"))

instance TCoElim (Const (TTerm i -> TTerm o)) where
  coElimToTerm fn = coerce (tLam "inp" (fn (tVar "inp")))

polyFnScheme :: Int -> TypeScheme
polyFnScheme n = TypeScheme
  { typeSchemeVariables = vars
  , typeSchemeBody = foldr step ret (init tvars)
  , typeSchemeConstraints = Nothing
  }
  where
    vars = [Name ("_a" <> show i) | i <- [0 .. n - 1]]
    tvars = fmap TypeVariable vars
    ret = last tvars
    step a b = TypeFunction FunctionType
      { functionTypeDomain = a
      , functionTypeCodomain = b
      }

hyloDef :: forall f a. Shape f
        => String
        -> String
        -> [String]
        -> ([TTerm a] -> ( TTerm a -> TTerm a
                         , f (TTerm a) -> TTerm a ))
        -> TermDefinition
hyloDef ns name outerArgNames k = TermDefinition
  { termDefinitionName = Name (ns <> "." <> name)
  , termDefinitionTerm = reduceTerm $ unTTerm $ foldr (~>) innerTerm outerArgNames
  , termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 2))
  }
  where
    vars = map var outerArgNames
    (coalg, alg) = k vars
    appliedSelf = foldl (\s n -> tApply s (var n)) (var (ns <> "." <> name)) outerArgNames
    innerTerm = "x" ~> withSelf appliedSelf (hyloT @f coalg alg (var "x"))

hyloModule :: forall f a. Shape f
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> ( TTerm a -> TTerm a
                            , f (TTerm a) -> TTerm a ))
           -> Module
hyloModule ns name deps outerArgNames k = Module
  { moduleDescription = Just "Hylomorphic recursive definition"
  , moduleNamespace = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions = [DefinitionTerm (hyloDef @f ns name outerArgNames k)]
  }

recModule :: forall f a. Shape f
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> ( TTerm a -> TTerm a
                           , f (TTerm a) -> TTerm a ))
          -> Module
recModule = hyloModule

cataModule :: forall f a. (Shape f, TElim f)
           => String
           -> String
           -> [Namespace]
           -> [String]
           -> ([TTerm a] -> Elim f a (TTerm a))
           -> Module
cataModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs -> (id, elimToAlg @f (k vs))

anaModule :: forall f a. (Shape f, TCoElim f)
          => String
          -> String
          -> [Namespace]
          -> [String]
          -> ([TTerm a] -> (TTerm a -> CoElim f a))
          -> Module
anaModule ns name deps outerArgNames k =
  recModule @f ns name deps outerArgNames $ \vs ->
    (coElimToTerm @f . k vs, buildLayer @f)
