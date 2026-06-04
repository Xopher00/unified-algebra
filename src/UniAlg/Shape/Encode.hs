{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module UniAlg.Shape.Encode
  ( Shape(..)
  , Exp(..)
  , ConstFn(..)
  , CoElim
  , TCoElim(..)
  ) where

import Data.Coerce (coerce)
import Data.Functor.Const (Const(..))
import Data.Functor.Identity (Identity(..))
import Data.Functor.Product (Product(..))
import Data.Functor.Sum (Sum(..))
import qualified Data.Kind as Kind

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Term.Internal
  ( tApply
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

instance Shape (Const (TTerm i -> TTerm o)) where
  matchLayer alg x =
    alg (Const (coerce . tApply (coerce x :: TTerm i)))
  buildLayer (Const f) =
    coerce (tLam "inp" (f (tVar "inp")))

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
      (\fl -> matchLayer @g (alg . Pair fl) (tSnd x))
      (tFst x)

  buildLayer (Pair l r) = tPair (buildLayer @f l) (buildLayer @g r)

newtype Exp r a = Exp { runExp :: r -> a }

instance Functor (Exp r) where
  fmap f (Exp g) = Exp (f . g)

instance Shape (Exp (TTerm i)) where
  matchLayer alg x =
    alg (Exp (coerce . tApply (coerce x :: TTerm i)))

  buildLayer (Exp g) =
    tLam "inp" (g (tVar "inp"))

-- | Constant-output exponential: @ConstFn i o@ represents @i → o@,
--   independent of the recursive carrier.  @fmap@ is a no-op so the
--   anamorphism never substitutes a self-call into this position.
newtype ConstFn i o a = ConstFn { runConstFn :: i -> o }

instance Functor (ConstFn i o) where
  fmap _ (ConstFn f) = ConstFn f

instance Shape (ConstFn (TTerm i) (TTerm o)) where
  matchLayer alg x =
    alg (ConstFn (coerce . tApply (coerce x :: TTerm i)))

  buildLayer (ConstFn f) =
    tLam "inp" (coerce (f (tVar "inp")))

-- | Type-level co-eliminator that converts a functor shape into the Haskell
-- value a coalgebra must produce for 'UniAlg.Architecture.anaModule' and
-- (after this refactor) 'UniAlg.Architecture.hyloModule'.
--
-- Each atom maps to a natural Haskell carrier:
--
-- @
-- CoElim (Const ())        a = ()
-- CoElim (Const (TTerm k)) a = TTerm k
-- CoElim Identity          a = TTerm a
-- CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
-- CoElim (Product f g)     a = (CoElim f a, CoElim g a)
-- CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a
-- @
type family CoElim (f :: Kind.Type -> Kind.Type) (a :: Kind.Type) :: Kind.Type where
  CoElim (Const ())        a = ()
  CoElim (Const (TTerm k)) a = TTerm k
  CoElim Identity          a = TTerm a
  CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
  CoElim (Product f g)     a = (CoElim f a, CoElim g a)
  CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a
  CoElim (Const (TTerm i -> TTerm o)) a = TTerm i -> TTerm o

-- | Witness that a 'CoElim'-shaped Haskell value can be re-encoded as a
-- 'TTerm'.  Used by 'UniAlg.Scheme.Internal.hyloT' to bridge between the
-- user-facing coalgebra (which returns explicit Haskell layer values) and
-- 'matchLayer' (which decodes from terms).
class Shape f => TCoElim f where
  coElimToTerm :: CoElim f a -> TTerm a

instance TCoElim (Const ()) where
  coElimToTerm () = TTerm Terms.unit

instance TCoElim (Const (TTerm k)) where
  coElimToTerm = coerce

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
