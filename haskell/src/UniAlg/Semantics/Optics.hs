{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}

module UniAlg.Semantics.Optics
  ( Lens
  , Traversal
  , Prism
  , Lens'
  , Traversal'
  , Prism'
  , Endo(..)
  , mkLens
  , mkPrism
  , view
  , set
  , over
  , review
  , preview
  , both
  , tBoth
  , _Just
  , toListOf
  ) where

import Data.Monoid
  ( First(..)
  )

import Data.Profunctor
  ( dimap
  )

import Data.Profunctor.Choice
  ( Choice
  , right'
  )

import Data.Tagged
  ( Tagged(..)
  )

import Hydra.Phantoms
  ( TTerm
  )

import UniAlg.Semantics.Functors
  ( Identity(..)
  , Const(..)
  )

import UniAlg.Semantics.Arrows
  ( TArr(..)
  )

import UniAlg.Semantics.Category
  ( fst
  , snd
  , pair
  )

import Prelude hiding (fst, snd)


-- ── Endo — unifies Haskell functions and TTerm arrows ────────────────────────
-- f is the full endomorphism type, not a type constructor:
--   Endo (a -> a) a        for native Haskell functions
--   Endo (TArr p p) (TTerm p)  for TTerm arrows

class Endo f a where
  endo :: f -> a -> a

instance Endo (a -> a) a where
  endo = id

instance Endo (TArr p p) (TTerm p) where
  endo = runTArr


-- ── Optics ───────────────────────────────────────────────────────────────────

type Lens s t a b =
  forall f. Functor f => (a -> f b) -> s -> f t


type Traversal s t a b =
  forall f. Applicative f => (a -> f b) -> s -> f t


type Prism s t a b =
  forall p f. (Choice p, Applicative f) => p a (f b) -> p s (f t)


type Lens' s a =
  Lens s s a a


type Traversal' s a =
  Traversal s s a a


type Prism' s a =
  Prism s s a a


mkLens :: (s -> a) -> (s -> b -> t) -> Lens s t a b
mkLens get set_ f s =
  fmap (set_ s) (f (get s))


mkPrism :: (b -> t) -> (s -> Either t a) -> Prism s t a b
mkPrism bt sta =
  dimap sta (either pure (fmap bt)) . right'


view :: Lens' s a -> s -> a
view l s =
  getConst (l Const s)


set :: Lens' s a -> a -> s -> s
set l b s =
  runIdentity (l (\_ -> Identity b) s)


over :: Endo f a => Lens' s a -> f -> s -> s
over l f s =
  runIdentity (l (Identity . endo f) s)


review :: Prism s t a b -> b -> t
review l b =
  runIdentity (unTagged (l (Tagged (Identity b))))


preview :: Prism' s a -> s -> Maybe a
preview l s =
  getFirst (getConst (l (Const . First . Just) s))


both :: Traversal' (a, a) a
both f (x, y) =
  (,) <$> f x <*> f y


tBoth :: Traversal' (TTerm (a, a)) (TTerm a)
tBoth f p =
  pair <$> f (fst p) <*> f (snd p)


_Just :: Traversal' (Maybe a) a
_Just f (Just a) = Just <$> f a
_Just _ Nothing  = pure Nothing


toListOf :: Traversal' s a -> s -> [a]
toListOf t s =
  getConst (t (\a -> Const [a]) s)
