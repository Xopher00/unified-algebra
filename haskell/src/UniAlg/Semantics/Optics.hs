{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}

{-|
Van Laarhoven optics over both Haskell values and 'TTerm' nodes.

Uses the standard profunctor / van Laarhoven encoding so that optics compose
with @('.')@ and work with the familiar 'view', 'set', 'over', 'review',
'preview' combinators.

=== The 'Endo' class

Lenses applied via 'over' need to accept either a plain Haskell @a -> a@
function or a 'TArr' morphism.  'Endo' provides a single @endo@ method that
dispatches on the type of the modifier, so 'over' works uniformly at both
levels.

=== Usage in algebra functions

Construct lenses with 'mkLens' and prisms with 'mkPrism', then compose them
with @('.')@.  Use 'tBoth' to traverse both components of a @'TTerm' (a, a)@.

@
_fst = mkLens fst (\\p w -> pair w (snd p))
_snd = mkLens snd (\\p w -> pair (fst p) w)

-- Nested access:
_wK = _snd . _fst   -- second field of outer pair, first of inner
@
-}
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

import UniAlg.Semantics.Term.Polynomial
  ( Identity(..)
  , Const(..)
  )

import UniAlg.Semantics.Term.Arrows
  ( TArr(..)
  )

import UniAlg.Semantics.Category
  ( fst
  , snd
  , pair
  )

import Prelude hiding (fst, snd)


-- ── Endo — unifies Haskell functions and TTerm arrows ────────────────────────

-- | Typeclass that unifies plain Haskell endomorphisms and 'TArr' morphisms
-- so that 'over' works at both levels.
--
-- @f@ is the full endomorphism type (not a type constructor):
--
-- * @'Endo' (a -> a) a@ — native Haskell modifier functions.
-- * @'Endo' ('TArr' p p) ('TTerm' p)@ — TTerm-level modifier morphisms.
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


-- | Build a 'Lens' from a getter and a setter.
mkLens :: (s -> a) -> (s -> b -> t) -> Lens s t a b
mkLens get set_ f s =
  fmap (set_ s) (f (get s))


-- | Build a 'Prism' from a constructor and a matcher.
mkPrism :: (b -> t) -> (s -> Either t a) -> Prism s t a b
mkPrism bt sta =
  dimap sta (either pure (fmap bt)) . right'


-- | Extract a value through a 'Lens'.
view :: Lens' s a -> s -> a
view l s =
  getConst (l Const s)


-- | Replace the focused value.
set :: Lens' s a -> a -> s -> s
set l b s =
  runIdentity (l (\_ -> Identity b) s)


-- | Modify the focused value with either a Haskell function or a 'TArr'
-- morphism (via the 'Endo' class).
over :: Endo f a => Lens' s a -> f -> s -> s
over l f s =
  runIdentity (l (Identity . endo f) s)


-- | Construct a value using a 'Prism'.
review :: Prism s t a b -> b -> t
review l b =
  runIdentity (unTagged (l (Tagged (Identity b))))


-- | Try to extract a value through a 'Prism'.
preview :: Prism' s a -> s -> Maybe a
preview l s =
  getFirst (getConst (l (Const . First . Just) s))


-- | Traverse both components of a Haskell pair.
both :: Traversal' (a, a) a
both f (x, y) =
  (,) <$> f x <*> f y


-- | Traverse both components of a @'TTerm' (a, a)@ pair node.
tBoth :: Traversal' (TTerm (a, a)) (TTerm a)
tBoth f p =
  pair <$> f (fst p) <*> f (snd p)


-- | Traverse the 'Just' branch of a @'Maybe'@.
_Just :: Traversal' (Maybe a) a
_Just f (Just a) = Just <$> f a
_Just _ Nothing  = pure Nothing


-- | Collect all focused values into a list.
toListOf :: Traversal' s a -> s -> [a]
toListOf t s =
  getConst (t (\a -> Const [a]) s)
