{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

{-|
Natural eliminator and co-eliminator typeclasses for arch-facing code.

'TElim' converts a naturally-typed algebra handler into the 'applyAlg'-style
algebra @f ('TTerm' a) -> 'TTerm' a@.  Arch authors write tuples of plain
functions; the library handles all @Data.Functor.*@ constructor wrangling.

'TCoElim' converts a naturally-typed coalgebra output into a 'TTerm'.
Arch authors produce plain tuples or 'Either's; the library assembles them
into the runtime F-shaped 'TTerm' representation.

Together, 'Elim' and 'CoElim' are duals:

* 'Elim' computes what the algebra __receives__ (one handler per constructor).
* 'CoElim' computes what the coalgebra __produces__ (one value per component).
-}
module UniAlg.Semantics.Schemes
  ( -- * Algebra eliminator
    TElim(..)
  , Elim

    -- * Coalgebra co-eliminator
  , TCoElim(..)
  , CoElim
  ) where

import Data.Coerce            ( coerce )

import Data.Functor.Identity  ( Identity(..) )
import Data.Functor.Const     ( Const(..) )
import Data.Functor.Product   ( Product(..) )
import Data.Functor.Sum       ( Sum(..) )

import Hydra.Phantoms         ( TTerm(..) )
import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Semantics.Term.Polynomial  ( TFunctor, Exp(..) )
import UniAlg.Semantics.Term.Builders    ( tPair, tLeft, tRight, tLam, tVar )


-- ── Elim — natural algebra handler ──────────────────────────────────────────

-- | Type family computing the natural handler type for functor @f@.
--
-- * 'Sum': a pair of handlers, one per branch.
-- * 'Product': curried — left result feeds into the right handler.
-- * 'Const ()': handler is just the result value (no argument).
-- * 'Const (TTerm k)': handler receives the constant as a 'TTerm'.
-- * 'Identity': handler receives the recursive result as a 'TTerm'.
-- * 'Exp (TTerm i)': handler receives a Haskell function @TTerm i -> TTerm a@.
type family Elim (f :: * -> *) (a :: *) (r :: *) :: * where
  Elim (Const ())        a r = r
  Elim (Const (TTerm k)) a r = TTerm k -> r
  Elim Identity          a r = TTerm a -> r
  Elim (Sum f g)         a r = (Elim f a r, Elim g a r)
  Elim (Product f g)     a r = Elim f a (Elim g a r)
  Elim (Exp (TTerm i))   a r = (TTerm i -> TTerm a) -> r

-- | Convert a natural 'Elim'-typed handler to the 'applyAlg'-style algebra.
--
-- Instances absorb all @Data.Functor.*@ constructor wrapping\/unwrapping.
-- Arch authors never see 'InL', 'InR', 'Pair', 'Const', or 'Identity'.
class TFunctor f => TElim f where
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


-- ── CoElim — natural coalgebra output ───────────────────────────────────────

-- | Type family computing the natural coalgebra output type for functor @f@.
--
-- Dual to 'Elim'.  Where 'Elim' computes one handler per constructor
-- (consumption), 'CoElim' computes one value per component (production):
--
-- * 'Const ()': produces nothing — @()@.
-- * 'Const (TTerm k)': produces the constant — @TTerm k@.
-- * 'Identity': produces the recursive seed — @TTerm a@.
-- * 'Sum': chooses a side — @Either@.
-- * 'Product': produces both — a pair.
-- * 'Exp (TTerm i)': produces a function — @TTerm i -> TTerm a@.
type family CoElim (f :: * -> *) (a :: *) :: * where
  CoElim (Const ())        a = ()
  CoElim (Const (TTerm k)) a = TTerm k
  CoElim Identity          a = TTerm a
  CoElim (Sum f g)         a = Either (CoElim f a) (CoElim g a)
  CoElim (Product f g)     a = (CoElim f a, CoElim g a)
  CoElim (Exp (TTerm i))   a = TTerm i -> TTerm a

-- | Convert a naturally-typed coalgebra output into a 'TTerm'.
--
-- Mirrors 'foldToTerm' but accepts 'CoElim f a' instead of @f ('TTerm' a)@.
-- Arch authors produce plain tuples or 'Either's; the library assembles them.
class TFunctor f => TCoElim f where
  coElimToTerm :: CoElim f a -> TTerm a

instance TCoElim (Const ()) where
  coElimToTerm () = TTerm Terms.unit

instance TCoElim (Const (TTerm k)) where
  coElimToTerm k = coerce k

instance TCoElim Identity where
  coElimToTerm x = x

instance (TCoElim f, TCoElim g) => TCoElim (Sum f g) where
  coElimToTerm (Left  fv) = tLeft  (coElimToTerm @f fv)
  coElimToTerm (Right gv) = tRight (coElimToTerm @g gv)

instance (TCoElim f, TCoElim g) => TCoElim (Product f g) where
  coElimToTerm (fv, gv) = tPair (coElimToTerm @f fv) (coElimToTerm @g gv)

instance TCoElim (Exp (TTerm i)) where
  coElimToTerm fn = tLam "inp" (fn (tVar "inp"))
