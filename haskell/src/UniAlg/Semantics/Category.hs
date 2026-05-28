{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

module UniAlg.Semantics.Category
  ( -- * Term-level arrow
    module UniAlg.Semantics.Arrows

    -- * TTerm-level operator aliases (identical syntax to Control.Arrow)
  , (>>>)
  , (&&&)
  , (***)
  , (|||)
  , (+++)

    -- * Value-level TTerm operations (overload Prelude)
  , fst
  , snd
  , pair
  , either
  , left
  , right
  , tApply

    -- * Structural morphisms
  , copy
  , delete
  , assoc
  , symmetry
  , merge
  , distributeLeft
  , distributeRight

    -- * Legacy
  , terminalObj
  , absurdMorphism
  ) where

import Prelude hiding (fst, snd, either, left, right)

import Data.Void (Void, absurd)

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms
import qualified Hydra.Dsl.Meta.Lib.Pairs   as Pairs
import qualified Hydra.Dsl.Meta.Lib.Eithers as Eithers

import UniAlg.Semantics.Arrows (TArr(..), reify)


-- ── TTerm-level operator aliases ─────────────────────────────────────────────

infixr 1 >>>
(>>>) :: TArr a b -> TArr b c -> TArr a c
TArr f >>> TArr g = TArr (\x -> g (f x))

infixr 3 &&&
(&&&) :: TArr a b -> TArr a c -> TArr a (b, c)
TArr f &&& TArr g = TArr (\x -> pair (f x) (g x))

infixr 3 ***
(***) :: TArr a b -> TArr c d -> TArr (a, c) (b, d)
TArr f *** TArr g = TArr (\p -> pair (f (fst p)) (g (snd p)))

infixr 2 |||
(|||) :: TArr a c -> TArr b c -> TArr (Either a b) c
TArr f ||| TArr g = TArr (either f g)

infixr 2 +++
(+++) :: TArr a b -> TArr c d -> TArr (Either a c) (Either b d)
TArr f +++ TArr g = TArr (\e -> either (\l -> left (f l)) (\r -> right (g r)) e)


-- ── Value-level TTerm operations ─────────────────────────────────────────────

fst :: TTerm (a, b) -> TTerm a
fst = Pairs.first

snd :: TTerm (a, b) -> TTerm b
snd = Pairs.second

pair :: TTerm a -> TTerm b -> TTerm (a, b)
pair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

either :: (TTerm a -> TTerm c) -> (TTerm b -> TTerm c) -> TTerm (Either a b) -> TTerm c
either f g = Eithers.either_ (reify f) (reify g)

left :: TTerm a -> TTerm (Either a b)
left x = TTerm (Terms.left (unTTerm x))

right :: TTerm b -> TTerm (Either a b)
right x = TTerm (Terms.right (unTTerm x))

-- | Type-erased term application. Needed when the phantom type of a TTerm
-- conflicts with the typed (@@) operator — e.g. completing a partial
-- self-call in a cataBody algebra where the recursive result has phantom
-- type TTerm a but must be applied to additional arguments.
tApply :: TTerm a -> TTerm a -> TTerm a
tApply f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))


-- ── Structural morphisms ──────────────────────────────────────────────────────

copy :: TArr a (a, a)
copy = TArr (\x -> pair x x)

delete :: TArr a ()
delete = TArr (\_ -> TTerm Terms.unit)

symmetry :: TArr (a, b) (b, a)
symmetry = TArr (\p -> pair (snd p) (fst p))

assoc :: TArr ((a, b), c) (a, (b, c))
assoc = TArr (\p -> pair (fst (fst p)) (pair (snd (fst p)) (snd p)))

merge :: TArr (Either a a) a
merge = TArr (\e -> either id id e)

distributeLeft :: TArr (a, Either b c) (Either (a, b) (a, c))
distributeLeft = TArr (\p ->
  either
    (\l -> left  (pair (fst p) l))
    (\r -> right (pair (fst p) r))
    (snd p))

distributeRight :: TArr (Either a b, c) (Either (a, c) (b, c))
distributeRight = TArr (\p ->
  either
    (\l -> left  (pair l (snd p)))
    (\r -> right (pair r (snd p)))
    (fst p))


-- ── Legacy ────────────────────────────────────────────────────────────────────

terminalObj :: ()
terminalObj = ()

absurdMorphism :: Void -> a
absurdMorphism = absurd
