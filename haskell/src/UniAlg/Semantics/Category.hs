{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

{-|
Structural morphisms and operator aliases.

Re-exports 'TArr' from "UniAlg.Semantics.Arrows" and adds:

* Operator aliases matching standard @Control.Arrow@ notation so that
  category-theoretic composition feels natural:
  @f '>>>' g@ for sequential composition, @f '&&&' g@ for @⟨f, g⟩@, etc.

* Value-level 'TTerm' operations (@'fst'@, @'snd'@, @'pair'@, @'either'@,
  @'left'@, @'right'@) that override their 'Prelude' counterparts.

* Structural morphisms for the cartesian and cocartesian structure:
  @'copy'@, @'delete'@, @'assoc'@, @'symmetry'@, @'merge'@,
  @'distributeLeft'@, @'distributeRight'@.

Import this module instead of "UniAlg.Semantics.Arrows" for user-facing DSL
work.
-}
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

-- | Sequential composition: @f '>>>' g@ is @g ∘ f@.
infixr 1 >>>
(>>>) :: TArr a b -> TArr b c -> TArr a c
TArr f >>> TArr g = TArr (\x -> g (f x))

-- | Fan-out: @f '&&&' g@ is @⟨f, g⟩ : a → b × c@.
infixr 3 &&&
(&&&) :: TArr a b -> TArr a c -> TArr a (b, c)
TArr f &&& TArr g = TArr (\x -> pair (f x) (g x))

-- | Parallel product: @f '***' g@ is @f × g : a × c → b × d@.
infixr 3 ***
(***) :: TArr a b -> TArr c d -> TArr (a, c) (b, d)
TArr f *** TArr g = TArr (\p -> pair (f (fst p)) (g (snd p)))

-- | Copairing: @f '|||' g@ is @[f, g] : a + b → c@.
infixr 2 |||
(|||) :: TArr a c -> TArr b c -> TArr (Either a b) c
TArr f ||| TArr g = TArr (either f g)

-- | Coproduct bimap: @f '+++' g@ is @f + g : a + c → b + d@.
infixr 2 +++
(+++) :: TArr a b -> TArr c d -> TArr (Either a c) (Either b d)
TArr f +++ TArr g = TArr (\e -> either (\l -> left (f l)) (\r -> right (g r)) e)


-- ── Value-level TTerm operations ─────────────────────────────────────────────
-- These shadow their Prelude counterparts and operate on symbolic TTerm values.

-- | First projection: @'fst' p@ extracts the left component of a pair term.
fst :: TTerm (a, b) -> TTerm a
fst = Pairs.first

-- | Second projection: @'snd' p@ extracts the right component of a pair term.
snd :: TTerm (a, b) -> TTerm b
snd = Pairs.second

-- | Pair constructor: @'pair' a b@ builds a product term.
pair :: TTerm a -> TTerm b -> TTerm (a, b)
pair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

-- | Case analysis on a coproduct term.
either :: (TTerm a -> TTerm c) -> (TTerm b -> TTerm c) -> TTerm (Either a b) -> TTerm c
either f g = Eithers.either_ (reify f) (reify g)

-- | Left injection into a coproduct term.
left :: TTerm a -> TTerm (Either a b)
left x = TTerm (Terms.left (unTTerm x))

-- | Right injection into a coproduct term.
right :: TTerm b -> TTerm (Either a b)
right x = TTerm (Terms.right (unTTerm x))

-- | Type-erased term application.
--
-- Use when the phantom type of a 'TTerm' conflicts with the typed @('@@')@
-- operator — e.g. completing a partial self-call inside a 'cataT' algebra
-- where the recursive result has a monomorphic phantom but must be applied
-- to additional arguments.
tApply :: TTerm a -> TTerm a -> TTerm a
tApply f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))


-- ── Structural morphisms ──────────────────────────────────────────────────────

-- | Diagonal / copy: @δ_A : a → a × a@.
copy :: TArr a (a, a)
copy = TArr (\x -> pair x x)

-- | Counit / delete: @ε_A : a → 1@.
delete :: TArr a ()
delete = TArr (\_ -> TTerm Terms.unit)

-- | Swap: @σ : a × b → b × a@.
symmetry :: TArr (a, b) (b, a)
symmetry = TArr (\p -> pair (snd p) (fst p))

-- | Associator: @α : (a × b) × c → a × (b × c)@.
assoc :: TArr ((a, b), c) (a, (b, c))
assoc = TArr (\p -> pair (fst (fst p)) (pair (snd (fst p)) (snd p)))

-- | Codiagonal / merge: @▽_A : a + a → a@.
merge :: TArr (Either a a) a
merge = TArr (\e -> either id id e)

-- | Left distributivity: @a × (b + c) → (a × b) + (a × c)@.
distributeLeft :: TArr (a, Either b c) (Either (a, b) (a, c))
distributeLeft = TArr (\p ->
  either
    (\l -> left  (pair (fst p) l))
    (\r -> right (pair (fst p) r))
    (snd p))

-- | Right distributivity: @(a + b) × c → (a × c) + (b × c)@.
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
