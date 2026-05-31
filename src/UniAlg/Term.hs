{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

module UniAlg.Term
  ( TArr(..)
  , reify
  , reify2
  , (>>>)
  , (&&&)
  , (***)
  , (|||)
  , (+++)
  , fst
  , snd
  , pair
  , either
  , left
  , right
  , copy
  , delete
  , assoc
  , symmetry
  , merge
  , distributeLeft
  , distributeRight
  , terminalObj
  , absurdMorphism
  ) where

import Prelude hiding (fst, snd, either, left, right, id, (.))

import qualified Control.Arrow as Arr
import qualified Control.Category as Cat
import Data.Void (Void, absurd)

import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms
import Hydra.Dsl.Meta.Phantoms (unaryFunction, var, (~>))
import qualified Hydra.Dsl.Meta.Lib.Pairs as Pairs
import qualified Hydra.Dsl.Meta.Lib.Eithers as Eithers


newtype TArr a b = TArr { runTArr :: TTerm a -> TTerm b }

instance Cat.Category TArr where
  id = TArr (\x -> x)
  TArr f . TArr g = TArr (\x -> f (g x))

instance Arr.Arrow TArr where
  arr _ = error "TArr: arr cannot inspect Haskell functions to generate code"
  first (TArr f) = TArr (\p -> pair (f (fst p)) (snd p))
  second (TArr f) = TArr (\p -> pair (fst p) (f (snd p)))
  TArr f *** TArr g = TArr (\p -> pair (f (fst p)) (g (snd p)))
  TArr f &&& TArr g = TArr (\x -> pair (f x) (g x))

instance Arr.ArrowChoice TArr where
  left (TArr f) = TArr (\e -> either (\l -> leftTerm (f l)) rightTerm e)
  right (TArr f) = TArr (\e -> either leftTerm (\r -> rightTerm (f r)) e)
  TArr f +++ TArr g = TArr (\e -> either (\l -> leftTerm (f l)) (\r -> rightTerm (g r)) e)
  TArr f ||| TArr g = TArr (either f g)

reify :: (TTerm a -> TTerm b) -> TTerm (a -> b)
reify = unaryFunction

reify2 :: (TTerm a -> TTerm b -> TTerm c) -> TTerm (a -> b -> c)
reify2 f = "x" ~> "y" ~> f (var "x") (var "y")

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
TArr f +++ TArr g = TArr (\e -> either (\l -> leftTerm (f l)) (\r -> rightTerm (g r)) e)

fst :: TTerm (a, b) -> TTerm a
fst = Pairs.first

snd :: TTerm (a, b) -> TTerm b
snd = Pairs.second

pair :: TTerm a -> TTerm b -> TTerm (a, b)
pair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

either :: (TTerm a -> TTerm c) -> (TTerm b -> TTerm c) -> TTerm (Either a b) -> TTerm c
either f g = Eithers.either_ (reify f) (reify g)

left :: TTerm a -> TTerm (Either a b)
left = leftTerm

right :: TTerm b -> TTerm (Either a b)
right = rightTerm

copy :: TArr a (a, a)
copy = TArr (\x -> pair x x)

delete :: TArr a ()
delete = TArr (\_ -> TTerm Terms.unit)

symmetry :: TArr (a, b) (b, a)
symmetry = TArr (\p -> pair (snd p) (fst p))

assoc :: TArr ((a, b), c) (a, (b, c))
assoc = TArr (\p -> pair (fst (fst p)) (pair (snd (fst p)) (snd p)))

merge :: TArr (Either a a) a
merge = TArr (\e -> either (\x -> x) (\x -> x) e)

distributeLeft :: TArr (a, Either b c) (Either (a, b) (a, c))
distributeLeft = TArr (\p ->
  either
    (\l -> leftTerm (pair (fst p) l))
    (\r -> rightTerm (pair (fst p) r))
    (snd p))

distributeRight :: TArr (Either a b, c) (Either (a, c) (b, c))
distributeRight = TArr (\p ->
  either
    (\l -> leftTerm (pair l (snd p)))
    (\r -> rightTerm (pair r (snd p)))
    (fst p))

terminalObj :: ()
terminalObj = ()

absurdMorphism :: Void -> a
absurdMorphism = absurd

leftTerm :: TTerm a -> TTerm (Either a b)
leftTerm x = TTerm (Terms.left (unTTerm x))

rightTerm :: TTerm b -> TTerm (Either a b)
rightTerm x = TTerm (Terms.right (unTTerm x))
