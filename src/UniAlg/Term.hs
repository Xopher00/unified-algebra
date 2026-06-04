{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

{-|
Categorical morphisms over 'TTerm'.

'TArr' is the core morphism type: a Haskell function @TTerm a -> TTerm b@
wrapped so it can carry 'Control.Category.Category' and
'Control.Arrow.Arrow' instances and compose with standard operators.

Because @TTerm@ values are Hydra IR nodes rather than real Haskell values,
'TArr' cannot implement 'Control.Arrow.arr' — there is no way to inspect a
bare Haskell function @a -> b@ and emit the equivalent Python source.  Any
standard library combinator that calls @arr@ internally will fail at
code-generation time with a runtime error.  Use the re-implemented
operators exported here (@'>>>'@, @'&&&'@, @'|||'@, etc.) instead; they
bypass @arr@ and build 'TTerm' nodes directly.

'reify' and 'reify2' convert plain Haskell functions over @TTerm@ values
into lambda @TTerm@ nodes — the only safe way to cross the @TTerm@ boundary.

The structural morphisms ('pair', 'fst', 'snd', 'either', 'left', 'right',
'copy', 'delete', 'symmetry', 'assoc', 'merge', 'distributeLeft',
'distributeRight') are the categorical building blocks from which algebra and
coalgebra bodies are assembled.
-}
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


-- | A morphism @a → b@ in the @TTerm@ category: a function that transforms
-- one Hydra IR term into another.  Compose with @'>>>'@, split with @'&&&'@,
-- sum with @'|||'@.
newtype TArr a b = TArr { runTArr :: TTerm a -> TTerm b }

instance Cat.Category TArr where
  id = TArr Cat.id
  TArr f . TArr g = TArr (f Cat.. g)

instance Arr.Arrow TArr where
  arr _ = error "TArr: arr cannot inspect Haskell functions to generate code"
  first (TArr f) = TArr (\p -> pair (f (fst p)) (snd p))
  second (TArr f) = TArr (\p -> pair (fst p) (f (snd p)))
  TArr f *** TArr g = TArr (\p -> pair (f (fst p)) (g (snd p)))
  TArr f &&& TArr g = TArr (\x -> pair (f x) (g x))

instance Arr.ArrowChoice TArr where
  left (TArr f) = TArr (either (leftTerm Cat.. f) rightTerm)
  right (TArr f) = TArr (either leftTerm (rightTerm Cat.. f))
  TArr f +++ TArr g = TArr (either (leftTerm Cat.. f) (rightTerm Cat.. g))
  TArr f ||| TArr g = TArr (either f g)

-- | Lift a Haskell function over @TTerm@ values into a lambda @TTerm@ node.
-- Use this when you need to pass a function as a first-class value inside
-- Hydra IR (e.g. as an argument to @either@).
reify :: (TTerm a -> TTerm b) -> TTerm (a -> b)
reify = unaryFunction

-- | Lift a two-argument Haskell function into a curried lambda @TTerm@ node.
reify2 :: (TTerm a -> TTerm b -> TTerm c) -> TTerm (a -> b -> c)
reify2 f = "x" ~> "y" ~> f (var "x") (var "y")

infixr 1 >>>
(>>>) :: TArr a b -> TArr b c -> TArr a c
TArr f >>> TArr g = TArr (g Cat.. f)

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
TArr f +++ TArr g = TArr (either (leftTerm Cat.. f) (rightTerm Cat.. g))

-- | First projection from a pair @TTerm@.
fst :: TTerm (a, b) -> TTerm a
fst = Pairs.first

-- | Second projection from a pair @TTerm@.
snd :: TTerm (a, b) -> TTerm b
snd = Pairs.second

-- | Construct a pair @TTerm@ from two terms.
pair :: TTerm a -> TTerm b -> TTerm (a, b)
pair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

-- | Eliminate a sum @TTerm@: apply @f@ on @Left@, @g@ on @Right@.
either :: (TTerm a -> TTerm c) -> (TTerm b -> TTerm c) -> TTerm (Either a b) -> TTerm c
either f g = Eithers.either_ (reify f) (reify g)

-- | Inject into the left branch of a sum @TTerm@.
left :: TTerm a -> TTerm (Either a b)
left = leftTerm

-- | Inject into the right branch of a sum @TTerm@.
right :: TTerm b -> TTerm (Either a b)
right = rightTerm

-- | Diagonal morphism: duplicate the input into a pair.
copy :: TArr a (a, a)
copy = TArr (\x -> pair x x)

-- | Terminal morphism: discard the input.
delete :: TArr a ()
delete = TArr (\_ -> TTerm Terms.unit)

-- | Swap the components of a pair.
symmetry :: TArr (a, b) (b, a)
symmetry = TArr (\p -> pair (snd p) (fst p))

-- | Re-associate a nested pair left-to-right.
assoc :: TArr ((a, b), c) (a, (b, c))
assoc = TArr (\p -> pair (fst (fst p)) (pair (snd (fst p)) (snd p)))

-- | Codiagonal: eliminate a sum where both branches have the same type.
merge :: TArr (Either a a) a
merge = TArr (either Cat.id Cat.id)

distributeLeft :: TArr (a, Either b c) (Either (a, b) (a, c))
distributeLeft = TArr (\p ->
  either
    (leftTerm Cat.. pair (fst p))
    (rightTerm Cat.. pair (fst p))
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
