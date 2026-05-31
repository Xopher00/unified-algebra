module UniAlg.Term.Internal
  ( tApp
  , tLam
  , tVar
  , tPair
  , tFst
  , tSnd
  , tEither
  , tLeft
  , tRight
  , tApply
  ) where

import Hydra.Kernel (Name(..), Term(..))
import Hydra.Phantoms (TTerm(..))
import qualified Hydra.Dsl.Terms as Terms

import Hydra.Sources.Libraries
  ( _eithers_either
  , _pairs_first
  , _pairs_second
  )


tApp :: TTerm a -> TTerm a -> TTerm a
tApp f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))

tLam :: String -> TTerm a -> TTerm a
tLam p body = TTerm (Terms.lambda p (unTTerm body))

tVar :: String -> TTerm a
tVar = TTerm . Terms.var

tPair :: TTerm a -> TTerm a -> TTerm a
tPair a b = TTerm (Terms.pair (unTTerm a) (unTTerm b))

tFst :: TTerm a -> TTerm a
tFst x = TTerm (Terms.apply (TermVariable _pairs_first) (unTTerm x))

tSnd :: TTerm a -> TTerm a
tSnd x = TTerm (Terms.apply (TermVariable _pairs_second) (unTTerm x))

tEither :: TTerm a -> TTerm a -> TTerm a -> TTerm a
tEither f g x = TTerm
  (Terms.apply
    (Terms.apply
      (Terms.apply (TermVariable _eithers_either) (unTTerm f))
      (unTTerm g))
    (unTTerm x))

tLeft :: TTerm a -> TTerm a
tLeft x = TTerm (Terms.left (unTTerm x))

tRight :: TTerm a -> TTerm a
tRight x = TTerm (Terms.right (unTTerm x))

tApply :: TTerm a -> TTerm a -> TTerm a
tApply f x = TTerm (Terms.apply (unTTerm f) (unTTerm x))
