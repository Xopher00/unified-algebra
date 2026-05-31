{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Recursive NN architecture: polynomial functor @F(X) = Tensor + (X × X)@.

Leaf = @W · a@ (contraction), node = @left + right@ (elementwise).
No library-native counterpart — structural test only.
-}
module Explore.Archs.TreeRnn
  ( RTreeF
  , treeCata
  , backendSeeds
  ) where

import Prelude hiding (either)
import Hydra.Kernel (Module(..))
import UniAlg

import Explore.Seed (SeedEntry(..), ArchClass(..), contraction)


type RTreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

real :: Semiring
real = Semiring "add" "multiply" (Just "divide")


treeCata :: SeedEntry
treeCata = SeedEntry "treeCata" CataArch $
  recModule @(RTreeF Tensor)
    "seed.tree" "fold_tree"
    [Namespace "numpy"] ["w"] $ \[w] ->
      ( id
      , \case InL (Const a)                        -> contraction real "hi,i->h" w a
              InR (Pair (Identity l) (Identity r)) -> add l r )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",       treeCata)
  , ("tensorflow",  treeCata)
  , ("torch",       treeCata)
  ]
