{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Recursive NN architecture: polynomial functor @F(X) = Tensor + (X × X)@.

Leaf = @W · a@ (contraction), node = @left + right@ (elementwise).
No library-native counterpart — structural test only.
-}
module TreeRnn
  ( RTreeF
  , treeCata
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Seed (SeedEntry(..), ArchClass(..), contraction)


type RTreeF a = Sum (Const (TTerm a)) (Product Identity Identity)

real :: Semiring
real = Semiring "add" "multiply" (Just "divide")


treeCata :: SeedEntry
treeCata = SeedEntry "treeCata" CataArch $
  cataModule @(RTreeF Tensor)
    "seed.tree" "fold_tree"
    [Namespace "numpy"] ["w"] $ \[w] ->
      ( \a   -> contraction real "hi,i->h" w a
      , \l r -> add l r
      )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",       treeCata)
  , ("tensorflow",  treeCata)
  , ("torch",       treeCata)
  ]
