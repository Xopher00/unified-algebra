{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Unfolding RNN: polynomial functor @F(X) = Tensor × X@.
Structural test only — trivial coalgebra repeats state as output.
-}
module StreamRnn
  ( StreamF
  , streamAna
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Seed (SeedEntry(..), ArchClass(..))


type StreamF o = Product (Const (TTerm o)) Identity


streamAna :: SeedEntry
streamAna = SeedEntry "streamAna" AnaArch $
  anaModule @(StreamF Tensor)
    "seed.stream" "unfold_stream"
    [Namespace "numpy"] [] $ \[] ->
      \s -> (s, s)


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy", streamAna)
  ]
