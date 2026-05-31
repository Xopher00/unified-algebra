{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Unfolding RNN: polynomial functor @F(X) = Tensor × X@.
Identity corecursion via 'foldToTerm'. Structural test only.
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
  recModule @(StreamF Tensor)
    "seed.stream" "unfold_stream"
    [Namespace "numpy"] [] $ \[] ->
      ( id :: TTerm Tensor -> TTerm Tensor
      , foldToTerm )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy", streamAna)
  ]
