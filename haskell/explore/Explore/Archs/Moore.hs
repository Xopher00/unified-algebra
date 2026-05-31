{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Moore machine: polynomial functor @F(X) = Tensor × (Tensor → X)@.
Uses the 'Exp' functor. Structural test only (2-tuple output).
-}
module Explore.Archs.Moore
  ( MooreF
  , mooreCata
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Explore.Seed (SeedEntry(..), ArchClass(..))


type MooreF o i = Product (Const (TTerm o)) (Exp (TTerm i))


mooreCata :: SeedEntry
mooreCata = SeedEntry "mooreCata" AnaArch $
  recModule @(MooreF Tensor Tensor)
    "seed.moore" "moore_step"
    [Namespace "numpy"] [] $ \[] ->
      ( id :: TTerm Tensor -> TTerm Tensor
      , foldToTerm )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy", mooreCata)
  ]
