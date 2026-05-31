{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Moore machine: polynomial functor @F(X) = Tensor × (Tensor → X)@.
Uses the 'Exp' functor. Structural test only (2-tuple output).
-}
module Moore
  ( MooreF
  , mooreAna
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Grammar (PolyF(..))
import Seed (SeedEntry(..), ArchClass(..))


type MooreF o i = Product (Const (TTerm o)) (Exp (TTerm i))


mooreAna :: SeedEntry
mooreAna = SeedEntry "mooreAna" AnaArch (KConst :*: ExpF Hole) $
  anaModule @(MooreF Tensor Tensor)
    "seed.moore" "moore_step"
    [Namespace "numpy"] [] $ \[] ->
      \s -> (s, \_ -> s)


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy", mooreAna)
  ]
