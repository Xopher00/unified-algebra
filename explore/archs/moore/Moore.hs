{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Moore machine: @F(X) = Output × (Input → X)@.

Coalgebra: @s ↦ (V·s, λinp. W·s + U·inp)@.
Output depends only on state; state transitions are input-driven.
-}
module Moore
  ( MooreF
  , mooreAna
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Grammar (PolyF(..))
import Seed (SeedEntry(..), ArchClass(..), contraction)


real :: Semiring
real = Semiring "add" "multiply" (Just "divide")

lin mat vec = contraction real "ij,j->i" mat vec
linSum a b s inp = add (lin a s) (lin b inp)

type MooreF o i = Product (Const o) (Exp i)


mooreAna :: SeedEntry
mooreAna = SeedEntry "mooreAna" AnaArch (KConst :*: ExpF Hole) $
  anaModule @(MooreF (TTerm Tensor) (TTerm Tensor))
    "seed.moore" "moore_step"
    [Namespace "numpy"] ["v", "w", "u"] $ \[v, w, u] ->
      \s -> ( lin v s
            , linSum w u s
            )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",      mooreAna)
  , ("tensorflow", mooreAna)
  , ("torch",      mooreAna)
  ]
