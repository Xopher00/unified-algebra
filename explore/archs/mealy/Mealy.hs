{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Mealy machine: @F(X) = (Input → Output) × (Input → X)@.

Functor: @Product (Const (i -> o)) (Exp i)@.
Coalgebra: @s ↦ (λinp. V·s + C·inp, λinp. W·s + U·inp)@.

'Const' wraps the output function: @fmap _ (Const f) = Const f@, so the
output branch never receives a self-call.  Only the 'Exp' branch recurses.
-}
module Mealy
  ( MealyF
  , mealyStep
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

-- | Mealy functor: output function (non-recursive) × transition function (recursive).
type MealyF i o = Product (Const (i -> o)) (Exp i)


mealyStep :: SeedEntry
mealyStep = SeedEntry "mealyStep" AnaArch (ExpF KConst :*: ExpF Hole) $
  anaModule @(MealyF (TTerm Tensor) (TTerm Tensor))
    "seed.mealy" "mealy_step"
    [Namespace "numpy"] ["v", "c", "w", "u"] $ \[v, c, w, u] ->
      \s -> ( linSum v c s
            , linSum w u s
            )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",      mealyStep)
  , ("tensorflow", mealyStep)
  , ("torch",      mealyStep)
  ]
