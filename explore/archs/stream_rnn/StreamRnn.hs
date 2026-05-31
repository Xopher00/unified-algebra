{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Unfolding RNN — lazy stream anamorphism.

Functor: @StreamLazyF(X) = Tensor × (() → X)@.  The continuation is wrapped
in a lambda so Python does not immediately recurse.

'streamLinear' uses 'StreamLazyF' with coalgebra @s ↦ (W·s, λ_. W·s)@,
producing a lazy stream of successive matrix-vector products.  N steps of
this stream equal @[W·s, W²·s, …, Wⁿ·s]@, which is the differential
reference used in the test harness.
-}
module StreamRnn
  ( StreamLazyF
  , streamLinear
  , backendSeeds
  ) where

import Hydra.Kernel (Module(..))
import UniAlg

import Grammar (PolyF(..))
import Seed (SeedEntry(..), ArchClass(..), contraction)


real :: Semiring
real = Semiring "add" "multiply" (Just "divide")

-- | Lazy stream functor — continuation is a thunk @() → X@.
type StreamLazyF o = Product (Const (TTerm o)) (Exp (TTerm ()))


-- | Linear streaming RNN: @unfold_stream_linear w s = (W·s, λ_. W·s)@.
streamLinear :: SeedEntry
streamLinear = SeedEntry "streamLinear" AnaArch (KConst :*: ExpF Hole) $
  anaModule @(StreamLazyF Tensor)
    "seed.stream_linear" "unfold_stream_linear"
    [Namespace "numpy"] ["w"] $ \[w] ->
      \s -> ( contraction real "hi,i->h" w s
            , \_ -> contraction real "hi,i->h" w s
            )


backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",      streamLinear)
  , ("tensorflow", streamLinear)
  , ("torch",      streamLinear)
  ]
