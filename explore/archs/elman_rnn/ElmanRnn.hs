{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

module ElmanRnn
  ( elmanRnn
  , backendSeeds
  ) where

import Prelude hiding (tanh, sigmoid)
import Hydra.Kernel (Module(..))
import UniAlg

import Grammar (PolyF(..))
import Seed (SeedEntry(..), ArchClass(..), contraction)

real :: Semiring
real = Semiring "add" "multiply" (Just "divide")

elmanRnn :: SeedEntry
elmanRnn = SeedEntry "elman_rnn" AnaArch (KConst :*: ExpF Hole) $
  anaModule @(Product (Const (TTerm Tensor)) (Exp (TTerm Tensor)))
    "seed.elman_rnn" "elman_rnn_step"
    [Namespace "numpy"] ["v","bv","w","u","bh"] $ \[v, bv, w, u, bh] ->
      \s ->
        let lin_vs = contraction real "oi,i->o" v s
            out = add lin_vs bv
        in  ( out
        , \inp ->
            let lin_ws = contraction real "hi,i->h" w s
                lin_ux = contraction real "hi,i->h" u inp
                sum1 = add lin_ws lin_ux
                sum2 = add sum1 bh
                h_next = tanh sum2
            in  h_next
        )

backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",       elmanRnn)
  , ("tensorflow",  elmanRnn)
  , ("torch",       elmanRnn)
  ]
