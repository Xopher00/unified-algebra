{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Folding RNN architecture: polynomial functor @F(X) = 1 + (Tensor × X)@.

Two seeds are exported:

  * 'seqCata' — linear activation; tested against @SimpleRNN(activation=\'linear\')@.
  * 'seqCataTanh' — tanh activation; tested against @torch.nn.RNN(nonlinearity=\'tanh\')@.
-}
module Explore.Archs.SeqRnn
  ( SeqF
  , seqCata
  , seqCataTanh
  , backendSeeds
  ) where

import Prelude hiding (either, tanh)
import Hydra.Kernel (Module(..))
import UniAlg

import Explore.Seed (SeedEntry(..), ArchClass(..), contraction)


type SeqF a = Sum (Const ()) (Product (Const (TTerm a)) Identity)

real :: Semiring
real = Semiring "add" "multiply" (Just "divide")


-- | General RNN cell: @h_t = W_in · x_t + W_rec · h_{t-1} + b@
seqCata :: SeedEntry
seqCata = SeedEntry "seqCata" CataArch $
  recModule @(SeqF Tensor)
    "seed.seq" "fold_seq"
    [Namespace "numpy"] ["wIn", "wRec", "b", "s0"] $ \[wIn, wRec, b, s0] ->
      ( id
      , \case InL (Const ())                    -> s0
              InR (Pair (Const a) (Identity s)) ->
                add (add (contraction real "hi,i->h" wIn a)
                         (contraction real "hj,j->h" wRec s)) b )


-- | RNN cell with tanh: @h_t = tanh(W_in · x_t + W_rec · h_{t-1} + b)@
seqCataTanh :: SeedEntry
seqCataTanh = SeedEntry "seqCataTanh" CataArch $
  recModule @(SeqF Tensor)
    "seed.seq_tanh" "fold_seq_tanh"
    [Namespace "numpy"] ["wIn", "wRec", "b", "s0"] $ \[wIn, wRec, b, s0] ->
      ( id
      , \case InL (Const ())                    -> s0
              InR (Pair (Const a) (Identity s)) ->
                tanh (add (add (contraction real "hi,i->h" wIn a)
                               (contraction real "hj,j->h" wRec s)) b) )


-- | numpy and tf use the linear seed; torch uses tanh to match torch.nn.RNN.
backendSeeds :: [(String, SeedEntry)]
backendSeeds =
  [ ("numpy",       seqCata)
  , ("tensorflow",  seqCata)
  , ("torch",       seqCataTanh)
  ]
