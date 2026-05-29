{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

{-|
Phase-0 seed catalogue.

Tensor operations use 'applyEquation' from "UniAlg.Domain.Tensors" for
contractions (matmul-like operations) and elementwise ops ('add', 'tanh')
for everything else.  Contraction expresses only index summation, never
addition or nonlinearity.
-}
module Explore.Seed
  ( SeedEntry(..)
  , ArchClass(..)
  , seeds
  , seqCata
  , seqCataTanh
  , treeCata
  , streamAna
  , mooreCata
  ) where

import Hydra.Kernel ( Module(..) )

import Prelude hiding (either, tanh)
import UniAlg

import Explore.Archs


data ArchClass
  = CataArch
  | AnaArch
  | HyloArch
  | NoStructure
  deriving (Eq, Show)


data SeedEntry = SeedEntry
  { seedLabel :: String
  , seedClass :: ArchClass
  , seedModule :: Module
  }


seeds :: [SeedEntry]
seeds =
  [ seqCata
  , treeCata
  , streamAna
  , mooreCata
  ]


-- ── Contraction helper ──────────────────────────────────────────────────────

real :: Semiring
real = Semiring "add" "multiply" (Just "divide")

contraction :: String -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
contraction eqStr w x = case applyEquation Forward real eq [w, x] of
  Right t -> t
  Left  e -> error ("contraction " <> eqStr <> " failed: " <> e)
  where Right eq = parseEquation eqStr


-- ── Folding RNN — SeqF cata ─────────────────────────────────────────────────

-- F(X) = 1 + (Tensor × X) = SeqF Tensor
-- General RNN cell: h_t = W_in · x_t + W_rec · h_{t-1} + b
-- W_in · x_t and W_rec · h_{t-1} are contractions ("hi,i->h").
-- Addition and bias are elementwise.
seqCata :: SeedEntry
seqCata = SeedEntry "seqCata" CataArch $
  recModule @(SeqF Tensor)
    "seed.seq" "fold_seq"
    [Namespace "numpy"] ["wIn", "wRec", "b", "s0"] $ \[wIn, wRec, b, s0] ->
      ( id
      , \case InL (Const ())                    -> s0
              InR (Pair (Const a) (Identity s)) ->
                add (add (contraction "hi,i->h" wIn a)
                         (contraction "hj,j->h" wRec s)) b )


-- ── Folding RNN with tanh — for torch.nn.RNN comparison ─────────────────────

-- h_t = tanh(W_in · x_t + W_rec · h_{t-1} + b)
seqCataTanh :: SeedEntry
seqCataTanh = SeedEntry "seqCataTanh" CataArch $
  recModule @(SeqF Tensor)
    "seed.seq_tanh" "fold_seq_tanh"
    [Namespace "numpy"] ["wIn", "wRec", "b", "s0"] $ \[wIn, wRec, b, s0] ->
      ( id
      , \case InL (Const ())                    -> s0
              InR (Pair (Const a) (Identity s)) ->
                tanh (add (add (contraction "hi,i->h" wIn a)
                               (contraction "hj,j->h" wRec s)) b) )


-- ── Recursive NN — RTreeF cata ──────────────────────────────────────────────

-- F(X) = Tensor + (X × X) = RTreeF Tensor
-- Leaf: W · a (contraction). Node: l + r (elementwise).
treeCata :: SeedEntry
treeCata = SeedEntry "treeCata" CataArch $
  recModule @(RTreeF Tensor)
    "seed.tree" "fold_tree"
    [Namespace "numpy"] ["w"] $ \[w] ->
      ( id
      , \case InL (Const a)                        -> contraction "hi,i->h" w a
              InR (Pair (Identity l) (Identity r)) -> add l r )


-- ── Unfolding RNN — StreamF ana ─────────────────────────────────────────────

streamAna :: SeedEntry
streamAna = SeedEntry "streamAna" AnaArch $
  recModule @(StreamF Tensor)
    "seed.stream" "unfold_stream"
    [Namespace "numpy"] [] $ \[] ->
      ( id :: TTerm Tensor -> TTerm Tensor
      , foldToTerm )


-- ── Moore machine — MooreF cata ─────────────────────────────────────────────

mooreCata :: SeedEntry
mooreCata = SeedEntry "mooreCata" AnaArch $
  recModule @(MooreF Tensor Tensor)
    "seed.moore" "moore_step"
    [Namespace "numpy"] [] $ \[] ->
      ( id :: TTerm Tensor -> TTerm Tensor
      , foldToTerm )
