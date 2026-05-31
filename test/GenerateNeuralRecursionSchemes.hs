{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module NeuralRecursionSchemes where

import Prelude
import qualified Prelude as P

import UniAlg hiding ((++))
import SeqRnn (SeqF)
import StreamRnn (StreamF)
import TreeRnn (RTreeF)


-- ── Folding RNN ───────────────────────────────────────────────────────────────
-- F(X) = 1 + (a × X)   =   SeqF a

type Seq a = Fix (SeqF a)

nil :: Seq a
nil = Fix (InL (Const ()))

cons :: TTerm a -> Seq a -> Seq a
cons a as = Fix (InR (Pair (Const a) (Identity as)))


foldRNN cell p = cata $ \case
  InL (Const ())                    -> cell p (Left ())
  InR (Pair (Const a) (Identity s)) -> cell p (Right (a, s))


linearCell
  :: (TTerm Tensor, TTerm Tensor)
  -> Either () (TTerm Tensor, TTerm Tensor)
  -> TTerm Tensor
linearCell (_w, s0) (Left ())      = s0
linearCell (w,  _)  (Right (a, s)) = add (multiply w a) s


foldRNNTerm :: TTerm (Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor)
foldRNNTerm =
  "w" ~> "s0" ~> "a1" ~> "a2" ~> "a3" ~>
    foldRNN
      linearCell
      (varPhantom "w", varPhantom "s0")
      (cons (varPhantom "a1") (cons (varPhantom "a2") (cons (varPhantom "a3") nil)))


-- ── Recursive NN ──────────────────────────────────────────────────────────────
-- F(X) = a + (X × X)   =   RTreeF a

type RTree a = Fix (RTreeF a)

rleaf :: TTerm a -> RTree a
rleaf a = Fix (InL (Const a))

rnode :: RTree a -> RTree a -> RTree a
rnode l r = Fix (InR (Pair (Identity l) (Identity r)))


foldTreeRNN cell p = cata $ \case
  InL (Const a)                        -> cell p (Left a)
  InR (Pair (Identity l) (Identity r)) -> cell p (Right (l, r))


treeCell
  :: TTerm Tensor
  -> Either (TTerm Tensor) (TTerm Tensor, TTerm Tensor)
  -> TTerm Tensor
treeCell w  (Left a)      = multiply w a
treeCell _  (Right (l, r)) = add l r


treeRNNTerm :: TTerm (Tensor -> Tensor -> Tensor -> Tensor -> Tensor)
treeRNNTerm =
  "w" ~> "a1" ~> "a2" ~> "a3" ~>
    foldTreeRNN
      treeCell
      (varPhantom "w")
      (rnode
        (rnode (rleaf (varPhantom "a1")) (rleaf (varPhantom "a2")))
        (rleaf (varPhantom "a3")))


-- ── Unfolding RNN / Stream ────────────────────────────────────────────────────
-- F(X) = o × X   =   StreamF o

type Stream' o = Fix (StreamF o)

unfoldRNN step p = ana $ \s ->
  let (o, s') = step p s
  in Pair (Const o) (Identity s')

takeS 0 _ = []
takeS n (Fix (Pair (Const o) (Identity rest))) = o : takeS (n P.- 1) rest


geoStep :: TTerm Tensor -> TTerm Tensor -> (TTerm Tensor, TTerm Tensor)
geoStep w s = (s, multiply w s)

geoRNNTerm :: TTerm (Tensor -> Tensor -> Tensor)
geoRNNTerm =
  "w" ~> "s0" ~>
    P.last (takeS 6 (unfoldRNN geoStep (varPhantom "w") (varPhantom "s0")))


-- ── Entry point ───────────────────────────────────────────────────────────────

main :: IO ()
main = do
  generatePython
    "/tmp/unialg-rnn-fold"
    "backends/numpy.json"
    "fold_rnn"
    [ ("foldRNN", foldRNNTerm) ]

  generatePython
    "/tmp/unialg-rnn-tree"
    "backends/numpy.json"
    "tree_rnn"
    [ ("treeRNN", treeRNNTerm) ]

  generatePython
    "/tmp/unialg-rnn-unfold"
    "backends/numpy.json"
    "geo_rnn"
    [ ("geoRNN", geoRNNTerm) ]
