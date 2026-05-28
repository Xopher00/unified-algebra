{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

module Main where

import Data.List (isInfixOf)
import System.Process (callProcess)

import Hydra.Kernel (Module(..), Namespace(..))

import UniAlg.Pipeline.Backend (backendContextSpec, loadBackendContext)
import UniAlg.Pipeline.Externals (backendExternalModules)
import UniAlg.Pipeline.Lowering (lowerModule)

import Prelude hiding (fst, snd, either, left, right)
import UniAlg

import TestUtils
  ( assertBool
  , pythonVenv
  , generateFor
  )


-- ── Shared backend ops ────────────────────────────────────────────────────────

multiply :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor
multiply = op2 "multiply"

add :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor
add = op2 "add"


-- ── Test: ListF catamorphism — sum elements ──────────────────────────────────
-- ListF (TTerm Tensor) x = 1 + (TTerm Tensor × x)
-- Base (Left (Const ())): return initial accumulator s0
-- Cons (Right (Pair (Const a) (Identity acc))): add element a to accumulator

testListCata :: IO ()
testListCata = do
  let ns      = "test_rec.fold"
      defName = "sum_list"

      sumAlg (InL (Const ()))                      = var "s0"
      sumAlg (InR (Pair (Const a) (Identity acc))) = backendOp "add" @@ a @@ acc

      mod_ = recModule ns defName [Namespace "numpy"] ["s0"] $
               cataT @(ListF (TTerm Tensor)) sumAlg

  putStrLn "=== cataT with ListF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "ListF: emitted as a def"         ("def sum_list"   `isInfixOf` py)
  assertBool "ListF: body contains self-call"  ("sum_list("      `isInfixOf` py)
  assertBool "ListF: either elimination present" ("either"       `isInfixOf` py)
  assertBool "ListF: backend op lowered"       ("numpy.add"      `isInfixOf` py)


-- ── Test: RTreeF catamorphism — weighted sum of binary tree leaves ────────────
-- RTreeF (TTerm Tensor) x = TTerm Tensor + (x × x)
-- Leaf (Left (Const a)): multiply weight w by leaf value a
-- Node (Right (Pair (Identity l) (Identity r))): add left and right results

testTreeCata :: IO ()
testTreeCata = do
  let ns      = "test_rec.tree"
      defName = "sum_tree"

      treeAlg (InL (Const a))                        = backendOp "multiply" @@ var "w" @@ a
      treeAlg (InR (Pair (Identity l) (Identity r))) = backendOp "add" @@ l @@ r

      mod_ = recModule ns defName [Namespace "numpy"] ["w"] $
               cataT @(RTreeF (TTerm Tensor)) treeAlg

  putStrLn "\n=== cataT with RTreeF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "RTreeF: emitted as a def"          ("def sum_tree"   `isInfixOf` py)
  assertBool "RTreeF: body contains self-call"   ("sum_tree("      `isInfixOf` py)
  assertBool "RTreeF: either elimination present" ("either"        `isInfixOf` py)
  assertBool "RTreeF: backend ops lowered"
    ("numpy.add" `isInfixOf` py && "numpy.multiply" `isInfixOf` py)


-- ── Test: StreamF anamorphism — geometric stream ─────────────────────────────
-- StreamF (TTerm Tensor) x = TTerm Tensor × x
-- Unfold: emit current state s, recurse on multiply(w, s)
-- No base case — stream continues indefinitely

testStreamAna :: IO ()
testStreamAna = do
  let ns      = "test_rec.stream"
      defName = "geo_stream"

      streamCoalg s = Pair (Const s) (Identity (backendOp "multiply" @@ var "w" @@ s))

      mod_ = recModule ns defName [Namespace "numpy"] ["w"] $
               anaT @(StreamF (TTerm Tensor)) streamCoalg

  putStrLn "\n=== anaT with StreamF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "StreamF: emitted as a def"       ("def geo_stream"  `isInfixOf` py)
  assertBool "StreamF: contains self-call"      ("geo_stream("     `isInfixOf` py)
  assertBool "StreamF: backend op lowered"      ("numpy.multiply"  `isInfixOf` py)


-- ── Test: SeqF hylomorphism — unfold then fold in one pass ───────────────────
-- Coalg: produce w*x at each step, recurse on x
-- Alg: accumulate with add, base case returns s0

testHylo :: IO ()
testHylo = do
  let ns      = "test_rec.hylo"
      defName = "hylo_sum"

      seqCoalg x = InR (Pair (Const (backendOp "multiply" @@ var "w" @@ x)) (Identity x))

      seqAlg (InL (Const ()))                      = var "s0"
      seqAlg (InR (Pair (Const a) (Identity acc))) = backendOp "add" @@ a @@ acc

      mod_ = recModule ns defName [Namespace "numpy"] ["w", "s0"] $
               hyloT @(SeqF (TTerm Tensor)) seqCoalg seqAlg

  putStrLn "\n=== hyloT with SeqF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "hylo_sum: emitted as a def"      ("def hylo_sum"    `isInfixOf` py)
  assertBool "hylo_sum: contains self-call"     ("hylo_sum("       `isInfixOf` py)
  assertBool "hylo_sum: backend ops lowered"
    ("numpy.add" `isInfixOf` py && "numpy.multiply" `isInfixOf` py)


-- ── Folding RNN — cataT ───────────────────────────────────────────────────────
-- F(X) = 1 + (Tensor × X)   =   SeqF Tensor
-- Algebra uses native Haskell pattern matching on SeqF constructors.

testFoldRNN :: IO ()
testFoldRNN = do
  let ns      = "neural.fold_rnn"
      defName = "fold_rnn"

      foldAlg (InL (Const ()))                    = var "s0"
      foldAlg (InR (Pair (Const a) (Identity s))) = add (multiply (var "w") a) s

      mod_ = recModule ns defName [Namespace "numpy"] ["w", "s0"] $
               cataT @(SeqF (TTerm Tensor)) foldAlg

  putStrLn "\n=== FoldRNN cataT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "fold_rnn: emitted as a def"        ("def fold_rnn"   `isInfixOf` py)
  assertBool "fold_rnn: contains self-call"       ("fold_rnn("      `isInfixOf` py)
  assertBool "fold_rnn: either dispatch present"  ("either"         `isInfixOf` py)
  assertBool "fold_rnn: numpy.multiply lowered"   ("numpy.multiply" `isInfixOf` py)
  assertBool "fold_rnn: numpy.add lowered"        ("numpy.add"      `isInfixOf` py)


-- ── Tree RNN — cataT ──────────────────────────────────────────────────────────
-- F(X) = Tensor + (X × X)   =   RTreeF Tensor
-- Leaf (Left a)        → multiply weight by leaf value
-- Node (Right (l, r))  → add left and right subtree results

testTreeRNN :: IO ()
testTreeRNN = do
  let ns      = "neural.tree_rnn"
      defName = "tree_rnn"

      treeAlg (InL (Const a))                        = multiply (var "w") a
      treeAlg (InR (Pair (Identity l) (Identity r))) = add l r

      mod_ = recModule ns defName [Namespace "numpy"] ["w"] $
               cataT @(RTreeF (TTerm Tensor)) treeAlg

  putStrLn "\n=== TreeRNN cataT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "tree_rnn: emitted as a def"        ("def tree_rnn"   `isInfixOf` py)
  assertBool "tree_rnn: contains self-call"       ("tree_rnn("      `isInfixOf` py)
  assertBool "tree_rnn: either dispatch present"  ("either"         `isInfixOf` py)
  assertBool "tree_rnn: numpy.multiply lowered"   ("numpy.multiply" `isInfixOf` py)
  assertBool "tree_rnn: numpy.add lowered"        ("numpy.add"      `isInfixOf` py)


-- ── Stream RNN — anaT ─────────────────────────────────────────────────────────
-- F(X) = Tensor × X   =   StreamF Tensor
-- Unfold hidden state: emit current state s, next state is tanh(W*s)

testStreamRNN :: IO ()
testStreamRNN = do
  let ns      = "neural.stream_rnn"
      defName = "stream_rnn"

      streamCoalg s = Pair (Const s) (Identity (multiply (var "w") s))

      mod_ = recModule ns defName [Namespace "numpy"] ["w"] $
               anaT @(StreamF (TTerm Tensor)) streamCoalg

  putStrLn "\n=== StreamRNN anaT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "stream_rnn: emitted as a def"       ("def stream_rnn"  `isInfixOf` py)
  assertBool "stream_rnn: contains self-call"      ("stream_rnn("     `isInfixOf` py)
  assertBool "stream_rnn: numpy.multiply lowered"  ("numpy.multiply"  `isInfixOf` py)


-- ── Hylo RNN — hyloT ──────────────────────────────────────────────────────────
-- F(X) = 1 + (Tensor × X)   =   SeqF Tensor
-- Refold: unfold input into sequence layers, fold with linear combination

testHyloRNN :: IO ()
testHyloRNN = do
  let ns      = "neural.hylo_rnn"
      defName = "hylo_rnn"

      hyloCoalg x = InR (Pair (Const (multiply (var "w") x)) (Identity x))

      hyloAlg (InL (Const ()))                    = var "s0"
      hyloAlg (InR (Pair (Const a) (Identity s))) = add a s

      mod_ = recModule ns defName [Namespace "numpy"] ["w", "s0"] $
               hyloT @(SeqF (TTerm Tensor)) hyloCoalg hyloAlg

  putStrLn "\n=== HyloRNN hyloT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "hylo_rnn: emitted as a def"        ("def hylo_rnn"    `isInfixOf` py)
  assertBool "hylo_rnn: contains self-call"       ("hylo_rnn("       `isInfixOf` py)
  assertBool "hylo_rnn: numpy.multiply lowered"   ("numpy.multiply"  `isInfixOf` py)
  assertBool "hylo_rnn: numpy.add lowered"        ("numpy.add"       `isInfixOf` py)


-- ── Main ──────────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  testListCata
  testTreeCata
  testStreamAna
  testHylo
  testFoldRNN
  testTreeRNN
  testStreamRNN
  testHyloRNN
  putStrLn "\n=== TF comparison ==="
  let scriptPath = "/home/xopher001/Documents/Research/doctoral_research/src/unialg/test/neural_comparison.py"
  callProcess pythonVenv [scriptPath]
