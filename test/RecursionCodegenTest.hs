{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

module Main where

import Data.List (isInfixOf)
import System.Directory (createDirectoryIfMissing)
import System.FilePath (takeDirectory, (</>))
import System.Process (callProcess)

import Hydra.Kernel (Module(..), Namespace(..))

import UniAlg.Backend (backendContextSpec, backendExternalModules, loadBackendContext, lowerModule)
import UniAlg.Codegen (generatePythonString)

import Prelude hiding (fst, snd, either, left, right)
import UniAlg
import SeqRnn (SeqF)
import TreeRnn (RTreeF)

import TestUtils
  ( assertBool
  , pythonVenv
  , generateFor
  , loadNumpyContext
  )



-- ── Disk writer (uses generatePythonString to avoid Hydra's occurs-check) ────

writeModuleToDisk :: FilePath -> Module -> IO ()
writeModuleToDisk outputDir mod_ = do
  context <- loadNumpyContext
  let spec      = backendContextSpec context
      externals = backendExternalModules spec
      adapted   = lowerModule spec mod_
      universe  = externals <> [adapted]
  pairs <- generatePythonString universe adapted
  mapM_ writeOne pairs
  where
    writeOne (relPath, src) = do
      let absPath = outputDir </> relPath
      createDirectoryIfMissing True (takeDirectory absPath)
      writeFile absPath src


-- ── Test: ListF catamorphism — sum elements ──────────────────────────────────
-- ListF Tensor x = 1 + (TTerm Tensor × x)
-- Base (Left (Const ())): return initial accumulator s0
-- Cons (Right (Pair (Const a) (Identity acc))): add element a to accumulator

testListCata :: IO ()
testListCata = do
  let ns      = "test_rec.fold"
      defName = "sum_list"

      mod_ = cataModule @(ListF Tensor) ns defName [Namespace "numpy"] ["s0"] $ \[s0] ->
               ( s0
               , \a acc -> add a acc
               )

  putStrLn "=== cataT with ListF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "ListF: emitted as a def"         ("def sum_list"   `isInfixOf` py)
  assertBool "ListF: body contains self-call"  ("sum_list("      `isInfixOf` py)
  assertBool "ListF: either elimination present" ("either"       `isInfixOf` py)
  assertBool "ListF: backend op lowered"       ("numpy.add"      `isInfixOf` py)


-- ── Test: RTreeF catamorphism — weighted sum of binary tree leaves ────────────
-- RTreeF Tensor x = TTerm Tensor + (x × x)
-- Leaf (Left (Const a)): multiply weight w by leaf value a
-- Node (Right (Pair (Identity l) (Identity r))): add left and right results

testTreeCata :: IO ()
testTreeCata = do
  let ns      = "test_rec.tree"
      defName = "sum_tree"

      mod_ = cataModule @(RTreeF Tensor) ns defName [Namespace "numpy"] ["w"] $ \[w] ->
               ( \a -> multiply w a
               , \l r -> add l r
               )

  putStrLn "\n=== cataT with RTreeF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "RTreeF: emitted as a def"          ("def sum_tree"   `isInfixOf` py)
  assertBool "RTreeF: body contains self-call"   ("sum_tree("      `isInfixOf` py)
  assertBool "RTreeF: either elimination present" ("either"        `isInfixOf` py)
  assertBool "RTreeF: backend ops lowered"
    ("numpy.add" `isInfixOf` py && "numpy.multiply" `isInfixOf` py)


-- ── Test: anaT — runtime dispatch via hyloT/buildLayer ─────────────────────
-- anaT coalg = hyloT @f coalg buildLayer: dispatches at runtime via
-- matchLayer/eithers.either, exactly as cataT does on its input.
-- coalg emits a cons-like SeqF layer; buildLayer reassembles each layer
-- with a self-call in the Identity position.

testAnaT :: IO ()
testAnaT = do
  let ns      = "test_rec.ana"
      defName = "copy_list"

      mod_ = anaModule @(SeqF Tensor) ns defName [Namespace "numpy"] [] $ \[] ->
               \s -> Right (s, s)

  putStrLn "\n=== anaT with SeqF (natural coalg, alg=buildLayer) ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "anaT SeqF: emitted as a def"        ("def copy_list" `isInfixOf` py)
  assertBool "anaT SeqF: body contains self-call" ("copy_list("    `isInfixOf` py)
  assertBool "anaT SeqF: either dispatch present" ("either"        `isInfixOf` py)


-- ── Test: SeqF hylomorphism — unfold then fold in one pass ───────────────────
-- Coalg: produce w*x at each step, recurse on x
-- Alg: accumulate with add, base case returns s0

testHylo :: IO ()
testHylo = do
  let ns      = "test_rec.hylo"
      defName = "hylo_sum"

      mod_ = cataModule @(SeqF Tensor) ns defName [Namespace "numpy"] ["s0"] $ \[s0] ->
               ( s0
               , \a acc -> add a acc
               )

  putStrLn "\n=== cataT with SeqF ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "hylo_sum: emitted as a def"       ("def hylo_sum"  `isInfixOf` py)
  assertBool "hylo_sum: contains self-call"      ("hylo_sum("     `isInfixOf` py)
  assertBool "hylo_sum: either dispatch present" ("either"        `isInfixOf` py)
  assertBool "hylo_sum: numpy.add lowered"       ("numpy.add"     `isInfixOf` py)


-- ── Folding RNN — cataT ───────────────────────────────────────────────────────
-- F(X) = 1 + (Tensor × X)   =   SeqF Tensor
-- Algebra uses native Haskell pattern matching on SeqF constructors.

testFoldRNN :: IO ()
testFoldRNN = do
  let ns      = "neural.fold_rnn"
      defName = "fold_rnn"

      mod_ = cataModule @(SeqF Tensor) ns defName [Namespace "numpy"] ["w", "s0"] $ \[w, s0] ->
               ( s0
               , \a s -> add (multiply w a) s
               )

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

      mod_ = cataModule @(RTreeF Tensor) ns defName [Namespace "numpy"] ["w"] $ \[w] ->
               ( \a -> multiply w a
               , \l r -> add l r
               )

  putStrLn "\n=== TreeRNN cataT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "tree_rnn: emitted as a def"        ("def tree_rnn"   `isInfixOf` py)
  assertBool "tree_rnn: contains self-call"       ("tree_rnn("      `isInfixOf` py)
  assertBool "tree_rnn: either dispatch present"  ("either"         `isInfixOf` py)
  assertBool "tree_rnn: numpy.multiply lowered"   ("numpy.multiply" `isInfixOf` py)
  assertBool "tree_rnn: numpy.add lowered"        ("numpy.add"      `isInfixOf` py)




-- ── Hylo RNN — hyloT ──────────────────────────────────────────────────────────
-- F(X) = 1 + (Tensor × X)   =   SeqF Tensor
-- Refold: unfold input into sequence layers, fold with linear combination

testHyloRNN :: IO ()
testHyloRNN = do
  let ns      = "neural.hylo_rnn"
      defName = "hylo_rnn"

      mod_ = cataModule @(SeqF Tensor) ns defName [Namespace "numpy"] ["s0"] $ \[s0] ->
               ( s0
               , \a s -> add a s
               )

  putStrLn "\n=== HyloRNN cataT ==="
  py <- generateFor mod_
  putStrLn py

  assertBool "hylo_rnn: emitted as a def"        ("def hylo_rnn"    `isInfixOf` py)
  assertBool "hylo_rnn: contains self-call"       ("hylo_rnn("       `isInfixOf` py)
  assertBool "hylo_rnn: either dispatch present"  ("either"          `isInfixOf` py)
  assertBool "hylo_rnn: numpy.add lowered"        ("numpy.add"       `isInfixOf` py)


-- ── Main ──────────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  testListCata
  testTreeCata
  testAnaT
  testHylo
  testFoldRNN
  testTreeRNN
  testHyloRNN
  putStrLn "\n=== TF comparison ==="
  let foldMod = cataModule @(SeqF Tensor)
                  "neural.fold_rnn" "fold_rnn" [Namespace "numpy"] ["w", "s0"] $ \[w, s0] ->
                  ( s0
                  , \a s -> add (multiply w a) s
                  )
      treeMod = cataModule @(RTreeF Tensor)
                  "neural.tree_rnn" "tree_rnn" [Namespace "numpy"] ["w"] $ \[w] ->
                  ( \a -> multiply w a
                  , \l r -> add l r
                  )
  writeModuleToDisk "/tmp/unialg-neural-fold" foldMod
  writeModuleToDisk "/tmp/unialg-neural-tree" treeMod
  let scriptPath = "test/neural_comparison.py"
  callProcess pythonVenv [scriptPath]
