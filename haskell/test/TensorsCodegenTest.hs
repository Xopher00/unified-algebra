{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.List (isInfixOf)

import Hydra.Generation (generateSources)

import Hydra.Kernel
  ( Module(..)
  , Namespace(..)
  )

import Hydra.Languages (hydraLanguage)

import qualified Hydra.Python.Coder as PythonCoder

import UniAlg.Pipeline.Backend
  ( BackendContext(..)
  , backendContextSpec
  , loadBackendContext
  )

import UniAlg.Pipeline.Externals (backendExternalModules)
import UniAlg.Pipeline.Lowering (lowerModule)
import UniAlg.Domain.Tensors

import TestUtils
  ( assertBool
  , backendsDir
  )


generate :: String -> Module -> IO String
generate outputDir targetModule = do
  loaded <- loadBackendContext backendsDir "numpy"
  context <- case loaded of
    Left err -> error ("Could not load backend context: " <> err)
    Right c  -> pure c

  let spec = backendContextSpec context
      externals = backendExternalModules spec
      adapted   = lowerModule spec targetModule
      universe  = externals <> [adapted]

  _ <- generateSources
    PythonCoder.moduleToPython
    hydraLanguage
    True True True True
    outputDir
    universe
    [adapted]

  readFile (outputDir <> "/unialg/tensors.py")


deps :: [Namespace]
deps = [Namespace "numpy", Namespace "numpy.linalg", Namespace "scipy.special"]

mkModule :: String -> Semiring -> Equation -> Orientation -> Module
mkModule defName sr eq orientation =
  case equationModule "unialg.tensors" defName deps sr eq orientation of
    Left err -> error err
    Right m   -> m { moduleDescription = Just "Tensor codegen test" }


main :: IO ()
main = do
  let sr = Semiring "add" "multiply" (Just "divide")

  -- ── Simple contract (arity-inferred reduce) ──────────────
  putStrLn "=== simple contract ==="
  let simpleModule = (contractModule "matmul_simple" sr) { moduleTermDependencies = deps }
  simple <- generate "/tmp/unialg-codegen-simple" simpleModule
  putStrLn simple

  assertBool "simple: imports numpy" ("import numpy" `isInfixOf` simple)
  assertBool "simple: calls numpy.sum" ("numpy.sum" `isInfixOf` simple)
  assertBool "simple: calls numpy.multiply" ("numpy.multiply" `isInfixOf` simple)

  -- ── Forward matmul equation ──────────────────────────────
  let Right matmulEq = parseEquation "ij,jk->ik"

  putStrLn "\n=== forward matmul (ij,jk->ik) ==="
  fwd <- generate "/tmp/unialg-codegen-fwd" (mkModule "matmul_fwd" sr matmulEq Forward)
  putStrLn fwd

  assertBool "fwd: calls numpy.multiply" ("numpy.multiply" `isInfixOf` fwd)
  assertBool "fwd: calls numpy.sum" ("numpy.sum" `isInfixOf` fwd)
  assertBool "fwd: calls numpy.expand_dims" ("numpy.expand_dims" `isInfixOf` fwd)
  assertBool "fwd: calls numpy.transpose" ("numpy.transpose" `isInfixOf` fwd)

  -- ── Adjoint matmul equation ──────────────────────────────
  putStrLn "\n=== adjoint matmul (ij,jk->ik) ==="
  adj <- generate "/tmp/unialg-codegen-adj" (mkModule "matmul_adj" sr matmulEq Adjoint)
  putStrLn adj

  assertBool "adj: calls numpy.divide" ("numpy.divide" `isInfixOf` adj)
  assertBool "adj: calls numpy.prod" ("numpy.prod" `isInfixOf` adj)

  -- ── Fused three-way equation ─────────────────────────────
  let Right inner = parseEquation "ij,jk->ik"
      Right outer_ = parseEquation "ik,kl->il"
      Right fused = fuseEquation outer_ 0 inner

  putStrLn "\n=== fused (ij,jk,kl->il) ==="
  fusedPy <- generate "/tmp/unialg-codegen-fused" (mkModule "chain_fused" sr fused Forward)
  putStrLn fusedPy

  assertBool "fused: calls numpy.multiply" ("numpy.multiply" `isInfixOf` fusedPy)
  assertBool "fused: calls numpy.sum" ("numpy.sum" `isInfixOf` fusedPy)
  assertBool "fused: defines chain_fused" ("chain_fused" `isInfixOf` fusedPy)

  -- ── Tropical semiring ────────────────────────────────────
  let tropical = Semiring "minimum" "add" Nothing
      Right matvecEq = parseEquation "ij,j->i"

  putStrLn "\n=== tropical matvec (ij,j->i) ==="
  trop <- generate "/tmp/unialg-codegen-trop" (mkModule "tropical_matvec" tropical matvecEq Forward)
  putStrLn trop

  assertBool "trop: calls numpy.add" ("numpy.add" `isInfixOf` trop)
  assertBool "trop: calls numpy.min" ("numpy.min" `isInfixOf` trop)
