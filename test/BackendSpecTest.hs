{-# LANGUAGE OverloadedStrings #-}

module Main where

import Hydra.Kernel
  ( Name(..)
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Backend

import TestUtils
  ( assertEqual
  , backendsDir
  )


main :: IO ()
main = do
  loaded <- loadBackendSpec (backendsDir <> "/numpy.json")

  spec <- case loaded of
    Left err -> error ("Could not load backend spec: " <> err)
    Right s -> pure s

  loadedContext <- loadBackendContext backendsDir "numpy"

  context <- case loadedContext of
    Left err -> error ("Could not load backend context: " <> err)
    Right c -> pure c

  let expectedMatmulBinding =
        BackendBinding
          { bindingPath = "numpy.matmul"
          }

  assertEqual
    "backend name"
    "numpy"
    (backend spec)

  assertEqual
    "backend context uses selected backend spec"
    spec
    (backendContextSpec context)

  assertEqual
    "lookup matmul by backend op key"
    (Just "numpy.matmul")
    (lookupOpPath "matmul" spec)

  assertEqual
    "lookup multiply by backend op key"
    (Just "numpy.multiply")
    (lookupOpPath "multiply" spec)

  assertEqual
    "do not lookup full Hydra name directly as backend op key"
    Nothing
    (lookupOpPath "unialg.backend.matmul" spec)

  assertEqual
    "missing backend op key"
    Nothing
    (lookupOpPath "unknown" spec)

  assertEqual
    "resolve symbolic BackendOp"
    (Just "numpy.matmul")
    (resolveBackendOp spec (backendOp "unialg.backend.matmul"))

  assertEqual
    "resolve raw Hydra Name"
    (Just "numpy.matmul")
    (resolveName spec (Name "unialg.backend.matmul"))

  assertEqual
    "resolve name through selected backend context"
    (Just expectedMatmulBinding)
    (resolveContextName context (Name "unialg.backend.matmul"))

  assertEqual
    "resolve binding"
    (Just expectedMatmulBinding)
    (resolveBinding spec (Name "unialg.backend.matmul"))

  assertEqual
    "resolve binding from Hydra variable term"
    (Just expectedMatmulBinding)
    (resolveTermBinding spec (Terms.var "unialg.backend.matmul"))

  assertEqual
    "do not resolve non-backend Hydra variable term"
    Nothing
    (resolveTermBinding spec (Terms.var "not.in.backend"))

  assertEqual
    "missing symbolic BackendOp"
    Nothing
    (resolveBackendOp spec (backendOp "unialg.backend.doesNotExist"))
