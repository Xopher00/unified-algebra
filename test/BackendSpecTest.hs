{-# LANGUAGE OverloadedStrings #-}

module Main where

import Hydra.Kernel
  ( Name(..)
  )

import Data.Text (Text)
import qualified Data.Text as T

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Backend

import TestUtils
  ( assertEqual
  , assertBool
  , backendsDir
  )


commonBackendOps :: [Text]
commonBackendOps =
  [ "einsum"
  , "log_softmax"
  , "reduce.logaddexp"
  , "structural.reshape"
  , "structural.squeeze"
  , "structural.stack"
  , "structural.concat"
  , "xor"
  , "sigmoid"
  , "softmax"
  , "erf"
  , "erfc"
  , "gammaln"
  , "digamma"
  ]


backendNames :: [Text]
backendNames =
  [ "numpy"
  , "cupy"
  , "jax"
  , "tensorflow"
  , "torch"
  ]


loadSpecOrFail :: Text -> IO BackendSpec
loadSpecOrFail backendName = do
  loaded <- loadBackendSpec (backendsDir <> "/" <> T.unpack backendName <> ".json")
  case loaded of
    Left err ->
      error ("Could not load backend spec " <> T.unpack backendName <> ": " <> err)

    Right s ->
      pure s


assertCommonBackendCoverage :: IO ()
assertCommonBackendCoverage =
  mapM_ assertBackend backendNames
  where
    assertBackend backendName = do
      spec <- loadSpecOrFail backendName
      mapM_
        (\opKey ->
          assertBool
            ("backend " <> backendName <> " resolves " <> opKey)
            (lookupOpPath opKey spec /= Nothing))
        commonBackendOps


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

  assertCommonBackendCoverage

  tensorflowSpec <- loadSpecOrFail "tensorflow"
  cupySpec <- loadSpecOrFail "cupy"

  assertEqual
    "tensorflow reduction logaddexp uses stable Keras logsumexp"
    (Just "tensorflow.keras.ops.logsumexp")
    (lookupOpPath "reduce.logaddexp" tensorflowSpec)

  assertEqual
    "tensorflow diagonal uses native linalg diag_part"
    (Just "tensorflow.linalg.diag_part")
    (lookupOpPath "structural.take_diagonal" tensorflowSpec)

  assertEqual
    "cupy reduction logaddexp uses stable SciPy-compatible logsumexp"
    (Just "cupyx.scipy.special.logsumexp")
    (lookupOpPath "reduce.logaddexp" cupySpec)

  assertEqual
    "numpy einsum resolves as contraction primitive"
    (Just "numpy.einsum")
    (lookupOpPath "einsum" spec)
