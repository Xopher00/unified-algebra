{-# LANGUAGE OverloadedStrings #-}

module TestUtils
  ( assertEqual
  , assertBool
  , backendsDir
  , pythonVenv
  , loadNumpyContext
  , generateFor
  ) where

import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel (Module)

import UniAlg.Backend
  ( BackendContext
  , backendContextSpec
  , loadBackendContext
  )

import UniAlg.Codegen
  ( generatePythonString
  )

import UniAlg.Backend
  ( backendExternalModules
  )

import UniAlg.Backend
  ( lowerModule
  )


-- | Test assertion for equality
assertEqual :: (Show a, Eq a) => Text -> a -> a -> IO ()
assertEqual label expected actual =
  if expected == actual
    then putStrLn ("PASS: " <> T.unpack label)
    else error $
      "FAIL: " <> T.unpack label
        <> "\n  expected: " <> show expected
        <> "\n  actual:   " <> show actual


-- | Test assertion for boolean condition
assertBool :: Text -> Bool -> IO ()
assertBool label condition =
  if condition
    then putStrLn ("PASS: " <> T.unpack label)
    else error ("FAIL: " <> T.unpack label)


-- | Backends directory path
backendsDir :: FilePath
backendsDir =
  "backends"


-- | Python venv path
pythonVenv :: FilePath
pythonVenv =
  ".venv/bin/python3"


-- | Load the numpy backend context, error on failure
loadNumpyContext :: IO BackendContext
loadNumpyContext = do
  loaded <- loadBackendContext backendsDir "numpy"
  case loaded of
    Left err -> error ("Could not load backend context: " <> err)
    Right c  -> pure c


-- | Generate Python code from a module (in-memory, no disk writes)
generateFor :: Module -> IO String
generateFor targetModule = do
  context <- loadNumpyContext
  let spec      = backendContextSpec context
      externals = backendExternalModules spec
      adapted   = lowerModule spec targetModule
      universe  = externals <> [adapted]
  pairs <- generatePythonString universe adapted
  case pairs of
    (_, src) : _ -> pure src
    []           -> error "generatePythonString returned no files"
