{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.List (isInfixOf)
import qualified Data.Text as T
import System.Directory (createDirectoryIfMissing)
import System.Environment (setEnv, lookupEnv)
import System.FilePath (takeDirectory, (</>))
import System.Process (callProcess)

import Hydra.Kernel (Module(..))

import UniAlg.Pipeline.Backend (backendContextSpec, loadBackendContext)
import UniAlg.Pipeline.Codegen (generatePythonString, evalPython)
import UniAlg.Pipeline.Externals (backendExternalModules)
import UniAlg.Pipeline.Lowering (lowerModule)

import Explore.Laws
import Explore.Seed

import TestUtils
  ( assertBool
  , backendsDir
  , generateFor
  , loadNumpyContext
  , pythonVenv
  )


-- ── Per-backend module generation ────────────────────────────────────────────

writeModuleForBackend :: String -> FilePath -> Module -> IO ()
writeModuleForBackend backend outDir mod_ = do
  loaded <- loadBackendContext backendsDir (T.pack backend)
  context <- case loaded of
    Left err -> error ("Could not load backend " <> backend <> ": " <> err)
    Right c  -> pure c
  let spec      = backendContextSpec context
      externals = backendExternalModules spec
      adapted   = lowerModule spec mod_
      universe  = externals <> [adapted]
  pairs <- generatePythonString universe adapted
  mapM_ (writeOne outDir) pairs
  where
    writeOne dir (relPath, src) = do
      let absPath = dir </> relPath
      createDirectoryIfMissing True (takeDirectory absPath)
      writeFile absPath src


-- ── Smoke-check the full seed set (numpy) ────────────────────────────────────

testSeeds :: IO ()
testSeeds = do
  putStrLn "\n=== Full seed set smoke check ==="
  mapM_ checkSeed seeds
  where
    checkSeed entry = do
      let label = T.pack (seedLabel entry)
      py <- generateFor (seedModule entry)
      assertBool (label <> ": non-empty Python emitted") (not (null py))
      assertBool (label <> ": contains def") ("def " `isInfixOf` py)


-- ── Stage 1 gate: Moore lowers ───────────────────────────────────────────────

testMoore :: IO ()
testMoore = do
  putStrLn "\n=== Moore machine (MooreF Tensor Tensor) ==="
  let entry = mooreCata
  py <- generateFor (seedModule entry)
  putStrLn py
  assertBool "moore: emitted as a def"        ("def moore_step"   `isInfixOf` py)
  assertBool "moore: body contains self-call" ("moore_step("      `isInfixOf` py)
  assertBool "moore: lambda emitted"          ("lambda"           `isInfixOf` py)
  result <- evalPython pythonVenv py "type(moore_step).__name__"
  case result of
    Left err -> error ("eval failed: " <> err)
    Right t  -> do
      putStrLn ("eval result: " <> t)
      assertBool "moore: evalPython succeeds" (not (null t))


-- ── Arm A: symbolic law checks ───────────────────────────────────────────────

testLaws :: IO ()
testLaws = do
  putStrLn "\n=== Arm A: symbolic law checks ==="
  mapM_ runCheck checkGrammarLaws
  mapM_ runCheck checkClassificationLaws
  mapM_ runCheck checkSeedMapping
  where
    runCheck (label, passed) =
      assertBool (T.pack label) passed


-- ── Arm B: per-library differential tests ────────────────────────────────────

testHarness :: IO ()
testHarness = do
  let tfDir    = "explore/generated/tf"
      torchDir = "explore/generated/torch"

  putStrLn "\n=== Arm B: generating per-backend modules ==="
  mapM_ (\e -> writeModuleForBackend "tensorflow" tfDir (seedModule e)) seeds
  writeModuleForBackend "torch" torchDir (seedModule seqCataTanh)
  mapM_ (\e -> writeModuleForBackend "torch" torchDir (seedModule e))
    [treeCata, streamAna, mooreCata]

  putStrLn "=== Formatting generated code ==="
  callProcess pythonVenv ["-m", "black", "--quiet", "explore/generated"]

  existing <- lookupEnv "PYTHONPATH"
  let newPath = tfDir <> ":" <> torchDir <> case existing of
        Just p  -> ":" <> p
        Nothing -> ""
  setEnv "PYTHONPATH" newPath

  putStrLn "=== Arm B: running differential harness ==="
  callProcess pythonVenv
    ["-m", "pytest", "explore/harness.py", "-v", "--tb=short"]


main :: IO ()
main = do
  testSeeds
  testMoore
  testLaws
  testHarness
  putStrLn "\n=== All explore gates: PASSED ==="
