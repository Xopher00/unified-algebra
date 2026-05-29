{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

module Main where

import Data.List (isInfixOf)

import Hydra.Generation (generateSources)
import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , TermDefinition(..)
  )
import Hydra.Languages (hydraLanguage)
import qualified Hydra.Dsl.Types as Types
import qualified Hydra.Dsl.Terms as Terms
import qualified Hydra.Python.Coder as PythonCoder

import Hydra.Phantoms (TTerm(..), unTTerm)

import UniAlg.Pipeline.Backend (backendContextSpec, loadBackendContext)
import UniAlg.Pipeline.Externals (backendExternalModules)
import UniAlg.Pipeline.Lowering (lowerModule)

import Prelude hiding (fst, snd, either, left, right, tanh)
import UniAlg

import TestUtils
  ( assertEqual
  , assertBool
  , backendsDir
  , pythonVenv
  )


-- ── IR structure tests ────────────────────────────────────────────────────────
-- These verify generated static bindings produce the same IR as op key @@ args.

testOp1Structure :: IO ()
testOp1Structure = do
  let x        = TTerm (Terms.var "x") :: TTerm Tensor
      expected = unTTerm (op "tanh" @@ x)
      actual   = unTTerm (tanh x)
  assertEqual "tanh x  ==  op \"tanh\" @@ x" expected actual

testOp2Structure :: IO ()
testOp2Structure = do
  let x        = TTerm (Terms.var "x") :: TTerm Tensor
      y        = TTerm (Terms.var "y") :: TTerm Tensor
      expected = unTTerm (op "multiply" @@ x @@ y)
      actual   = unTTerm (multiply x y)
  assertEqual "multiply x y  ==  op \"multiply\" @@ x @@ y" expected actual

testOp3Structure :: IO ()
testOp3Structure = do
  let x        = TTerm (Terms.var "x") :: TTerm Tensor
      y        = TTerm (Terms.var "y") :: TTerm Tensor
      z        = TTerm (Terms.var "z") :: TTerm Tensor
      expected = unTTerm (op "clip" @@ x @@ y @@ z)
      actual   = unTTerm (clip x y z)
  assertEqual "clip x y z  ==  op \"clip\" @@ x @@ y @@ z" expected actual


-- ── Codegen validation tests ──────────────────────────────────────────────────
-- Verify that op-built expressions lower correctly against the numpy backend.

generate :: String -> String -> Module -> IO String
generate outputDir pyPath targetModule = do
  loaded <- loadBackendContext backendsDir "numpy"
  context <- case loaded of
    Left err -> error ("Could not load backend context: " <> err)
    Right c  -> pure c
  let spec      = backendContextSpec context
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
  readFile (outputDir <> "/" <> pyPath)


mkOpModule :: [Namespace] -> String -> Name -> TTerm a -> Module
mkOpModule deps ns defName body = Module
  { moduleDescription    = Just "op arity codegen test"
  , moduleNamespace      = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions =
      [ DefinitionTerm $ TermDefinition
          { termDefinitionName      = defName
          , termDefinitionTerm      = unTTerm body
          , termDefinitionTypeScheme =
              Just $ Types.poly ["a"] $
                Types.var "a" Types.~> Types.var "a"
          }
      ]
  }


testOp1Codegen :: IO ()
testOp1Codegen = do
  let deps = [Namespace "numpy"]
      body = "x" ~> tanh (varPhantom "x")
      mod_ = mkOpModule deps "test.ops.unary" (Name "test.ops.unary.test_tanh") body
  putStrLn "\n=== op1 codegen: tanh ==="
  py <- generate "/tmp/unialg-op-arity-1" "test/ops/unary.py" mod_
  putStrLn py
  assertBool "op1: numpy.tanh in output" ("numpy.tanh" `isInfixOf` py)
  assertBool "op1: def emitted"          ("def test_tanh" `isInfixOf` py)


testOp2Codegen :: IO ()
testOp2Codegen = do
  -- multiply is a pure numpy binary op
  let body = "x" ~> "y" ~> multiply (varPhantom "x") (varPhantom "y")
      mod_ = Module
        { moduleDescription    = Just "op2 codegen test"
        , moduleNamespace      = Namespace "test.ops.binary"
        , moduleTermDependencies = [Namespace "numpy"]
        , moduleTypeDependencies = []
        , moduleDefinitions =
            [ DefinitionTerm $ TermDefinition
                { termDefinitionName      = Name "test.ops.binary.test_multiply"
                , termDefinitionTerm      = unTTerm body
                , termDefinitionTypeScheme =
                    Just $ Types.poly ["a"] $
                      Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a"
                }
            ]
        }
  putStrLn "\n=== op2 codegen: multiply ==="
  py <- generate "/tmp/unialg-op-arity-2" "test/ops/binary.py" mod_
  putStrLn py
  assertBool "op2: numpy.multiply in output" ("numpy.multiply" `isInfixOf` py)
  assertBool "op2: two parameters" ("def test_multiply" `isInfixOf` py)


testOp3Codegen :: IO ()
testOp3Codegen = do
  -- clip takes three tensor arguments (value, min, max)
  let body = "x" ~> "lo" ~> "hi" ~> clip (varPhantom "x") (varPhantom "lo") (varPhantom "hi")
      mod_ = Module
        { moduleDescription    = Just "op3 codegen test"
        , moduleNamespace      = Namespace "test.ops.ternary"
        , moduleTermDependencies = [Namespace "numpy"]
        , moduleTypeDependencies = []
        , moduleDefinitions =
            [ DefinitionTerm $ TermDefinition
                { termDefinitionName      = Name "test.ops.ternary.test_clip"
                , termDefinitionTerm      = unTTerm body
                , termDefinitionTypeScheme =
                    Just $ Types.poly ["a"] $
                      Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a"
                }
            ]
        }
  putStrLn "\n=== op3 codegen: clip ==="
  py <- generate "/tmp/unialg-op-arity-3" "test/ops/ternary.py" mod_
  putStrLn py
  assertBool "op3: clip in output" ("clip" `isInfixOf` py)
  assertBool "op3: three parameters" ("def test_clip" `isInfixOf` py)


-- ── Main ──────────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  putStrLn "=== op arity: IR structure ==="
  testOp1Structure
  testOp2Structure
  testOp3Structure
  testOp1Codegen
  testOp2Codegen
  testOp3Codegen
