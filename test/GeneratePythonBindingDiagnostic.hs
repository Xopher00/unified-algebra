{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Exception
  ( SomeException
  , try
  )

import Data.List
  ( isInfixOf
  )

import Hydra.Generation
  ( generateSources
  )

import Hydra.Kernel
  ( Application(..)
  , Definition(..)
  , Graph
  , Lambda(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , Term(..)
  , TermDefinition(..)
  )

import Hydra.Languages
  ( hydraLanguage
  )

import qualified Hydra.Python.Coder as PythonCoder

import TestUtils
  ( assertBool
  )


app2 :: Term -> Term -> Term -> Term
app2 f x y =
  TermApplication $
    Application
      (TermApplication $ Application f x)
      y


lambda2 :: Name -> Name -> Term -> Term
lambda2 x y body =
  TermLambda $
    Lambda x Nothing $
      TermLambda $
        Lambda y Nothing body


mkTermDef :: Name -> Term -> Definition
mkTermDef name body =
  DefinitionTerm $
    TermDefinition
      { termDefinitionName = name
      , termDefinitionTerm = body
      , termDefinitionTypeScheme = Nothing
      }


-- Already-adapted target. This intentionally references numpy.matmul directly.
-- That removes UniAlg lowering from the test and isolates Hydra's graph/codegen.

forwardDef :: Definition
forwardDef =
  mkTermDef
    (Name "demo.forward")
    (lambda2
      (Name "x")
      (Name "w")
      (app2
        (TermVariable $ Name "numpy.matmul")
        (TermVariable $ Name "x")
        (TermVariable $ Name "w")))


demoModule :: Module
demoModule =
  Module
    { moduleDescription = Just "Diagnostic target module"
    , moduleNamespace = Namespace "demo"
    , moduleTermDependencies = [Namespace "numpy"]
    , moduleTypeDependencies = []
    , moduleDefinitions = [forwardDef]
    }


-- Minimal candidate universe binding for numpy.matmul.
--
-- This is not meant to be the final external-binding design.
-- It is only a diagnostic control:
--
--   if adding this changes the error, the failure is really about graph binding.
--   if adding this does not change the error, the binding is not being seen.

numpyMatmulDef :: Definition
numpyMatmulDef =
  mkTermDef
    (Name "numpy.matmul")
    (lambda2
      (Name "x")
      (Name "w")
      (TermVariable $ Name "x"))


numpyModule :: Module
numpyModule =
  Module
    { moduleDescription = Just "Diagnostic numpy universe module"
    , moduleNamespace = Namespace "numpy"
    , moduleTermDependencies = []
    , moduleTypeDependencies = []
    , moduleDefinitions = [numpyMatmulDef]
    }


runGeneration :: FilePath -> [Module] -> [Module] -> IO (Either String Int)
runGeneration outputDir universe targets = do
  result <- try $
    generateSources
      PythonCoder.moduleToPython
      hydraLanguage
      True
      True
      True
      True
      outputDir
      universe
      targets

  pure $ case result of
    Left err ->
      Left (show (err :: SomeException))

    Right count ->
      Right count


main :: IO ()
main = do
  missingBinding <-
    runGeneration
      "/tmp/unialg-diag-missing-binding"
      [demoModule]
      [demoModule]

  case missingBinding of
    Left err ->
      assertBool
        "without numpy.matmul in universe, Hydra reports missing binding"
        ("no such binding: numpy.matmul" `isInfixOf` err)

    Right n ->
      error $
        "Expected missing-binding failure, but Hydra generated "
          <> show n
          <> " files"

  withBinding <-
    runGeneration
      "/tmp/unialg-diag-with-binding"
      [numpyModule, demoModule]
      [demoModule]

  case withBinding of
    Left err -> do
      assertBool
        "with numpy.matmul in universe, error changes from missing binding"
        (not ("no such binding: numpy.matmul" `isInfixOf` err))

      putStrLn "INFO: adding a universe binding changed the Hydra error to:"
      putStrLn err

    Right n ->
      assertBool
        "with numpy.matmul in universe, Hydra gets past missing binding"
        (n > 0)