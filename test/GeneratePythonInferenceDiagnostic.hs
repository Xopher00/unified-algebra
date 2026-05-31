{-# LANGUAGE OverloadedStrings #-}

module Main where
import qualified Hydra.Dsl.Types as Types
import Data.List
  ( isInfixOf
  )
import qualified Data.Map.Strict as Map

import Hydra.Generation
  ( generateSources
  )

import Hydra.Kernel
  ( Application(..)
  , Definition(..)
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

import UniAlg.Backend
  ( BackendContext(..)
  , OpSpec(..)
  , backendContextSpec
  , loadBackendContext
  , ops
  )

import UniAlg.Backend
  ( backendExternalModules
  )

import UniAlg.Backend
  ( lowerModule
  )

import TestUtils
  ( assertBool
  , backendsDir
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


forwardDef :: Definition
forwardDef =
  DefinitionTerm $
    TermDefinition
      { termDefinitionName = Name "demo.forward"
      , termDefinitionTerm =
          lambda2
            (Name "x")
            (Name "w")
            (app2
              (TermVariable $ Name "unialg.backend.matmul")
              (TermVariable $ Name "x")
              (TermVariable $ Name "w"))
      , termDefinitionTypeScheme =
          Just $
            Types.poly ["a"] $
              Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a"
      }


demoModule :: Module
demoModule =
  Module
    { moduleDescription = Just "Inference diagnostic"
    , moduleNamespace = Namespace "demo"
    , moduleTermDependencies = [Namespace "numpy"]
    , moduleTypeDependencies = []
    , moduleDefinitions = [forwardDef]
    }


generateDemo :: Bool -> FilePath -> BackendContext -> IO String
generateDemo doInfer outputDir context = do
  let spec =
        backendContextSpec context

      adaptedUniverse =
        fmap (lowerModule spec) [demoModule]

      adaptedTargets =
        fmap (lowerModule spec) [demoModule]

      fullUniverse =
        backendExternalModules spec <> adaptedUniverse

  _ <-
    generateSources
      PythonCoder.moduleToPython
      hydraLanguage
      doInfer
      True
      True
      True
      outputDir
      fullUniverse
      adaptedTargets

  readFile (outputDir <> "/demo.py")


main :: IO ()
main = do
  loaded <- loadBackendContext backendsDir "numpy"

  context <- case loaded of
    Left err ->
      error ("Could not load backend context: " <> err)

    Right c ->
      pure c

  let spec =
        backendContextSpec context

      matmulArity =
        arity =<< Map.lookup "matmul" (ops spec)

  assertBool
    "backend spec says matmul has arity 2"
    (matmulArity == Just 2)

  withInfer <-
    generateDemo
      True
      "/tmp/unialg-infer-true"
      context

  withoutInfer <-
    generateDemo
      False
      "/tmp/unialg-infer-false"
      context

  putStrLn "\n--- with doInfer=True ---"
  putStrLn withInfer

  putStrLn "\n--- with doInfer=False ---"
  putStrLn withoutInfer

  assertBool
    "doInfer=True preserves matmul arguments"
    ("numpy.matmul(x, w)" `isInfixOf` withInfer)

  assertBool
    "doInfer=False preserves matmul arguments"
    ("numpy.matmul(x, w)" `isInfixOf` withoutInfer)
