{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.List
  ( isInfixOf
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

import UniAlg.Pipeline.Codegen
  ( loadBackendAndWritePython
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


forwardTerm :: Term
forwardTerm =
  lambda2
    (Name "x")
    (Name "w")
    (app2
      (TermVariable $ Name "unialg.backend.matmul")
      (TermVariable $ Name "x")
      (TermVariable $ Name "w"))


forwardDef :: Definition
forwardDef =
  DefinitionTerm $
    TermDefinition
      { termDefinitionName = Name "demo.forward"
      , termDefinitionTerm = forwardTerm
      , termDefinitionTypeScheme = Nothing
      }


demoModule :: Module
demoModule =
  Module
    { moduleDescription = Just "UniAlg Hydra Python generation demo"
    , moduleNamespace = Namespace "demo"
    , moduleTermDependencies =
        [ Namespace "numpy"
        ]
    , moduleTypeDependencies = []
    , moduleDefinitions =
        [ forwardDef
        ]
    }


outputDir :: FilePath
outputDir =
  "/tmp/unialg-hydra-python"


generatedFile :: FilePath
generatedFile =
  outputDir <> "/demo.py"


main :: IO ()
main = do
  filesWritten <-
    loadBackendAndWritePython
      backendsDir
      "numpy"
      outputDir
      [demoModule]
      [demoModule]

  case filesWritten of
    Left err ->
      error ("Hydra Python generation failed: " <> err)

    Right count ->
      assertBool
        "Hydra wrote at least one Python file"
        (count > 0)

  generated <- readFile generatedFile

  assertBool
    "generated Python imports numpy"
    ("import numpy" `isInfixOf` generated)

  assertBool
    "generated Python defines forward"
    ("def forward" `isInfixOf` generated)

  assertBool
    "generated Python preserves backend call arguments"
    ("return numpy.matmul(x, w)" `isInfixOf` generated)

  assertBool
    "generated Python does not drop backend call arguments"
    (not ("return numpy.matmul()" `isInfixOf` generated))