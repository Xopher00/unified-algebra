{-# LANGUAGE OverloadedStrings #-}

module Main where

import Hydra.Kernel
  ( Application(..)
  , Definition(..)
  , Module(..)
  , Name(..)
  , Term(..)
  , TermDefinition(..)
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Backend
  ( lowerDefinition
  , lowerDefinitions
  , loadBackendSpec
  , lowerModule
  , lowerName
  , lowerTerm
  )

import TestUtils
  ( assertEqual
  , backendsDir
  )


app2 :: Term -> Term -> Term -> Term
app2 f x y =
  TermApplication $
    Application
      (TermApplication $ Application f x)
      y


mkTermDef :: Term -> TermDefinition
mkTermDef body =
  TermDefinition
    { termDefinitionName = Name "demo.forward"
    , termDefinitionTerm = body
    , termDefinitionTypeScheme = Nothing
    }


main :: IO ()
main = do
  loaded <- loadBackendSpec (backendsDir <> "/numpy.json")

  spec <- case loaded of
    Left err -> error ("Could not load backend spec: " <> err)
    Right s -> pure s

  -- lowerName tests
  assertEqual
    "lower backend matmul name"
    (Name "numpy.matmul")
    (lowerName spec (Name "unialg.backend.matmul"))

  assertEqual
    "leave ordinary name unchanged"
    (Name "my.model.forward")
    (lowerName spec (Name "my.model.forward"))

  assertEqual
    "leave unknown backend name unchanged"
    (Name "unialg.backend.notInSpec")
    (lowerName spec (Name "unialg.backend.notInSpec"))

  -- lowerTerm tests
  assertEqual
    "lower backend variable term"
    (Terms.var "numpy.matmul")
    (lowerTerm spec (Terms.var "unialg.backend.matmul"))

  assertEqual
    "leave ordinary variable term unchanged"
    (Terms.var "my.model.forward")
    (lowerTerm spec (Terms.var "my.model.forward"))

  let symbolicMatmul =
        Terms.var "unialg.backend.matmul" `Terms.apply`
          Terms.var "x" `Terms.apply`
          Terms.var "w"

      loweredMatmul =
        Terms.var "numpy.matmul" `Terms.apply`
          Terms.var "x" `Terms.apply`
          Terms.var "w"

  assertEqual
    "lower backend op inside applied term"
    loweredMatmul
    (lowerTerm spec symbolicMatmul)

  -- lowerDefinition tests
  let symbolicBody =
        app2
          (TermVariable $ Name "unialg.backend.matmul")
          (TermVariable $ Name "x")
          (TermVariable $ Name "w")

      loweredBody =
        app2
          (TermVariable $ Name "numpy.matmul")
          (TermVariable $ Name "x")
          (TermVariable $ Name "w")

      symbolicDef =
        DefinitionTerm (mkTermDef symbolicBody)

      loweredDef =
        DefinitionTerm (mkTermDef loweredBody)

  assertEqual
    "lower backend op inside term definition"
    loweredDef
    (lowerDefinition spec symbolicDef)

  assertEqual
    "lower backend op inside definition list"
    [loweredDef]
    (lowerDefinitions spec [symbolicDef])

  -- lowerModule tests
  let symbolicModule =
        Module
          { moduleDescription = Just "demo module"
          , moduleNamespace = error "namespace should not be forced in this test"
          , moduleTermDependencies = []
          , moduleTypeDependencies = []
          , moduleDefinitions = [symbolicDef]
          }

      loweredModule =
        lowerModule spec symbolicModule

  assertEqual
    "lower backend op inside module definitions"
    [loweredDef]
    (moduleDefinitions loweredModule)
