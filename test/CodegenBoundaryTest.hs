{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Set (Set)
import qualified Data.Set as Set

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

import UniAlg.Backend
  ( backendContextSpec
  , loadBackendContext
  , lowerModule
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
    { moduleDescription = Just "UniAlg codegen boundary test"
    , moduleNamespace = Namespace "demo"
    , moduleTermDependencies = []
    , moduleTypeDependencies = []
    , moduleDefinitions = [forwardDef]
    }


definitionName :: Definition -> Maybe Name
definitionName def =
  case def of
    DefinitionTerm termDef ->
      Just (termDefinitionName termDef)

    DefinitionType _ ->
      Nothing


moduleDefinitionNames :: Module -> Set Name
moduleDefinitionNames modu =
  Set.fromList $
    foldMap (maybeToList . definitionName) (moduleDefinitions modu)


moduleReferencedVariables :: Module -> Set Name
moduleReferencedVariables modu =
  Set.unions $
    fmap definitionReferencedVariables (moduleDefinitions modu)


definitionReferencedVariables :: Definition -> Set Name
definitionReferencedVariables def =
  case def of
    DefinitionTerm termDef ->
      termReferencedVariables (termDefinitionTerm termDef)

    DefinitionType _ ->
      Set.empty


termReferencedVariables :: Term -> Set Name
termReferencedVariables term =
  case term of
    TermVariable name ->
      Set.singleton name

    TermApplication (Application f x) ->
      Set.union
        (termReferencedVariables f)
        (termReferencedVariables x)

    TermLambda (Lambda _ _ body) ->
      termReferencedVariables body

    TermList terms ->
      Set.unions (fmap termReferencedVariables terms)

    TermMaybe maybeTerm ->
      maybe Set.empty termReferencedVariables maybeTerm

    TermPair (left, right) ->
      Set.union
        (termReferencedVariables left)
        (termReferencedVariables right)

    TermEither eitherTerm ->
      case eitherTerm of
        Left left ->
          termReferencedVariables left

        Right right ->
          termReferencedVariables right

    _ ->
      Set.empty


maybeToList :: Maybe a -> [a]
maybeToList maybeValue =
  case maybeValue of
    Nothing ->
      []

    Just value ->
      [value]


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

      adaptedTargets =
        fmap (lowerModule spec) [demoModule]

      adaptedUniverse =
        fmap (lowerModule spec) [demoModule]

      targetRefs =
        Set.unions (fmap moduleReferencedVariables adaptedTargets)

      universeDefs =
        Set.unions (fmap moduleDefinitionNames adaptedUniverse)

  assertBool
    "adapted target references numpy.matmul"
    (Set.member (Name "numpy.matmul") targetRefs)

  assertBool
    "adapted target no longer references unialg.backend.matmul"
    (not $ Set.member (Name "unialg.backend.matmul") targetRefs)

  assertBool
    "adapted universe does not define numpy.matmul"
    (not $ Set.member (Name "numpy.matmul") universeDefs)
