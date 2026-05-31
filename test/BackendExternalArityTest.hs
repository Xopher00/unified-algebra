{-# LANGUAGE OverloadedStrings #-}

module Main where

import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Name(..)
  , TermDefinition(..)
  )

import UniAlg.Backend
  ( backendExternalModules
  )

import UniAlg.Backend
  ( loadBackendSpec
  )

import TestUtils
  ( assertBool
  , backendsDir
  )


findTermDefinition :: Name -> [Module] -> Maybe TermDefinition
findTermDefinition wanted modules =
  firstJust $
    concatMap
      (\m -> fmap definitionTermMaybe (moduleDefinitions m))
      modules
  where
    definitionTermMaybe def =
      case def of
        DefinitionTerm td
          | termDefinitionName td == wanted ->
              Just td

        _ ->
          Nothing


firstJust :: [Maybe a] -> Maybe a
firstJust values =
  case values of
    [] ->
      Nothing

    Nothing : rest ->
      firstJust rest

    Just x : _ ->
      Just x


main :: IO ()
main = do
  loaded <- loadBackendSpec (backendsDir <> "/numpy.json")

  spec <- case loaded of
    Left err ->
      error ("Could not load backend spec: " <> err)

    Right s ->
      pure s

  let externalModules =
        backendExternalModules spec

      maybeMatmul =
        findTermDefinition (Name "numpy.matmul") externalModules

  assertBool
    "backend externals define numpy.matmul"
    (case maybeMatmul of
      Nothing -> False
      Just _ -> True)

  assertBool
    "numpy.matmul external has a type scheme"
    (case maybeMatmul of
      Nothing ->
        False

      Just td ->
        case termDefinitionTypeScheme td of
          Nothing ->
            False

          Just _ ->
            True)
