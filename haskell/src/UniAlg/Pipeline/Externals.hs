{-# LANGUAGE OverloadedStrings #-}

module UniAlg.Pipeline.Externals
  ( backendExternalModules
  ) where

import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as Text

import Hydra.Kernel
  ( Application(..)
  , Definition(..)
  , Lambda(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , Term(..)
  , TermDefinition(..)
  , TypeScheme
  )

import qualified Hydra.Dsl.Types as Types

import UniAlg.Pipeline.Backend
  ( BackendSpec(..)
  , OpSpec(..)
  )


-- | Create universe-only Hydra modules for backend external symbols.
--
-- Example backend paths:
--
--   numpy.matmul
--   numpy.multiply
--
-- become definitions in a universe module:
--
--   module numpy
--     numpy.matmul = \x1 x2 -> numpy.matmul x1 x2
--     numpy.multiply = \x1 x2 -> numpy.multiply x1 x2
--
-- These modules should be added to the Hydra generation universe, but NOT to
-- the modules-to-generate list.

backendExternalModules :: BackendSpec -> [Module]
backendExternalModules spec =
  fmap mkModule groupedByNamespace
  where
    backendOps :: [(Text, OpSpec)]
    backendOps =
      dedupeByPath $
        fmap (\op -> (path op, op)) $
          Map.elems (ops spec)

    groupedByNamespace :: [(Text, [(Text, OpSpec)])]
    groupedByNamespace =
      Map.toList $
        Map.fromListWith (<>)
          [ (pathNamespace p, [(p, op)])
          | (p, op) <- backendOps
          ]

    mkModule :: (Text, [(Text, OpSpec)]) -> Module
    mkModule (ns, entries) =
      Module
        { moduleDescription =
            Just ("External backend bindings for " <> Text.unpack (backend spec))
        , moduleNamespace =
            Namespace (Text.unpack ns)
        , moduleTermDependencies =
            []
        , moduleTypeDependencies =
            []
        , moduleDefinitions =
            fmap externalDefinition entries
        }


externalDefinition :: (Text, OpSpec) -> Definition
externalDefinition (p, op) =
  DefinitionTerm $
    TermDefinition
      { termDefinitionName =
          textName p
      , termDefinitionTerm =
          externalTerm p op
      , termDefinitionTypeScheme =
          Just (externalTypeScheme op)
      }


-- | Build an arity-shaped external term.
--
-- For a binary backend op such as numpy.matmul, this creates:
--
--   \x1 x2 -> numpy.matmul x1 x2
--
-- This is an eta-expanded external declaration that gives Hydra inference and
-- codegen the correct application shape while preserving the external call.

externalTerm :: Text -> OpSpec -> Term
externalTerm p op =
  etaExpandExternal (textName p) (arity op)


etaExpandExternal :: Name -> Maybe Int -> Term
etaExpandExternal name maybeArity =
  case maybeArity of
    Just n | n > 0 ->
      etaExpand n name

    _ ->
      TermVariable name


etaExpand :: Int -> Name -> Term
etaExpand n name =
  foldr
    (\param body ->
      TermLambda $
        Lambda param Nothing body)
    applied
    params
  where
    params :: [Name]
    params =
      fmap
        (\i -> Name ("x" <> show i))
        [1 .. n]

    applied :: Term
    applied =
      foldl
        (\f x ->
          TermApplication $
            Application f (TermVariable x))
        (TermVariable name)
        params


externalTypeScheme :: OpSpec -> TypeScheme
externalTypeScheme op =
  case arity op of
    Just 1 ->
      Types.poly ["a"] $
        Types.var "a" Types.~> Types.var "a"

    Just 2 ->
      Types.poly ["a"] $
        Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a"

    Just 3 ->
      Types.poly ["a"] $
        Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a" Types.~> Types.var "a"

    _ ->
      Types.poly ["a"] $
        Types.var "a"


pathNamespace :: Text -> Text
pathNamespace p =
  case Text.splitOn "." p of
    [] ->
      p

    [_single] ->
      p

    segments ->
      Text.intercalate "." (init segments)


textName :: Text -> Name
textName =
  Name . Text.unpack


dedupeByPath :: [(Text, OpSpec)] -> [(Text, OpSpec)]
dedupeByPath entries =
  Map.toList $
    Map.fromList entries
