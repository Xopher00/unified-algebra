{-# LANGUAGE OverloadedStrings #-}

module UniAlg.Pipeline.Lowering
  ( lowerName
  , lowerTerm
  , lowerDefinition
  , lowerDefinitions
  , lowerModule
  , unialgModuleLowerer
  ) where

import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Name(..)
  , Term(..)
  , TermDefinition(..)
  )

import qualified Hydra.Rewriting as Rewriting

import UniAlg.Pipeline.Backend
  ( BackendContext(..)
  , BackendSpec
  , backendContextSpec
  , resolveName
  )


lowerName :: BackendSpec -> Name -> Name
lowerName spec name =
  case resolveName spec name of
    Nothing ->
      name

    Just resolved ->
      textToName resolved


-- | Rewrite symbolic UniAlg backend primitive names throughout a Hydra term.
--
-- Example:
--
--   unialg.backend.matmul x w
--   -> numpy.matmul x w
--
-- Still Hydra IR — not Python source.

lowerTerm :: BackendSpec -> Term -> Term
lowerTerm spec =
  Rewriting.rewriteTerm rewrite
  where
    rewrite descend term =
      case term of
        TermVariable name ->
          TermVariable (lowerName spec name)

        _ ->
          descend term


lowerDefinition :: BackendSpec -> Definition -> Definition
lowerDefinition spec def =
  case def of
    DefinitionTerm termDef ->
      DefinitionTerm $
        termDef
          { termDefinitionTerm =
              lowerTerm spec (termDefinitionTerm termDef)
          }

    DefinitionType _ ->
      def


lowerDefinitions :: BackendSpec -> [Definition] -> [Definition]
lowerDefinitions spec =
  fmap (lowerDefinition spec)


lowerModule :: BackendSpec -> Module -> Module
lowerModule spec modu =
  modu
    { moduleDefinitions =
        lowerDefinitions spec (moduleDefinitions modu)
    }


-- | Lower UniAlg backend symbols in a module before delegating to Hydra's
-- Python coder. Rewrites Hydra IR only:
--
--   unialg.backend.matmul -> numpy.matmul

unialgModuleLowerer
  :: BackendContext
  -> Module
  -> [Definition]
  -> (Module, [Definition])
unialgModuleLowerer context modu defs =
  let spec = backendContextSpec context
  in ( lowerModule spec modu
     , lowerDefinitions spec defs
     )


textToName :: Text -> Name
textToName =
  Name . T.unpack
