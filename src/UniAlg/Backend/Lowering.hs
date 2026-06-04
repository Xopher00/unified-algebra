{-|
Backend lowering: rewrite symbolic UniAlg op names to backend-specific paths.

DSL-level morphisms reference operations symbolically, e.g.:

@
unialg.backend.matmul x w
@

Before Hydra generates Python source, these names must be rewritten to the
chosen backend's qualified paths, e.g.:

@
numpy.matmul x w
@

This module performs that rewrite at the Hydra IR level — it operates on
'Term', 'Definition', and 'Module' values, not on Python source text.
The result is still Hydra IR; actual Python emission is handled by
@Hydra.Python.Coder@.

The entry point for most uses is 'lowerModule'.
-}
module UniAlg.Backend.Lowering
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

import UniAlg.Backend.Spec
  ( BackendContext(..)
  , BackendSpec
  , backendContextSpec
  , resolveName
  )


-- | Resolve a single Hydra 'Name'.  Returns the backend path if the name is
-- a known @unialg.backend.*@ symbol; otherwise returns the name unchanged.
lowerName :: BackendSpec -> Name -> Name
lowerName spec name =
  maybe name textToName (resolveName spec name)


-- | Rewrite all symbolic UniAlg backend names throughout a Hydra 'Term'.
--
-- @
-- unialg.backend.matmul x w  →  numpy.matmul x w
-- @
--
-- The result is still Hydra IR, not Python source.  Unknown names are left
-- unchanged so that non-backend references (Hydra stdlib names, user
-- definitions) pass through unmodified.
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


-- | Lower backend names inside a single 'Definition'.  Type definitions pass through unchanged.
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


-- | Lower backend names across a list of 'Definition's.
lowerDefinitions :: BackendSpec -> [Definition] -> [Definition]
lowerDefinitions spec =
  fmap (lowerDefinition spec)


-- | Lower backend names in all definitions of a 'Module'.
--
-- This is the main entry point used by 'writePythonWithBackend' in
-- "UniAlg.Codegen".
lowerModule :: BackendSpec -> Module -> Module
lowerModule spec modu =
  modu
    { moduleDefinitions =
        lowerDefinitions spec (moduleDefinitions modu)
    }


-- | Adapter for Hydra's @generateSources@ callback signature.
--
-- Lowers UniAlg backend symbols in a module and its definitions before
-- delegating to Hydra's Python coder.
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
