{-# LANGUAGE OverloadedStrings #-}

{-|
Backend specification loading and symbolic op resolution.

A /backend/ is a JSON file (e.g. @numpy.json@, @jax.json@) that maps
logical UniAlg op keys to backend-specific qualified paths:

@
{ "backend": "numpy",
  "ops": {
    "matmul":       { "path": "numpy.matmul",       "arity": 2 },
    "reduce.add":   { "path": "numpy.sum",           "arity": 2 },
    "structural.expand_dims": { "path": "numpy.expand_dims", "arity": 2 }
  }
}
@

At DSL-definition time, morphisms reference ops symbolically as
@unialg.backend.matmul@.  The lowering pass (see "UniAlg.Backend")
rewrites these names to their backend-specific equivalents (@numpy.matmul@)
before Hydra generates Python source.

'BackendOp' and 'call' are the construction side; the @resolve*@ and @lookup*@
functions are the query side used by lowering.
-}
module UniAlg.Backend.Spec
  ( BackendOp
  , backendOp
  , backendOpName
  , call
  , BackendSpec(..)
  , OpSpec(..)
  , BackendBinding(..)
  , BackendContext(..)
  , loadBackendSpec
  , loadBackendContext
  , lookupOpPath
  , lookupOpSymbol
  , resolveName
  , resolveBackendOp
  , resolveBinding
  , resolveTermBinding
  , resolveContextName
  ) where

import Data.Aeson (eitherDecode)
import qualified Data.ByteString.Lazy as BL
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel
  ( Name(..)
  , Term(..)
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Core.BackendSpec
  ( BackendSpec(..)
  , OpSpec(..)
  )


-- | A symbolic reference to a backend operation.
--
-- Stores the logical op name (e.g. @\"whatever.the.loaded.backend.supports\"@).
-- The actual backend path (e.g. @numpy.matmul@) is resolved later by
-- 'resolveBackendOp' during lowering.
newtype BackendOp = BackendOp
  { backendOpName :: Name -- ^ The logical Hydra 'Name' for this op.
  } deriving (Eq, Ord, Show)


-- | Construct a 'BackendOp' from an arbitrary logical op name.
backendOp :: Text -> BackendOp
backendOp =
  BackendOp . Name . T.unpack


-- | Build a Hydra 'Term' that applies a 'BackendOp' to a list of arguments.
--
-- The result is a curried application: @primitive(name) arg1 arg2 ...@
call :: BackendOp -> [Term] -> Term
call (BackendOp name) args =
  foldl (Terms.@@) (Terms.primitive name) args


-- BackendSpec and OpSpec are defined in UniAlg.Core.BackendSpec and re-exported.

-- | A resolved backend binding: the backend-specific qualified path that
-- Hydra codegen treats as an ordinary module-qualified name.
--
-- Examples: @numpy.matmul@, @jax.numpy.matmul@, @torch.matmul@.
--
-- This is /not/ Python source — it is a Hydra name that the Python coder
-- renders as a qualified attribute access.
data BackendBinding = BackendBinding
  { bindingPath :: Text -- ^ Qualified path, e.g. @\"numpy.matmul\"@.
  } deriving (Eq, Show)


-- | A loaded backend, ready to use in lowering and codegen.
newtype BackendContext = BackendContext
  { backendContextSpec :: BackendSpec -- ^ The underlying 'BackendSpec'.
  } deriving (Eq, Show)


-- | Load and decode a backend JSON file.
loadBackendSpec :: FilePath -> IO (Either String BackendSpec)
loadBackendSpec filePath =
  eitherDecode <$> BL.readFile filePath


-- | Load a named backend from a directory.
--
-- Reads @backendDir/backendName.json@ and wraps it in a 'BackendContext'.
loadBackendContext :: FilePath -> Text -> IO (Either String BackendContext)
loadBackendContext backendDir backendName = do
  loaded <- loadBackendSpec filePath
  pure (BackendContext <$> loaded)
  where
    filePath = backendDir <> "/" <> T.unpack backendName <> ".json"


-- | Look up the backend path for a logical op key (e.g. @\"matmul\"@).
lookupOpPath :: Text -> BackendSpec -> Maybe Text
lookupOpPath opKey spec =
  path <$> Map.lookup opKey (ops spec)


-- | Alias for 'lookupOpPath'.
lookupOpSymbol :: Text -> BackendSpec -> Maybe Text
lookupOpSymbol =
  lookupOpPath


-- | Resolve a Hydra 'Name' (e.g. @unialg.backend.matmul@) to its backend path.
resolveName :: BackendSpec -> Name -> Maybe Text
resolveName spec name =
  lookupOpPath (nameToBackendOpKey name) spec


-- | Resolve a 'BackendOp' to its backend path.
resolveBackendOp :: BackendSpec -> BackendOp -> Maybe Text
resolveBackendOp spec op =
  resolveName spec (backendOpName op)


-- | Resolve a Hydra 'Name' to a 'BackendBinding'.
resolveBinding :: BackendSpec -> Name -> Maybe BackendBinding
resolveBinding spec name =
  fmap (\resolvedPath -> BackendBinding { bindingPath = resolvedPath })
       (resolveName spec name)


-- | Resolve a 'TermVariable' to a 'BackendBinding'; returns 'Nothing' for
-- any other term shape.
resolveTermBinding :: BackendSpec -> Term -> Maybe BackendBinding
resolveTermBinding spec term =
  case term of
    TermVariable name ->
      resolveBinding spec name

    _ ->
      Nothing


-- | Resolve a Hydra 'Name' using a loaded 'BackendContext'.
resolveContextName :: BackendContext -> Name -> Maybe BackendBinding
resolveContextName context =
  resolveBinding (backendContextSpec context)


nameToBackendOpKey :: Name -> Text
nameToBackendOpKey (Name raw) =
  case T.stripPrefix "unialg.backend." fullName of
    Just opKey ->
      opKey

    Nothing ->
      fullName
  where
    fullName = T.pack raw
