{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module UniAlg.Pipeline.Backend
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

import Data.Aeson
import qualified Data.ByteString.Lazy as BL
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

import Hydra.Kernel
  ( Name(..)
  , Term(..)
  )

import qualified Hydra.Dsl.Terms as Terms


newtype BackendOp = BackendOp
  { backendOpName :: Name
  } deriving (Eq, Ord, Show)


backendOp :: Text -> BackendOp
backendOp =
  BackendOp . Name . T.unpack


call :: BackendOp -> [Term] -> Term
call (BackendOp name) args =
  foldl (Terms.@@) (Terms.primitive name) args


data OpSpec = OpSpec
  { path :: Text
  , arity :: Maybe Int
  , kind :: Maybe Text
  } deriving (Eq, Show, Generic)

instance FromJSON OpSpec where
  parseJSON = withObject "OpSpec" $ \o ->
    OpSpec
      <$> o .:  "path"
      <*> o .:? "arity"
      <*> o .:? "kind"


data BackendSpec = BackendSpec
  { backend :: Text
  , ops :: Map Text OpSpec
  } deriving (Eq, Show, Generic)

instance FromJSON BackendSpec


-- | A resolved backend binding.
--
-- Not Python source code — the backend-specific Hydra/Python path that
-- Hydra codegen can later see as an ordinary qualified name:
--
--   numpy.matmul
--   jax.numpy.matmul
--   torch.matmul

data BackendBinding = BackendBinding
  { bindingPath :: Text
  } deriving (Eq, Show)


newtype BackendContext = BackendContext
  { backendContextSpec :: BackendSpec
  } deriving (Eq, Show)


loadBackendSpec :: FilePath -> IO (Either String BackendSpec)
loadBackendSpec filePath =
  eitherDecode <$> BL.readFile filePath


loadBackendContext :: FilePath -> Text -> IO (Either String BackendContext)
loadBackendContext backendDir backendName = do
  loaded <- loadBackendSpec filePath
  pure (BackendContext <$> loaded)
  where
    filePath = backendDir <> "/" <> T.unpack backendName <> ".json"


lookupOpPath :: Text -> BackendSpec -> Maybe Text
lookupOpPath opKey spec =
  path <$> Map.lookup opKey (ops spec)


-- Temporary compatibility alias.
lookupOpSymbol :: Text -> BackendSpec -> Maybe Text
lookupOpSymbol =
  lookupOpPath


resolveName :: BackendSpec -> Name -> Maybe Text
resolveName spec name =
  lookupOpPath (nameToBackendOpKey name) spec


resolveBackendOp :: BackendSpec -> BackendOp -> Maybe Text
resolveBackendOp spec op =
  resolveName spec (backendOpName op)


resolveBinding :: BackendSpec -> Name -> Maybe BackendBinding
resolveBinding spec name = do
  resolvedPath <- resolveName spec name
  pure BackendBinding
    { bindingPath = resolvedPath
    }


resolveTermBinding :: BackendSpec -> Term -> Maybe BackendBinding
resolveTermBinding spec term =
  case term of
    TermVariable name ->
      resolveBinding spec name

    _ ->
      Nothing


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
