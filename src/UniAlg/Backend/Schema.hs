{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Backend specification data model.

'BackendSpec' and 'OpSpec' are the parsed form of a backend JSON file.
They live in 'UniAlg.Backend' so that both the runtime resolution\/lowering
pipeline and the compile-time op generator ('UniAlg.Backend.Generate') can
depend on them without introducing a layer inversion.
-}
module UniAlg.Backend.Schema
  ( OpSpec(..)
  , BackendSpec(..)
  ) where

import Data.Aeson
import Data.Map.Strict (Map)
import Data.Text (Text)
import GHC.Generics (Generic)


-- | A single entry in a backend JSON spec.
data OpSpec = OpSpec
  { path  :: Text        -- ^ Qualified backend path, e.g. @\"numpy.matmul\"@.
  , arity :: Maybe Int   -- ^ Number of tensor arguments (used for eta-expansion and op generation).
  , kind  :: Maybe Text  -- ^ Optional op category tag (currently unused).
  } deriving (Eq, Show, Generic)

instance FromJSON OpSpec where
  parseJSON = withObject "OpSpec" $ \o ->
    OpSpec
      <$> o .:  "path"
      <*> o .:? "arity"
      <*> o .:? "kind"


-- | The full deserialized backend JSON file.
data BackendSpec = BackendSpec
  { backend :: Text             -- ^ Backend name, e.g. @\"numpy\"@.
  , ops     :: Map Text OpSpec  -- ^ Map from logical op key to 'OpSpec'.
  } deriving (Eq, Show, Generic)

instance FromJSON BackendSpec
