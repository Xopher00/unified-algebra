{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}

-- JSON schema for architecture specifications emitted by agents.
-- Three cleanly separated blocks:
--   arch  — recursion structure (PolyF, class, monad, weight-tying)
--   cell  — explicit primitive ops only; no framework class names
--   ref   — torch/tf/numpy reference used solely by the Python harness
module ArchSpec
  ( ArchSpec(..)
  , ArchBlock(..)
  , CellBody(..)
  , RefBlock(..)
  , PolyFSpec(..)
  , ArchClassSpec(..)
  , CellExprSpec(..)
  , Binding(..)
  , ResultSpec(..)
  , RefEntry(..)
  , SemiringSpec(..)
  ) where

import Data.Aeson
import Data.Map.Strict (Map)
import GHC.Generics


-- Open semiring: agent supplies all operation names and identities directly.
-- JSON example (real):
--   {"label": "real", "add": "add", "multiply": "multiply", "divide": "divide",
--    "zero": "0.0", "one": "1.0"}
-- JSON example (tropical / min-plus):
--   {"label": "tropical", "add": "minimum", "multiply": "add",
--    "zero": "float('inf')", "one": "0.0"}
--
-- zero  = additive identity  (x + zero = x)
-- one   = multiplicative identity  (x * one = x)
-- Both are Python literal strings; the test harness evaluates them as numpy scalars.
-- divide is optional (not all semirings have a division operation).
data SemiringSpec = SemiringSpec
  { srLabel :: String       -- Haskell variable name for this semiring
  , srAdd   :: String       -- name of the addition operation (Haskell op key)
  , srMul   :: String       -- name of the multiplication operation
  , srDiv   :: Maybe String -- name of the division operation, if any
  , srZero  :: String       -- additive identity as a Python literal
  , srOne   :: String       -- multiplicative identity as a Python literal
  } deriving (Show, Eq, Generic)

instance FromJSON SemiringSpec where
  parseJSON = withObject "SemiringSpec" $ \o ->
    SemiringSpec
      <$> o .:  "label"
      <*> o .:  "add"
      <*> o .:  "multiply"
      <*> o .:? "divide"
      <*> o .:  "zero"
      <*> o .:  "one"


data ArchClassSpec = CataSpec | AnaSpec | HyloSpec | NoStructureSpec
  deriving (Show, Eq, Generic)

instance FromJSON ArchClassSpec where
  parseJSON = withText "ArchClassSpec" $ \t -> case t of
    "Cata"        -> pure CataSpec
    "Ana"         -> pure AnaSpec
    "Hylo"        -> pure HyloSpec
    "NoStructure" -> pure NoStructureSpec
    _             -> fail ("Unknown arch class: " ++ show t)


-- Value-level PolyF AST matching Grammar.PolyF.
data PolyFSpec
  = PFUnit
  | PFConst
  | PFHole
  | PFSum     PolyFSpec PolyFSpec
  | PFProduct PolyFSpec PolyFSpec
  | PFExp     PolyFSpec
  deriving (Show, Eq, Generic)

instance FromJSON PolyFSpec where
  parseJSON = withObject "PolyFSpec" $ \o -> do
    tag <- o .: "tag"
    case (tag :: String) of
      "KUnit"   -> pure PFUnit
      "KConst"  -> pure PFConst
      "Hole"    -> pure PFHole
      "Sum"     -> PFSum     <$> o .: "left" <*> o .: "right"
      "Product" -> PFProduct <$> o .: "left" <*> o .: "right"
      "Exp"     -> PFExp     <$> o .: "arg"
      t         -> fail ("Unknown PolyF tag: " ++ t)


-- One primitive op in ANF (administrative normal form).
-- Each Binding names an intermediate value.
-- Args reference earlier binding names, param names, or special vars (state, input).
data CellExprSpec
  = CEContraction { ceSemiring :: SemiringSpec, ceEquation :: String, ceArgs :: [String] }
  | CEElemOp      { ceOp :: String, ceOpArgs :: [String] }
  | CEActivation  { ceKind :: String, ceActArg :: String }
  deriving (Show, Eq, Generic)

instance FromJSON CellExprSpec where
  parseJSON = withObject "CellExprSpec" $ \o -> do
    tag <- o .: "tag"
    case (tag :: String) of
      "Contraction" -> CEContraction <$> o .: "semiring" <*> o .: "equation" <*> o .: "args"
      "ElemOp"      -> CEElemOp      <$> o .: "op"       <*> o .: "args"
      "Activation"  -> CEActivation  <$> o .: "kind"     <*> o .: "arg"
      t             -> fail ("Unknown CellExpr tag: " ++ t)

data Binding = Binding
  { bindName :: String
  , bindExpr :: CellExprSpec
  } deriving (Show, Eq, Generic)

instance FromJSON Binding where
  parseJSON = withObject "Binding" $ \o ->
    Binding <$> o .: "name" <*> o .: "expr"


-- Arch-class-specific result structure.
-- All binding names refer to entries in CellBody.cellBindings.
data ResultSpec
  -- Ana: coalgebra s ↦ (output, λinp. next_state)
  = ResAna
      { resStateVar          :: String    -- lambda var for state
      , resInputVar          :: String    -- lambda var for input
      , resOutputBindings    :: [String]  -- subset of binding names used in output
      , resOutput            :: String    -- binding name for output value
      , resNextStateBindings :: [String]  -- subset used in next-state (may ref state+input)
      , resNextState         :: String    -- binding name for next-state value
      }
  -- CataConst: algebra (base_value, \step_vars -> step_result)
  -- base is a constant param reference (list-fold style)
  | ResCataConst
      { resBase         :: String    -- param name used directly as base case
      , resStepVars     :: [String]  -- lambda vars for step (e.g. ["a", "s"])
      , resStepBindings :: [String]  -- binding names used in step
      , resStepResult   :: String    -- binding name for step result
      }
  -- CataFn: algebra (\base_var -> base_result, \step_vars -> step_result)
  -- base is a function of its input (tree-fold style)
  | ResCataFn
      { resBaseVar      :: String    -- lambda var for base-case input
      , resBaseBindings :: [String]  -- binding names used in base
      , resBase         :: String    -- binding name for base result
      , resStepVars     :: [String]  -- lambda vars for step (e.g. ["l", "r"])
      , resStepBindings :: [String]  -- binding names used in step
      , resStepResult   :: String    -- binding name for step result
      }
  -- Pure: stateless function λinp. result
  | ResPure
      { resInputVar     :: String
      , resPureBindings :: [String]
      , resResult       :: String
      }
  deriving (Show, Eq, Generic)

instance FromJSON ResultSpec where
  parseJSON = withObject "ResultSpec" $ \o -> do
    tag <- o .: "tag"
    case (tag :: String) of
      "Ana"       -> ResAna
                       <$> o .: "state_var"
                       <*> o .: "input_var"
                       <*> o .: "output_bindings"
                       <*> o .: "output"
                       <*> o .: "next_state_bindings"
                       <*> o .: "next_state"
      "CataConst" -> ResCataConst
                       <$> o .: "base"
                       <*> o .: "step_vars"
                       <*> o .: "step_bindings"
                       <*> o .: "step_result"
      "CataFn"    -> ResCataFn
                       <$> o .: "base_var"
                       <*> o .: "base_bindings"
                       <*> o .: "base"
                       <*> o .: "step_vars"
                       <*> o .: "step_bindings"
                       <*> o .: "step_result"
      "Pure"      -> ResPure
                       <$> o .: "input_var"
                       <*> o .: "pure_bindings"
                       <*> o .: "result"
      t           -> fail ("Unknown result tag: " ++ t)


-- Arch block: recursion structure only.
-- Framework class names are FORBIDDEN here.
data ArchBlock = ArchBlock
  { archClass  :: ArchClassSpec
  , archPolyF  :: PolyFSpec
  , archMonad  :: Maybe String  -- e.g. "G x Z_w" for translation-equivariant
  , archTying  :: [[String]]    -- equivalence classes of param names sharing weights
  , archLax    :: Bool
  } deriving (Show, Eq, Generic)

instance FromJSON ArchBlock where
  parseJSON = withObject "ArchBlock" $ \o ->
    ArchBlock
      <$> o .: "class"
      <*> o .: "poly_f"
      <*> o .:? "monad"
      <*> o .:? "weight_tying" .!= []
      <*> o .:? "lax"          .!= False


-- Cell body block: ordered ANF bindings + arch-class result structure.
-- No framework class names anywhere in this block.
data CellBody = CellBody
  { cellParams   :: [String]   -- outer parameter names (weights, biases, etc.)
  , cellBindings :: [Binding]  -- all named intermediate values
  , cellResult   :: ResultSpec
  } deriving (Show, Eq, Generic)

instance FromJSON CellBody where
  parseJSON = withObject "CellBody" $ \o ->
    CellBody
      <$> o .: "params"
      <*> o .: "bindings"
      <*> o .: "result"


-- Single framework reference entry.
data RefEntry
  = RefSingleClass { refClass :: String, refKwargs :: Map String String }
  | RefComposed    { refComponents :: [String] }
  | RefNumpyOnly   { refEquation :: String }
  deriving (Show, Eq, Generic)

instance FromJSON RefEntry where
  parseJSON = withObject "RefEntry" $ \o -> do
    tag <- o .: "tag"
    case (tag :: String) of
      "SingleClass" -> RefSingleClass <$> o .: "class" <*> o .:? "kwargs" .!= mempty
      "Composed"    -> RefComposed    <$> o .: "components"
      "NumpyOnly"   -> RefNumpyOnly   <$> o .: "equation"
      t             -> fail ("Unknown ref entry tag: " ++ t)


-- Reference block: consumed only by the Python renderer.
-- The Haskell renderer ignores this block entirely.
data RefBlock = RefBlock
  { refStrength   :: String          -- "single-class" | "composed" | "numpy-only"
  , refTorch      :: Maybe RefEntry
  , refTensorflow :: Maybe RefEntry
  , refNumpy      :: Maybe RefEntry
  } deriving (Show, Eq, Generic)

instance FromJSON RefBlock where
  parseJSON = withObject "RefBlock" $ \o ->
    RefBlock
      <$> o .:? "strength"   .!= "numpy-only"
      <*> o .:? "torch"
      <*> o .:? "tensorflow"
      <*> o .:? "numpy"


-- Top-level spec. The label determines the arch directory and module name.
data ArchSpec = ArchSpec
  { specLabel :: String
  , specArch  :: ArchBlock
  , specCell  :: CellBody
  , specRef   :: RefBlock
  } deriving (Show, Eq, Generic)

instance FromJSON ArchSpec where
  parseJSON = withObject "ArchSpec" $ \o ->
    ArchSpec
      <$> o .: "label"
      <*> o .: "arch"
      <*> o .: "cell"
      <*> o .: "ref"
