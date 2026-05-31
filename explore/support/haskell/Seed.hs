{-# LANGUAGE OverloadedStrings #-}

{-|
Shared types and utilities for the Explore seed catalogue.

Individual seeds live in their own modules under "explore/archs/*/".
This module provides only the common types and the 'contraction' helper
parameterized on an explicit 'Semiring'.
-}
module Seed
  ( SeedEntry(..)
  , ArchClass(..)
  , contraction
  ) where

import Hydra.Kernel ( Module(..) )
import UniAlg
import Grammar (PolyF)


data ArchClass
  = CataArch
  | AnaArch
  | HyloArch
  | NoStructure
  deriving (Eq, Show)


data SeedEntry = SeedEntry
  { seedLabel  :: String
  , seedClass  :: ArchClass
  , seedPolyF  :: PolyF
  , seedModule :: Module
  }


-- | Tensor contraction via 'applyEquation', parameterized on the semiring.
contraction :: Semiring -> String -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
contraction sr eqStr w x = case applyEquation Forward sr eq [w, x] of
  Right t -> t
  Left  e -> error ("contraction " <> eqStr <> " failed: " <> e)
  where Right eq = parseEquation eqStr
