{-# LANGUAGE OverloadedStrings #-}

{-|
Shared types and utilities for the Explore seed catalogue.

Individual seeds live in their own modules under "explore/archs/*/".
This module provides the common types and the 'contraction' /
'adjointContraction' helpers parameterized on an explicit 'Semiring'.
-}
module Seed
  ( SeedEntry(..)
  , ArchClass(..)
  , contraction
  , adjointContraction
  ) where

import Hydra.Kernel ( Module(..) )
import UniAlg
import Grammar (PolyF)


-- | The recursion-scheme direction of an architecture.
--
--   * 'CataArch' — catamorphism (fold): consumes a recursive input structure.
--     Example: @SeqRnn@ folds a list; @TreeRnn@ folds a binary tree.
--
--   * 'AnaArch' — anamorphism (unfold): generates a corecursive output from a seed.
--     Example: @Moore@, @Mealy@, @StreamRnn@.
--
--   * 'HyloArch' — hylomorphism (unfold then fold): the coalgebra decomposes
--     the input, the algebra reassembles it.  The coalgebra typically uses the
--     right adjoint of the algebra's forward operation (e.g. @subtract@ paired
--     with @add@).  Example: @EdgeConv@.
--
--   * 'NoStructure' — architectures that do not fit a single recursion scheme.
data ArchClass
  = CataArch
  | AnaArch
  | HyloArch
  | NoStructure
  deriving (Eq, Show)


-- | A catalog entry for one generated architecture.
--
-- Each 'SeedEntry' fixes four coordinates on the architecture classification lattice:
--
--   * @'seedLabel'@ — unique identifier used to name the generated Python function.
--   * @'seedClass'@ — recursion direction ('CataArch', 'AnaArch', 'HyloArch').
--   * @'seedPolyF'@ — the polynomial endofunctor @F@ as a value-level 'PolyF'
--     expression (used by the catalog agent for enumeration and classification).
--   * @'seedModule'@ — the Hydra 'Module' carrying the fully built Hydra IR.
--
-- Construct entries with 'cataModule', 'anaModule', or 'hyloModule' and
-- wrap them in this record.
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

-- | Adjoint contraction: uses the semiring's adjoint op for element products
-- and @times@ for reduction.  Fails at runtime if the semiring has no adjoint
-- (i.e. 'semiringAdjoint' is 'Nothing').
--
-- Typical use in a hylomorphism coalgebra whose algebra uses 'contraction':
--
-- @
-- real = Semiring "add" "multiply" (Just "subtract")
-- -- algebra:   contraction real "ij,j->i" w x  →  sum(w * x)
-- -- coalgebra: adjointContraction real "ij,j->i" w x  →  prod(w - x)
-- @
adjointContraction :: Semiring -> String -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
adjointContraction sr eqStr w x = case applyEquation Adjoint sr eq [w, x] of
  Right t -> t
  Left  e -> error ("adjointContraction " <> eqStr <> " failed: " <> e)
  where Right eq = parseEquation eqStr
