{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
UniAlg — algebraic neural architecture DSL.

A mathematician writes a program in ordinary Haskell using the combinators
exported here.  GHC type-checks it.  When evaluated, the Haskell expressions
silently build Hydra IR (@'TTerm'@ / @'Term'@ nodes).  The resulting IR is
then lowered through a backend spec (e.g. @numpy.json@) and rendered to
Python source by Hydra's Python coder.

The generated Python is semantically equivalent to the same computation
expressed in a canonical ML library — numerical agreement has been verified
against NumPy and TensorFlow.

=== Workflow

@
-- 1. Define morphisms using standard categorical combinators:
proj w x = let Right t = applyEquation Forward real matvecEq [w, x] in t

-- 2. Compose recursion schemes for sequential or tree structure:
-- Pure catamorphism (coalg = id):
archMod = recModule \@(SeqF Layer)
            \"transformer\" \"stack\"
            [Namespace \"numpy\"] [\"x\", \"tokens\"]
            id stackAlg
-- Hylomorphism (explicit coalgebra):
archMod = recModule \@MyF
            \"neural\" \"arch\"
            [Namespace \"numpy\"] [\"w\"]
            myCoalg myAlg

-- 3. Generate Python:
writePythonWithBackend context outputDir [attnMod] [attnMod]
@

=== Module map

* "UniAlg.Semantics.Term.Arrows"   — 'TArr': the core morphism type.
* "UniAlg.Semantics.Category" — structural morphisms and operator aliases.
* "UniAlg.Semantics.Term.Polynomial" — 'TFunctor', polynomial functor atoms ('Identity', 'Const', 'Product', 'Sum', 'Exp').
  Architecture aliases ('SeqF', 'RTreeF', 'StreamF', 'MooreF') live in "Explore.Archs".
* "UniAlg.Semantics.Recursion"— 'cataT', 'anaT', 'hyloT', 'withSelf'.
* "UniAlg.Semantics.Optics"   — van Laarhoven optics over 'TTerm' values.
* "UniAlg.Domain.Tensors"     — Einstein notation, semirings, tensor contraction.
* "UniAlg.Pipeline.Codegen"   — 'recModule', 'writePythonWithBackend', 'evalPython'.
* "UniAlg.Pipeline.Backend"   — backend spec loading and op resolution.
* "UniAlg.Pipeline.Lowering"  — Hydra IR rewriting.
* "UniAlg.Core.Reduce"        — Hydra IR simplification.
* "UniAlg.Core.Ops"           — auto-generated backend op surface ('add', 'tanh', 'op', …).
-}
module UniAlg
  ( -- * Hydra phantom DSL surface
    module Hydra.Dsl.Meta.Phantoms
  , module Hydra.Phantoms
  , Name(..)
  , Namespace(..)
  , varPhantom

    -- * Auto-generated backend op surface (arity-typed, from spec)
  , module UniAlg.Core.Ops

    -- * Tensor semiring DSL
  , module UniAlg.Domain.Tensors

    -- * TTerm lambda builders
  , reify
  , reify2

    -- * Category / product / coproduct helpers
  , module UniAlg.Semantics.Category

    -- * Polynomial functor atoms and aliases
  , module UniAlg.Semantics.Term.Polynomial

    -- * Natural eliminators (algebra and coalgebra)
  , module UniAlg.Semantics.Schemes

    -- * Fixed points and recursion schemes
  , module UniAlg.Semantics.Recursion

    -- * Optics
  , module UniAlg.Semantics.Optics

    -- * Backend-aware Hydra generation
  , writePythonWithBackend
  , writePythonWithBackendRec
  , loadBackendAndWritePython
  , loadBackendAndWritePythonRec
  , generatePython
  ) where

import Hydra.Kernel
  ( Name(..)
  , Namespace(..)
  )

import Hydra.Dsl.Meta.Phantoms hiding
  ( compose
  , left
  , pair
  , right
  , set
  )

import Hydra.Dsl.Meta.Terms
  ( varPhantom
  )

import Hydra.Phantoms

import UniAlg.Core.Ops

import UniAlg.Pipeline.Codegen
  ( generatePythonTerms
  , loadBackendAndWritePython
  , loadBackendAndWritePythonRec
  , writePythonWithBackend
  , writePythonWithBackendRec
  )

import UniAlg.Domain.Tensors

import UniAlg.Semantics.Term.Arrows
  ( reify
  , reify2
  )

import UniAlg.Semantics.Category

import UniAlg.Semantics.Term.Polynomial

import UniAlg.Semantics.Schemes

import UniAlg.Semantics.Recursion

import UniAlg.Semantics.Optics


-- | Generate Python for a flat list of named definitions.  Alias for
-- 'generatePythonTerms'.
generatePython :: FilePath -> FilePath -> String -> [(String, TTerm a)] -> IO ()
generatePython =
  generatePythonTerms
