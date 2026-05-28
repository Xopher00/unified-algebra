{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE OverloadedStrings #-}

module UniAlg
  ( -- * Hydra phantom DSL surface
    module Hydra.Dsl.Meta.Phantoms
  , module Hydra.Phantoms
  , Name(..)
  , varPhantom

    -- * UniAlg symbolic architecture surface
  , op

    -- * Tensor semiring DSL
  , module UniAlg.Domain.Tensors

    -- * TTerm lambda builders
  , reify
  , reify2

    -- * Category / product / coproduct helpers
  , module UniAlg.Semantics.Category

    -- * Polynomial functor atoms and aliases
  , module UniAlg.Semantics.Functors

    -- * Fixed points and recursion schemes
  , module UniAlg.Semantics.Recursion

    -- * Optics
  , module UniAlg.Semantics.Optics

    -- * Backend-aware Hydra generation
  , writePythonWithBackend
  , loadBackendAndWritePython
  , generatePython
  , recursiveDef
  , recursiveModule
  , recDef
  , recModule
  ) where

import Hydra.Kernel
  ( Name(..)
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

import UniAlg.Pipeline.Codegen
  ( generatePythonTerms
  , loadBackendAndWritePython
  , writePythonWithBackend
  , recursiveDef
  , recursiveModule
  , recDef
  , recModule
  )

import UniAlg.Domain.Tensors

import UniAlg.Semantics.Arrows
  ( reify
  , reify2
  )

import UniAlg.Semantics.Category

import UniAlg.Semantics.Functors

import UniAlg.Semantics.Recursion

import UniAlg.Semantics.Optics


op :: String -> TTerm a
op =
  varPhantom


generatePython :: FilePath -> FilePath -> String -> [(String, TTerm a)] -> IO ()
generatePython =
  generatePythonTerms
