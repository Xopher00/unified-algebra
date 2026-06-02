{-# LANGUAGE OverloadedStrings #-}

module UniAlg
  ( module Hydra.Dsl.Meta.Phantoms
  , module Hydra.Phantoms
  , Name(..)
  , Namespace(..)
  , varPhantom
  , module UniAlg.Core.Ops
  , module UniAlg.Term
  , module UniAlg.Shape
  , module UniAlg.Scheme
  , module UniAlg.Architecture
  , module UniAlg.RuntimeArchitecture
  , module UniAlg.Tensor
  , module UniAlg.Optics
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

import UniAlg.Architecture
import UniAlg.RuntimeArchitecture
import UniAlg.Codegen
  ( generatePythonTerms
  , loadBackendAndWritePython
  , loadBackendAndWritePythonRec
  , writePythonWithBackend
  , writePythonWithBackendRec
  )
import UniAlg.Core.Ops
import UniAlg.Optics
import UniAlg.Scheme
import UniAlg.Shape
import UniAlg.Tensor
import UniAlg.Term


generatePython :: FilePath -> FilePath -> String -> [(String, TTerm a)] -> IO ()
generatePython =
  generatePythonTerms
