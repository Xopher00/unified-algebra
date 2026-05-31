{-# LANGUAGE OverloadedStrings #-}

{-|
Assembled catalogue of all architecture seeds.

Collects unique seeds from each architecture module for use by
"Explore.Generate" (seed lookup) and "ExploreTest" (smoke checks).
-}
module Explore.Catalogue
  ( seeds
  , allArchSeeds
  ) where

import Data.Function (on)
import Data.List (nubBy)

import Explore.Seed (SeedEntry(..), seedLabel)
import qualified Explore.Archs.SeqRnn    as SeqRnn
import qualified Explore.Archs.TreeRnn   as TreeRnn
import qualified Explore.Archs.StreamRnn as StreamRnn
import qualified Explore.Archs.Moore     as Moore


-- | All @(archDir, [(backend, SeedEntry)])@ pairs for code generation.
allArchSeeds :: [(String, [(String, SeedEntry)])]
allArchSeeds =
  [ ("seq_rnn",    SeqRnn.backendSeeds)
  , ("tree_rnn",   TreeRnn.backendSeeds)
  , ("stream_rnn", StreamRnn.backendSeeds)
  , ("moore",      Moore.backendSeeds)
  ]


-- | Unique seeds across all architectures (for smoke checks).
seeds :: [SeedEntry]
seeds = nubBy ((==) `on` seedLabel) $
  concatMap (map snd . snd) allArchSeeds
