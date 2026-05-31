{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel
  ( Name(..)
  , Term
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Backend


assertEqual :: (Eq a, Show a) => Text -> a -> a -> IO ()
assertEqual label expected actual =
  if expected == actual
    then putStrLn ("PASS: " <> T.unpack label)
    else error $
      "FAIL: " <> T.unpack label
        <> "\n  expected: " <> show expected
        <> "\n  actual:   " <> show actual


main :: IO ()
main = do
  let opName = "whatever.the.loaded.backend.supports"
      op = backendOp opName

      x :: Term
      x = Terms.var "x"

      w :: Term
      w = Terms.var "w"

      actual :: Term
      actual = call op [x, w]

      expected :: Term
      expected =
        foldl
          (Terms.@@)
          (Terms.primitive (Name (T.unpack opName)))
          [x, w]

  assertEqual
    "backendOp stores arbitrary logical op name"
    (Name (T.unpack opName))
    (backendOpName op)

  assertEqual
    "call builds a Hydra primitive application"
    expected
    actual
