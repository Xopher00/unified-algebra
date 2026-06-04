{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.List (isInfixOf, isPrefixOf, tails)
import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel
  ( Name(..)
  , Term(..)
  )

import Hydra.Phantoms (TTerm(..), unTTerm)

import UniAlg.Tensor


assertBool :: Text -> Bool -> IO ()
assertBool label condition =
  if condition
    then putStrLn ("PASS: " <> T.unpack label)
    else error ("FAIL: " <> T.unpack label)


assertEqual :: (Eq a, Show a) => Text -> a -> a -> IO ()
assertEqual label expected actual =
  if expected == actual
    then putStrLn ("PASS: " <> T.unpack label)
    else error $
      "FAIL: " <> T.unpack label
        <> "\n  expected: " <> show expected
        <> "\n  actual:   " <> show actual


containsName :: String -> TTerm a -> Bool
containsName needle t =
  needle `isInfixOf` show (unTTerm t)


countName :: String -> TTerm a -> Int
countName needle t =
  length (filter (isPrefixOf needle) (tails (show (unTTerm t))))


ix :: Char -> Index
ix = Index


main :: IO ()
main = do
  -- ── Semiring basics ──────────────────────────────────────
  let sr = Semiring "add" "multiply" (Just "multiply") (Just "divide") 0 1

  assertEqual "semiring plus is add" "add" (semiringPlus sr)
  assertEqual "semiring times is multiply" "multiply" (semiringTimes sr)
  assertEqual "semiring adjoint plus is multiply"
    (Just "multiply") (semiringAdjointPlus sr)
  assertEqual "semiring adjoint times is divide"
    (Just "divide") (semiringAdjointTimes sr)

  -- ── Simple contract ──────────────────────────────────────
  assertBool "contract references unialg.backend.multiply"
    (containsName "unialg.backend.multiply" (contract sr))

  assertBool "contract references unialg.backend.reduce.add"
    (containsName "unialg.backend.reduce.add" (contract sr))

  -- ── Adjoint contract ─────────────────────────────────────
  assertBool "adjoint contract references unialg.backend.divide"
    (containsName "unialg.backend.divide" (adjointContract sr))

  assertBool "adjoint contract references unialg.backend.reduce.multiply"
    (containsName "unialg.backend.reduce.multiply" (adjointContract sr))

  -- ── Op registry ──────────────────────────────────────────
  assertEqual "op builds correct variable name"
    (TermVariable (Name "unialg.backend.multiply"))
    (unTTerm (op "multiply" :: TTerm a))

  assertEqual "op builds correct variable name for reduce ops"
    (TermVariable (Name "unialg.backend.reduce.add"))
    (unTTerm (op "reduce.add" :: TTerm a))

  -- ── Equation parsing ─────────────────────────────────────
  let Right matmul = parseEquation "ij,jk->ik"

  assertEqual "matmul inputs"
    [[ix 'i', ix 'j'], [ix 'j', ix 'k']]
    (eqInputs matmul)

  assertEqual "matmul output"
    [ix 'i', ix 'k']
    (eqOutput matmul)

  assertEqual "matmul reduced"
    [ix 'j']
    (eqReduced matmul)

  let Right matvec = parseEquation "ij,j->i"

  assertEqual "matvec inputs"
    [[ix 'i', ix 'j'], [ix 'j']]
    (eqInputs matvec)

  assertEqual "matvec reduced" [ix 'j'] (eqReduced matvec)

  let Right outer = parseEquation "i,j->ij"

  assertEqual "outer product reduced" [] (eqReduced outer)

  let Right single = parseEquation "i->i"

  assertEqual "single input equation inputs" [[ix 'i']] (eqInputs single)
  assertEqual "single input equation output" [ix 'i'] (eqOutput single)
  assertEqual "single input equation reduced" [] (eqReduced single)

  let Right trace_ = parseEquation "ii->"

  assertEqual "trace reduced" [ix 'i'] (eqReduced trace_)

  let Right batched = parseEquation "bij,bjk->bik"

  assertEqual "batched matmul reduced" [ix 'j'] (eqReduced batched)

  let Right three = parseEquation "ij,jk,kl->il"

  assertEqual "three-operand reduced" [ix 'j', ix 'k'] (eqReduced three)

  -- ── Error cases ──────────────────────────────────────────
  assertBool "missing arrow is error"
    (case parseEquation "ij,jk" of Left e -> "->'" `isInfixOf` e; _ -> False)

  assertBool "output label not in input is error"
    (case parseEquation "ij->iz" of Left e -> "not in" `isInfixOf` e; _ -> False)

  assertBool "duplicate output label is error"
    (case parseEquation "i->ii" of Left e -> "unique" `isInfixOf` e; _ -> False)

  assertBool "multiple arrows is error"
    (case parseEquation "i->j->i" of Left e -> "exactly one" `isInfixOf` e; _ -> False)

  assertBool "malformed arrow fragment is error"
    (case parseEquation "i-j->i" of Left e -> "arrow" `isInfixOf` e; _ -> False)

  assertBool "empty input list is error"
    (case parseEquation "->i" of Left e -> "input list" `isInfixOf` e; _ -> False)

  assertBool "empty operand is error"
    (case parseEquation "i,,j->ij" of Left e -> "operands" `isInfixOf` e; _ -> False)

  -- ── Orientation ──────────────────────────────────────────
  assertEqual "forward orientation" Forward Forward
  assertEqual "adjoint orientation" Adjoint Adjoint

  -- ── compileEquation Forward ──────────────────────────────
  let Right fwdTerm = compileEquation Forward sr matmul

  assertBool "forward contractEq references multiply"
    (containsName "unialg.backend.multiply" fwdTerm)

  assertBool "forward contractEq references reduce.add"
    (containsName "unialg.backend.reduce.add" fwdTerm)

  assertBool "forward contractEq references expand_dims"
    (containsName "unialg.backend.structural.expand_dims" fwdTerm)

  assertBool "forward contractEq references transpose"
    (containsName "unialg.backend.structural.transpose" fwdTerm)

  -- ── compileEquation Adjoint ──────────────────────────────
  let Right adjTerm = compileEquation Adjoint sr matmul

  assertBool "adjoint contractEq references divide"
    (containsName "unialg.backend.divide" adjTerm)

  assertBool "adjoint contractEq references reduce.multiply"
    (containsName "unialg.backend.reduce.multiply" adjTerm)

  let Right singleTerm = compileEquation Forward sr single
  assertBool "single input compile references transpose"
    (containsName "unialg.backend.structural.transpose" singleTerm)

  let Right outerTerm = compileEquation Forward sr outer
  assertBool "outer product compile has no reduction"
    (not (containsName "unialg.backend.reduce." outerTerm))

  let Right swapEq = parseEquation "ij->ji"
      Right swapTerm = compileEquation Forward sr swapEq
  assertBool "axis swap compile references transpose"
    (containsName "unialg.backend.structural.transpose" swapTerm)
  assertBool "axis swap compile has no reduction"
    (not (containsName "unialg.backend.reduce." swapTerm))

  let Right rowSumEq = parseEquation "ij->i"
      Right rowSumTerm = compileEquation Forward sr rowSumEq
  assertEqual "single-axis reduction count"
    1
    (countName "unialg.backend.reduce.add" rowSumTerm)

  let Right chain4 = parseEquation "ij,jk,kl,lm->im"
      Right chain4Term = compileEquation Forward sr chain4
  assertBool "four-factor chain compiles with reductions"
    (containsName "unialg.backend.reduce.add" chain4Term)

  assertBool "zero-input equation compile is error"
    (case compileEquation Forward sr (Equation [] [] []) of Left e -> "at least one" `isInfixOf` e; _ -> False)

  -- ── applyEquation ────────────────────────────────────────
  let x = TTerm (TermVariable (Name "unialg.backend.x_tensor")) :: TTerm Tensor
      w = TTerm (TermVariable (Name "unialg.backend.w_tensor")) :: TTerm Tensor
      Right applied = applyEquation Forward sr matmul [x, w]

  assertBool "applyEquation produces a term with multiply"
    (containsName "unialg.backend.multiply" applied)

  assertBool "applyEquation produces a term with reduce.add"
    (containsName "unialg.backend.reduce.add" applied)

  assertBool "applyEquation produces a term with x_tensor"
    (containsName "unialg.backend.x_tensor" applied)

  assertBool "applyEquation produces a term with w_tensor"
    (containsName "unialg.backend.w_tensor" applied)

  assertBool "applyEquation too few args is error"
    (case applyEquation Forward sr matmul [x] of Left e -> "expected 2" `isInfixOf` e; _ -> False)

  assertBool "applyEquation too many args is error"
    (case applyEquation Forward sr matmul [x, w, x] of Left e -> "expected 2" `isInfixOf` e; _ -> False)

