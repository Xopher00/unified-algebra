{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.List (isInfixOf)
import Data.Text (Text)
import qualified Data.Text as T

import Hydra.Kernel
  ( Name(..)
  , Term(..)
  )

import Hydra.Phantoms (TTerm(..), unTTerm)

import UniAlg.Core.Ops (op)
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


ix :: Char -> Index
ix = Index


main :: IO ()
main = do
  -- ── Semiring basics ──────────────────────────────────────
  let sr = Semiring "add" "multiply" (Just "divide")

  assertEqual "semiring plus is add" "add" (semiringPlus sr)
  assertEqual "semiring times is multiply" "multiply" (semiringTimes sr)
  assertEqual "semiring adjoint is divide" (Just "divide") (semiringAdjoint sr)

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

  -- ── Equation fusion ──────────────────────────────────────
  let Right inner = parseEquation "ij,jk->ik"
      Right outer_ = parseEquation "ik,kl->il"
      Right fused = fuseEquation outer_ 0 inner

  assertEqual "fused inputs"
    [[ix 'i', ix 'j'], [ix 'j', ix 'k'], [ix 'k', ix 'l']]
    (eqInputs fused)

  assertEqual "fused output" [ix 'i', ix 'l'] (eqOutput fused)

  assertEqual "fused reduced" [ix 'j', ix 'k'] (eqReduced fused)

  let Right inner2 = parseEquation "km,ml->kl"
      Right fused2 = fuseEquation outer_ 1 inner2

  assertEqual "fused at slot 1 inputs"
    [[ix 'i', ix 'k'], [ix 'k', ix 'm'], [ix 'm', ix 'l']]
    (eqInputs fused2)

  assertEqual "fused at slot 1 output" [ix 'i', ix 'l'] (eqOutput fused2)

  assertBool "slot out of range is error"
    (case fuseEquation outer_ 5 inner of Left _ -> True; _ -> False)

  assertBool "mismatched inner output is error"
    (case fuseEquation outer_ 0 (let Right e = parseEquation "ab,bc->ac" in e)
     of Left _ -> True; _ -> False)

  -- ── FusionTree ────────────────────────────────────────────
  assertEqual "fuseTree single-child equals fuseEquation outer_ 0 inner"
    (fuseEquation outer_ 0 inner)
    (fuseTree (fusionNode outer_ [fusionLeaf inner]))

  -- Multi-child ordering: parent ab,cd->ad; child0 ax,xb->ab at slot 0;
  -- child1 cy,yd->cd at slot 1. Returns Left under the old reverse, Right here.
  let Right parent2 = parseEquation "ab,cd->ad"
      Right child0  = parseEquation "ax,xb->ab"
      Right child1  = parseEquation "cy,yd->cd"
      Right multi   = fuseTree (fusionNode parent2 [fusionLeaf child0, fusionLeaf child1])
  assertEqual "fuseTree multi-child inputs"
    [[ix 'a', ix 'x'], [ix 'x', ix 'b'], [ix 'c', ix 'y'], [ix 'y', ix 'd']]
    (eqInputs multi)
  assertEqual "fuseTree multi-child output" [ix 'a', ix 'd'] (eqOutput multi)

  let Right cTree = compileTree Forward sr (fusionNode outer_ [fusionLeaf inner])
  assertBool "compileTree references multiply"
    (containsName "unialg.backend.multiply" cTree)
  assertBool "compileTree references reduce.add"
    (containsName "unialg.backend.reduce.add" cTree)

  let ta = TTerm (TermVariable (Name "unialg.backend.ta")) :: TTerm Tensor
      tb = TTerm (TermVariable (Name "unialg.backend.tb")) :: TTerm Tensor
      tc = TTerm (TermVariable (Name "unialg.backend.tc")) :: TTerm Tensor
      Right aTree = applyTree Forward sr (fusionNode outer_ [fusionLeaf inner]) [ta, tb, tc]
  assertBool "applyTree references multiply"
    (containsName "unialg.backend.multiply" aTree)
  assertBool "applyTree references reduce.add"
    (containsName "unialg.backend.reduce.add" aTree)
  assertBool "applyTree references ta"
    (containsName "unialg.backend.ta" aTree)
