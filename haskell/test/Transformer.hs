{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}

module Main where

import Prelude hiding (fst, snd, either, left, right)
import UniAlg


real :: Semiring
real = Semiring "add" "multiply" (Just "divide")

Right matvecEq = parseEquation "ij,j->i"

proj :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor
proj w x = let Right t = applyEquation Forward real matvecEq [w, x] in t

add :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor
add = op2 "add"

mul :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor
mul = op2 "multiply"

_fst :: Lens' (TTerm (a, b)) (TTerm a)
_fst = mkLens fst (\p w -> pair w (snd p))

_snd :: Lens' (TTerm (a, b)) (TTerm b)
_snd = mkLens snd (\p w -> pair (fst p) w)


-- ── Sequence helpers ─────────────────────────────────────────────────────────

type Seq a = Fix (SeqF a)

nil :: Seq a
nil = Fix (InL (Const ()))

cons :: a -> Seq a -> Seq a
cons a as = Fix (InR (Pair (Const a) (Identity as)))

zipLayers
  :: [TTerm Tensor] -> [TTerm Tensor] -> [TTerm Tensor]
  -> [TTerm Tensor] -> [TTerm Tensor]
  -> Seq (TTerm (Tensor, (Tensor, (Tensor, (Tensor, Tensor)))))
zipLayers wQs wKs wVs wUps wDowns =
  foldr cons nil (zipWith5 (\q k v u d -> pair q (pair k (pair v (pair u d)))) wQs wKs wVs wUps wDowns)


-- ── FFN ──────────────────────────────────────────────────────────────────────

ffn :: TTerm Tensor -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
ffn wUp wDown x = add (proj wDown (op1 "tanh" (proj wUp x))) x


-- ── Attention head: fold (k,v) pairs, accumulate score·value ─────────────────
-- outer params: q — query; z0 — accumulator

attnAlg (InL (Const ()))                       = var "z0"
attnAlg (InR (Pair (Const kv) (Identity acc))) =
  let score = mul (var "q") (view _fst kv)
  in  add (view _snd (over _snd (mul score) kv)) acc

attnMod = recModule "transformer.attn" "attend" [Namespace "numpy"] ["q", "z0"] $
            cataT @(SeqF (TTerm (Tensor, Tensor))) attnAlg


-- ── Layer weight lenses ───────────────────────────────────────────────────────

type Layer = TTerm (Tensor, (Tensor, (Tensor, (Tensor, Tensor))))

_wQ    = _fst
_wK    = _snd . _fst
_wV    = _snd . _snd . _fst
_wUp   = _snd . _snd . _snd . _fst
_wDown = _snd . _snd . _snd . _snd


-- ── Block and layer stack ─────────────────────────────────────────────────────

blockAlg (InL (Const ()))                        = var "x"
blockAlg (InR (Pair (Const layer) (Identity h))) =
  let attended = var "transformer.attn.attend" @@ proj (view _wQ layer) h @@ h @@ var "tokens"
  in  ffn (view _wUp layer) (view _wDown layer) attended

stackMod = recModule "transformer.stack" "stack" [Namespace "numpy", Namespace "transformer.attn"] ["x", "tokens"] $
             cataT @(SeqF Layer) blockAlg
