module UniAlg.Core.Reduce
  ( reduceTerm
  ) where

import Hydra.Kernel
  ( Application(..)
  , Lambda(..)
  , Name(..)
  , Term(..)
  )


-- ── Primitive names ─────────────────────────────────────────────────────────

eitherPrim :: Name
eitherPrim = Name "hydra.lib.eithers.either"

firstPrim :: Name
firstPrim = Name "hydra.lib.pairs.first"

secondPrim :: Name
secondPrim = Name "hydra.lib.pairs.second"


-- ── Top-level reducer ───────────────────────────────────────────────────────

reduceTerm :: Term -> Term
reduceTerm = go
  where
    go term = case term of

      -- ── Either fusion ─────────────────────────────────────────────────
      -- either(f, g, either(λl.Left(l), λr.Right(h r), x))
      --   → either(f, λr. g(h r), x)
      TermApplication (Application
        (TermApplication (Application
          (TermApplication (Application (TermVariable p1) f))
          g))
        (TermApplication (Application
          (TermApplication (Application
            (TermApplication (Application (TermVariable p2) lBranch))
            rBranch))
          x)))
        | p1 == eitherPrim
        , p2 == eitherPrim
        , isIdentityLeft lBranch ->
          let rFused = fuseBranches g rBranch
          in go $ applyEither (go f) rFused (go x)

      -- ── Pair projection ───────────────────────────────────────────────
      -- first(pair(a, b)) → a
      TermApplication (Application (TermVariable p) (TermPair (a, _)))
        | p == firstPrim ->
          go a

      -- second(pair(a, b)) → b
      TermApplication (Application (TermVariable p) (TermPair (_, b)))
        | p == secondPrim ->
          go b

      -- ── Beta reduction ────────────────────────────────────────────────
      -- (λx. body)(arg) → body[x := arg]
      TermApplication (Application (TermLambda (Lambda param _ body)) arg) ->
        go (subst param arg body)

      -- ── Recurse into subterms ─────────────────────────────────────────
      TermApplication (Application f x) ->
        TermApplication (Application (go f) (go x))

      TermLambda (Lambda param dom body) ->
        TermLambda (Lambda param dom (go body))

      TermPair (a, b) ->
        TermPair (go a, go b)

      TermEither (Left t) ->
        TermEither (Left (go t))

      TermEither (Right t) ->
        TermEither (Right (go t))

      TermList ts ->
        TermList (fmap go ts)

      _ -> term


-- ── Helpers ─────────────────────────────────────────────────────────────────

isIdentityLeft :: Term -> Bool
isIdentityLeft (TermLambda (Lambda param _ (TermEither (Left (TermVariable v))))) =
  param == v
isIdentityLeft _ = False


fuseBranches :: Term -> Term -> Term
fuseBranches g (TermLambda (Lambda param dom (TermEither (Right innerBody)))) =
  reduceTerm $ TermLambda (Lambda param dom (applyTerm g innerBody))
fuseBranches g rBranch =
  reduceTerm $ composeLam g rBranch


composeLam :: Term -> Term -> Term
composeLam g rBranch =
  let v = Name "__r"
  in TermLambda (Lambda v Nothing
       (applyTerm g (applyTerm rBranch (TermVariable v))))


applyTerm :: Term -> Term -> Term
applyTerm f x = TermApplication (Application f x)


applyEither :: Term -> Term -> Term -> Term
applyEither f g x =
  applyTerm (applyTerm (applyTerm (TermVariable eitherPrim) f) g) x


subst :: Name -> Term -> Term -> Term
subst target replacement = go'
  where
    go' term = case term of
      TermVariable v
        | v == target -> replacement
        | otherwise   -> term

      TermApplication (Application f x) ->
        TermApplication (Application (go' f) (go' x))

      TermLambda (Lambda param dom body)
        | param == target -> term
        | otherwise       -> TermLambda (Lambda param dom (go' body))

      TermPair (a, b) ->
        TermPair (go' a, go' b)

      TermEither (Left t) ->
        TermEither (Left (go' t))

      TermEither (Right t) ->
        TermEither (Right (go' t))

      TermList ts ->
        TermList (fmap go' ts)

      _ -> term
