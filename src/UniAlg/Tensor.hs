{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

{-|
Einstein-notation tensor contractions over semirings, compiled to Hydra IR.

Parse an equation string with 'parseEquation' (any arity), then compile to
a 'TTerm' with 'applyEquation' or to a codegen 'Module' with 'equationModule'.
-}
module UniAlg.Tensor where

import Control.Monad (foldM)
import qualified Data.List.Split as Split
import Data.List (nub, sortOn)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set

import Hydra.Kernel (Module(..), Namespace(..))
import Hydra.Phantoms (TTerm(..))
import Hydra.Dsl.Meta.Phantoms (var, (@@), definitionInNamespace, toDefinition)
import UniAlg.Term (reify2)

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Reduce (reduceTerm)
import UniAlg.Backend (genBackendOps, lookupOp)

$(genBackendOps "backends/numpy.json")

op :: String -> TTerm a
op = lookupOp opRegistry


-- | Phantom type tag for tensor-typed 'TTerm' values.
data Tensor


-- | Whether to use the forward or adjoint semiring operations.
--
-- 'Forward' uses @times@ for element products and @plus@ for reduction.
-- 'Adjoint' uses @adjointTimes@ for element products and @adjointPlus@ for
-- reduction (i.e. the dual contraction for backward passes).
data Orientation
  = Forward
  | Adjoint
  deriving (Eq, Show)


-- | An algebraic structure parameterising tensor contractions.
--
-- @
-- real     = Semiring \"add\" \"multiply\" (Just \"multiply\") (Just \"divide\") 0 1
-- tropical = Semiring \"maximum\" \"add\" Nothing Nothing (-1/0) 0
-- @
data Semiring = Semiring
  { semiringPlus         :: String        -- ^ Op key for reduction (e.g. @\"add\"@).
  , semiringTimes        :: String        -- ^ Op key for element products (e.g. @\"multiply\"@).
  , semiringAdjointPlus  :: Maybe String  -- ^ Adjoint reduction op key, if any.
  , semiringAdjointTimes :: Maybe String  -- ^ Adjoint element-product op key, if any.
  , semiringPlusId       :: Double        -- ^ Additive identity (neutral element of @plus@).
  , semiringTimesId      :: Double        -- ^ Multiplicative identity (neutral element of @times@).
  } deriving (Eq, Show)


-- | A parsed Einstein notation equation.
data Equation = Equation
  { eqInputs  :: [[Index]]  -- ^ Index set for each input operand.
  , eqOutput  :: [Index]    -- ^ Output index labels.
  , eqReduced :: [Index]    -- ^ Indices that are summed over (contracted).
  } deriving (Eq, Show)


-- | A single Einstein index label (a single character, e.g. @\'i\'@, @\'j\'@).
newtype Index = Index Char
  deriving (Eq, Ord, Show)

type Factor = ([Index], TTerm Tensor)


-- ── Equation ─────────────────────────────────────────────────────────────────

mkEquation :: [[Index]] -> [Index] -> Either String Equation
mkEquation inputs output = do
  let distinctIn = nub (concat inputs)
      inSet      = Set.fromList distinctIn
      outSet     = Set.fromList output
  check (Set.size outSet == length output) "output labels must be unique"
  check (outSet `Set.isSubsetOf` inSet)    "output labels not in any input"
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = output
    , eqReduced = filter (`Set.notMember` outSet) distinctIn
    }

-- | Parse an Einstein notation string into an 'Equation'.
--
-- Returns @'Left' err@ if the string is malformed (missing arrow, duplicate
-- output labels, output labels absent from all inputs).
--
-- @
-- parseEquation \"ij,jk->ik\"  -- matrix multiply: reduces j
-- parseEquation \"i,j->ij\"    -- outer product: no reduction
-- @
parseEquation :: String -> Either String Equation
parseEquation raw = do
  (lhs, rhs) <-
    case Split.splitOn "->" (filter (/= ' ') raw) of
      [lhs', rhs'] -> Right (lhs', rhs')
      _            -> Left "equation must contain exactly one '->'"
  check (not (null lhs)) "equation input list must not be empty"
  check (all (\c -> c /= '-' && c /= '>') (lhs <> rhs))
    "index labels cannot contain arrow characters"
  let inputParts = Split.splitOn "," lhs
  check (not (any null inputParts)) "equation operands must not be empty"
  mkEquation
    (fmap (fmap Index) inputParts)
    (fmap Index rhs)


-- ── Simple contractions (no equation, arity-inferred reduce) ─────────────────

-- | Curried binary contraction over the given semiring.
contract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
contract sr =
  reify2 $ \x y ->
    op (reduceKey (semiringPlus sr)) @@ (op (semiringTimes sr) @@ x @@ y)


-- | Adjoint contraction: uses @adjoint@ for element product and @times@
-- for reduction.  Fails at runtime if the semiring has no adjoint.
adjointContract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
adjointContract sr =
  case adjointKeys sr of
    Left err -> error ("adjointContract: " <> err)
    Right (plusKey, timesKey) -> reify2 $ \x y ->
      op (reduceKey plusKey) @@ (op timesKey @@ x @@ y)


-- ── Equation-aware contractions ──────────────────────────────────────────────

-- | Compile an 'Equation' to a curried 'TTerm' lambda.
compileEquation :: Orientation -> Semiring -> Equation -> Either String (TTerm a)
compileEquation orientation sr eq = do
  (plusKey, timesKey) <- orientationKeys orientation sr
  let params = ["t" <> show i | i <- [0 .. length (eqInputs eq) - 1]]
      factors = zip (eqInputs eq) (fmap var params)
  (firstFactor, rest) <- contractPlan eq factors
  factor <- foldM (stepFactor plusKey timesKey) firstFactor rest
  body <- snd <$> (alignFactor (eqOutput eq) =<< reduceFactor plusKey (eqOutput eq) factor)
  pure (TTerm (Terms.lambdas params (unTTerm body)))


-- | Compile an 'Equation' and immediately apply it to input tensors.
--
-- Equivalent to @'compileEquation'@ followed by partial application of
-- @args@, with 'reduceTerm' applied to beta-reduce the result.
-- This is the main entry point for constructing tensor contraction terms
-- to embed in algebras or pass directly to 'writePythonWithBackend'.
applyEquation :: Orientation -> Semiring -> Equation -> [TTerm Tensor] -> Either String (TTerm Tensor)
applyEquation orientation sr eq args = do
  check (length args == length (eqInputs eq)) $
    "expected " <> show (length (eqInputs eq)) <> " tensor arguments, got " <> show (length args)
  compiled <- compileEquation orientation sr eq
  pure $ TTerm $ reduceTerm $ foldl Terms.apply (unTTerm compiled) (fmap unTTerm args)


contractPlan :: Equation -> [Factor] -> Either String (Factor, [(Factor, [Index])])
contractPlan eq factors =
  case factors of
    []     -> Left "equation must have at least one input"
    f : fs -> Right (f, zip fs (fmap liveAfter (drop 2 suffixes)))
  where
    output = eqOutput eq
    live = output ++ eqReduced eq
    suffixes = scanr (Set.union . Set.fromList) Set.empty (fmap fst factors)
    liveAfter later =
      filter (`Set.member` (Set.fromList output `Set.union` later)) live


stepFactor :: String -> String -> Factor -> (Factor, [Index]) -> Either String Factor
stepFactor plusKey timesKey acc (factor, live) =
  reduceFactor plusKey live =<< mulFactor timesKey acc factor


mulFactor :: String -> Factor -> Factor -> Either String Factor
mulFactor timesKey left right = do
  let ctx = unionLabels (fst left) (fst right)
  l <- alignFactor ctx left
  r <- alignFactor ctx right
  pure (ctx, op timesKey @@ snd l @@ snd r)


alignFactor :: [Index] -> Factor -> Either String Factor
alignFactor target (labels, term) = do
  let expanded = unionLabels labels target
      posOf = positions expanded
      axes = [length labels .. length expanded - 1]
      unsqueezed = foldl (\acc ax ->
                     op "structural.expand_dims" @@ acc @@ TTerm (Terms.int32 ax))
                   term axes
  perm <- maybe (Left "target labels must be present after expansion") Right $
    traverse (`Map.lookup` posOf) target
  pure (target, op "structural.transpose" @@ unsqueezed @@
    TTerm (Terms.list (fmap Terms.int32 perm)))


reduceFactor :: String -> [Index] -> Factor -> Either String Factor
reduceFactor plusKey target (labels, term) =
  foldM reduceAxis (labels, term) axes
  where
    posOf = positions labels
    axes = sortOn (negate . fst)
      [(axis, label)
      | label <- nub (filter (`Set.notMember` Set.fromList target) labels)
      , Just axis <- [Map.lookup label posOf]]
    reduceAxis (ls, t) (axis, _) =
      Right
        ( take axis ls ++ drop (axis + 1) ls
        , op (reduceKey plusKey) @@ t @@ TTerm (Terms.int32 axis)
        )


-- ── Module builder ───────────────────────────────────────────────────────────

contractModule :: String -> Semiring -> Module
contractModule defName sr = moduleFromTerm "unialg.tensors" defName [] (contract sr)


-- | Build a codegen-ready 'Module' from a pre-parsed 'Equation'.
equationModule :: String -> String -> [Namespace] -> Semiring -> Equation -> Orientation -> Either String Module
equationModule ns name deps sr eq orient = do
  t <- compileEquation orient sr eq
  pure (moduleFromTerm ns name deps t)


moduleFromTerm :: String -> String -> [Namespace] -> TTerm a -> Module
moduleFromTerm ns name deps t = Module
  { moduleDescription      = Just "Tensor operation"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      =
      [toDefinition (definitionInNamespace (Namespace ns) name t)]
  }


-- ── Utilities ────────────────────────────────────────────────────────────────

note :: e -> Maybe a -> Either e a
note e = maybe (Left e) Right

check :: Bool -> String -> Either String ()
check True  _ = pure ()
check False e = Left e

unionLabels :: [Index] -> [Index] -> [Index]
unionLabels xs ys = xs ++ filter (`Set.notMember` Set.fromList xs) ys

positions :: [Index] -> Map.Map Index Int
positions labels = Map.fromList (zip labels [0..])

reduceKey :: String -> String
reduceKey key = "reduce." <> key

orientationKeys :: Orientation -> Semiring -> Either String (String, String)
orientationKeys Forward sr = Right (semiringPlus sr, semiringTimes sr)
orientationKeys Adjoint sr = adjointKeys sr

adjointKeys :: Semiring -> Either String (String, String)
adjointKeys sr =
  (,)
    <$> note "Adjoint orientation requires adjointPlus"  (semiringAdjointPlus sr)
    <*> note "Adjoint orientation requires adjointTimes" (semiringAdjointTimes sr)
