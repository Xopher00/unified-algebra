{-# LANGUAGE OverloadedStrings #-}

module UniAlg.Domain.Tensors
  ( Tensor
  , Index(..)
  , Orientation(..)
  , Semiring(..)
  , Equation(..)
  , AlignmentPlan(..)
  , FusionTree
  , parseEquation
  , fuseEquation
  , fusionLeaf
  , fusionNode
  , fuseTree
  , alignmentPlan
  , alignmentPlans
  , reducedAxes
  , targetVars
  , backendOp
  , reduceOp
  , structuralOp
  , op1
  , op2
  , op3
  , contract
  , adjointContract
  , compileEquation
  , applyEquation
  , contractModule
  , tensorOp
  , equationModule
  , tensorOpModule
  ) where

import Data.List (elemIndex, isInfixOf, nub)
import Data.Maybe (fromMaybe)

import Hydra.Kernel
  ( Definition(..)
  , Module(..)
  , Namespace(..)
  )

import Hydra.Phantoms
  ( TTerm(..)
  )

import Hydra.Dsl.Meta.Phantoms
  ( var
  , (@@)
  , definitionInNamespace
  , toDefinition
  )

import UniAlg.Semantics.Arrows
  ( reify2
  )

import qualified Hydra.Dsl.Terms as Terms

import UniAlg.Core.Reduce
  ( reduceTerm
  )

import UniAlg.Semantics.Functors
  ( Const(..)
  , Product(..)
  , RoseF
  )

import UniAlg.Semantics.Recursion
  ( Fix(..)
  , cata
  )


data Tensor


newtype Index = Index Char
  deriving (Eq, Ord, Show)


data Orientation
  = Forward
  | Adjoint
  deriving (Eq, Show)


data Semiring = Semiring
  { semiringPlus    :: String
  , semiringTimes   :: String
  , semiringAdjoint :: Maybe String
  } deriving (Eq, Show)


-- ── Equation ─────────────────────────────────────────────────────────────────

data Equation = Equation
  { eqInputs  :: [[Index]]
  , eqOutput  :: [Index]
  , eqReduced :: [Index]
  } deriving (Eq, Show)


data AlignmentPlan = AlignmentPlan
  { unsqueezeAxes :: [Int]
  , perm          :: [Int]
  } deriving (Eq, Show)


parseEquation :: String -> Either String Equation
parseEquation raw = do
  (lhs, rhs) <- splitArrow (filter (/= ' ') raw)
  let inputs = (fmap . fmap) Index (splitOn ',' lhs)
      output = fmap Index rhs
      allIn  = nub (concat inputs)
  check (nub output == output) "output labels must be unique"
  check (all (`elem` allIn) output) "output labels not in any input"
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = output
    , eqReduced = filter (`notElem` output) allIn
    }


splitArrow :: String -> Either String (String, String)
splitArrow s
  | "->" `isInfixOf` s = Right (a, drop 2 b)
  | otherwise          = Left "equation must contain '->'"
  where (a, b) = break (== '-') s


check :: Bool -> String -> Either String ()
check True  _ = pure ()
check False e = Left e


targetVars :: Equation -> [Index]
targetVars eq = eqOutput eq ++ eqReduced eq


fuseEquation :: Equation -> Int -> Equation -> Either String Equation
fuseEquation outer slot inner = do
  check (slot >= 0 && slot < length (eqInputs outer))
    ("slot " <> show slot <> " out of range for equation with "
      <> show (length (eqInputs outer)) <> " inputs")
  check (eqOutput inner == eqInputs outer !! slot)
    "inner output labels must match the replaced outer input"
  let inputs  = take slot (eqInputs outer)
             ++ eqInputs inner
             ++ drop (slot + 1) (eqInputs outer)
      allIn   = nub (concat inputs)
      outSet  = eqOutput outer
      reduced = filter (`notElem` outSet) $
                  nub (eqReduced inner ++ eqReduced outer)
  pure Equation
    { eqInputs  = inputs
    , eqOutput  = outSet
    , eqReduced = filter (`elem` allIn) reduced
    }


-- ── Fusion tree ──────────────────────────────────────────────────────────────
-- RoseF (Const Equation) = Product (Const Equation) []
-- Every node carries an Equation + list of child sub-trees.
-- A leaf is a node with no children.
-- Catamorphism fuses children into parent slots at build time;
-- the result is a flat Equation passed to compileEquation.

type FusionTree = Fix (RoseF (Const Equation))


fusionLeaf :: Equation -> FusionTree
fusionLeaf eq = Fix (Pair (Const eq) [])


fusionNode :: Equation -> [FusionTree] -> FusionTree
fusionNode eq children = Fix (Pair (Const eq) children)


fuseTree :: FusionTree -> Either String Equation
fuseTree = cata algebra
  where
    algebra :: RoseF (Const Equation) (Either String Equation) -> Either String Equation
    algebra (Pair (Const eq) childResults) =
      -- Highest slot first: keeps lower slot indices stable as each fusion
      -- prepends inner inputs at the targeted position.
      foldr step (Right eq) (reverse (zip [0..] childResults))

    step :: (Int, Either String Equation) -> Either String Equation -> Either String Equation
    step (slot, child) acc = do
      a <- acc
      c <- child
      fuseEquation a slot c


alignmentPlan :: Equation -> Int -> AlignmentPlan
alignmentPlan eq i = AlignmentPlan
  { unsqueezeAxes = [length inp .. length expanded - 1]
  , perm          = fmap (\v -> fromMaybe 0 (elemIndex v expanded)) target
  }
  where
    inp      = eqInputs eq !! i
    target   = targetVars eq
    existing = nub inp
    expanded = inp ++ filter (`notElem` existing) target


alignmentPlans :: Equation -> [AlignmentPlan]
alignmentPlans eq = fmap (alignmentPlan eq) [0 .. length (eqInputs eq) - 1]


reducedAxes :: Equation -> [Int]
reducedAxes eq =
  [length (eqOutput eq) .. length (targetVars eq) - 1]


-- ── Backend op references ────────────────────────────────────────────────────

backendOp :: String -> TTerm a
backendOp key = var ("unialg.backend." <> key)


reduceOp :: String -> TTerm a
reduceOp key = var ("unialg.backend.reduce." <> key)


structuralOp :: String -> TTerm a
structuralOp key = var ("unialg.backend.structural." <> key)


-- | Lift a named backend op to a unary Haskell function.
op1 :: String -> TTerm Tensor -> TTerm Tensor
op1 key a = backendOp key @@ a

-- | Lift a named backend op to a binary Haskell function.
op2 :: String -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
op2 key a b = backendOp key @@ a @@ b

-- | Lift a named backend op to a ternary Haskell function.
op3 :: String -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor -> TTerm Tensor
op3 key a b c = backendOp key @@ a @@ b @@ c


-- ── Simple contractions (no equation, arity-inferred reduce) ─────────────────

contract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
contract sr =
  reify2 $ \x y ->
    reduceOp (semiringPlus sr) @@ op2 (semiringTimes sr) x y


adjointContract :: Semiring -> TTerm (Tensor -> Tensor -> Tensor)
adjointContract sr =
  case semiringAdjoint sr of
    Nothing  -> error "adjointContract: semiring has no adjoint"
    Just adj -> reify2 $ \x y ->
      reduceOp (semiringTimes sr) @@ op2 adj x y


-- ── Equation-aware contractions ──────────────────────────────────────────────

align :: AlignmentPlan -> TTerm a -> TTerm a
align plan t = transpose_ (unsqueeze_ t)
  where
    unsqueeze_ t' =
      foldl (\acc ax -> structuralOp "expand_dims" @@ acc @@ TTerm (Terms.int32 ax))
            t' (unsqueezeAxes plan)
    transpose_ t' =
      structuralOp "transpose" @@ t' @@ TTerm (Terms.list (fmap Terms.int32 (perm plan)))


orientedOps :: Orientation -> Semiring -> Either String (String, String)
orientedOps Forward sr = pure (semiringTimes sr, "reduce." <> semiringPlus sr)
orientedOps Adjoint sr =
  case semiringAdjoint sr of
    Nothing  -> Left "Adjoint orientation requires a semiring with an adjoint"
    Just adj -> pure (adj, "reduce." <> semiringTimes sr)


compileEquation :: Orientation -> Semiring -> Equation -> Either String (TTerm a)
compileEquation orientation sr eq = do
  (timesKey, reduceKey) <- orientedOps orientation sr
  let params   = ["t" <> show i | i <- [0 .. length (eqInputs eq) - 1]]
      aligned  = zipWith (\plan p -> align plan (var p)) (alignmentPlans eq) params
      product_ = foldl1 (\a b -> backendOp timesKey @@ a @@ b) aligned
      body     = foldl (\acc ax -> backendOp reduceKey @@ acc @@ TTerm (Terms.int32 ax))
                       product_ (reducedAxes eq)
  pure (TTerm (Terms.lambdas params (unTTerm body)))


applyEquation :: Orientation -> Semiring -> Equation -> [TTerm Tensor] -> Either String (TTerm Tensor)
applyEquation orientation sr eq args = do
  compiled <- compileEquation orientation sr eq
  pure $ TTerm $ reduceTerm $ foldl Terms.apply (unTTerm compiled) (fmap unTTerm args)


-- ── Module builder ───────────────────────────────────────────────────────────

contractModule :: String -> Semiring -> Module
contractModule defName sr = moduleFromTerm "unialg.tensors" defName [] (contract sr)


-- | Compile a tensor equation to a composable TTerm value.
-- The result can be applied with (@@), passed as a leaf into algebras,
-- or composed with other TTerm functions.
tensorOp :: Semiring -> String -> Orientation -> Either String (TTerm a)
tensorOp sr eqStr orient = do
  eq <- parseEquation eqStr
  compileEquation orient sr eq


-- | Build a codegen-ready Module from a pre-parsed Equation.
equationModule :: String -> String -> [Namespace] -> Semiring -> Equation -> Orientation -> Either String Module
equationModule ns name deps sr eq orient = do
  t <- compileEquation orient sr eq
  pure (moduleFromTerm ns name deps t)


-- | Build a codegen-ready Module from a tensor equation string.
tensorOpModule :: String -> String -> [Namespace] -> Semiring -> String -> Orientation -> Either String Module
tensorOpModule ns name deps sr eqStr orient = do
  t <- tensorOp sr eqStr orient
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

splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn delim s =
  case break (== delim) s of
    (chunk, [])     -> [chunk]
    (chunk, _:rest) -> chunk : splitOn delim rest
