{-# LANGUAGE TemplateHaskell #-}

{-|
Template Haskell generator for backend op bindings.

Reads a backend JSON spec at compile time and emits:

* A typed binding for every op with a known arity (1 or more).
  The binding is eta-expanded to the required arity.
* An @opRegistry :: Map String Term@ of bare, unapplied base terms keyed by
  the raw spec key, for use by the tensor contraction compiler.

=== Stage restriction

This module must be compiled before the splice site.
-}
module UniAlg.Backend.Generate
  ( opBase
  , genBackendOps
  , lookupOp
  ) where

import Control.Monad (replicateM)
import qualified Data.ByteString.Lazy as BL
import Data.Char (toUpper)
import Data.List (foldl')
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T

import Language.Haskell.TH
import Language.Haskell.TH.Syntax (addDependentFile)

import Data.Aeson (eitherDecode)

import Hydra.Kernel (Name(..), Term(..))
import Hydra.Phantoms (TTerm(..))
import Hydra.Dsl.Meta.Phantoms ((@@), var)

import UniAlg.Backend.Schema


-- | Build a base symbolic 'TTerm' for a backend op.
-- The name @unialg.backend.\<key\>@ is resolved to a backend path by
-- the lowering pass.
opBase :: String -> TTerm a
opBase key = var ("unialg.backend." <> key)

-- | Generate arity-typed op bindings and an @opRegistry@ from a backend JSON spec.
--
-- Reads @jsonPath@ at compile time, tracks it as a dependency (so changing
-- the JSON triggers recompilation), and emits:
--
-- * @name :: TTerm a -> … -> TTerm a@ + @name = opBase "key" @@ arg1 @@ ... @@ argN@
--   for each op with arity 1 or more. Dotted keys (e.g. @reduce.*@, @structural.*@)
--   are mangled to camelCase.  Reserved words and common Prelude clashes get a
--   trailing @_@.
--
-- * @opRegistry :: Map String Term@ mapping every raw key to its bare
--   'opBase' term.  Used by the contraction compiler for runtime key lookup.
genBackendOps :: FilePath -> Q [Dec]
genBackendOps jsonPath = do
  addDependentFile jsonPath
  content <- runIO (BL.readFile jsonPath)
  spec <- case eitherDecode content of
    Left err -> fail ("genBackendOps: failed to parse " <> jsonPath <> ": " <> err)
    Right s  -> return (s :: BackendSpec)
  let entries = Map.toList (ops spec)
  staticDecls <- fmap concat (mapM genBinding entries)
  regDecls    <- genRegistryDecl entries
  return (staticDecls <> regDecls)


genBinding :: (Text, OpSpec) -> Q [Dec]
genBinding (rawKey, opSpec) =
  case arity opSpec of
    Just n | n >= 1 -> do
      let key    = T.unpack rawKey
          ident  = mangleKey key
          nm     = mkName ident
          keyLit = litE (stringL key)
      argNames <- replicateM n (newName "x")
      let body = foldl (\e arg -> [| $e @@ $arg |])
                       [| opBase $(keyLit) |]
                       (map varE argNames)
      sig <- sigD nm (buildArrowType n)
      def <- funD nm [clause (map varP argNames) (normalB body) []]
      return [sig, def]
    _ -> do
      reportWarning
        ("genBackendOps: skipping '" <> T.unpack rawKey
         <> "' (arity missing or 0)")
      return []


-- Build  forall a. TTerm a -> … -> TTerm a  with @n@ arrows.
buildArrowType :: Int -> Q Type
buildArrowType n = do
  a <- newName "a"
  let ttermA  = AppT (ConT ''TTerm) (VarT a)
      tterms  = replicate (n + 1) ttermA
      body    = foldr1 (AppT . AppT ArrowT) tterms
  return $ ForallT [PlainTV a SpecifiedSpec] [] body


genRegistryDecl :: [(Text, OpSpec)] -> Q [Dec]
genRegistryDecl entries = do
  let keyStrs = map (T.unpack . fst) entries
      pairFor k = tupE
        [ litE (stringL k)
        , [| TermVariable (Name ("unialg.backend." <> $(litE (stringL k)))) |]
        ]
      listExpr = listE (map pairFor keyStrs)
  sig <- sigD (mkName "opRegistry") [t| Map String Term |]
  def <- valD (varP (mkName "opRegistry")) (normalB [| Map.fromList $(listExpr) |]) []
  return [sig, def]


-- ── Name mangling ─────────────────────────────────────────────────────────────

-- | Mangle a raw backend key (which may contain @.@ or @_@) into a valid
-- lowercase camelCase Haskell identifier.  Appends @_@ to Haskell reserved
-- words and common Prelude function clashes.
mangleKey :: String -> String
mangleKey key = protect camelCased
  where
    segments   = splitOnAny "._" key
    camelCased = case segments of
      []     -> "_op"
      (s:ss) -> toLowerFirst s <> foldl' (\acc seg -> acc <> capitalize seg) "" ss

    capitalize []     = []
    capitalize (c:cs) = toUpper c : cs

    toLowerFirst []     = []
    toLowerFirst (c:cs) = c : cs  -- first segment stays as-is (already lowercase in JSON)

    protect s
      | s `elem` reserved = s <> "_"
      | otherwise         = s

    reserved =
      [ "where", "let", "in", "do", "of", "case", "if", "then", "else"
      , "module", "import", "type", "data", "class", "instance", "deriving"
      , "newtype", "infixl", "infixr", "infix", "default", "foreign"
      , "and", "or", "not"  -- Prelude function clashes
      ]


splitOnAny :: String -> String -> [String]
splitOnAny _    []  = [""]
splitOnAny seps str =
  case break (`elem` seps) str of
    (chunk, [])     -> [chunk]
    (chunk, _:rest) -> chunk : splitOnAny seps rest


-- | Resolve a raw backend op key against a registry produced by a
-- 'genBackendOps' splice. Errors at construction time on unknown keys.
lookupOp :: Map String Term -> String -> TTerm a
lookupOp registry key =
  TTerm $ Map.findWithDefault
    (error ("lookupOp: unknown backend op key " <> show key))
    key
    registry
