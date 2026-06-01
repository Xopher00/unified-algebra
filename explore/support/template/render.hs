{-# LANGUAGE OverloadedStrings #-}

-- Renders an ArchSpec JSON file into explore/archs/<label>/<Module>.hs
-- and updates unialg.cabal.
--
-- Usage: cabal run explore-render -- <spec-file.json>

module Main (main) where

import ArchSpec

import Data.Aeson         (eitherDecodeFileStrict)
import Data.Char          (toLower, toUpper)
import Data.List          (intercalate, isPrefixOf, nubBy, sort)
import qualified Data.Map.Strict as Map
import System.Directory   (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.Exit        (die)
import System.FilePath    ((</>))
import System.IO          (openFile, IOMode(..), hGetContents, hClose)


-- ── entry point ──────────────────────────────────────────────────────────────

main :: IO ()
main = do
  args <- getArgs
  specFile <- case args of
    [f] -> pure f
    _   -> die "Usage: explore-render <spec.json>"
  result <- eitherDecodeFileStrict specFile
  spec <- case result of
    Left  e -> die ("JSON parse error: " ++ e)
    Right s -> pure s
  let label     = specLabel spec
      modName   = toPascalCase label
      archDir   = "explore" </> "archs" </> label
      hsFile    = archDir </> (modName ++ ".hs")
  createDirectoryIfMissing True archDir
  writeFile hsFile (renderHs spec modName)
  putStrLn ("Wrote " ++ hsFile)
  updateCabal label modName
  putStrLn "Updated unialg.cabal"
  putStrLn "Run: runghc explore/gen-catalogue.hs"


-- ── cabal editing ─────────────────────────────────────────────────────────────

updateCabal :: String -> String -> IO ()
updateCabal label modName = do
  h   <- openFile "unialg.cabal" ReadMode
  src <- hGetContents h
  let ls  = lines src
      ls' = insertSrcDir label (insertModule modName ls)
      out = unlines ls'
  length out `seq` hClose h
  writeFile "unialg.cabal" out

-- Insert "    explore/archs/<label>" into hs-source-dirs block of library explore.
insertSrcDir :: String -> [String] -> [String]
insertSrcDir label = go False
  where
    newEntry = "    explore/archs/" ++ label
    go _ [] = []
    go inExplore (l:ls)
      | "library explore" `isPrefixOf` l       = l : go True ls
      | inExplore && "  hs-source-dirs:" `isPrefixOf` l =
          let srcLines  = takeWhile isSrcLine ls
              restLines = dropWhile isSrcLine ls
          in l : insertAlpha newEntry srcLines ++ go False restLines
      | otherwise = l : go inExplore ls
    isSrcLine l = "    explore/" `isPrefixOf` l

-- Insert into exposed-modules block of library explore.
insertModule :: String -> [String] -> [String]
insertModule modName = go False
  where
    go _ [] = []
    go inExplore (l:ls)
      | "library explore" `isPrefixOf` l         = l : go True ls
      | inExplore && "  exposed-modules:" `isPrefixOf` l =
          let modLines  = takeWhile isModLine ls
              restLines = dropWhile isModLine ls
          in l : insertAlpha ("    " ++ modName) modLines ++ go False restLines
      | otherwise = l : go inExplore ls
    isModLine l = "    " `isPrefixOf` l && not ("build-depends" `isPrefixOf` (dropWhile (== ' ') l))

insertAlpha :: String -> [String] -> [String]
insertAlpha new existing
  | new `elem` existing = existing
  | otherwise           = sort (new : existing)


-- ── name utilities ────────────────────────────────────────────────────────────

toPascalCase :: String -> String
toPascalCase s = concatMap capitalise (splitOn '_' s)
  where capitalise []     = []
        capitalise (c:cs) = toUpper c : cs

splitOn :: Char -> String -> [String]
splitOn _ "" = [""]
splitOn c (x:xs)
  | x == c    = "" : splitOn c xs
  | otherwise = let (w:ws) = splitOn c xs in (x:w) : ws

snakeLabel :: String -> String
snakeLabel = map toLower

seedName :: String -> String
seedName label = toCamelCase label
  where
    toCamelCase s = case splitOn '_' s of
      []     -> ""
      (w:ws) -> map toLower w ++ concatMap capitalise ws
    capitalise []     = []
    capitalise (c:cs) = toUpper c : cs


-- ── PolyF rendering ───────────────────────────────────────────────────────────

-- Value-level PolyF expression (Grammar.PolyF constructors).
pfVal :: PolyFSpec -> String
pfVal PFUnit          = "KUnit"
pfVal PFConst         = "KConst"
pfVal PFHole          = "Hole"
pfVal (PFSum     a b) = "(" ++ pfVal a ++ " :+: " ++ pfVal b ++ ")"
pfVal (PFProduct a b) = "(" ++ pfVal a ++ " :*: " ++ pfVal b ++ ")"
pfVal (PFExp     a)   = "ExpF " ++ pfVal a

-- Type-level functor expression for the TypeApplications @ site.
pfType :: PolyFSpec -> String
pfType PFUnit          = "Const ()"
pfType PFConst         = "Const (TTerm Tensor)"
pfType PFHole          = "Identity"
pfType (PFSum     a b) = "Sum ("     ++ pfType a ++ ") (" ++ pfType b ++ ")"
pfType (PFProduct a b) = "Product (" ++ pfType a ++ ") (" ++ pfType b ++ ")"
pfType (PFExp     _)   = "Exp (TTerm Tensor)"


-- ── semiring rendering ────────────────────────────────────────────────────────

srVal :: SemiringSpec -> String
srVal sr =
  "Semiring " ++ show (srAdd sr) ++ " " ++ show (srMul sr) ++ " " ++
  case srDiv sr of
    Nothing -> "Nothing"
    Just d  -> "(Just " ++ show d ++ ")"

srName :: SemiringSpec -> String
srName = srLabel

-- Collect all semirings used in the cell body.
usedSemirings :: CellBody -> [SemiringSpec]
usedSemirings body =
  nubBy (\a b -> srLabel a == srLabel b)
    [ ceSemiring e | Binding _ e <- cellBindings body, isCEContraction e ]
  where isCEContraction CEContraction{} = True
        isCEContraction _               = False


-- ── expression rendering ──────────────────────────────────────────────────────

renderExpr :: CellExprSpec -> String
renderExpr (CEContraction sr eq args) =
  "contraction " ++ srName sr ++ " \"" ++ eq ++ "\" " ++ unwords args
renderExpr (CEElemOp op args) =
  case args of
    [a, b] -> op ++ " " ++ a ++ " " ++ b
    _      -> op ++ " " ++ unwords args
renderExpr (CEActivation kind arg) =
  kind ++ " " ++ arg


-- ── module rendering ──────────────────────────────────────────────────────────

renderHs :: ArchSpec -> String -> String
renderHs spec modName =
  unlines $
    [ "{-# LANGUAGE OverloadedStrings #-}"
    , "{-# LANGUAGE TypeApplications  #-}"
    , ""
    , "module " ++ modName
    , "  ( " ++ sn
    , "  , backendSeeds"
    , "  ) where"
    , ""
    , "import Prelude hiding (tanh, sigmoid)"
    , "import Hydra.Kernel (Module(..))"
    , "import UniAlg"
    , ""
    , "import Grammar (PolyF(..))"
    , "import Seed (SeedEntry(..), ArchClass(..), contraction)"
    , ""
    ]
    ++ srDecls
    ++ [ ""
       , sn ++ " :: SeedEntry"
       , sn ++ " = SeedEntry " ++ show (specLabel spec)
               ++ " " ++ archClassStr
               ++ " " ++ pfVal polyF ++ " $"
       , "  " ++ moduleCall ++ " @(" ++ pfType polyF ++ ")"
       , "    " ++ show ("seed." ++ snakeLabel (specLabel spec))
               ++ " " ++ show (snakeLabel (specLabel spec) ++ "_step")
       , "    [Namespace \"numpy\"] " ++ show params ++ " $ \\" ++ paramsList ++ " ->"
       ]
    ++ bodyLines
    ++ [ ""
       , "backendSeeds :: [(String, SeedEntry)]"
       , "backendSeeds ="
       , "  [ (\"numpy\",       " ++ sn ++ ")"
       , "  , (\"tensorflow\",  " ++ sn ++ ")"
       , "  , (\"torch\",       " ++ sn ++ ")"
       , "  ]"
       ]
  where
    sn       = seedName (specLabel spec)
    body     = specCell spec
    polyF    = archPolyF (specArch spec)
    params   = cellParams body
    paramsList = "[" ++ intercalate ", " params ++ "]"
    bmap     = Map.fromList [(bindName b, bindExpr b) | b <- cellBindings body]
    srs      = usedSemirings body
    srDecls  = [ line | sr <- srs
                      , line <- [ srName sr ++ " :: Semiring"
                                , srName sr ++ " = " ++ srVal sr ] ]

    archClassStr = case archClass (specArch spec) of
      CataSpec        -> "CataArch"
      AnaSpec         -> "AnaArch"
      HyloSpec        -> "HyloArch"
      NoStructureSpec -> "NoStructure"

    moduleCall = case archClass (specArch spec) of
      CataSpec        -> "cataModule"
      AnaSpec         -> "anaModule"
      HyloSpec        -> "hyloModule"
      NoStructureSpec -> "anaModule"  -- fallback; pure arches use simplest available

    bodyLines = case cellResult body of
      ResAna sv iv outBs out nsBs ns ->
        [ "      \\" ++ sv ++ " ->"
        , renderLetIn bmap outBs
            ("( " ++ out)
            "        "
        , "        , \\" ++ iv ++ " ->"
        , renderLetIn bmap nsBs ns "            "
        , "        )"
        ]
      ResCataConst base stepVs stepBs stepRes ->
        [ "      ( " ++ base
        , "      , \\" ++ unwords stepVs ++ " ->"
        , renderLetIn bmap stepBs stepRes "          "
        , "      )"
        ]
      ResCataFn bv baseBs base stepVs stepBs stepRes ->
        [ "      ( \\" ++ bv ++ " ->"
        , renderLetIn bmap baseBs base "          "
        , "      , \\" ++ unwords stepVs ++ " ->"
        , renderLetIn bmap stepBs stepRes "          "
        , "      )"
        ]
      ResPure iv pureBs res ->
        [ "      \\" ++ iv ++ " ->"
        , renderLetIn bmap pureBs res "        "
        ]

renderLetIn :: Map.Map String CellExprSpec -> [String] -> String -> String -> String
renderLetIn bmap []    result indent = indent ++ result
renderLetIn bmap names result indent =
  indent ++ "let " ++ firstName ++ " = " ++ lookupExpr (head names) ++ "\n"
  ++ concatMap (\n -> indent ++ "    " ++ n ++ " = " ++ lookupExpr n ++ "\n") (tail names)
  ++ indent ++ "in  " ++ result
  where
    lookupExpr n = case Map.lookup n bmap of
      Just e  -> renderExpr e
      Nothing -> n
    firstName = head names
