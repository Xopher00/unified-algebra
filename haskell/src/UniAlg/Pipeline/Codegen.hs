{-# LANGUAGE ImplicitParams    #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes        #-}

module UniAlg.Pipeline.Codegen
  ( writePythonWithBackend
  , loadBackendAndWritePython
  , generatePythonTerms
  , recursiveDef
  , recursiveModule
  , recDef
  , recModule
  , evalPython
  , generatePythonString
  ) where

import Data.Text (Text)
import qualified Data.Text as Text
import System.Exit (ExitCode(..))
import System.FilePath
  ( dropExtension
  , takeBaseName
  , takeDirectory
  )
import System.Process (readProcessWithExitCode)

import qualified Data.Map as M
import qualified Hydra.Codegen as CodeGeneration
import qualified Hydra.Context as Context
import Hydra.Dsl.Bootstrap (bootstrapGraph)
import Hydra.Generation
  ( generateSources
  )

import Hydra.Kernel
  ( Definition(..)
  , FunctionType(..)
  , Module(..)
  , Name(..)
  , Namespace(..)
  , TermDefinition(..)
  , Type(..)
  , TypeScheme(..)
  )

import Hydra.Languages
  ( hydraLanguage
  )

import Hydra.Phantoms
  ( TTerm
  , unTTerm
  )

import Hydra.Dsl.Meta.Phantoms
  ( var
  , (~>)
  )

import UniAlg.Semantics.Arrows
  ( reify
  )

import UniAlg.Semantics.Category
  ( tApply
  )

import UniAlg.Semantics.Recursion
  ( withSelf
  )

import qualified Hydra.Python.Coder as PythonCoder

import UniAlg.Pipeline.Backend
  ( BackendContext(..)
  , backendContextSpec
  , loadBackendContext
  )

import UniAlg.Pipeline.Externals
  ( backendExternalModules
  )

import UniAlg.Pipeline.Lowering
  ( lowerModule
  )

import UniAlg.Core.Reduce
  ( reduceTerm
  )


writePythonWithBackend
  :: BackendContext
  -> FilePath
  -> [Module]
  -> [Module]
  -> IO Int
writePythonWithBackend context outputDir universeModules modulesToGenerate =
  generateSources
    PythonCoder.moduleToPython
    hydraLanguage
    True
    True
    True
    True
    outputDir
    fullUniverse
    adaptedTargets
  where
    spec =
      backendContextSpec context

    adaptedUniverse =
      fmap (lowerModule spec) universeModules

    adaptedTargets =
      fmap (lowerModule spec) modulesToGenerate

    fullUniverse =
      backendExternalModules spec <> adaptedUniverse


loadBackendAndWritePython
  :: FilePath
  -> Text
  -> FilePath
  -> [Module]
  -> [Module]
  -> IO (Either String Int)
loadBackendAndWritePython backendDir backendName outputDir universeModules modulesToGenerate = do
  loaded <- loadBackendContext backendDir backendName

  case loaded of
    Left err ->
      pure (Left err)

    Right context ->
      Right <$>
        writePythonWithBackend
          context
          outputDir
          universeModules
          modulesToGenerate


generatePythonTerms
  :: FilePath
  -> FilePath
  -> String
  -> [(String, TTerm a)]
  -> IO ()
generatePythonTerms outputDir backendJson moduleName defs = do
  context <- loadContextFromJson backendJson

  let externals =
        backendExternalModules (backendContextSpec context)

      backendNamespaces =
        fmap moduleNamespace externals

      module_ =
        Module
          { moduleDescription =
              Just ("UniAlg generated module " <> moduleName)
          , moduleNamespace =
              Namespace moduleName
          , moduleTermDependencies =
              backendNamespaces
          , moduleTypeDependencies =
              []
          , moduleDefinitions =
              fmap (definitionFromTTerm moduleName) defs
          }

  _ <-
    writePythonWithBackend
      context
      outputDir
      [module_]
      [module_]

  pure ()


definitionFromTTerm :: String -> (String, TTerm a) -> Definition
definitionFromTTerm moduleName (localName, term) =
  DefinitionTerm $
    TermDefinition
      { termDefinitionName =
          Name (moduleName <> "." <> localName)
      , termDefinitionTerm =
          unTTerm term
      , termDefinitionTypeScheme =
          Nothing
      }


recursiveDef :: String -> String -> TTerm a -> TermDefinition
recursiveDef ns defName body = TermDefinition
  { termDefinitionName       = Name qualName
  , termDefinitionTerm       = reduceTerm (unTTerm body)
  , termDefinitionTypeScheme = Nothing
  }
  where qualName = ns <> "." <> defName


recursiveModule :: String -> String -> [Namespace] -> TTerm a -> Module
recursiveModule ns defName deps body = Module
  { moduleDescription      = Just "Recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (recursiveDef ns defName body)]
  }


-- ── Recursion-scheme aware builders ──────────────────────────────────────────
-- outerArgNames: shared parameters prepended to every recursive self-call.

-- Builds forall _a0 .. _a(n-1). _a0 -> .. -> _a(n-1).
-- Gives recursive definitions an explicit type scheme so Hydra skips
-- inference and avoids the occurs-check on equi-recursive function bodies.
polyFnScheme :: Int -> TypeScheme
polyFnScheme n = TypeScheme
  { typeSchemeVariables   = vars
  , typeSchemeBody        = foldr step ret (init tvars)
  , typeSchemeConstraints = Nothing
  }
  where
    vars  = [Name ("_a" <> show i) | i <- [0 .. n - 1]]
    tvars = fmap TypeVariable vars
    ret   = last tvars
    step a b = TypeFunction FunctionType
                 { functionTypeDomain   = a
                 , functionTypeCodomain = b }
-- These become the leading lambda parameters of the generated Python function.
-- Users write: recModule ns "f" deps ["w"] $ cataT @F myAlg
-- and never touch withSelf, applyAlg, or TTerm layer construction.

recDef :: String -> String -> [String] -> ((?self :: TTerm a) => TTerm a -> TTerm a) -> TermDefinition
recDef ns name outerArgNames body =
  (recursiveDef ns name $
    foldr (~>) innerTerm outerArgNames)
    { termDefinitionTypeScheme = Just (polyFnScheme (length outerArgNames + 2)) }
  where
    appliedSelf = foldl (\s n -> tApply s (var n)) (var (ns <> "." <> name)) outerArgNames
    innerTerm   = "x" ~> withSelf appliedSelf (body (var "x"))


recModule :: String -> String -> [Namespace] -> [String] -> ((?self :: TTerm a) => TTerm a -> TTerm a) -> Module
recModule ns name deps outerArgNames body = Module
  { moduleDescription      = Just "Recursive definition"
  , moduleNamespace        = Namespace ns
  , moduleTermDependencies = deps
  , moduleTypeDependencies = []
  , moduleDefinitions      = [DefinitionTerm (recDef ns name outerArgNames body)]
  }


-- | Generate Python for a module as a string — no disk writes.
-- Returns a map of relative file path to source content (same keys
-- generateSources would write). Fails in IO on a codegen error.
generatePythonString :: [Module] -> Module -> IO ([(FilePath, String)])
generatePythonString universe target =
  let cx = Context.Context [] [] M.empty
  in case CodeGeneration.generateSourceFiles
           PythonCoder.moduleToPython
           hydraLanguage
           False False False False
           bootstrapGraph
           universe
           [target]
           cx of
       Left err    -> fail ("generatePythonString: " <> show err)
       Right pairs -> pure pairs


-- | Pipe generated Python code to the interpreter and evaluate an expression.
-- The expression is appended as @print(<expr>)@ so its repr appears on stdout.
-- Returns @Right stdout@ on success, @Left stderr@ on non-zero exit.
evalPython
  :: FilePath    -- ^ Python executable (e.g. path to venv python3)
  -> String      -- ^ Generated module source (as returned by 'generate')
  -> String      -- ^ Python expression to evaluate
  -> IO (Either String String)
evalPython pyBin code expr = do
  let script = code <> "\nprint(" <> expr <> ")"
  (exitCode, out, err) <- readProcessWithExitCode pyBin ["-"] script
  pure $ case exitCode of
    ExitSuccess   -> Right (dropTrailingNewline out)
    ExitFailure _ -> Left err
  where
    dropTrailingNewline s = reverse (dropWhile (== '\n') (reverse s))


loadContextFromJson :: FilePath -> IO BackendContext
loadContextFromJson backendJson = do
  loaded <-
    loadBackendContext
      (takeDirectory backendJson)
      (Text.pack . dropExtension $ takeBaseName backendJson)

  case loaded of
    Left err ->
      error ("Could not load backend context: " <> err)

    Right context ->
      pure context
