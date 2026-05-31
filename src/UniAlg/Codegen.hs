{-# LANGUAGE OverloadedStrings #-}

{-|
Python code generation pipeline.

Workflow:

1. Load a backend JSON spec ('loadBackendContext').
2. Lower symbolic @unialg.backend.*@ names to concrete paths ('lowerModule').
3. Inject eta-expanded external module declarations ('backendExternalModules').
4. Call Hydra's Python coder (@Hydra.Python.Coder.moduleToPython@).

For non-recursive definitions use 'writePythonWithBackend' or 'generatePythonTerms'.
For recursive definitions use 'UniAlg.Architecture.hyloModule' or
'UniAlg.Architecture.cataModule' \/ 'UniAlg.Architecture.anaModule'.
'generatePythonString' generates Python in-memory; 'evalPython' pipes it straight to the interpreter.
-}
module UniAlg.Codegen
  ( writePythonWithBackend
  , writePythonWithBackendRec
  , loadBackendAndWritePython
  , loadBackendAndWritePythonRec
  , generatePythonTerms
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
  , Module(..)
  , Name(..)
  , Namespace(..)
  , TermDefinition(..)
  )

import Hydra.Languages
  ( hydraLanguage
  )

import Hydra.Phantoms
  ( TTerm
  , unTTerm
  )

import qualified Hydra.Python.Coder as PythonCoder

import UniAlg.Backend
  ( BackendContext(..)
  , backendContextSpec
  , backendExternalModules
  , loadBackendContext
  , lowerModule
  )


-- Private worker: inferTypes=True for flat definitions, False for recursive ones
-- (recursive modules carry a polyFnScheme annotation so Hydra skips inference).
writePython' :: Bool -> BackendContext -> FilePath -> [Module] -> [Module] -> IO Int
writePython' inferTypes context outputDir universeModules modulesToGenerate =
  generateSources
    PythonCoder.moduleToPython hydraLanguage
    inferTypes True True True
    outputDir fullUniverse adaptedTargets
  where
    spec            = backendContextSpec context
    adaptedUniverse = fmap (lowerModule spec) universeModules
    adaptedTargets  = fmap (lowerModule spec) modulesToGenerate
    fullUniverse    = backendExternalModules spec <> adaptedUniverse


-- | Generate Python files for a set of modules using a loaded backend.
--
-- @universeModules@ are available to the Hydra type system but not generated.
-- @modulesToGenerate@ are lowered and emitted as @.py@ files under @outputDir@.
-- Returns the number of files written.
writePythonWithBackend :: BackendContext -> FilePath -> [Module] -> [Module] -> IO Int
writePythonWithBackend = writePython' True


-- | Like 'writePythonWithBackend' but skips Hydra type inference.
-- Use for modules built with 'hyloModule' or 'hyloDef'.
writePythonWithBackendRec :: BackendContext -> FilePath -> [Module] -> [Module] -> IO Int
writePythonWithBackendRec = writePython' False


-- Private worker: load backend then call writePython'.
loadAndWrite' :: Bool -> FilePath -> Text -> FilePath -> [Module] -> [Module] -> IO (Either String Int)
loadAndWrite' inferTypes backendDir backendName outputDir us ms = do
  loaded <- loadBackendContext backendDir backendName
  case loaded of
    Left err      -> pure (Left err)
    Right context -> Right <$> writePython' inferTypes context outputDir us ms


-- | Load a backend by name and generate Python files.
-- Returns @'Left' err@ if the backend JSON cannot be loaded.
loadBackendAndWritePython :: FilePath -> Text -> FilePath -> [Module] -> [Module] -> IO (Either String Int)
loadBackendAndWritePython = loadAndWrite' True


-- | Like 'loadBackendAndWritePython' but skips Hydra type inference.
-- Use for modules built with 'hyloModule' or 'hyloDef'.
loadBackendAndWritePythonRec :: FilePath -> Text -> FilePath -> [Module] -> [Module] -> IO (Either String Int)
loadBackendAndWritePythonRec = loadAndWrite' False


-- | Generate Python for a flat list of named 'TTerm' definitions.
--
-- Wraps the definitions in a single 'Module' keyed by @moduleName@, lowers
-- backend names using the spec at @backendJson@, and writes @.py@ files
-- under @outputDir@.
generatePythonTerms
  :: FilePath              -- ^ Output directory.
  -> FilePath              -- ^ Path to the backend JSON file.
  -> String                -- ^ Module namespace (e.g. @\"demo\"@).
  -> [(String, TTerm a)]   -- ^ @(localName, term)@ pairs.
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


-- | Generate Python for a module as an in-memory @('FilePath', 'String')@
-- list — no disk writes.
--
-- Returns the same @(relative path, source)@ pairs that
-- @generateSources@ would write.  Useful for testing generated output
-- without touching the filesystem.  Fails in 'IO' on a codegen error.
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


-- | Pipe a generated Python module to the interpreter and evaluate one expression.
--
-- Appends @print(\<expr\>)@ to the module source and feeds the result to
-- @pyBin@ via @stdin@.  Returns @'Right' stdout@ on success or
-- @'Left' stderr@ on a non-zero exit.  Useful for quick numerical
-- correctness checks in tests.
evalPython
  :: FilePath    -- ^ Python executable (e.g. @\"../.venv/bin/python3\"@).
  -> String      -- ^ Generated module source.
  -> String      -- ^ Python expression whose @repr@ to print.
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
