# Running the Haskell tests

## Requirements

- GHC 9.8–9.12 (tested on 9.10.2). The system GHC is too old; use ghcup.
- cabal 3.14+
- Python venv at `../.venv/` relative to this repo's `haskell/` directory (i.e. `unialg/.venv/`), with `numpy` and `tensorflow` installed.

Install GHC via ghcup if needed:

```
ghcup install ghc 9.10.2
ghcup set ghc 9.10.2
```

## Running the tests

**Always run from the `haskell/` directory**, not the repo root. Cabal resolves backend JSON paths relative to the package root.

```bash
cd haskell/
cabal test --with-compiler=ghc-9.10.2 -j1
```

The `-j1` flag is required. Parallel builds cause an internal library (`test-utils`) to be missing from the package database when test executables try to link against it.

## Path assumptions

All file paths in tests are relative to `haskell/`:

- Backend specs: `backends/numpy.json` etc.
- Python executable: `../.venv/bin/python3`
- Generated output: `/tmp/unialg-neural-*/` (created at test time)
- Comparison script: `test/neural_comparison.py`

Do not use absolute paths. Do not run `cabal test` from the repo root or a subdirectory of `haskell/`.

## First run

The first run compiles hydra from source (`hydra-src/hydra-0.15.0`), which takes several minutes. Subsequent runs are fast.
