#!/usr/bin/env python3
"""
run_one.py — one iteration of the catalog pipeline (Arm A).

Usage:
    UV_CACHE_DIR=$TMPDIR/uv-cache uv run python explore/support/agents/run_one.py

Mechanical steps (shell): get candidate, introspect class, run renderers,
run tests, append row, mark state, commit.
LLM calls: Phase 1 (derive spec), Phase 3 failure classification only.
"""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


# ── paths ─────────────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parents[3]
AGENTS   = Path(__file__).parent
TEMPLATE = REPO / "explore/support/template"
SPECS    = TEMPLATE / "specs"
RENDER_BIN = (
    REPO / "dist-newstyle/build/x86_64-linux/ghc-9.10.2"
    / "unialg-0.1.0.0/x/explore-render/build/explore-render/explore-render"
)
CABAL_FILE = REPO / "unialg.cabal"

UV_ENV = {**os.environ,
          "UV_CACHE_DIR": os.environ.get("TMPDIR", "/tmp") + "/uv-cache"}

PHASE1_AGENT  = "catalog-phase1"
PHASE3_AGENT  = "catalog-phase3-classifier"
MAX_FIX_ATTEMPTS = 2

# ── shell helper ──────────────────────────────────────────────────────────────

def sh(cmd: list[str], *, cwd: Path = REPO) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, env=UV_ENV, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

# ── LLM helper ────────────────────────────────────────────────────────────────

def llm(agent: str, user: str) -> str:
    """Call Claude via CLI using existing OAuth credentials and a named agent."""
    r = subprocess.run(
        ["claude", "--print", "--agent", agent, user],
        capture_output=True, text=True, cwd=REPO,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI error (agent={agent}): {r.stderr[:500]}")
    return r.stdout.strip()

def extract_json(text: str) -> dict | None:
    m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None

# ── class introspection (mechanical) ─────────────────────────────────────────

_INTROSPECT = """
import json, inspect, sys
name = {name!r}
parts = name.split(".")
try:
    if parts[0] == "torch":
        import torch.nn as nn
        cls = getattr(nn, parts[-1])
        call_attr = "forward"
    else:
        import tensorflow as tf
        cls = getattr(tf.keras.layers, parts[-1])
        call_attr = "call"
    init_sig = str(inspect.signature(cls.__init__))
    call_sig = str(inspect.signature(getattr(cls, call_attr)))
    doc = (cls.__doc__ or "").strip().split("\\n\\n")[0].strip()[:800]
    print(json.dumps({{"init": init_sig, "call": call_sig, "doc": doc}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
"""

def introspect(class_name: str) -> dict:
    rc, out, err = sh(["uv", "run", "python", "-c",
                       _INTROSPECT.format(name=class_name)])
    if rc != 0:
        raise RuntimeError(f"introspect failed for {class_name}: {err.strip()}")
    return json.loads(out)

# ── Phase 1: LLM derives spec ─────────────────────────────────────────────────

def phase1(class_name: str) -> tuple[str | None, dict | None]:
    """Return (label, spec) or (None, None) on skip."""
    print(f"[phase1] introspecting {class_name}")
    info = introspect(class_name)

    user = (
        f"CLASS: {class_name}\n\n"
        f"Init signature: {info['init']}\n"
        f"Call signature: {info['call']}\n"
        f"Docstring: {info['doc']}\n\n"
        "Return either:\n"
        "  SKIP: <one-line reason>\n"
        "or a complete ArchSpec JSON object in a ```json block."
    )

    print(f"[phase1] calling LLM (agent={PHASE1_AGENT})")
    response = llm(PHASE1_AGENT, user)

    first_line = response.strip().splitlines()[0] if response.strip() else ""
    if first_line.upper().startswith("SKIP"):
        reason = first_line.split(":", 1)[-1].strip()
        print(f"[phase1] skip: {reason}")
        return None, None

    spec = extract_json(response)
    if spec is None:
        print("[phase1] ERROR: could not extract JSON from response")
        print(response[:600])
        return None, None

    label = spec.get("label")
    if not label:
        print("[phase1] ERROR: spec missing label field")
        return None, None

    print(f"[phase1] derived spec: {label}")
    return label, spec

# ── Phase 2: mechanical renderers ─────────────────────────────────────────────

def phase2(spec_path: Path, label: str) -> bool:
    """Render Haskell + Python, regenerate catalogue, build. True = success."""

    if not RENDER_BIN.exists():
        print("[phase2] building explore-render binary first")
        rc, _, err = sh(["cabal", "build", "explore-render"])
        if rc != 0:
            print(f"[phase2] cabal build explore-render FAILED:\n{err[-1000:]}")
            return False

    # Save cabal snapshot so we can roll back on failure
    cabal_snapshot = CABAL_FILE.read_text()

    rc, out, err = sh([str(RENDER_BIN), str(spec_path)])
    if rc != 0:
        print(f"[phase2] render.hs FAILED:\n{err}")
        return False
    print(f"[phase2] {out.strip()}")

    rc, out, err = sh(
        ["uv", "run", "python", str(TEMPLATE / "render_py.py"), str(spec_path)],
        cwd=REPO,
    )
    if rc != 0:
        print(f"[phase2] render_py.py FAILED:\n{err}")
        _rollback(cabal_snapshot, label)
        return False
    print(f"[phase2] {out.strip()}")

    rc, out, err = sh(["uv", "run", "runghc", "explore/gen-catalogue.hs"])
    if rc != 0:
        print(f"[phase2] gen-catalogue FAILED:\n{err}")
        _rollback(cabal_snapshot, label)
        return False
    print(f"[phase2] {out.strip()}")

    rc, _, err = sh(["cabal", "build", "explore"])
    if rc != 0:
        print(f"[phase2] cabal build FAILED (template gap):\n{err[-2000:]}")
        _rollback(cabal_snapshot, label)
        return False
    print("[phase2] cabal build OK")
    return True


def _rollback(cabal_snapshot: str, label: str) -> None:
    """Undo a failed render: restore cabal, remove generated arch dir."""
    CABAL_FILE.write_text(cabal_snapshot)
    arch_dir = REPO / f"explore/archs/{label}"
    if arch_dir.exists():
        shutil.rmtree(arch_dir)
    print(f"[phase2] rolled back cabal and removed {arch_dir.name}/")

# ── Phase 3: test + optional LLM fix ─────────────────────────────────────────

def phase3(spec_path: Path, label: str) -> bool:
    """Run tests. On failure call LLM classifier; retry up to MAX_FIX_ATTEMPTS."""

    # cabal test generates Hydra Python that pytest depends on
    rc, _, err = sh(["cabal", "test", "explore-test"])
    if rc != 0:
        print(f"[phase3] cabal test FAILED:\n{err[-800:]}")
        return False

    arch_dir = REPO / f"explore/archs/{label}"
    rc, out, err = sh(["uv", "run", "pytest", str(arch_dir), "-v", "--tb=short"])
    test_output = out + err

    if rc == 0:
        print("[phase3] pytest PASSED")
        return True

    print("[phase3] pytest FAILED — invoking classifier")

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        spec = json.loads(spec_path.read_text())
        user = (
            f"LABEL: {label}\n\n"
            f"SPEC cell block:\n{json.dumps(spec['cell'], indent=2)}\n\n"
            f"TEST OUTPUT (last 3000 chars):\n{test_output[-3000:]}\n\n"
            "Return either:\n"
            "  ESCALATE: <reason>\n"
            "or a corrected cell object in a ```json block."
        )

        print(f"[phase3] fix attempt {attempt}/{MAX_FIX_ATTEMPTS} (agent={PHASE3_AGENT})")
        response = llm(PHASE3_AGENT, user)

        first_line = response.strip().splitlines()[0] if response.strip() else ""
        if first_line.upper().startswith("ESCALATE"):
            print(f"[phase3] escalate: {first_line}")
            return False

        fixed_cell = extract_json(response)
        if fixed_cell is None:
            print("[phase3] could not extract fix — escalating")
            return False

        # Apply fix, re-render, re-test
        spec["cell"] = fixed_cell
        spec_path.write_text(json.dumps(spec, indent=2))

        if not phase2(spec_path, label):
            print("[phase3] re-render failed after fix — escalating")
            return False

        rc, out, err = sh(["uv", "run", "pytest", str(arch_dir), "-v", "--tb=short"])
        test_output = out + err
        if rc == 0:
            print(f"[phase3] pytest PASSED after fix (attempt {attempt})")
            return True
        print(f"[phase3] still failing after attempt {attempt}")

    return False

# ── record result ─────────────────────────────────────────────────────────────

def record_pass(spec_path: Path, class_name: str, label: str) -> None:
    sh(["uv", "run", "python", str(TEMPLATE / "append_row.py"), str(spec_path)])
    sh(["uv", "run", "python", str(AGENTS / "arm_a_state.py"),
        "mark", class_name, "pass"])
    sh(["git", "add",
        f"explore/archs/{label}/",
        f"explore/support/template/specs/{label}.json",
        "explore/support/family_tree.csv",
        "unialg.cabal",
        "explore/support/haskell/Catalogue.hs"])
    rc, _, err = sh(["git", "commit", "-m",
                     f"catalog: add {label} (Arm A, {class_name})"])
    if rc == 0:
        print(f"[done] committed {label}")
    else:
        print(f"[done] commit skipped or failed: {err.strip()}")


def record_fail(class_name: str) -> None:
    sh(["uv", "run", "python", str(AGENTS / "arm_a_state.py"),
        "mark", class_name, "fail"])
    print(f"[done] marked {class_name} fail")


def record_skip(class_name: str) -> None:
    sh(["uv", "run", "python", str(AGENTS / "arm_a_state.py"),
        "mark", class_name, "skip"])
    print(f"[done] marked {class_name} skip")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Get next candidate
    rc, out, _ = sh(["uv", "run", "python", str(AGENTS / "arm_a_state.py"), "next"])
    candidate = json.loads(out)
    if candidate.get("done"):
        print("All Arm A candidates exhausted.")
        return
    class_name = candidate["class"]
    print(f"\n{'='*60}")
    print(f"CANDIDATE: {class_name}")
    print(f"{'='*60}")

    # Phase 1: LLM
    label, spec = phase1(class_name)
    if spec is None or label is None:
        record_skip(class_name)
        return

    spec_path = SPECS / f"{label}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"[phase1] wrote {spec_path.name}")

    # Pre-test filter
    rc, out, err = sh(["uv", "run", "python", str(TEMPLATE / "filters.py"),
                       str(spec_path)])
    if rc != 0:
        print(f"[filter] rejected: {(out + err).strip()}")
        record_skip(class_name)
        return
    print("[filter] passed")

    # Phase 2: mechanical renderers
    if not phase2(spec_path, label):
        record_fail(class_name)
        return

    # Phase 3: test (+ LLM fix on failure)
    if phase3(spec_path, label):
        record_pass(spec_path, class_name, label)
    else:
        record_fail(class_name)


if __name__ == "__main__":
    main()
