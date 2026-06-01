"""
Arm A state: track which torch.nn / tf.keras.layers classes have been attempted.

The class list is populated on first run by introspecting the live modules.
Classes are walked alphabetically. Status per class:
  pending   — not yet attempted
  pass      — numpy test green; row in family_tree.csv
  fail      — test red (diagnosis in family_tree notes)
  skip      — not a single recurrent/feedforward cell (e.g. container, loss)

State is persisted to arm_a_state.json in the same directory.

Usage:
    python explore/support/agents/arm_a_state.py init
        Populate the class list from torch.nn and tf.keras.layers.
        Safe to re-run: only adds classes not already present.

    python explore/support/agents/arm_a_state.py next
        Print the next pending class name and its framework.

    python explore/support/agents/arm_a_state.py mark <class> pass|fail|skip
        Record the outcome for a class.

    python explore/support/agents/arm_a_state.py status
        Print counts by status.
"""

from __future__ import annotations
import inspect
import json
import sys
from pathlib import Path

STATE_FILE = Path(__file__).parent / "arm_a_state.json"

# Module path prefixes whose entire contents are non-cell classes.
_SKIP_MODULES = {
    # torch
    "torch.nn.modules.activation",
    "torch.nn.modules.pooling",
    "torch.nn.modules.normalization",
    "torch.nn.modules.dropout",
    "torch.nn.modules.padding",
    "torch.nn.modules.sparse",
    "torch.nn.modules.distance",
    "torch.nn.modules.channelshuffle",
    "torch.nn.modules.fold",
    "torch.nn.modules.upsampling",
    "torch.nn.modules.pixelshuffle",
    "torch.nn.modules.flatten",
    "torch.nn.modules.loss",
    "torch.nn.parallel",
    # keras / tensorflow
    "keras.src.layers.activations",
    "keras.src.layers.preprocessing",
    "keras.src.layers.regularization",
    "keras.src.layers.core",          # Lambda, Masking, InputLayer, base Layer
}

# Exact class names that are not cells (base classes, meta types, utilities).
_SKIP_EXACT = {
    # base / abstract
    "Layer", "Module", "Container", "RNNBase", "RNNCellBase",
    # non-layer objects
    "Parameter", "UninitializedParameter", "InputSpec",
    # cross-framework / saved-model utilities
    "JaxLayer", "TFSMLayer",
}

# Name substrings that identify non-cell classes in both frameworks.
_SKIP_SUBSTRINGS = (
    # containers / wrappers
    "Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict",
    "DataParallel", "Bidirectional", "Wrapper", "FlaxLayer",
    "Stacked", "TimeDistributed",
    # normalization
    "Norm", "Normalization",
    # pooling
    "Pool", "Pooling",
    # dropout / regularization
    "Dropout", "Regularization",
    # padding / cropping / spatial
    "Pad", "Padding", "Cropping", "Upsample", "UpSampling",
    "Flatten", "Reshape", "Permute", "Repeat",
    # augmentation / preprocessing (Random* not caught by module path)
    "Augment", "AutoContrast", "AugMix", "CutMix", "Equalization",
    "Solarize", "Posterize", "Sharpen", "Grayscale",
    "CenterCrop", "RandomCrop", "RandomFlip", "RandomRotation",
    "RandomHeight", "RandomWidth", "RandomBrightness",
    "Rescaling", "Resizing", "CategoryEncoding", "Discretization",
    # loss / activation (name-based fallback for classes not in skip modules)
    "Loss", "ReLU", "Relu", "Softmax", "Sigmoid", "Tanh",
    # Conv aliases (Convolution* duplicates Conv*)
    "Convolution",
    # misc utility
    "Identity", "Buffer", "Shuffle", "Merge",
    "Dot", "Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum",
    "Concatenate", "ZeroPadding", "Lazy",
)


def _is_candidate(name: str, obj) -> bool:
    if not inspect.isclass(obj):
        return False
    if name.startswith("_"):
        return False
    if name in _SKIP_EXACT:
        return False
    # Filter by defining submodule.
    mod = getattr(obj, "__module__", "") or ""
    if any(mod.startswith(m) for m in _SKIP_MODULES):
        return False
    # Filter by name substrings.
    if any(s in name for s in _SKIP_SUBSTRINGS):
        return False
    return True


# ── init ──────────────────────────────────────────────────────────────────────

def cmd_init(state: dict) -> None:
    existing = set(state.get("classes", {}).keys())
    added = 0

    try:
        import torch.nn as tnn
        for name, obj in sorted(inspect.getmembers(tnn)):
            key = f"torch.nn.{name}"
            if _is_candidate(name, obj) and key not in existing:
                state["classes"][key] = "pending"
                added += 1
    except ImportError:
        print("[warn] torch not available; skipping torch.nn", file=sys.stderr)

    try:
        import tensorflow as tf
        layers = tf.keras.layers
        for name, obj in sorted(inspect.getmembers(layers)):
            key = f"tf.keras.layers.{name}"
            if _is_candidate(name, obj) and key not in existing:
                state["classes"][key] = "pending"
                added += 1
    except ImportError:
        print("[warn] tensorflow not available; skipping tf.keras.layers",
              file=sys.stderr)

    print(f"[ok] added {added} new classes ({len(state['classes'])} total)")


# ── next ──────────────────────────────────────────────────────────────────────

def cmd_next(state: dict) -> None:
    for name, status in sorted(state.get("classes", {}).items()):
        if status == "pending":
            print(json.dumps({"class": name}, indent=2))
            return
    print(json.dumps({"done": True, "message": "no pending classes"}))


# ── mark ──────────────────────────────────────────────────────────────────────

def cmd_mark(state: dict, class_name: str, result: str) -> None:
    if result not in ("pass", "fail", "skip"):
        print(f"result must be pass|fail|skip, got {result!r}", file=sys.stderr)
        sys.exit(1)
    if class_name not in state["classes"]:
        print(f"[warn] {class_name!r} not in state; adding as {result}")
        state["classes"][class_name] = result
    else:
        state["classes"][class_name] = result
    print(f"[ok] marked {class_name!r} as {result}")


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status(state: dict) -> None:
    from collections import Counter
    counts = Counter(state.get("classes", {}).values())
    total = sum(counts.values())
    for status in ("pass", "fail", "skip", "pending"):
        n = counts.get(status, 0)
        pct = f"{100*n//total}%" if total else "—"
        print(f"  {status:8s}  {n:4d}  {pct}")
    print(f"  {'total':8s}  {total:4d}")


# ── state I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"classes": {}}


def _save(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    state = _load()

    if args[0] == "init":
        cmd_init(state)
        _save(state)

    elif args[0] == "next":
        cmd_next(state)

    elif args[0] == "mark":
        if len(args) < 3:
            print("Usage: mark <class> pass|fail|skip", file=sys.stderr)
            sys.exit(1)
        cmd_mark(state, args[1], args[2])
        _save(state)

    elif args[0] == "status":
        cmd_status(state)

    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
