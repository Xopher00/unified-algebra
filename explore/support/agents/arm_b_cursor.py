"""
Arm B(a) cursor: enumerate (PolyF × ArchClass × Semiring × Activation) candidates.

The cursor persists state to arm_b_state.json in the same directory.
It advances exhaustively through depth N before starting depth N+1.
After each depth completes, it computes per-subspace hit rates and
skips subspaces with zero hits in the next depth (adaptive policy).

Usage:
    python explore/support/agents/arm_b_cursor.py next
        Print the next untried candidate as a JSON description.

    python explore/support/agents/arm_b_cursor.py mark <label> pass|fail|skip
        Record the outcome for the candidate identified by <label>.
        'pass'  = numpy test green; row appended to family_tree.csv
        'fail'  = test red (classification: see phase3 report)
        'skip'  = pre-test filter rejected it

    python explore/support/agents/arm_b_cursor.py stats
        Print per-depth hit rates.
"""

from __future__ import annotations
import json
import sys
from itertools import product as iproduct
from pathlib import Path

STATE_FILE = Path(__file__).parent / "arm_b_state.json"

# ── standard axes ─────────────────────────────────────────────────────────────

SEMIRINGS = [
    {"label": "real",       "add": "add",        "multiply": "multiply"},
    {"label": "tropical",   "add": "minimum",    "multiply": "add"},
    {"label": "boolean",    "add": "logical_or", "multiply": "logical_and"},
    {"label": "semilattice","add": "maximum",    "multiply": "minimum"},
]

ACTIVATIONS = ["tanh", "relu", "sigmoid", None]

# ── PolyF enumeration (Python mirror of Grammar.enumerate) ───────────────────

def _enum(depth: int) -> list[dict]:
    """All PolyF up to `depth` as tag-dict trees."""
    atoms = [{"tag": "KUnit"}, {"tag": "KConst"}, {"tag": "Hole"}]
    if depth == 0:
        return atoms
    prev = _enum(depth - 1)
    new  = list(prev)
    for a, b in iproduct(prev, prev):
        new.append({"tag": "Sum",     "left": a, "right": b})
        new.append({"tag": "Product", "left": a, "right": b})
    for a in prev:
        new.append({"tag": "Exp", "arg": a})
    return new


def new_at_depth(depth: int) -> list[dict]:
    """PolyF introduced at exactly `depth` (not present at depth-1)."""
    if depth == 0:
        return _enum(0)
    all_prev = {json.dumps(n, sort_keys=True) for n in _enum(depth - 1)}
    return [n for n in _enum(depth)
            if json.dumps(n, sort_keys=True) not in all_prev]


# ── ArchClass classification ──────────────────────────────────────────────────

def _has_tag(node: dict, tag: str) -> bool:
    if node["tag"] == tag:
        return True
    return any(_has_tag(node[k], tag) for k in ("left", "right", "arg")
               if k in node)


def _arity(node: dict) -> int:
    tag = node["tag"]
    if tag in ("KUnit", "KConst"):  return 0
    if tag == "Hole":               return 1
    if tag == "Sum":    return _arity(node["left"]) + _arity(node["right"])
    if tag == "Product":return _arity(node["left"]) + _arity(node["right"])
    if tag == "Exp":    return _arity(node["arg"])
    return 0


def classify(node: dict) -> str:
    if _arity(node) == 0:            return "NoStructure"
    if _has_tag(node, "Exp"):        return "Ana"
    # Sum where at least one arm is a base (arity-0) → Cata
    if node["tag"] == "Sum":
        if _arity(node["left"]) == 0 or _arity(node["right"]) == 0:
            return "Cata"
    if _has_tag(node, "Sum"):
        # nested sum — check for any base arm anywhere
        def has_base_sum(n: dict) -> bool:
            if n["tag"] == "Sum":
                if _arity(n["left"]) == 0 or _arity(n["right"]) == 0:
                    return True
                return has_base_sum(n["left"]) or has_base_sum(n["right"])
            return False
        if has_base_sum(node):
            return "Cata"
    return "Hylo"


# ── candidate key (hashable identity) ────────────────────────────────────────

def _key(poly_f: dict, sr: dict, activation: str | None) -> str:
    return json.dumps({
        "poly_f":     poly_f,
        "sr_add":     sr["add"],
        "sr_mul":     sr["multiply"],
        "activation": activation,
    }, sort_keys=True)


# ── state I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"current_depth": 0, "tried": {}, "depth_stats": {},
            "skipped_subspaces": []}


def _save(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── adaptive subspace filter ──────────────────────────────────────────────────

def _subspace_key(sr: dict, arch_class: str) -> str:
    return f"{arch_class}|{sr['add']}|{sr['multiply']}"


def _should_skip(state: dict, sr: dict, arch_class: str) -> bool:
    return _subspace_key(sr, arch_class) in state.get("skipped_subspaces", [])


def _update_depth_stats(state: dict, depth: int) -> None:
    """Called when all candidates at `depth` are exhausted."""
    tried_at_depth = {k: v for k, v in state["tried"].items()
                      if v.get("depth") == depth}
    stats: dict[str, dict] = {}
    for entry in tried_at_depth.values():
        sk = _subspace_key(
            {"add": entry["sr_add"], "multiply": entry["sr_mul"]},
            entry["arch_class"],
        )
        s = stats.setdefault(sk, {"total": 0, "hits": 0})
        s["total"] += 1
        if entry["result"] == "pass":
            s["hits"] += 1

    state["depth_stats"][str(depth)] = stats

    # Mark zero-hit subspaces as skipped for subsequent depths.
    for sk, s in stats.items():
        if s["total"] > 0 and s["hits"] == 0:
            if sk not in state["skipped_subspaces"]:
                state["skipped_subspaces"].append(sk)


# ── next candidate ────────────────────────────────────────────────────────────

def cmd_next(state: dict) -> None:
    depth = state["current_depth"]

    while True:
        candidates = new_at_depth(depth)
        for poly_f in candidates:
            arch_class = classify(poly_f)
            if arch_class == "NoStructure":
                continue
            for sr in SEMIRINGS:
                if _should_skip(state, sr, arch_class):
                    continue
                for activation in ACTIVATIONS:
                    key = _key(poly_f, sr, activation)
                    if key not in state["tried"]:
                        # Found the next untried candidate.
                        out = {
                            "depth":       depth,
                            "poly_f":      poly_f,
                            "arch_class":  arch_class,
                            "semiring":    sr,
                            "activation":  activation,
                            "_cursor_key": key,
                        }
                        print(json.dumps(out, indent=2))
                        return

        # This depth is exhausted — update stats, advance.
        _update_depth_stats(state, depth)
        depth += 1
        state["current_depth"] = depth
        _save(state)
        if depth > 6:
            print(json.dumps({"done": True, "message": "depth limit reached"}))
            return


# ── mark outcome ──────────────────────────────────────────────────────────────

def cmd_mark(state: dict, label: str, result: str,
             cursor_key: str, depth: int,
             arch_class: str, sr_add: str, sr_mul: str) -> None:
    if result not in ("pass", "fail", "skip"):
        print(f"result must be pass|fail|skip, got {result!r}", file=sys.stderr)
        sys.exit(1)
    state["tried"][cursor_key] = {
        "label":      label,
        "result":     result,
        "depth":      depth,
        "arch_class": arch_class,
        "sr_add":     sr_add,
        "sr_mul":     sr_mul,
    }
    _save(state)
    print(f"[ok] marked {label!r} as {result}")


# ── stats ─────────────────────────────────────────────────────────────────────

def cmd_stats(state: dict) -> None:
    for depth, subspaces in state.get("depth_stats", {}).items():
        print(f"depth {depth}:")
        for sk, s in subspaces.items():
            rate = s["hits"] / s["total"] if s["total"] else 0
            print(f"  {sk:40s}  {s['hits']}/{s['total']}  ({rate:.0%})")
    skipped = state.get("skipped_subspaces", [])
    if skipped:
        print(f"\nSkipped subspaces: {skipped}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    state = _load()

    if args[0] == "next":
        cmd_next(state)

    elif args[0] == "mark":
        if len(args) < 7:
            print(
                "Usage: mark <label> pass|fail|skip <cursor_key> "
                "<depth> <arch_class> <sr_add> <sr_mul>",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_mark(state, args[1], args[2], args[3],
                 int(args[4]), args[5], args[6], args[7])
        _save(state)

    elif args[0] == "stats":
        cmd_stats(state)

    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
