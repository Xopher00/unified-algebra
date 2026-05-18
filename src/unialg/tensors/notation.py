"""Tensor extension — surface notation layer.

Pure parsed data structures.  No Morphism, no Type, no Hydra, no backend.

``Equation``         — parsed einsum string (input labels, output, contracted)
``AlignmentPlan``    — per-operand unsqueeze+transpose plan derived from equation
``SemiringDecl``     — parsed ``algebra`` declaration
``ContractExpr``     — parsed ``contract[sr]("eq")`` expression
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlignmentPlan:
    """Per-operand reshape plan: insert size-1 dims then permute."""
    unsqueeze_axes: tuple[int, ...]
    perm: tuple[int, ...]


@dataclass(frozen=True)
class Equation:
    """Parsed einsum equation — pure index structure."""
    inputs: tuple[tuple[str, ...], ...]
    output: tuple[str, ...]
    reduced: tuple[str, ...]

    @staticmethod
    def parse(s: str) -> Equation:
        s = s.strip()
        if "->" not in s:
            raise ValueError(f"equation must contain '->': {s!r}")
        lhs, rhs = s.split("->", 1)
        input_strs = lhs.split(",")
        inputs = tuple(tuple(c for c in inp.strip()) for inp in input_strs)
        output = tuple(c for c in rhs.strip())

        seen_output: set[str] = set()
        duplicate_output: list[str] = []
        for label in output:
            if label in seen_output and label not in duplicate_output:
                duplicate_output.append(label)
            seen_output.add(label)
        if duplicate_output:
            raise ValueError(f"output labels must be unique: {tuple(duplicate_output)}")

        all_input_labels: set[str] = set()
        for inp in inputs:
            all_input_labels.update(inp)
        output_set = set(output)

        if not output_set.issubset(all_input_labels):
            extra = output_set - all_input_labels
            raise ValueError(f"output labels not in any input: {extra}")

        seen: list[str] = []
        seen_set: set[str] = set()
        for inp in inputs:
            for label in inp:
                if label not in seen_set:
                    seen.append(label)
                    seen_set.add(label)
        reduced = tuple(l for l in seen if l not in output_set)

        return Equation(inputs=inputs, output=output, reduced=reduced)

    def diagonal_axes(self, i: int) -> list[tuple[int, int]]:
        """Pairs of axes with the same label in operand i."""
        inp = self.inputs[i]
        seen: dict[str, int] = {}
        pairs: list[tuple[int, int]] = []
        for j, label in enumerate(inp):
            if label in seen:
                pairs.append((seen[label], j))
            else:
                seen[label] = j
        return pairs

    def post_diagonal_labels(self, i: int) -> tuple[str, ...]:
        """Labels after diagonal extraction, in the axis order numpy.diagonal produces.

        For each pair of repeated axes: both are removed and the diagonal
        is appended at the end. This matches numpy.diagonal(axis1, axis2).
        """
        diag_pairs = self.diagonal_axes(i)
        if not diag_pairs:
            return self.inputs[i]
        labels = list(self.inputs[i])
        positions = list(range(len(labels)))

        for a1_orig, a2_orig in diag_pairs:
            a1_cur = positions[a1_orig]
            a2_cur = positions[a2_orig]
            if a1_cur is None or a2_cur is None:
                continue
            if a1_cur > a2_cur:
                a1_cur, a2_cur = a2_cur, a1_cur

            diag_label = labels[a1_cur]
            del labels[a2_cur]
            del labels[a1_cur]
            labels.append(diag_label)

            for k in range(len(positions)):
                p = positions[k]
                if p is None:
                    continue
                if p == a1_cur or p == a2_cur:
                    positions[k] = None
                elif p > a2_cur:
                    positions[k] = p - 2
                elif p > a1_cur:
                    positions[k] = p - 1
            positions[a1_orig] = len(labels) - 1

        return tuple(labels)

    def target_vars(self) -> tuple[str, ...]:
        return self.output + self.reduced

    def alignment_plan(self, i: int) -> AlignmentPlan:
        """Compute unsqueeze+transpose plan for input operand ``i``.

        Target dim order is ``output ++ reduced``.  For each target var
        not present in this operand, insert a size-1 dim.  Then permute
        to match target ordering.

        Uses post-diagonal labels (unique) so the plan is valid after
        any diagonal extraction has collapsed repeated axes.
        """
        inp = self.post_diagonal_labels(i)
        target = self.target_vars()

        existing_vars = set(inp)
        expanded_vars: list[str] = list(inp)
        unsqueeze_axes: list[int] = []

        for tv in target:
            if tv not in existing_vars:
                axis = len(expanded_vars)
                unsqueeze_axes.append(axis)
                expanded_vars.append(tv)

        var_to_pos = {v: pos for pos, v in enumerate(expanded_vars)}
        perm = tuple(var_to_pos[tv] for tv in target)

        return AlignmentPlan(
            unsqueeze_axes=tuple(unsqueeze_axes),
            perm=perm,
        )

    def replace_input(
        self,
        slot: int,
        new_inputs: tuple[tuple[str, ...], ...],
        new_reduced: tuple[str, ...],
    ) -> Equation:
        """Replace input at ``slot`` with ``new_inputs``, merging reduced vars."""
        inputs = self.inputs[:slot] + new_inputs + self.inputs[slot + 1:]
        all_input_labels: set[str] = set()
        for inp in inputs:
            all_input_labels.update(inp)
        output_set = set(self.output)
        reduced = tuple(
            l for l in dict.fromkeys(list(new_reduced) + list(self.reduced))
            if l in all_input_labels and l not in output_set
        )
        return Equation(inputs=inputs, output=self.output, reduced=reduced)


@dataclass(frozen=True)
class SemiringDecl:
    """Parsed ``algebra`` declaration — pure surface data."""
    name: str
    plus: str
    times: str
    zero: str | float
    one: str | float
    adjoint: str | None = None


@dataclass(frozen=True)
class ContractExpr:
    """Parsed ``contract[sr]("eq")`` expression."""
    semiring_name: str
    equation_str: str
    adjoint: bool = False
    _domain_tag: str = field(default="tensors", init=False, repr=False)


# ---------------------------------------------------------------------------
# Cursor-based parsers — called by the extension dispatch hooks
# ---------------------------------------------------------------------------

def _require_str(fields, key) -> str:
    from unialg.syntax.parse import ParseError
    v = fields[key]
    if not isinstance(v, str):
        raise ParseError(f"{key} must be an op name, not a literal")
    return v


def parse_algebra(cursor, prog):
    """Parse ``algebra name(plus=op, times=op, zero=val, one=val, ...)``."""
    from unialg.syntax.parse import ParseError

    name_tok = cursor.expect("NAME", "semiring name")
    cursor.expect("LPAREN", "'('")

    fields: dict[str, str | float] = {}
    while cursor.peek()[0] != "RPAREN":
        key_tok = cursor.expect("NAME", "field name")
        cursor.expect("EQ", "'='")

        val_tok = cursor.advance()
        negate = False
        if val_tok[0] == "MINUS":
            negate = True
            val_tok = cursor.advance()

        if val_tok[0] == "NAME":
            val: str | float = val_tok[1]
            if negate:
                if val == "inf":
                    val = float("-inf")
                else:
                    raise ParseError(f"cannot negate name {val!r}")
        elif val_tok[0] == "FLOAT":
            val = -val_tok[1] if negate else val_tok[1]
        elif val_tok[0] == "INT":
            val = float(-val_tok[1]) if negate else float(val_tok[1])
        else:
            raise ParseError(f"expected value, got {val_tok[0]!r}")

        fields[key_tok[1]] = val

        if cursor.peek()[0] == "COMMA":
            cursor.advance()

    cursor.expect("RPAREN", "')'")

    required = {"plus", "times", "zero", "one"}
    missing = required - fields.keys()
    if missing:
        raise ParseError(f"algebra {name_tok[1]!r} missing fields: {missing}")

    optional_str = {}
    if "adjoint" in fields:
        adj = fields["adjoint"]
        if not isinstance(adj, str):
            raise ParseError("adjoint must be an op name, not a literal")
        optional_str["adjoint"] = adj

    decl = SemiringDecl(
        name=name_tok[1],
        plus=_require_str(fields, "plus"),
        times=_require_str(fields, "times"),
        zero=fields["zero"],
        one=fields["one"],
        **optional_str,
    )

    prog.extensions.setdefault("tensors", []).append(decl)


def parse_contract(cursor):
    """Parse ``contract[sr]("eq")`` or ``contract[sr, adjoint]("eq")``."""
    from unialg.syntax.parse import ParseError

    cursor.expect("LBRACKET", "'['")
    sr_tok = cursor.expect("NAME", "semiring name")
    sr_name = sr_tok[1]

    adjoint = False
    if cursor.peek()[0] == "COMMA":
        cursor.advance()
        adj_tok = cursor.expect("NAME", "'adjoint'")
        if adj_tok[1] != "adjoint":
            raise ParseError(f"expected 'adjoint', got {adj_tok[1]!r}")
        adjoint = True

    cursor.expect("RBRACKET", "']'")
    cursor.expect("LPAREN", "'('")
    eq_tok = cursor.expect("STRING", "equation string")
    cursor.expect("RPAREN", "')'")

    return ContractExpr(
        semiring_name=sr_name,
        equation_str=eq_tok[1],
        adjoint=adjoint,
    )
