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


def _shift_position(p: int | None, a1: int, a2: int) -> int | None:
    """Adjust a position index after two elements at a1 < a2 were deleted."""
    if p is None or p == a1 or p == a2:
        return None
    if p > a2:
        return p - 2
    if p > a1:
        return p - 1
    return p


@dataclass(frozen=True)
class Equation:
    """Parsed einsum equation — pure index structure."""
    inputs: tuple[tuple[str, ...], ...]
    output: tuple[str, ...]
    reduced: tuple[str, ...]

    @staticmethod
    def _parse_components(s: str) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
        lhs, rhs = s.split("->", 1)
        inputs = tuple(tuple(c for c in inp.strip()) for inp in lhs.split(","))
        output = tuple(c for c in rhs.strip())
        return inputs, output

    @staticmethod
    def _check_output_unique(output: tuple[str, ...]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for label in output:
            if label in seen and label not in duplicates:
                duplicates.append(label)
            seen.add(label)
        if duplicates:
            raise ValueError(f"output labels must be unique: {tuple(duplicates)}")

    @staticmethod
    def _compute_reduced(
        inputs: tuple[tuple[str, ...], ...], output_set: set[str]
    ) -> tuple[str, ...]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for inp in inputs:
            for label in inp:
                if label not in seen_set:
                    seen.append(label)
                    seen_set.add(label)
        return tuple(l for l in seen if l not in output_set)

    @staticmethod
    def parse(s: str) -> Equation:
        s = s.strip()
        if "->" not in s:
            raise ValueError(f"equation must contain '->': {s!r}")
        inputs, output = Equation._parse_components(s)
        Equation._check_output_unique(output)
        output_set = set(output)
        all_input_labels: set[str] = {label for inp in inputs for label in inp}
        if not output_set.issubset(all_input_labels):
            extra = output_set - all_input_labels
            raise ValueError(f"output labels not in any input: {extra}")
        reduced = Equation._compute_reduced(inputs, output_set)
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
            positions = [_shift_position(p, a1_cur, a2_cur) for p in positions]
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


def _parse_value(cursor) -> str | float:
    """Parse one field value token, handling optional leading minus sign."""
    from unialg.syntax.parse import ParseError
    val_tok = cursor.advance()
    negate = val_tok[0] == "MINUS"
    if negate:
        val_tok = cursor.advance()
    if val_tok[0] == "NAME":
        if negate:
            if val_tok[1] == "inf":
                return float("-inf")
            raise ParseError(f"cannot negate name {val_tok[1]!r}")
        return val_tok[1]
    if val_tok[0] == "FLOAT":
        return -val_tok[1] if negate else val_tok[1]
    if val_tok[0] == "INT":
        return float(-val_tok[1]) if negate else float(val_tok[1])
    raise ParseError(f"expected value, got {val_tok[0]!r}")


def _validate_algebra_fields(name: str, fields: dict) -> dict[str, str]:
    """Check required fields present and adjoint (if given) is a name, not a literal."""
    from unialg.syntax.parse import ParseError
    missing = {"plus", "times", "zero", "one"} - fields.keys()
    if missing:
        raise ParseError(f"algebra {name!r} missing fields: {missing}")
    optional_str: dict[str, str] = {}
    if "adjoint" in fields:
        adj = fields["adjoint"]
        if not isinstance(adj, str):
            raise ParseError("adjoint must be an op name, not a literal")
        optional_str["adjoint"] = adj
    return optional_str


def parse_algebra(cursor, prog):
    """Parse ``algebra name(plus=op, times=op, zero=val, one=val, ...)``."""
    name_tok = cursor.expect("NAME", "semiring name")
    cursor.expect("LPAREN", "'('")

    fields: dict[str, str | float] = {}
    while cursor.peek()[0] != "RPAREN":
        key_tok = cursor.expect("NAME", "field name")
        cursor.expect("EQ", "'='")
        fields[key_tok[1]] = _parse_value(cursor)
        if cursor.peek()[0] == "COMMA":
            cursor.advance()

    cursor.expect("RPAREN", "')'")
    optional_str = _validate_algebra_fields(name_tok[1], fields)

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
