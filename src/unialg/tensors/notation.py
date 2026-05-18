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

        for i, inp in enumerate(inputs):
            if len(inp) != len(set(inp)):
                dupes = [l for l in inp if inp.count(l) > 1]
                raise ValueError(
                    f"repeated labels {set(dupes)} in operand {i}: "
                    f"diagonal/trace semantics not yet supported"
                )

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

    def target_vars(self) -> tuple[str, ...]:
        return self.output + self.reduced

    def alignment_plan(self, i: int) -> AlignmentPlan:
        """Compute unsqueeze+transpose plan for input operand ``i``.

        Target dim order is ``output ++ reduced``.  For each target var
        not present in this operand, insert a size-1 dim.  Then permute
        to match target ordering.
        """
        inp = self.inputs[i]
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
