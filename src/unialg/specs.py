"""Composition spec types for assemble_graph.

Each spec carries its own validation logic via constraints() and sort_terms().
validation.py calls spec.constraints(eq_by_name) and spec.sort_terms() uniformly
rather than dispatching through a type table.
"""

from __future__ import annotations

from dataclasses import dataclass

from hydra.typing import TypeConstraint

import unialg.algebra as alg


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    """Base for all composition specs."""
    name: str

    def sort_terms(self) -> list:
        return [getattr(self, a) for a in ("domain_sort", "codomain_sort", "state_sort")
                if getattr(self, a, None) is not None]

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        raise NotImplementedError

    @staticmethod
    def _require_eq(eq_by_name: dict, name: str, label: str) -> None:
        if name not in eq_by_name:
            raise ValueError(f"{label} equation '{name}' not found")

    @staticmethod
    def _eq_sort_type(eq_by_name: dict, eq_name: str, field: str):
        """Return the Hydra Type for an equation's domain ('d') or codomain ('c')."""
        v = eq_by_name[eq_name]
        return alg.sort_type_from_term(v.domain_sort if field == "d" else v.codomain_sort)


@dataclass
class PathSpec(Spec):
    """Sequential path composition."""
    eq_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term
    params: dict[str, list] | None = None
    residual: bool = False
    residual_semiring: str | None = None

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        cs = []
        if self.domain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.eq_names[0], "d"),
                alg.sort_type_from_term(self.domain_sort),
                f"Path domain != '{self.eq_names[0]}' domain"))
        for a, b in zip(self.eq_names, self.eq_names[1:]):
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, a, "c"),
                self._eq_sort_type(eq_by_name, b, "d"),
                f"'{a}' codomain != '{b}' domain"))
        if self.codomain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.eq_names[-1], "c"),
                alg.sort_type_from_term(self.codomain_sort),
                f"Path codomain != '{self.eq_names[-1]}' codomain"))
        return cs


@dataclass
class FanSpec(Spec):
    """Parallel fan composition."""
    branch_names: list[str]
    merge_name: str
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        cs = []
        md = alg.sort_type_from_term(eq_by_name[self.merge_name].domain_sort)
        if self.domain_sort is not None:
            for b in self.branch_names:
                cs.append(TypeConstraint(
                    self._eq_sort_type(eq_by_name, b, "d"),
                    alg.sort_type_from_term(self.domain_sort),
                    f"Fan branch '{b}' domain mismatch"))
        for b in self.branch_names:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, b, "c"),
                md,
                f"Fan branch '{b}' codomain != merge domain"))
        if self.codomain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.merge_name, "c"),
                alg.sort_type_from_term(self.codomain_sort),
                f"Fan merge codomain mismatch"))
        return cs


@dataclass
class FoldSpec(Spec):
    """Fold (catamorphism)."""
    step_name: str
    init_term: object  # core.Term
    domain_sort: object  # core.Term
    state_sort: object  # core.Term

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Fold step")
        return [TypeConstraint(
            self._eq_sort_type(eq_by_name, self.step_name, "c"),
            alg.sort_type_from_term(self.state_sort),
            f"Fold step codomain != state sort")]


@dataclass
class UnfoldSpec(Spec):
    """Unfold (anamorphism)."""
    step_name: str
    n_steps: int
    domain_sort: object  # core.Term
    state_sort: object  # core.Term

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Unfold step")
        ds = alg.sort_type_from_term(self.domain_sort)
        return [
            TypeConstraint(self._eq_sort_type(eq_by_name, self.step_name, "d"), ds,
                           f"Unfold step domain != state sort"),
            TypeConstraint(self._eq_sort_type(eq_by_name, self.step_name, "c"), ds,
                           f"Unfold step codomain != state sort"),
        ]


@dataclass
class LensPathSpec(Spec):
    """Bidirectional lens path."""
    lens_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term
    params: dict[str, list] | None = None

    def sort_terms(self) -> list:
        return []

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        return []  # validated separately by _build_lens_by_name


@dataclass
class LensFanSpec(Spec):
    """Bidirectional lens fan."""
    lens_names: list[str]
    merge_lens_name: str
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term

    def sort_terms(self) -> list:
        return []

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        return []  # validated separately by _build_lens_by_name


@dataclass
class FixpointSpec(Spec):
    """Fixpoint iteration."""
    step_name: str
    predicate_name: str
    epsilon: float
    max_iter: int
    domain_sort: object  # core.Term

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Fixpoint step")
        self._require_eq(eq_by_name, self.predicate_name, "Fixpoint predicate")
        ds = alg.sort_type_from_term(self.domain_sort)
        pred_cod = self._eq_sort_type(eq_by_name, self.predicate_name, "c")
        if pred_cod == ds:
            raise TypeError(
                f"Fixpoint predicate '{self.predicate_name}' returns the same type as the state. "
                f"The predicate must return a scalar residual (float32), not an endomorphism."
            )
        return [
            TypeConstraint(self._eq_sort_type(eq_by_name, self.step_name, "d"), ds,
                           f"Fixpoint step domain != state sort"),
            TypeConstraint(self._eq_sort_type(eq_by_name, self.step_name, "c"), ds,
                           f"Fixpoint step codomain != state sort"),
            TypeConstraint(self._eq_sort_type(eq_by_name, self.predicate_name, "d"), ds,
                           f"Fixpoint predicate domain != state sort"),
        ]
