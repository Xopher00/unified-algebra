"""Composition spec types for assemble_graph.

Each spec carries its own validation logic via constraints() and sort_terms(),
and its own build() method which registers any required primitives and returns
the (Name, Term) pairs to bind into the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.dsl.python import FrozenDict
from hydra.typing import TypeConstraint

from unialg.terms import unify_or_raise
from unialg.assembly.compositions import (
    PathComposition, FanComposition, FoldComposition, UnfoldComposition, FixpointComposition,
)
from unialg.assembly._primitives import (
    unfold_n_primitive, fixpoint_primitive,
    lens_fwd_primitive, lens_bwd_primitive, residual_add_primitive,
)


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

    def validate(self, eq_by_name: dict, schema_types) -> None:
        cs = self.constraints(eq_by_name)
        if cs:
            unify_or_raise(cs, schema_types)

    def build(self, primitives: dict, native_fns: dict, **kwargs) -> list:
        for prim, fn in self._primitives(**kwargs):
            primitives.setdefault(prim.name, prim)
            if fn is not None:
                native_fns.setdefault(prim.name, fn)
        return self._compose()

    def _primitives(self, **kwargs) -> list[tuple]:
        return []

    def _compose(self) -> list:
        raise NotImplementedError

    @staticmethod
    def _require_eq(eq_by_name: dict, name: str, label: str) -> None:
        if name not in eq_by_name:
            raise ValueError(f"{label} equation '{name}' not found")

    @staticmethod
    def _eq_sort_type(eq_by_name: dict, eq_name: str, field: str):
        """Return the Hydra Type for an equation's domain ('d') or codomain ('c')."""
        v = eq_by_name[eq_name]
        return (v.domain_sort if field == "d" else v.codomain_sort).type_

    @staticmethod
    def _sort_type(sort_term):
        return sort_term.type_

    @staticmethod
    def _endomorphism(eq_by_name: dict, eq_name: str, sort_type, label: str) -> list:
        return [
            TypeConstraint(Spec._eq_sort_type(eq_by_name, eq_name, "d"), sort_type,
                           f"{label} domain != state sort"),
            TypeConstraint(Spec._eq_sort_type(eq_by_name, eq_name, "c"), sort_type,
                           f"{label} codomain != state sort"),
        ]

    @staticmethod
    def _bidi(eq_by_name: dict, fwd_name: str, bwd_name: str, label: str) -> list:
        return [
            TypeConstraint(Spec._eq_sort_type(eq_by_name, fwd_name, "d"),
                           Spec._eq_sort_type(eq_by_name, bwd_name, "c"),
                           f"{label} fwd.domain != bwd.codomain"),
            TypeConstraint(Spec._eq_sort_type(eq_by_name, fwd_name, "c"),
                           Spec._eq_sort_type(eq_by_name, bwd_name, "d"),
                           f"{label} fwd.codomain != bwd.domain"),
        ]


@dataclass
class PathSpec(Spec):
    """Sequential path composition."""
    COMPOSITION = PathComposition
    eq_names: list[str]
    domain_sort: object
    codomain_sort: object
    params: dict[str, list] | None = None
    residual: bool = False
    residual_semiring: str | None = None

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        cs = []
        if self.domain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.eq_names[0], "d"),
                self._sort_type(self.domain_sort),
                f"Path domain != '{self.eq_names[0]}' domain"))
        for a, b in zip(self.eq_names, self.eq_names[1:]):
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, a, "c"),
                self._eq_sort_type(eq_by_name, b, "d"),
                f"'{a}' codomain != '{b}' domain"))
        if self.codomain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.eq_names[-1], "c"),
                self._sort_type(self.codomain_sort),
                f"Path codomain != '{self.eq_names[-1]}' codomain"))
        return cs

    def _primitives(self, **kwargs) -> list[tuple]:
        if not self.residual:
            return []
        sr_name = self.residual_semiring or "default"
        resolved_semirings = kwargs.get("resolved_semirings", {})
        if sr_name not in resolved_semirings:
            raise ValueError(
                f"Residual path references semiring '{sr_name}' but it was not resolved")
        prim, fn = residual_add_primitive(sr_name, resolved_semirings[sr_name], kwargs["coder"])
        return [(prim, fn)]

    def _compose(self) -> list:
        return [self.COMPOSITION(self.name, self.eq_names, self.params,
                     residual=self.residual, residual_semiring=self.residual_semiring)]


@dataclass
class FanSpec(Spec):
    """Parallel fan composition."""
    COMPOSITION = FanComposition
    branch_names: list[str]
    merge_name: str
    domain_sort: object
    codomain_sort: object

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        cs = []
        md = self._sort_type(eq_by_name[self.merge_name].domain_sort)
        if self.domain_sort is not None:
            for b in self.branch_names:
                cs.append(TypeConstraint(
                    self._eq_sort_type(eq_by_name, b, "d"),
                    self._sort_type(self.domain_sort),
                    f"Fan branch '{b}' domain mismatch"))
        for b in self.branch_names:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, b, "c"),
                md,
                f"Fan branch '{b}' codomain != merge domain"))
        if self.codomain_sort is not None:
            cs.append(TypeConstraint(
                self._eq_sort_type(eq_by_name, self.merge_name, "c"),
                self._sort_type(self.codomain_sort),
                f"Fan merge codomain mismatch"))
        return cs

    def _compose(self) -> list:
        return [self.COMPOSITION(self.name, self.branch_names, self.merge_name)]


@dataclass
class FoldSpec(Spec):
    """Fold (catamorphism)."""
    COMPOSITION = FoldComposition
    step_name: str
    init_term: object  # core.Term
    domain_sort: object
    state_sort: object

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Fold step")
        return [TypeConstraint(
            self._eq_sort_type(eq_by_name, self.step_name, "c"),
            self._sort_type(self.state_sort),
            f"Fold step codomain != state sort")]

    def _compose(self) -> list:
        return [self.COMPOSITION(self.name, self.step_name, self.init_term)]


@dataclass
class UnfoldSpec(Spec):
    """Unfold (anamorphism)."""
    COMPOSITION = UnfoldComposition
    step_name: str
    n_steps: int
    domain_sort: object
    state_sort: object

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Unfold step")
        return self._endomorphism(eq_by_name, self.step_name, self._sort_type(self.domain_sort), "Unfold step")

    def _primitives(self, **kwargs) -> list[tuple]:
        return [(unfold_n_primitive, None)]

    def _compose(self) -> list:
        return [self.COMPOSITION(self.name, self.step_name, self.n_steps)]


@dataclass
class LensPathSpec(PathSpec):
    """Bidirectional lens path. eq_names = forward chain; bwd_eq_names = backward chain."""
    bwd_eq_names: list[str] = field(default_factory=list)
    has_residual: bool = False

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        for n in self.eq_names + self.bwd_eq_names:
            self._require_eq(eq_by_name, n, f"LensPath '{self.name}'")

        # Residual skips chain checks (intermediate sorts are ProductSorts)
        if self.has_residual:
            cs = []
            if self.domain_sort is not None and self.eq_names:
                cs.append(TypeConstraint(
                    self._eq_sort_type(eq_by_name, self.eq_names[0], "d"),
                    self._sort_type(self.domain_sort),
                    f"LensPath '{self.name}' domain mismatch"))
        else:
            cs = super().constraints(eq_by_name)

        for fwd_name, bwd_name in zip(self.eq_names, self.bwd_eq_names):
            cs.extend(self._bidi(eq_by_name, fwd_name, bwd_name, f"'{fwd_name}'"))
        return cs

    def _primitives(self, **kwargs) -> list[tuple]:
        if not self.has_residual:
            return []
        return [(lens_fwd_primitive, None), (lens_bwd_primitive, None)]

    def _compose(self) -> list:
        fwd, bwd = self.COMPOSITION.build_lens(self.name, self.eq_names, self.bwd_eq_names, self.params, self.has_residual)
        return [fwd, bwd]


@dataclass
class LensFanSpec(FanSpec):
    """Bidirectional lens fan. branch_names/merge_name = forward; bwd fields = backward."""
    bwd_branch_names: list[str] = field(default_factory=list)
    merge_bwd_name: str = ""

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        for n in self.branch_names + self.bwd_branch_names + [self.merge_name, self.merge_bwd_name]:
            self._require_eq(eq_by_name, n, f"LensFan '{self.name}'")
        cs = super().constraints(eq_by_name)
        for fwd_name, bwd_name in zip(self.branch_names, self.bwd_branch_names):
            cs.extend(self._bidi(eq_by_name, fwd_name, bwd_name, f"LensFan '{self.name}' branch '{fwd_name}'"))
        cs.extend(self._bidi(eq_by_name, self.merge_name, self.merge_bwd_name, f"LensFan '{self.name}' merge"))
        return cs

    def _compose(self) -> list:
        fwd, bwd = self.COMPOSITION.build_lens(self.name, self.branch_names, self.bwd_branch_names, self.merge_name, self.merge_bwd_name)
        return [fwd, bwd]


@dataclass
class FixpointSpec(Spec):
    """Fixpoint iteration."""
    COMPOSITION = FixpointComposition
    step_name: str
    predicate_name: str
    epsilon: float
    max_iter: int
    domain_sort: object  # core.Term

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Fixpoint step")
        self._require_eq(eq_by_name, self.predicate_name, "Fixpoint predicate")
        ds = self._sort_type(self.domain_sort)
        pred_cod = self._eq_sort_type(eq_by_name, self.predicate_name, "c")
        if pred_cod == ds:
            raise TypeError(
                f"Fixpoint predicate '{self.predicate_name}' returns the same type as the state. "
                f"The predicate must return a scalar residual (float32), not an endomorphism."
            )
        return [
            *self._endomorphism(eq_by_name, self.step_name, ds, "Fixpoint step"),
            TypeConstraint(self._eq_sort_type(eq_by_name, self.predicate_name, "d"), ds,
                           f"Fixpoint predicate domain != state sort"),
        ]

    def _primitives(self, **kwargs) -> list[tuple]:
        return [(fixpoint_primitive(self.epsilon, self.max_iter), None)]

    def _compose(self) -> list:
        return [self.COMPOSITION(self.name, self.step_name, self.predicate_name, self.epsilon, self.max_iter)]
