"""Composition specs: validation, primitive registration, and Hydra term construction."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import hydra.core as core
from hydra.typing import TypeConstraint

from ._validation import unify_or_raise
from .compositions import (
    PathComposition, FanComposition, FoldComposition, UnfoldComposition, FixpointComposition,
    ParallelComposition,
)
from ._primitives import (
    unfold_n_primitive, fixpoint_primitive,
    lens_fwd_primitive, lens_bwd_primitive, residual_add_primitive,
)


@dataclass
class Spec:
    """Base for all composition specs."""
    name: str
    COMPOSITION = None
    _COMPOSE_FIELDS = ()

    @classmethod
    def from_parsed(cls, decl, get_sort, **kw):
        name = decl[1]
        sig = decl[2]
        sorts = (get_sort(sig[0]), get_sort(sig[1])) if isinstance(sig, tuple) else (get_sort(sig),)
        sort_fields = [f.name for f in fields(cls) if f.name.endswith('_sort')]
        return cls(name=name, **dict(zip(sort_fields, sorts)), **cls._parse_rest(decl[3:], **kw))

    @classmethod
    def _parse_rest(cls, rest, **kw):
        return {}

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
        return [self.COMPOSITION(self.name, *[getattr(self, f) for f in self._COMPOSE_FIELDS])]

    @staticmethod
    def _require_eq(eq_by_name: dict, name: str, label: str) -> None:
        if name not in eq_by_name:
            raise ValueError(f"{label} op '{name}' not found")

    @staticmethod
    def _eq_sort_type(eq_by_name: dict, eq_name: str, field: str):
        v = eq_by_name[eq_name]
        return (v.domain_sort if field == "d" else v.codomain_sort).type_

    @staticmethod
    def _boundary(eq_by_name, domain_sort, codomain_sort, domain_eqs, codomain_eq=None):
        cs = []
        if domain_sort is not None:
            ds = domain_sort.type_
            for eq in domain_eqs:
                cs.append(TypeConstraint(
                    Spec._eq_sort_type(eq_by_name, eq, "d"), ds, f"'{eq}' domain mismatch"))
        if codomain_sort is not None and codomain_eq is not None:
            cs.append(TypeConstraint(
                Spec._eq_sort_type(eq_by_name, codomain_eq, "c"), codomain_sort.type_,
                f"'{codomain_eq}' codomain mismatch"))
        return cs

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

    @staticmethod
    def _decompose_lenses(names, get_lens):
        fwd, bwd, has_res = [], [], False
        for ln in names:
            lv = get_lens(ln)
            fwd.append(lv.forward); bwd.append(lv.backward)
            if lv.residual_sort is not None: has_res = True
        return fwd, bwd, has_res


@dataclass
class PathSpec(Spec):
    COMPOSITION = PathComposition
    _COMPOSE_FIELDS = ('eq_names', 'params', 'residual', 'residual_semiring')
    eq_names: list[str]
    domain_sort: object
    codomain_sort: object
    params: dict[str, list] | None = None
    residual: bool = False
    residual_semiring: str | None = None
    bwd_eq_names: list[str] = field(default_factory=list)
    has_residual: bool = False

    @classmethod
    def _parse_rest(cls, rest, expand_ref=None, get_lens=None, **_):
        if get_lens is not None:
            fwd, bwd, has_res = Spec._decompose_lenses(rest[0], get_lens)
            return dict(eq_names=fwd, bwd_eq_names=bwd, has_residual=has_res)
        eq_names = [expand_ref(en) for en in rest[0]]
        attrs = rest[1]
        residual = attrs.get('residual', False)
        return dict(eq_names=eq_names, residual=residual,
                    residual_semiring=attrs.get('algebra') if residual else None)

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        all_names = self.eq_names + self.bwd_eq_names
        for n in all_names:
            self._require_eq(eq_by_name, n, f"Seq '{self.name}'")
        if self.bwd_eq_names and self.has_residual:
            cs = self._boundary(eq_by_name, self.domain_sort, None, [self.eq_names[0]])
        else:
            cs = self._boundary(eq_by_name, self.domain_sort, self.codomain_sort,
                                [self.eq_names[0]], self.eq_names[-1])
            for a, b in zip(self.eq_names, self.eq_names[1:]):
                cs.append(TypeConstraint(
                    self._eq_sort_type(eq_by_name, a, "c"),
                    self._eq_sort_type(eq_by_name, b, "d"),
                    f"'{a}' codomain != '{b}' domain"))
        for fwd, bwd in zip(self.eq_names, self.bwd_eq_names):
            cs.extend(self._bidi(eq_by_name, fwd, bwd, f"'{fwd}'"))
        return cs

    def _primitives(self, **kwargs) -> list[tuple]:
        prims = []
        if self.residual:
            sr_name = self.residual_semiring or "default"
            resolved_semirings = kwargs.get("resolved_semirings", {})
            if sr_name not in resolved_semirings:
                raise ValueError(
                    f"Residual path references semiring '{sr_name}' but it was not resolved")
            prim, fn = residual_add_primitive(sr_name, resolved_semirings[sr_name], kwargs["coder"])
            prims.append((prim, fn))
        if self.bwd_eq_names and self.has_residual:
            prims.extend([(lens_fwd_primitive, None), (lens_bwd_primitive, None)])
        return prims

    def _compose(self) -> list:
        if not self.bwd_eq_names:
            return super()._compose()
        fwd, bwd = PathComposition.build_lens(
            self.name, self.eq_names, self.bwd_eq_names, self.params, self.has_residual)
        return [fwd, bwd]


@dataclass
class FanSpec(Spec):
    COMPOSITION = FanComposition
    _COMPOSE_FIELDS = ('branch_names', 'merge_names')
    branch_names: list[str]
    merge_names: list[str]
    domain_sort: object
    codomain_sort: object
    bwd_branch_names: list[str] = field(default_factory=list)
    merge_bwd_name: str = ""

    @classmethod
    def _parse_rest(cls, rest, expand_ref=None, get_lens=None, **_):
        if get_lens is not None:
            fwd, bwd, _ = Spec._decompose_lenses(rest[0], get_lens)
            mlv = get_lens(rest[1][0])
            return dict(branch_names=fwd, merge_names=[mlv.forward],
                        bwd_branch_names=bwd, merge_bwd_name=mlv.backward)
        branches = [expand_ref(bn) for bn in rest[0]]
        return dict(branch_names=branches, merge_names=rest[1])

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        all_names = self.branch_names + self.bwd_branch_names
        if self.merge_bwd_name:
            all_names += [self.merge_names[0], self.merge_bwd_name]
        declared = [n for n in self.merge_names if n in eq_by_name]
        for n in all_names:
            self._require_eq(eq_by_name, n, f"Branch '{self.name}'")
        if not declared:
            cs = self._boundary(eq_by_name, self.domain_sort, self.codomain_sort,
                                self.branch_names)
        else:
            first_merge = declared[0]
            cs = self._boundary(eq_by_name, self.domain_sort, self.codomain_sort,
                                self.branch_names, first_merge)
            md = self._eq_sort_type(eq_by_name, first_merge, "d")
            for b in self.branch_names:
                cs.append(TypeConstraint(
                    self._eq_sort_type(eq_by_name, b, "c"), md,
                    f"Branch '{b}' codomain != merge domain"))
        for fwd, bwd in zip(self.branch_names, self.bwd_branch_names):
            cs.extend(self._bidi(eq_by_name, fwd, bwd,
                                 f"LensBranch '{self.name}' branch '{fwd}'"))
        if self.merge_bwd_name:
            cs.extend(self._bidi(eq_by_name, self.merge_names[0], self.merge_bwd_name,
                                 f"LensBranch '{self.name}' merge"))
        return cs

    def _compose(self) -> list:
        if not self.bwd_branch_names:
            return super()._compose()
        fwd, bwd = FanComposition.build_lens(
            self.name, self.branch_names, self.bwd_branch_names,
            self.merge_names[0], self.merge_bwd_name)
        return [fwd, bwd]


@dataclass
class FoldSpec(Spec):
    COMPOSITION = FoldComposition
    _COMPOSE_FIELDS = ('step_name', 'init_term')
    step_name: str
    init_term: object = None
    domain_sort: object = None
    state_sort: object = None

    @classmethod
    def _parse_rest(cls, rest, **_):
        attrs = rest[0]
        init_val = float(attrs.get('init', 0.0))
        init_term = core.TermLiteral(value=core.LiteralFloat(value=init_val))
        return dict(step_name=attrs['step'], init_term=init_term)

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Scan step")
        return [TypeConstraint(
            self._eq_sort_type(eq_by_name, self.step_name, "c"),
            self.state_sort.type_,
            f"Scan step codomain != state sort")]


@dataclass
class UnfoldSpec(Spec):
    COMPOSITION = UnfoldComposition
    _COMPOSE_FIELDS = ('step_name', 'n_steps')
    step_name: str
    n_steps: int
    domain_sort: object = None
    state_sort: object = None

    @classmethod
    def _parse_rest(cls, rest, **_):
        return dict(step_name=rest[0]['step'], n_steps=int(rest[0]['steps']))

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Unroll step")
        return self._endomorphism(eq_by_name, self.step_name, self.domain_sort.type_, "Unroll step")

    def _primitives(self, **kwargs) -> list[tuple]:
        return [(unfold_n_primitive, None)]


@dataclass
class FixpointSpec(Spec):
    COMPOSITION = FixpointComposition
    _COMPOSE_FIELDS = ('step_name', 'predicate_name', 'epsilon', 'max_iter')
    step_name: str
    predicate_name: str = ""
    epsilon: float = 1e-6
    max_iter: int = 100
    domain_sort: object = None

    @classmethod
    def _parse_rest(cls, rest, **_):
        attrs = rest[0]
        return dict(step_name=attrs.get('step'), predicate_name=attrs.get('predicate'),
                    epsilon=float(attrs.get('epsilon', 1e-6)),
                    max_iter=int(attrs.get('max_iter', 100)))

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.step_name, "Fixpoint step")
        self._require_eq(eq_by_name, self.predicate_name, "Fixpoint predicate")
        ds = self.domain_sort.type_
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


@dataclass
class ParallelSpec(Spec):
    """Bimap / monoidal product: routes pair components independently.

    Categorical reading: f ⊗ g : (A, B) -> (C, D)
    """
    COMPOSITION = ParallelComposition
    _COMPOSE_FIELDS = ('left_name', 'right_name')
    left_name: str
    right_name: str
    domain_sort: object = None    # the ProductSort (A, B)
    codomain_sort: object = None  # the ProductSort (C, D)

    @classmethod
    def from_parsed(cls, decl, get_sort, **kw):
        # decl = ('parallel', name, (left_name, right_name))
        _, name, (left_name, right_name) = decl
        return cls(name=name, left_name=left_name, right_name=right_name)

    def constraints(self, eq_by_name: dict) -> list[TypeConstraint]:
        self._require_eq(eq_by_name, self.left_name,  f"Parallel '{self.name}' left")
        self._require_eq(eq_by_name, self.right_name, f"Parallel '{self.name}' right")
        return []

LensPathSpec = PathSpec
LensFanSpec = FanSpec
