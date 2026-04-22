"""Composition spec types for assemble_graph.

Pure data — no logic. Lives at Layer 2 so validation.py and _assembly.py
can import directly without cycling through graph.py.
"""

from typing import NamedTuple


class PathSpec(NamedTuple):
    """Convenience spec for a sequential path composition."""
    name: str
    eq_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term
    params: dict[str, list] | None = None
    residual: bool = False
    residual_semiring: str | None = None


class FanSpec(NamedTuple):
    """Convenience spec for a parallel fan composition."""
    name: str
    branch_names: list[str]
    merge_name: str
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term


class FoldSpec(NamedTuple):
    """Convenience spec for a fold (catamorphism)."""
    name: str
    step_name: str
    init_term: object  # core.Term
    domain_sort: object  # core.Term
    state_sort: object  # core.Term


class UnfoldSpec(NamedTuple):
    """Convenience spec for an unfold (anamorphism)."""
    name: str
    step_name: str
    n_steps: int
    domain_sort: object  # core.Term
    state_sort: object  # core.Term


class LensPathSpec(NamedTuple):
    """Convenience spec for a bidirectional lens path."""
    name: str
    lens_names: list[str]
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term
    params: dict[str, list] | None = None


class LensFanSpec(NamedTuple):
    """Convenience spec for a bidirectional lens fan."""
    name: str
    lens_names: list[str]
    merge_lens_name: str
    domain_sort: object  # core.Term
    codomain_sort: object  # core.Term


class FixpointSpec(NamedTuple):
    """Convenience spec for a fixpoint iteration."""
    name: str
    step_name: str
    predicate_name: str
    epsilon: float
    max_iter: int
    domain_sort: object  # core.Term
