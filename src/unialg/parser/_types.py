"""Data types produced and consumed by the .ua parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NamedCell:
    """A named morphism expression destined for graph registration."""
    name: str
    cell: object
    matchers: dict | None = None


@dataclass
class UASpec:
    """Parsed .ua program before compilation."""
    semirings: dict[str, Any] = field(default_factory=dict)
    sorts: dict[str, Any] = field(default_factory=dict)
    equations: list[Any] = field(default_factory=list)
    defines: list[Any] = field(default_factory=list)
    backend_name: str | None = None
    share_groups: dict[str, list[str]] = field(default_factory=dict)
    cells: list[Any] = field(default_factory=list)
