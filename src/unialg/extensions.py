"""Generic extension registry — dispatch hooks for domain modules.

Domain modules self-register at import time. The core parser, semantic
constructor, and lowering pipeline delegate to registered domains via
the protocol defined here.  The core never imports domain-specific code;
it only knows about the protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .semantics.morphisms import Morphism


@dataclass(frozen=True)
class DomainProtocol:
    """Interface a domain module must provide."""

    construct: Callable[[list, dict], Any]
    construct_expr: Callable[[Any, dict], "Morphism"]
    refs: Callable[[Any], set[str]]
    finalize: Callable[["Morphism", dict], "Morphism"] | None = None


_keyword_handlers: dict[str, Callable] = {}
_expr_handlers: dict[str, Callable] = {}
_domain_protocols: dict[str, DomainProtocol] = {}


def register_keyword(keyword: str, handler: Callable) -> None:
    _keyword_handlers[keyword] = handler


def register_expr_form(name: str, handler: Callable) -> None:
    _expr_handlers[name] = handler


def register_domain(tag: str, protocol: DomainProtocol) -> None:
    _domain_protocols[tag] = protocol


def get_keyword_handler(keyword: str) -> Callable | None:
    return _keyword_handlers.get(keyword)


def get_expr_handler(name: str) -> Callable | None:
    return _expr_handlers.get(name)


def get_domain_protocol(tag: str) -> DomainProtocol | None:
    return _domain_protocols.get(tag)


def registered_keywords() -> frozenset[str]:
    return frozenset(_keyword_handlers)


def registered_domains() -> frozenset[str]:
    return frozenset(_domain_protocols)
