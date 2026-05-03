"""Parser for the .ua DSL. Grammar in _grammar.py, resolution in _resolver.py."""

from ._parse import parse_ua as parse_ua
from ._parse import parse_ua_spec as parse_ua_spec
from ._resolve_cells import NamedCell as NamedCell
from ._resolver import UASpec as UASpec

__all__ = ["parse_ua_spec", "parse_ua", "UASpec", "NamedCell"]
