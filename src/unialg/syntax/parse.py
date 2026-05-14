"""Public parser API for the unialg surface syntax.

  parse_morphism(src, env=None) -> MorphismExpr
  parse_functor(src, env=None)  -> PolyExpr

env maps previously-defined names to their expression nodes so the
parser resolves references inline. Unresolved names become Ref / PolyRef
placeholder nodes.

dom/cod fields on parsed ContextualBinary nodes are TypeUnit() placeholders.
Semantics (morphisms.py) owns type derivation.
"""
from __future__ import annotations

from unialg.syntax.expressions import MorphismExpr, PolyExpr
from unialg.syntax._lex import tokenize_morphism, tokenize_functor
from unialg.syntax._morphism_grammar import make_morphism_grammar, Env as MEnv
from unialg.syntax._functor_grammar import make_functor_grammar, Env as FEnv
from unialg.syntax._pratt import PrattParser, ParseError  # re-export

__all__ = ["parse_morphism", "parse_functor", "ParseError"]


def parse_morphism(src: str, env: MEnv | None = None) -> MorphismExpr:
    tokens = tokenize_morphism(src)
    nud, led, bp = make_morphism_grammar(env)
    p = PrattParser(tokens, label="morphism", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]


def parse_functor(src: str, env: FEnv | None = None) -> PolyExpr:
    tokens = tokenize_functor(src)
    nud, led, bp = make_functor_grammar(env)
    p = PrattParser(tokens, label="functor", binding_powers=bp, nud=nud, led=led)
    return p.parse_all()  # type: ignore[return-value]
