"""Hydra term-navigation helpers.

Utilities for extracting values from Hydra record terms at the Python level.
For field access inside Hydra lambda terms, use hydra.dsl.terms.project() directly.
"""


def record_fields(term) -> dict[str, object]:
    """Extract a Hydra record's fields as a {name_str: Term} dict."""
    return {f.name.value: f.term for f in term.value.fields}


def string_value(term) -> str:
    """Extract a plain string from a Hydra TermLiteral(LiteralString(str))."""
    return term.value.value




def bind_composition(kind, name, var_name, body):
    """Wrap a body term in a lambda and return (Name, lambda_term)."""
    import hydra.core as core
    from hydra.dsl.meta.phantoms import lam, TTerm
    if not isinstance(body, TTerm):
        body = TTerm(body)
    term = lam(var_name, body).value
    return (core.Name(f"ua.{kind}.{name}"), term)
