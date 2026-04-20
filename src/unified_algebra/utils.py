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


def float_value(term) -> float:
    """Extract a float from a Hydra TermLiteral(LiteralFloat(FloatValueFloat64(float)))."""
    return term.value.value


def eq_name(eq_term) -> str:
    """Extract an equation's name from its Hydra record."""
    return string_value(record_fields(eq_term)["name"])


def lens_fields(term) -> dict[str, object]:
    """Extract name/forward/backward/residualSort from a lens record."""
    fields = record_fields(term)
    return {
        "name": string_value(fields["name"]),
        "forward": string_value(fields["forward"]),
        "backward": string_value(fields["backward"]),
        "residualSort": fields.get("residualSort"),
    }
