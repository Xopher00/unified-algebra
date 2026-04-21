"""Text parser for the unified-algebra DSL (.ua files).

Layer role: syntactic sugar over the stable Python API.  Parses `.ua` source
text into the same Python objects that hand-written code would produce, then
delegates to compile_program() for compilation.

Entry points:
    parse_ua_spec(text)          -> UASpec  (introspection / testing)
    parse_ua(text, backend)      -> Program (ready to call)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .program import Program


# ---------------------------------------------------------------------------
# UASpec — the parse tree after resolution
# ---------------------------------------------------------------------------


@dataclass
class UASpec:
    """Parsed .ua program before compilation.

    Fields contain the resolved DSL objects (semiring terms, sort terms, etc.)
    in declaration order.  Suitable for passing directly to compile_program().
    """
    semirings: dict[str, Any] = field(default_factory=dict)   # name -> semiring Term
    sorts: dict[str, Any] = field(default_factory=dict)        # name -> sort Term
    equations: list[Any] = field(default_factory=list)         # equation Terms
    specs: list[Any] = field(default_factory=list)             # PathSpec|FanSpec|FoldSpec|…
    lenses: list[Any] = field(default_factory=list)            # lens Terms


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser():
    """Build the full .ua program parser.  Returns the top-level Parser[list[tuple]]."""
    import hydra.parsers as P
    from hydra.parsers import Nothing

    # -------------------------------------------------------------------
    # Primitives
    # -------------------------------------------------------------------

    # Whitespace — inline only (no newlines, which are meaningful)
    _iws = P.many(P.satisfy(lambda c: chr(c) in ' \t'))

    def _tok(p):
        """Wrap p with inline-whitespace on both sides."""
        return P.bind(_iws, lambda _:
               P.bind(p, lambda x:
               P.bind(_iws, lambda __:
               P.pure(x))))

    def sym(s):
        return _tok(P.string(s))

    _newline = P.satisfy(lambda c: chr(c) in '\n\r')
    _not_nl = P.satisfy(lambda c: chr(c) not in '\n\r')

    # End-of-line: inline ws + optional comment + newline
    _eol = P.bind(_iws, lambda _:
           P.bind(P.optional(P.bind(P.char(ord('#')), lambda _:
                             P.bind(P.many(_not_nl), lambda _:
                             P.pure(None)))), lambda _:
           P.bind(_newline, lambda _:
           P.pure(None))))

    # Blank/comment lines — must match lines that are ONLY whitespace/comment.
    # Three variants:
    #   1. pure newline (empty line)
    #   2. '#' comment at column 0 + newline
    #   3. spaces/tabs + optional '#' comment + newline
    # Using alt ensures each variant backtracks cleanly.
    _hash_comment_nl = P.bind(P.char(ord('#')), lambda _:
                       P.bind(P.many(_not_nl), lambda _:
                       P.bind(_newline, lambda _: P.pure(None))))
    _ws_line = P.bind(P.some(P.satisfy(lambda c: chr(c) in ' \t')), lambda _:
               P.bind(P.alt(_hash_comment_nl, P.bind(_newline, lambda _: P.pure(None))),
               lambda x: P.pure(x)))
    _blank_line = P.alt(P.bind(_newline, lambda _: P.pure(None)),
                  P.alt(_hash_comment_nl, _ws_line))
    skip_blank = P.many(_blank_line)

    # Identifier
    _id_start = P.satisfy(lambda c: chr(c).isalpha() or chr(c) == '_')
    _id_rest = P.many(P.satisfy(lambda c: chr(c).isalnum() or chr(c) == '_'))
    _raw_ident = P.bind(_id_start, lambda c:
                 P.bind(_id_rest, lambda cs:
                 P.pure(chr(c) + ''.join(chr(x) for x in cs))))
    ident = _tok(_raw_ident)

    # String literal  "…"
    _dq = P.char(ord('"'))
    _not_dq = P.satisfy(lambda c: c != ord('"'))
    _raw_str = P.bind(_dq, lambda _:
               P.bind(P.many(_not_dq), lambda cs:
               P.bind(_dq, lambda _:
               P.pure(''.join(chr(c) for c in cs)))))
    string_lit = _tok(_raw_str)

    # Number: optional minus, digits, optional decimal, or 'inf'/'-inf'
    _digit = P.satisfy(lambda c: chr(c).isdigit())
    _digits = P.some(_digit)
    _decimal = P.bind(P.char(ord('.')), lambda _:
               P.bind(_digits, lambda ds:
               P.pure('.' + ''.join(chr(c) for c in ds))))
    _plain_num = P.bind(_digits, lambda ds:
                 P.bind(P.optional(_decimal), lambda mf:
                 P.pure(float(
                     ''.join(chr(c) for c in ds) +
                     (mf.value if not isinstance(mf, Nothing) else '')
                 ))))
    _inf = P.bind(P.string('inf'), lambda _: P.pure(float('inf')))
    _neg_inf = P.bind(P.string('-inf'), lambda _: P.pure(float('-inf')))
    number_lit = _tok(P.alt(_neg_inf, P.alt(_inf, _plain_num)))

    # -------------------------------------------------------------------
    # Indented attribute block
    #
    # Parses lines of the form:
    #   <indent> <key> = <value> <eol>
    # where value is either a string literal or an identifier.
    #
    # Key insight for hydra.parsers alt/many: the attr parser must fail
    # *without* consuming input when there is no indent.  P.some(ws_char)
    # satisfies this — it fails immediately if the first char is not a
    # space/tab.
    # -------------------------------------------------------------------

    _indent = P.bind(P.some(P.satisfy(lambda c: chr(c) in ' \t')), lambda _: P.pure(None))
    _comma = sym(',')

    def _indented_kv():
        """Parse one indented key=value line, returning (key, value)."""
        return P.bind(_indent, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.alt(
                   P.bind(string_lit, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
                   P.bind(ident, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
               ))))

    def _indented_list(key):
        """Parse one indented key=[a,b,c] line, returning list of idents."""
        _ident_list = P.bind(sym('['), lambda _:
                      P.bind(P.sep_by1(ident, _comma), lambda items:
                      P.bind(sym(']'), lambda _:
                      P.pure(list(items)))))
        return P.bind(_indent, lambda _:
               P.bind(sym(key), lambda _:
               P.bind(sym('='), lambda _:
               P.bind(_ident_list, lambda v:
               P.bind(_eol, lambda _:
               P.pure(v))))))

    # -------------------------------------------------------------------
    # Sort type signatures — factored over the separator
    # -------------------------------------------------------------------

    def _sig(sep):
        return P.bind(ident, lambda dom:
               P.bind(sep, lambda _:
               P.bind(ident, lambda cod:
               P.pure((dom, cod)))))

    _sort_sig = _sig(sym('->'))
    _lens_sig  = _sig(sym('<->'))

    # -------------------------------------------------------------------
    # Individual declaration parsers
    # -------------------------------------------------------------------

    def _kv_ident(k):
        return P.bind(sym(k), lambda _:
               P.bind(sym('='), lambda _:
               P.bind(ident, lambda v:
               P.pure(v))))

    def _kv_num(k):
        return P.bind(sym(k), lambda _:
               P.bind(sym('='), lambda _:
               P.bind(number_lit, lambda v:
               P.pure(v))))

    _sr_args = P.bind(sym('('), lambda _:
               P.bind(_kv_ident('plus'), lambda plus:
               P.bind(_comma, lambda _:
               P.bind(_kv_ident('times'), lambda times:
               P.bind(_comma, lambda _:
               P.bind(_kv_num('zero'), lambda zero:
               P.bind(_comma, lambda _:
               P.bind(_kv_num('one'), lambda one:
               P.bind(sym(')'), lambda _:
               P.pure(dict(plus=plus, times=times, zero=zero, one=one)))))))))))

    # semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
    semiring_decl = P.bind(sym('semiring'), lambda _:
                    P.bind(ident, lambda name:
                    P.bind(_sr_args, lambda kw_args:
                    P.bind(_eol, lambda _:
                    P.pure(('semiring', name, kw_args))))))

    # sort hidden(real) | sort hidden(real, batched)
    _batched_flag = P.bind(_comma, lambda _:
                    P.bind(sym('batched'), lambda _:
                    P.pure(True)))

    sort_decl = P.bind(sym('sort'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym('('), lambda _:
                P.bind(ident, lambda sr_name:
                P.bind(P.optional(_batched_flag), lambda mb:
                P.bind(sym(')'), lambda _:
                P.bind(_eol, lambda _:
                P.pure(('sort', name, sr_name, not isinstance(mb, Nothing) and mb.value is True)))))))))

    # equation linear : hidden -> hidden
    #   einsum = "ij,j->i"
    #   semiring = real
    equation_decl = P.bind(sym('equation'), lambda _:
                    P.bind(ident, lambda name:
                    P.bind(sym(':'), lambda _:
                    P.bind(_sort_sig, lambda sig:
                    P.bind(_eol, lambda _:
                    P.bind(P.many(_indented_kv()), lambda kv_list:
                    P.pure(('equation', name, sig, dict(kv_list)))))))))

    # path layer : hidden -> hidden = linear >> bias >> relu
    def _sep_by_gg(p):
        _gg = sym('>>')
        return P.bind(p, lambda first:
               P.bind(P.many(P.bind(_gg, lambda _: p)), lambda rest:
               P.pure([first] + list(rest))))

    path_decl = P.bind(sym('path'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym(':'), lambda _:
                P.bind(_sort_sig, lambda sig:
                P.bind(sym('='), lambda _:
                P.bind(_sep_by_gg(ident), lambda eq_names:
                P.bind(_eol, lambda _:
                P.pure(('path', name, sig, eq_names)))))))))

    # fan split : hidden -> hidden
    #   branches = [linear, relu_branch]
    #   merge = add_merge
    fan_decl = P.bind(sym('fan'), lambda _:
               P.bind(ident, lambda name:
               P.bind(sym(':'), lambda _:
               P.bind(_sort_sig, lambda sig:
               P.bind(_eol, lambda _:
               P.bind(_indented_list('branches'), lambda branches:
               P.bind(_indented_kv(), lambda merge_kv:
               P.pure(('fan', name, sig, branches, merge_kv[1])))))))))

    # fold rnn : hidden -> hidden
    #   step = layer
    fold_decl = P.bind(sym('fold'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym(':'), lambda _:
                P.bind(_sort_sig, lambda sig:
                P.bind(_eol, lambda _:
                P.bind(_indented_kv(), lambda step_kv:
                P.pure(('fold', name, sig, step_kv[1]))))))))

    # lens backprop : hidden <-> hidden
    #   fwd = linear
    #   bwd = linear_bwd
    lens_decl = P.bind(sym('lens'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym(':'), lambda _:
                P.bind(_lens_sig, lambda sig:
                P.bind(_eol, lambda _:
                P.bind(_indented_kv(), lambda fwd_kv:
                P.bind(_indented_kv(), lambda bwd_kv:
                P.pure(('lens', name, sig, fwd_kv[1], bwd_kv[1])))))))))

    # lens_path pipe : hidden <-> hidden = backprop >> backprop2
    lens_path_decl = P.bind(sym('lens_path'), lambda _:
                     P.bind(ident, lambda name:
                     P.bind(sym(':'), lambda _:
                     P.bind(_lens_sig, lambda sig:
                     P.bind(sym('='), lambda _:
                     P.bind(_sep_by_gg(ident), lambda lens_names:
                     P.bind(_eol, lambda _:
                     P.pure(('lens_path', name, sig, lens_names)))))))))

    # -------------------------------------------------------------------
    # Any declaration (order matters for alt: longer keywords before shorter)
    # -------------------------------------------------------------------
    decl = P.alt(semiring_decl,
           P.alt(sort_decl,
           P.alt(equation_decl,
           P.alt(path_decl,
           P.alt(fan_decl,
           P.alt(fold_decl,
           P.alt(lens_path_decl,   # must come before lens_decl
                 lens_decl)))))))

    # -------------------------------------------------------------------
    # Full program: blank lines + declarations
    # -------------------------------------------------------------------
    program = P.bind(skip_blank, lambda _:
              P.bind(P.many(P.bind(decl, lambda d:
                            P.bind(skip_blank, lambda _:
                            P.pure(d)))), lambda decls:
              P.pure(list(decls))))

    return program


# ---------------------------------------------------------------------------
# Reference resolution pass
# ---------------------------------------------------------------------------

def _resolve_spec(raw_decls: list[tuple]) -> UASpec:
    """Second pass: resolve name references, build DSL terms.

    Processes declarations in dependency order:
    1. semirings (no deps)
    2. sorts (depend on semirings by name)
    3. equations (depend on sorts + semirings by name)
    4. compositions (depend on equations and lenses by name)
    """
    from .semiring import semiring as mk_semiring
    from .sort import sort as mk_sort
    from .morphism import equation as mk_equation
    from .composition import lens as mk_lens
    from .specs import PathSpec, FanSpec, FoldSpec, LensPathSpec

    semirings: dict[str, Any] = {}
    sorts: dict[str, Any] = {}
    equations_by_name: dict[str, Any] = {}
    equations_list: list[Any] = []
    specs: list[Any] = []
    lenses: list[Any] = []
    lenses_by_name: dict[str, Any] = {}

    def _lookup(name: str, d: dict, label: str) -> Any:
        if name not in d:
            raise ValueError(
                f"Unknown {label} {name!r} — declared {label}s: {list(d)}"
            )
        return d[name]

    def _get_sr(name):   return _lookup(name, semirings, 'semiring')
    def _get_sort(name): return _lookup(name, sorts, 'sort')
    def _get_eq(name):   return _lookup(name, equations_by_name, 'equation')

    for decl in raw_decls:
        kind = decl[0]

        if kind == 'semiring':
            _, name, kw_args = decl
            sr_term = mk_semiring(name, plus=kw_args['plus'], times=kw_args['times'],
                                  zero=kw_args['zero'], one=kw_args['one'])
            semirings[name] = sr_term

        elif kind == 'sort':
            _, name, sr_name, batched = decl
            sr_term = _get_sr(sr_name)
            sort_term = mk_sort(name, sr_term, batched=batched)
            sorts[name] = sort_term

        elif kind == 'equation':
            _, name, (dom_name, cod_name), attr_dict = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            einsum = attr_dict.get('einsum', None) or None
            nl = attr_dict.get('nonlinearity', None) or None
            sr_name = attr_dict.get('semiring', None)
            sr_term = _get_sr(sr_name) if sr_name else None
            eq_term = mk_equation(name, einsum, dom_sort, cod_sort,
                                  sr_term, nonlinearity=nl)
            equations_by_name[name] = eq_term
            equations_list.append(eq_term)

        elif kind == 'path':
            _, name, (dom_name, cod_name), eq_names = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for en in eq_names:
                _get_eq(en)
            specs.append(PathSpec(
                name=name,
                eq_names=eq_names,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

        elif kind == 'fan':
            _, name, (dom_name, cod_name), branches, merge = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for bn in branches:
                _get_eq(bn)
            _get_eq(merge)
            specs.append(FanSpec(
                name=name,
                branch_names=branches,
                merge_name=merge,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

        elif kind == 'fold':
            _, name, (dom_name, state_name), step = decl
            dom_sort = _get_sort(dom_name)
            state_sort = _get_sort(state_name)
            _get_eq(step)
            specs.append(FoldSpec(
                name=name,
                step_name=step,
                init_term=None,
                domain_sort=dom_sort,
                state_sort=state_sort,
            ))

        elif kind == 'lens':
            _, name, (_, _), fwd, bwd = decl
            _get_eq(fwd)
            _get_eq(bwd)
            lens_term = mk_lens(name, fwd, bwd)
            lenses.append(lens_term)
            lenses_by_name[name] = lens_term

        elif kind == 'lens_path':
            _, name, (dom_name, cod_name), lens_names = decl
            dom_sort = _get_sort(dom_name)
            cod_sort = _get_sort(cod_name)
            for ln in lens_names:
                if ln not in lenses_by_name:
                    raise ValueError(
                        f"Unknown lens {ln!r} — declared lenses: {list(lenses_by_name)}"
                    )
            specs.append(LensPathSpec(
                name=name,
                lens_names=lens_names,
                domain_sort=dom_sort,
                codomain_sort=cod_sort,
            ))

    return UASpec(
        semirings=semirings,
        sorts=sorts,
        equations=equations_list,
        specs=specs,
        lenses=lenses,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ua_spec(text: str) -> UASpec:
    """Parse .ua source text and return a UASpec without compiling.

    Useful for introspection and testing without needing a backend.

    Args:
        text: .ua source text

    Returns:
        UASpec with semirings, sorts, equations, specs, and lenses populated.

    Raises:
        SyntaxError: if the text cannot be parsed
        ValueError:  if sort/semiring references are invalid
    """
    import hydra.parsers as P
    import hydra.parsing as HP

    program_parser = _build_parser()
    result = P.run_parser(program_parser, text)

    if isinstance(result, HP.ParseResultFailure):
        err = result.value
        snippet = repr(err.remainder[:40]) if err.remainder else "<end of input>"
        raise SyntaxError(
            f"Parse error: {err.message} at {snippet}"
        )

    raw_decls = result.value.value
    remainder = result.value.remainder.strip()
    if remainder:
        snippet = repr(remainder[:40])
        raise SyntaxError(f"Unexpected input near {snippet}")

    return _resolve_spec(raw_decls)


def parse_ua(text: str, backend) -> "Program":
    """Parse a .ua program text and compile it to a Program.

    Args:
        text:    .ua source text
        backend: Backend (numpy_backend() or pytorch_backend())

    Returns:
        A compiled Program, callable by entry point name.

    Raises:
        SyntaxError: if the text cannot be parsed
        ValueError:  if sort junctions or references are invalid
    """
    from .program import compile_program

    spec = parse_ua_spec(text)

    return compile_program(
        spec.equations,
        backend=backend,
        specs=spec.specs,
        lenses=spec.lenses if spec.lenses else None,
    )
