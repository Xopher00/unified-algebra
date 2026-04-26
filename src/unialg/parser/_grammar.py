"""Grammar rules for the .ua DSL.

Builds Hydra parser combinators that transform .ua source text into
raw declaration tuples.  No semantic resolution — that lives in _resolver.py.
"""
from __future__ import annotations

from functools import reduce


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

    # Template reference: ident optionally followed by [prefix]
    # e.g., "proj[q]" -> ('_tpl', 'proj', 'q')
    # e.g., "linear"  -> 'linear'  (plain string)
    _bracket_suffix = P.bind(P.char(ord('[')), lambda _:
                      P.bind(_raw_ident, lambda prefix:
                      P.bind(_tok(P.char(ord(']'))), lambda _:
                      P.pure(prefix))))
    ident_or_tpl = P.bind(ident, lambda name:
                   P.bind(P.optional(_bracket_suffix), lambda mb:
                   P.pure(name if isinstance(mb, Nothing) else ('_tpl', name, mb.value))))

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

    # Attr parser must fail without consuming when no indent (P.some guarantees this)
    _indent = P.bind(P.some(P.satisfy(lambda c: chr(c) in ' \t')), lambda _: P.pure(None))
    _comma = sym(',')

    # Boolean literals
    _true = P.bind(P.string('true'), lambda _: P.pure(True))
    _false = P.bind(P.string('false'), lambda _: P.pure(False))
    _bool_lit = _tok(P.alt(_false, _true))

    _any_value = P.alt(string_lit, P.alt(_bool_lit, P.alt(number_lit, ident)))

    def _indented_kv():
        return P.bind(_indent, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.bind(_any_value, lambda v:
               P.bind(_eol, lambda _:
               P.pure((k, v)))))))

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

    def _kv(k, vp):
        return P.bind(sym(k), lambda _:
               P.bind(sym('='), lambda _: vp))

    _sr_args = P.bind(sym('('), lambda _:
               P.bind(_kv('plus', ident), lambda plus:
               P.bind(_comma, lambda _:
               P.bind(_kv('times', ident), lambda times:
               P.bind(_comma, lambda _:
               P.bind(_kv('zero', number_lit), lambda zero:
               P.bind(_comma, lambda _:
               P.bind(_kv('one', number_lit), lambda one:
               P.bind(sym(')'), lambda _:
               P.pure(dict(plus=plus, times=times, zero=zero, one=one)))))))))))

    algebra_decl = P.bind(sym('algebra'), lambda _:
                   P.bind(ident, lambda name:
                   P.bind(_sr_args, lambda kw_args:
                   P.bind(_eol, lambda _:
                   P.pure(('algebra', name, kw_args))))))

    _batched_flag = P.bind(_comma, lambda _:
                    P.bind(sym('batched'), lambda _:
                    P.pure(True)))

    spec_decl = P.bind(sym('spec'), lambda _:
                P.bind(ident, lambda name:
                P.bind(sym('('), lambda _:
                P.bind(ident, lambda sr_name:
                P.bind(P.optional(_batched_flag), lambda mb:
                P.bind(sym(')'), lambda _:
                P.bind(_eol, lambda _:
                P.pure(('spec', name, sr_name, not isinstance(mb, Nothing) and mb.value is True)))))))))

    def _kv_decl(keyword, sig_parser, tag):
        return P.bind(sym(keyword), lambda _:
               P.bind(ident, lambda name:
               P.bind(sym(':'), lambda _:
               P.bind(sig_parser, lambda sig:
               P.bind(_eol, lambda _:
               P.bind(P.many(_indented_kv()), lambda kv_list:
               P.pure((tag, name, sig, dict(kv_list)))))))))

    op_decl = P.bind(sym('op'), lambda _:
              P.bind(P.optional(sym('~')), lambda tilde:
              P.bind(ident, lambda name:
              P.bind(sym(':'), lambda _:
              P.bind(_sort_sig, lambda sig:
              P.bind(_eol, lambda _:
              P.bind(P.many(_indented_kv()), lambda kv_list:
              P.pure(('op', name, sig,
                      {**dict(kv_list),
                       **({"template": True} if not isinstance(tilde, Nothing) else {})}
                     )))))))))

    def _sep_by_gg(p):
        _gg = sym('>>')
        return P.bind(p, lambda first:
               P.bind(P.many(P.bind(_gg, lambda _: p)), lambda rest:
               P.pure([first] + list(rest))))

    def _sep_by_pipe(p):
        _pipe = sym('|')
        return P.bind(p, lambda first:
               P.bind(P.many(P.bind(_pipe, lambda _: p)), lambda rest:
               P.pure([first] + list(rest))))

    seq_decl = P.bind(sym('seq'), lambda _:
               P.bind(ident, lambda name:
               P.bind(P.optional(P.char(ord('+'))), lambda plus:
               P.bind(sym(':'), lambda _:
               P.bind(_sort_sig, lambda sig:
               P.bind(sym('='), lambda _:
               P.bind(_sep_by_gg(ident_or_tpl), lambda eq_names:
               P.bind(_eol, lambda _:
               P.bind(P.many(_indented_kv()), lambda kv_list:
               P.pure(('seq', name, sig, eq_names,
                       {**dict(kv_list),
                        **({"residual": True} if not isinstance(plus, Nothing) else {})}
                      )))))))))))

    def _branch_like(keyword, sig_parser, tag):
        return P.bind(sym(keyword), lambda _:
               P.bind(ident, lambda name:
               P.bind(sym(':'), lambda _:
               P.bind(sig_parser, lambda sig:
               P.bind(sym('='), lambda _:
               P.bind(_sep_by_pipe(ident_or_tpl), lambda branches:
               P.bind(_eol, lambda _:
               P.bind(_indented_kv(), lambda merge_kv:
               P.pure((tag, name, sig, branches, merge_kv[1]))))))))))

    branch_decl = _branch_like('branch', _sort_sig, 'branch')
    scan_decl = _kv_decl('scan', _sort_sig, 'scan')
    unroll_decl = _kv_decl('unroll', _sort_sig, 'unroll')
    fixpoint_decl = _kv_decl('fixpoint', ident, 'fixpoint')
    lens_decl = _kv_decl('lens', _lens_sig, 'lens')

    lens_seq_decl = P.bind(sym('lens_seq'), lambda _:
                    P.bind(ident, lambda name:
                    P.bind(sym(':'), lambda _:
                    P.bind(_lens_sig, lambda sig:
                    P.bind(sym('='), lambda _:
                    P.bind(_sep_by_gg(ident), lambda lens_names:
                    P.bind(_eol, lambda _:
                    P.pure(('lens_seq', name, sig, lens_names)))))))))

    lens_branch_decl = _branch_like('lens_branch', _lens_sig, 'lens_branch')

    decl = reduce(P.alt, [
        algebra_decl, spec_decl, op_decl, seq_decl,
        branch_decl, scan_decl, unroll_decl, fixpoint_decl,
        lens_branch_decl, lens_seq_decl, lens_decl,
    ])

    # -------------------------------------------------------------------
    # Full program: blank lines + declarations
    # -------------------------------------------------------------------
    program = P.bind(skip_blank, lambda _:
              P.bind(P.many(P.bind(decl, lambda d:
                            P.bind(skip_blank, lambda _:
                            P.pure(d)))), lambda decls:
              P.pure(list(decls))))

    return program
