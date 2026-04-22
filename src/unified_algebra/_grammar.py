"""Grammar rules for the .ua DSL.

Builds Hydra parser combinators that transform .ua source text into
raw declaration tuples.  No semantic resolution — that lives in _resolver.py.
"""
from __future__ import annotations


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

    # Boolean literals
    _true = P.bind(P.string('true'), lambda _: P.pure(True))
    _false = P.bind(P.string('false'), lambda _: P.pure(False))
    _bool_lit = _tok(P.alt(_false, _true))

    def _indented_kv():
        """Parse one indented key=value line, returning (key, value).

        Value may be a string literal, boolean, number, or identifier.
        """
        return P.bind(_indent, lambda _:
               P.bind(ident, lambda k:
               P.bind(sym('='), lambda _:
               P.alt(
                   P.bind(string_lit, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
               P.alt(
                   P.bind(_bool_lit, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
               P.alt(
                   P.bind(number_lit, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
                   P.bind(ident, lambda v:
                          P.bind(_eol, lambda _:
                          P.pure((k, v)))),
               ))))))

    def _indented_list(key):
        """Parse one indented key=[a,b,c] line, returning list of idents or template refs."""
        _ident_list = P.bind(sym('['), lambda _:
                      P.bind(P.sep_by1(ident_or_tpl, _comma), lambda items:
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
                P.bind(_sep_by_gg(ident_or_tpl), lambda eq_names:
                P.bind(_eol, lambda _:
                P.bind(P.many(_indented_kv()), lambda kv_list:
                P.pure(('path', name, sig, eq_names, dict(kv_list)))))))))))

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

    # unfold stream : hidden -> hidden
    #   step = transition
    #   n_steps = 10
    unfold_decl = P.bind(sym('unfold'), lambda _:
                  P.bind(ident, lambda name:
                  P.bind(sym(':'), lambda _:
                  P.bind(_sort_sig, lambda sig:
                  P.bind(_eol, lambda _:
                  P.bind(_indented_kv(), lambda step_kv:
                  P.bind(_indented_kv(), lambda nsteps_kv:
                  P.pure(('unfold', name, sig, step_kv[1], int(nsteps_kv[1]))))))))))

    # fixpoint converge : hidden
    #   step = step_eq
    #   predicate = residual_eq
    #   epsilon = 0.001
    #   max_iter = 100
    fixpoint_decl = P.bind(sym('fixpoint'), lambda _:
                    P.bind(ident, lambda name:
                    P.bind(sym(':'), lambda _:
                    P.bind(ident, lambda sort_name:
                    P.bind(_eol, lambda _:
                    P.bind(P.many(_indented_kv()), lambda kv_list:
                    P.pure(('fixpoint', name, sort_name, dict(kv_list)))))))))

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

    # lens_fan attention : hidden <-> hidden
    #   branches = [backprop1, backprop2]
    #   merge = merge_lens
    lens_fan_decl = P.bind(sym('lens_fan'), lambda _:
                    P.bind(ident, lambda name:
                    P.bind(sym(':'), lambda _:
                    P.bind(_lens_sig, lambda sig:
                    P.bind(_eol, lambda _:
                    P.bind(_indented_list('branches'), lambda branches:
                    P.bind(_indented_kv(), lambda merge_kv:
                    P.pure(('lens_fan', name, sig, branches, merge_kv[1])))))))))

    # -------------------------------------------------------------------
    # Any declaration (order matters for alt: longer keywords before shorter)
    # -------------------------------------------------------------------
    decl = P.alt(semiring_decl,
           P.alt(sort_decl,
           P.alt(equation_decl,
           P.alt(path_decl,
           P.alt(fan_decl,
           P.alt(fold_decl,
           P.alt(unfold_decl,
           P.alt(fixpoint_decl,
           P.alt(lens_fan_decl,      # must come before lens_path and lens
           P.alt(lens_path_decl,     # must come before lens_decl
                 lens_decl))))))))))

    # -------------------------------------------------------------------
    # Full program: blank lines + declarations
    # -------------------------------------------------------------------
    program = P.bind(skip_blank, lambda _:
              P.bind(P.many(P.bind(decl, lambda d:
                            P.bind(skip_blank, lambda _:
                            P.pure(d)))), lambda decls:
              P.pure(list(decls))))

    return program
