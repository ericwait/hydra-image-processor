#pragma once

// MSVC preprocessor doesn't expand variadic arguments default without extra indirection
#define _MSVC_EXPAND(...) __VA_ARGS__

// Expand/evaluate inputs (this is recursion compatible)
#ifdef _MSC_VER
	#define PRP_EXPAND(...) _MSVC_EXPAND(__VA_ARGS__)
#else
	#define PRP_EXPAND(...) __VA_ARGS__
#endif

// Expand (upt to 64 times) to fully evaluate nested argument macros
#define _EVAL_1(...) __VA_ARGS__
#define _EVAL_4(...) _EVAL_1(_EVAL_1(_EVAL_1(_EVAL_1(__VA_ARGS__))))
#define _EVAL_16(...) _EVAL_4(_EVAL_4(_EVAL_4(_EVAL_4(__VA_ARGS__))))
#define PRP_EVAL(...) _EVAL_16(_EVAL_16(_EVAL_16(_EVAL_16(__VA_ARGS__))))


// Expand and concatenate arguments
#define _PRIM_CAT(a, ...) _MSVC_EXPAND(a ## __VA_ARGS__)
#define PRP_CAT(a,...) _MSVC_EXPAND(_PRIM_CAT(a,__VA_ARGS__))

// If conditional (arguments must evaluate to 0/1)
#define IIF(cond) PRP_CAT(IIF_, cond)
#define IIF_0(t, f) f
#define IIF_1(t, f) t

// General token check returns 0 unless macro evaluates to PROBE(~)
#define CHECK_N(x, n, ...) n
#define CHECK(...) _MSVC_EXPAND(CHECK_N(__VA_ARGS__, 0,))
#define PROBE(x) x, 1,

// Eat next argument
#define EAT(...)


// Helper macros for use as specialized FOREACH arguments
// Remove parentheses from an arg (e.g. (a) -> a)
#define _UNWRAP_IMPL(...) __VA_ARGS__
#define _UNWRAP_PAREN(x) _UNWRAP_IMPL x

// Wrap an element or set of elements in parenthesis (e.g. a,b -> (a,b))
#define _WRAP_PAREN(...) _MSVC_EXPAND((__VA_ARGS__))

// Don't do anything to a set of elements
#define _IDENTITY(...) __VA_ARGS__

// Place a comma
#define _DELIM_COMMA() ,
#define _DELIM_EMPTY()

// Helpers for generalized FOREACH macro (applies macro to each argument)
// NOTE: Expands in reverse to avoid comma issues
#define _MAPL_1(_apply,_wrap_out,_wrap_arg,_delim, x) _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_2(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_1(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_3(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_2(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_4(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_3(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_5(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_4(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_6(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_5(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_7(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_6(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_8(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_7(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_9(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_8(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))
#define _MAPL_10(_apply,_wrap_out,_wrap_arg,_delim, x, ...) _MSVC_EXPAND(_MAPL_9(_apply,_wrap_out,_wrap_arg,_delim, __VA_ARGS__)) _delim() _wrap_out(PRP_EXPAND(_apply _wrap_arg(x)))

#define _NTH_ARG_10(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define _REV_FOREACH(_apply,_wrap,_unwrap,_delim, ...)		\
			_MSVC_EXPAND(_NTH_ARG_10(__VA_ARGS__,				\
				_MAPL_10, _MAPL_9, _MAPL_8, _MAPL_7, _MAPL_6,	\
				_MAPL_5, _MAPL_4, _MAPL_3, _MAPL_2, _MAPL_1)	\
				(_apply,_wrap,_unwrap,_delim, __VA_ARGS__))


// PRP_FOREACH_IOD - Most general for-each construct
//  _apply - macro transform to apply to each argument
//  _wrap_out - macro to wrap _apply macro's output (e.g. in parentheses)
//  _wrap_arg - macro to wrap input arguments in parentheses
//  _delim - empty function-like macro to generate delimiters (e.g. _COMMA() -> ,)
#define PRP_FOREACH_IOD(_apply, _wrap_out, _wrap_arg, _delim, ...)		\
			_REV_FOREACH(_apply, _wrap_out, _IDENTITY, _delim,			\
				_REV_FOREACH(_IDENTITY, _WRAP_PAREN, _wrap_arg, _DELIM_COMMA, __VA_ARGS__))

// Default FOREACH takes paren-wrapped input returns unwrapped output
#define PRP_FOREACH(_apply, ...) PRP_FOREACH_IOD(_apply, _IDENTITY, _IDENTITY, _DELIM_COMMA, __VA_ARGS__)
// Composable FOREACH takes wrapped input returns wrapped output
#define PRP_FOREACH_C(_apply, ...) PRP_FOREACH_IOD(_apply, _WRAP_PAREN, _IDENTITY, _DELIM_COMMA, __VA_ARGS__)

// Do full expansion (this is the dumbest but simplest way to make an arg selector)
#define _SEL3_000(arg1,arg2,arg3)
#define _SEL3_100(arg1,arg2,arg3) arg1
#define _SEL3_010(arg1,arg2,arg3) arg2
#define _SEL3_001(arg1,arg2,arg3) arg3
#define _SEL3_110(arg1,arg2,arg3) arg1 arg2
#define _SEL3_101(arg1,arg2,arg3) arg1 arg3
#define _SEL3_011(arg1,arg2,arg3) arg2 arg3
#define _SEL3_111(arg1,arg2,arg3) arg1 arg2 arg3

// Select any subset of 3 arguments
#define PRP_SEL3(sel1,sel2,sel3) PRP_CAT(PRP_CAT(PRP_CAT(_SEL3_, sel1), sel2), sel3)
