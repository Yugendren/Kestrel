#ifndef KYTY_COMMON_COMMON_H_
#define KYTY_COMMON_COMMON_H_

#include "common/config.h" // IWYU pragma: export

#if __cplusplus < 201703L
#undef __cplusplus
#define __cplusplus 201703L
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,cert-dcl51-cpp,cert-dcl37-c,bugprone-reserved-identifier)
#define __USE_MINGW_ANSI_STDIO 1
#endif

// IWYU pragma: begin_exports
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
// IWYU pragma: end_exports

#define KYTY_CLASS_NO_COPY(name)                                                                   \
public:                                                                                            \
	name(const name&)                = delete; /* NOLINT(bugprone-macro-parentheses) */            \
	name& operator=(const name&)     = delete; /* NOLINT(bugprone-macro-parentheses) */            \
	name(name&&) noexcept            = delete; /* NOLINT(bugprone-macro-parentheses) */            \
	name& operator=(name&&) noexcept = delete; /* NOLINT(bugprone-macro-parentheses) */

#define KYTY_CLASS_DEFAULT_COPY(name)                                                              \
public:                                                                                            \
	name(const name&)                = default; /* NOLINT(bugprone-macro-parentheses) */           \
	name& operator=(const name&)     = default; /* NOLINT(bugprone-macro-parentheses) */           \
	name(name&&) noexcept            = default; /* NOLINT(bugprone-macro-parentheses) */           \
	name& operator=(name&&) noexcept = default; /* NOLINT(bugprone-macro-parentheses) */

#if KYTY_COMPILER == KYTY_COMPILER_GCC
#define KYTY_FORMAT_PRINTF(a, b) __attribute__((format(gnu_printf, a, b)))
#elif KYTY_COMPILER == KYTY_COMPILER_CLANG
#define KYTY_FORMAT_PRINTF(a, b) __attribute__((format(printf, a, b)))
#endif

// Code-layout hints for hot paths (GCC and Clang, including clang-cl, understand both).
// KYTY_FORCE_INLINE: small accessors that sit on per-draw paths and must not become calls.
// KYTY_COLD_NOINLINE: failure handlers, kept out of line and out of the hot text.
// KYTY_UNLIKELY_COND: a condition that is false in every working run.
#define KYTY_FORCE_INLINE     inline __attribute__((always_inline))
#define KYTY_COLD_NOINLINE    __attribute__((cold, noinline))
#define KYTY_UNLIKELY_COND(x) __builtin_expect(static_cast<bool>(x), false)

#endif /* KYTY_COMMON_COMMON_H_ */
