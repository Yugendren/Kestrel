#ifndef KYTY_COMMON_ASSERT_H_
#define KYTY_COMMON_ASSERT_H_

#include "common/common.h"
#include "common/logging/log.h"

#include <cstdlib>
#include <string_view>

namespace Common {

#ifdef __clang__
int DbgExitHandler(char const* file, int line, std::string_view text)
    __attribute__((analyzer_noreturn));
int DbgExitHandler(char const* file, int line, fmt::text_style style, std::string_view text)
    __attribute__((analyzer_noreturn));
int DbgExitIfHandler(char const* expr, char const* file, int line)
    __attribute__((analyzer_noreturn));
int DbgNotImplementedHandler(char const* expr, char const* file, int line)
    __attribute__((analyzer_noreturn));
[[noreturn]] void DbgExit(int status);
#else
int  DbgExitHandler(char const* file, int line, std::string_view text);
int  DbgExitHandler(char const* file, int line, fmt::text_style style, std::string_view text);
int  DbgExitIfHandler(char const* expr, char const* file, int line);
int  DbgNotImplementedHandler(char const* expr, char const* file, int line);
[[noreturn]] void DbgExit(int status);
#endif

// EXIT_IF/EXIT_NOT_IMPLEMENTED sit on every hot path of the emulator (thousands per draw) and
// never fire in a working run. Their failure call goes through one cold, never-inlined, noreturn
// function and the condition is marked unlikely, so the compiler moves the failure block out of
// the hot code instead of laying `lea`/`call` sequences inline between the fast-path
// instructions (i-cache density), and needs no code after the call.
// Report through DbgExitIfHandler/DbgNotImplementedHandler, then DbgExit(321). Defined here
// rather than in assert.cpp so test targets that stub the handlers keep linking without it.
[[noreturn]] KYTY_COLD_NOINLINE inline void DbgExitIfFailed(char const* expr, char const* file,
                                                            int line) {
	DbgExitIfHandler(expr, file, line);
	DbgExit(321);
}
[[noreturn]] KYTY_COLD_NOINLINE inline void DbgNotImplementedFailed(char const* expr,
                                                                    char const* file, int line) {
	DbgNotImplementedHandler(expr, file, line);
	DbgExit(321);
}

} // namespace Common

#define EXIT_HALT() (Common::DbgExit(321), 1)

#ifndef KYTY_FINAL
#define EXIT_IF(x)                                                                                 \
	((void)(KYTY_UNLIKELY_COND(x) && (Common::DbgExitIfFailed(#x, __FILE__, __LINE__), 0)))
#else
#define EXIT_IF(x)                                                                                 \
	do {                                                                                           \
		constexpr bool kyty_exit_if_disabled = false && (x);                                       \
		(void)kyty_exit_if_disabled;                                                               \
	} while (0)
#endif

#define EXIT(...)                                                                                  \
	do {                                                                                           \
		((void)(Common::DbgExitHandler(__FILE__, __LINE__, ::fmt::sprintf(__VA_ARGS__)) &&         \
		        (EXIT_HALT(), 1)));                                                                \
	} while (0)

#define EXIT_COLOR(style, ...)                                                                     \
	do {                                                                                           \
		((void)(Common::DbgExitHandler(__FILE__, __LINE__, (style),                                \
		                               ::fmt::sprintf(__VA_ARGS__)) &&                             \
		        (EXIT_HALT(), 1)));                                                                \
	} while (0)

#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((void)(KYTY_UNLIKELY_COND(x) && (Common::DbgNotImplementedFailed(#x, __FILE__, __LINE__), 0)))
#define KYTY_NOT_IMPLEMENTED EXIT_NOT_IMPLEMENTED(true)

#endif /* KYTY_COMMON_ASSERT_H_ */
