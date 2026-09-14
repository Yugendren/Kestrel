#ifndef KYTY_COMMON_LOGGING_LOG_H_
#define KYTY_COMMON_LOGGING_LOG_H_

#include "common/common.h"

#include <atomic>
#include <cstdint>
#include <fmt/color.h>
#include <fmt/printf.h>
#include <string_view>

namespace Log {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name               = "Log";
	static constexpr auto        initialize         = Log::Initialize;
	static constexpr auto        shutdown           = Log::Shutdown;
	static constexpr auto        emergency_shutdown = Log::Shutdown;
};

enum class Direction { Silent, Console, File };

Direction GetDirection();
bool      IsSilent();
void      Write(std::string_view text);
void      Write(fmt::text_style style, std::string_view text);
void      WriteFatal(std::string_view text);
void      WriteFatal(fmt::text_style style, std::string_view text);
void      Flush();

namespace Color {

inline constexpr auto Default       = fmt::text_style {};
inline constexpr auto Red           = fmt::fg(fmt::terminal_color::red);
inline constexpr auto Green         = fmt::fg(fmt::terminal_color::green);
inline constexpr auto Yellow        = fmt::fg(fmt::terminal_color::yellow);
inline constexpr auto Magenta       = fmt::fg(fmt::terminal_color::magenta);
inline constexpr auto Cyan          = fmt::fg(fmt::terminal_color::cyan);
inline constexpr auto White         = fmt::fg(fmt::terminal_color::white);
inline constexpr auto BrightRed     = fmt::fg(fmt::terminal_color::bright_red);
inline constexpr auto BrightGreen   = fmt::fg(fmt::terminal_color::bright_green);
inline constexpr auto BrightYellow  = fmt::fg(fmt::terminal_color::bright_yellow);
inline constexpr auto BrightMagenta = fmt::fg(fmt::terminal_color::bright_magenta);
inline constexpr auto BrightWhite   = fmt::fg(fmt::terminal_color::bright_white);

} // namespace Color

} // namespace Log

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF(...)                                                                                  \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write(::fmt::sprintf(__VA_ARGS__));                                             \
		}                                                                                          \
	} while (false)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_COLOR(style, ...)                                                                     \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                  \
			::Log::Write((style), ::fmt::sprintf(__VA_ARGS__));                                    \
		}                                                                                          \
	} while (false)

// Per-call-site sampled logging: writes the first `first` occurrences and then every
// `every`-th one. Statements on per-packet or per-end-of-pipe paths run thousands of times
// per frame; writing every one of them costs more than the work being logged as soon as the
// log goes to a file, while a sample still shows that the path is being taken.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_SAMPLED(first, every, ...)                                                            \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                   \
			static ::std::atomic<uint64_t> log_sample_count {0};                                    \
			const auto log_sample_index =                                                           \
			    log_sample_count.fetch_add(1, ::std::memory_order_relaxed);                         \
			if (log_sample_index < (first) || log_sample_index % (every) == 0) {                     \
				::Log::Write(::fmt::sprintf(__VA_ARGS__));                                          \
			}                                                                                       \
		}                                                                                           \
	} while (false)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOGF_COLOR_SAMPLED(style, first, every, ...)                                               \
	do {                                                                                           \
		if (!::Log::IsSilent()) {                                                                   \
			static ::std::atomic<uint64_t> log_sample_count {0};                                    \
			const auto log_sample_index =                                                           \
			    log_sample_count.fetch_add(1, ::std::memory_order_relaxed);                         \
			if (log_sample_index < (first) || log_sample_index % (every) == 0) {                     \
				::Log::Write((style), ::fmt::sprintf(__VA_ARGS__));                                 \
			}                                                                                       \
		}                                                                                           \
	} while (false)

#endif /* KYTY_COMMON_LOGGING_LOG_H_ */
