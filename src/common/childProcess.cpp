#include "common/childProcess.h"

#include <cstdlib>
#include <system_error>

// Platform-neutral part of ChildProcess: executable lookup. Spawning, pipes and
// GetExecutableDirectory() live in platform/sys{Linux,Windows}ChildProcess.cpp.

namespace Common {

namespace {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr wchar_t          PATH_SEPARATOR = L';';
constexpr std::string_view EXE_SUFFIX     = ".exe";
#else
constexpr char             PATH_SEPARATOR = ':';
constexpr std::string_view EXE_SUFFIX     = "";
#endif

bool IsExecutableFile(const std::filesystem::path& path) {
	std::error_code ec;
	if (!std::filesystem::is_regular_file(path, ec)) {
		return false;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return true;
#else
	const auto perms = std::filesystem::status(path, ec).permissions();
	return !ec && (perms & (std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec |
	                        std::filesystem::perms::others_exec)) != std::filesystem::perms::none;
#endif
}

// PATH as a native string: on Windows the narrow getenv() is in the ANSI code page and would
// mangle non-ASCII directories, so read the wide variable.
std::filesystem::path::string_type SearchPath() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const wchar_t* value = _wgetenv(L"PATH");
#else
	const char* value = std::getenv("PATH");
#endif
	return value != nullptr ? value : std::filesystem::path::string_type();
}

} // namespace

std::optional<std::filesystem::path> FindExecutable(std::string_view name) {
	// Tool names are ASCII, so the narrow-string path constructor is encoding-safe here.
	const std::filesystem::path file_name(std::string(name) + std::string(EXE_SUFFIX));

	if (auto local = GetExecutableDirectory() / file_name; IsExecutableFile(local)) {
		return local;
	}

	const auto search_path = SearchPath();
	size_t     begin       = 0;
	while (begin <= search_path.size()) {
		size_t end = search_path.find(PATH_SEPARATOR, begin);
		if (end == std::filesystem::path::string_type::npos) {
			end = search_path.size();
		}
		// An empty PATH entry means the current directory; skip it: running a binary that
		// happens to sit in the game's working directory is not what the user asked for.
		if (end > begin) {
			auto candidate = std::filesystem::path(search_path.substr(begin, end - begin)) / file_name;
			if (IsExecutableFile(candidate)) {
				return candidate;
			}
		}
		begin = end + 1;
	}
	return std::nullopt;
}

} // namespace Common
