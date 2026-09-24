#ifndef KYTY_COMMON_CHILDPROCESS_H_
#define KYTY_COMMON_CHILDPROCESS_H_

#include "common/common.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Common {

// A helper process (e.g. ffmpeg) with a writable stdin and stdout+stderr redirected to a log
// file. Nothing else of the emulator is inherited: no other file descriptors / handles, and on
// Windows the child sits in a kill-on-close job object so it cannot outlive the emulator.
// Writing to a child that has exited returns false instead of raising SIGPIPE.
class ChildProcess final {
public:
	// argv[0] is the executable path (no PATH search; see FindExecutable). log_path empty =
	// discard output. Returns nullptr and sets *error (if non-null) when the spawn fails.
	[[nodiscard]] static std::unique_ptr<ChildProcess> Start(
	    const std::vector<std::string>& argv, const std::filesystem::path& log_path,
	    std::string* error);

	// Kills the child if it is still running, then reaps it.
	~ChildProcess();
	KYTY_CLASS_NO_COPY(ChildProcess);

	bool WriteStdin(std::string_view data);
	void CloseStdin();

	// Waits up to timeout_ms for the child to exit; returns its exit code once it has.
	[[nodiscard]] std::optional<int> Wait(uint32_t timeout_ms);
	[[nodiscard]] bool               IsRunning();
	void                             Kill();

private:
	struct Impl; // pid + stdin fd (POSIX) / process, job and pipe handles (Windows)
	explicit ChildProcess(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> m_impl;
};

// Directory containing the running emulator executable.
[[nodiscard]] std::filesystem::path GetExecutableDirectory();

// Looks for `name` (".exe" appended on Windows) next to the executable first, then on PATH.
[[nodiscard]] std::optional<std::filesystem::path> FindExecutable(std::string_view name);

} // namespace Common

#endif /* KYTY_COMMON_CHILDPROCESS_H_ */
