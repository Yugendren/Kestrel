#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_LINUX
// #error "KYTY_PLATFORM != KYTY_PLATFORM_LINUX"
#else

#include "common/childProcess.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

extern char** environ; // NOLINT(readability-redundant-declaration)

// POSIX side of ChildProcess. The emulator reserves a huge guest address space and runs many
// threads, so fork() (page-table copy, async-signal-safety in the child) is out of the question;
// posix_spawn uses vfork/CLONE_VM semantics and executes the file actions in the child for us.

namespace Common {

namespace {

// How often Wait() polls waitpid(WNOHANG); short enough that Stop() -> Saved feels instant.
constexpr uint32_t WAIT_POLL_MS = 5;

class UniqueFd final {
public:
	UniqueFd() = default;
	explicit UniqueFd(int fd): m_fd(fd) {}
	~UniqueFd() { Reset(); }
	KYTY_CLASS_NO_COPY(UniqueFd);

	[[nodiscard]] int Get() const { return m_fd; }

	void Reset(int fd = -1) {
		if (m_fd >= 0) {
			close(m_fd);
		}
		m_fd = fd;
	}

private:
	int m_fd = -1;
};

// Owns the posix_spawn attribute/file-action objects so every early return destroys them.
class SpawnSetup final {
public:
	SpawnSetup() {
		posix_spawn_file_actions_init(&m_actions);
		posix_spawnattr_init(&m_attr);
	}
	~SpawnSetup() {
		posix_spawnattr_destroy(&m_attr);
		posix_spawn_file_actions_destroy(&m_actions);
	}
	KYTY_CLASS_NO_COPY(SpawnSetup);

	posix_spawn_file_actions_t* Actions() { return &m_actions; }
	posix_spawnattr_t*          Attr() { return &m_attr; }

private:
	posix_spawn_file_actions_t m_actions {};
	posix_spawnattr_t          m_attr {};
};

std::string ErrnoMessage(const char* what, int err) {
	return std::string(what) + ": " + std::strerror(err);
}

// stdin channel. A socketpair rather than a pipe so writes can use MSG_NOSIGNAL (Linux) /
// SO_NOSIGPIPE (macOS): a dead ffmpeg must surface as a false return, never as a SIGPIPE that
// takes the emulator down.
bool CreateStdinChannel(UniqueFd* parent_end, UniqueFd* child_end, std::string* error) {
	int fds[2] = {-1, -1};
#if defined(__APPLE__)
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
		*error = ErrnoMessage("socketpair", errno);
		return false;
	}
	parent_end->Reset(fds[0]);
	child_end->Reset(fds[1]);
	int one = 1;
	fcntl(fds[0], F_SETFD, FD_CLOEXEC);
	fcntl(fds[1], F_SETFD, FD_CLOEXEC);
	setsockopt(fds[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
	if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0) {
		*error = ErrnoMessage("socketpair", errno);
		return false;
	}
	parent_end->Reset(fds[0]);
	child_end->Reset(fds[1]);
#endif
	// The child only reads and we only write; half-closing makes that explicit.
	shutdown(fds[0], SHUT_RD);
	shutdown(fds[1], SHUT_WR);
	return true;
}

int ExitCodeFromStatus(int status) {
	if (WIFEXITED(status)) {
		return WEXITSTATUS(status);
	}
	if (WIFSIGNALED(status)) {
		return 128 + WTERMSIG(status); // shell convention
	}
	return -1;
}

} // namespace

struct ChildProcess::Impl {
	pid_t              pid = -1;
	UniqueFd           stdin_fd;
	std::optional<int> exit_code; // set once reaped

	// One non-blocking reap attempt; true once the child is gone.
	bool TryReap() {
		if (exit_code.has_value()) {
			return true;
		}
		int status = 0;
		for (;;) {
			const pid_t r = waitpid(pid, &status, WNOHANG);
			if (r == pid) {
				exit_code = ExitCodeFromStatus(status);
				return true;
			}
			if (r == 0) {
				return false;
			}
			if (errno != EINTR) {
				// ECHILD: reaped elsewhere (e.g. SIGCHLD set to SIG_IGN); the process is gone.
				exit_code = -1;
				return true;
			}
		}
	}
};

ChildProcess::ChildProcess(std::unique_ptr<Impl> impl): m_impl(std::move(impl)) {}

ChildProcess::~ChildProcess() {
	Kill();
}

std::unique_ptr<ChildProcess> ChildProcess::Start(const std::vector<std::string>& argv,
                                                  const std::filesystem::path&    log_path,
                                                  std::string*                    error) {
	std::string local_error;
	if (error == nullptr) {
		error = &local_error;
	}
	if (argv.empty()) {
		*error = "empty command line";
		return nullptr;
	}

	auto impl = std::make_unique<Impl>();
	UniqueFd child_stdin;
	if (!CreateStdinChannel(&impl->stdin_fd, &child_stdin, error)) {
		return nullptr;
	}

	SpawnSetup setup;
	auto*      actions = setup.Actions();
	const std::string log = log_path.empty() ? std::string("/dev/null") : log_path.string();
	// dup2 clears FD_CLOEXEC on the target, so fd 0 survives the exec while the original
	// socket end (still close-on-exec) does not.
	posix_spawn_file_actions_adddup2(actions, child_stdin.Get(), STDIN_FILENO);
	posix_spawn_file_actions_addopen(actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
	posix_spawn_file_actions_adddup2(actions, STDOUT_FILENO, STDERR_FILENO);

	short flags = POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
#if defined(__APPLE__)
	// Everything not dup2'ed above is closed at exec.
	flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
	// Our own fds are O_CLOEXEC, but third-party code (drivers, SDL, spdlog) may not be.
	posix_spawn_file_actions_addclosefrom_np(actions, STDERR_FILENO + 1);
#endif
	auto* attr = setup.Attr();
	// Own process group: a Ctrl+C in the emulator's terminal goes to the foreground group and
	// must not kill ffmpeg before it has written the last fragment.
	posix_spawnattr_setpgroup(attr, 0);
	// The spawning thread may block signals (the emulator masks some for guest threads) and
	// may ignore SIGPIPE; the helper starts from a clean slate.
	sigset_t empty_mask;
	sigemptyset(&empty_mask);
	posix_spawnattr_setsigmask(attr, &empty_mask);
	sigset_t default_signals;
	sigfillset(&default_signals);
	posix_spawnattr_setsigdefault(attr, &default_signals);
	posix_spawnattr_setflags(attr, flags);

	std::vector<char*> c_argv;
	c_argv.reserve(argv.size() + 1);
	for (const auto& arg: argv) {
		c_argv.push_back(const_cast<char*>(arg.c_str())); // NOLINT(cppcoreguidelines-pro-type-const-cast)
	}
	c_argv.push_back(nullptr);

	const int result = posix_spawn(&impl->pid, argv[0].c_str(), actions, attr, c_argv.data(), environ);
	if (result != 0) {
		*error = ErrnoMessage(("cannot run " + argv[0]).c_str(), result);
		return nullptr;
	}
	return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

bool ChildProcess::WriteStdin(std::string_view data) {
	const int fd = m_impl->stdin_fd.Get();
	if (fd < 0) {
		return false;
	}
#if defined(__APPLE__)
	constexpr int SEND_FLAGS = 0; // SO_NOSIGPIPE is set on the socket
#else
	constexpr int SEND_FLAGS = MSG_NOSIGNAL;
#endif
	while (!data.empty()) {
		const ssize_t n = send(fd, data.data(), data.size(), SEND_FLAGS);
		if (n < 0) {
			if (errno == EINTR) {
				continue;
			}
			return false;
		}
		data.remove_prefix(static_cast<size_t>(n));
	}
	return true;
}

void ChildProcess::CloseStdin() {
	m_impl->stdin_fd.Reset();
}

std::optional<int> ChildProcess::Wait(uint32_t timeout_ms) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	for (;;) {
		if (m_impl->TryReap()) {
			return m_impl->exit_code;
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			return std::nullopt;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_POLL_MS));
	}
}

bool ChildProcess::IsRunning() {
	return !m_impl->TryReap();
}

void ChildProcess::Kill() {
	if (m_impl->TryReap()) {
		return;
	}
	kill(m_impl->pid, SIGKILL);
	int status = 0;
	pid_t r    = -1;
	do {
		r = waitpid(m_impl->pid, &status, 0);
	} while (r < 0 && errno == EINTR);
	m_impl->exit_code = r == m_impl->pid ? ExitCodeFromStatus(status) : -1;
}

std::filesystem::path GetExecutableDirectory() {
#if defined(__APPLE__)
	uint32_t size = 0;
	_NSGetExecutablePath(nullptr, &size);
	std::string buffer(size, '\0');
	if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
		std::error_code ec;
		auto            path = std::filesystem::canonical(buffer.c_str(), ec);
		if (!ec) {
			return path.parent_path();
		}
	}
#else
	std::error_code ec;
	auto            path = std::filesystem::read_symlink("/proc/self/exe", ec);
	if (!ec) {
		return path.parent_path();
	}
#endif
	std::error_code cwd_ec;
	return std::filesystem::current_path(cwd_ec);
}

} // namespace Common

#endif
