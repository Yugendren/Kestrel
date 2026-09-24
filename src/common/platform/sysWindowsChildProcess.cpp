#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// #error "KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS"
#else

// Keep <windows.h> lean and macro-free for the rest of this TU (min/max, MemoryBarrier...).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include "common/childProcess.h"

#include <string>
#include <system_error>
#include <utility>
#include <vector>

// Windows side of ChildProcess. The child inherits exactly two handles (stdin pipe read end and
// the log file) via PROC_THREAD_ATTRIBUTE_HANDLE_LIST, so emulator handles never leak into it,
// and it is placed in a kill-on-close job object before it runs a single instruction, so it dies
// with the emulator even when the emulator crashes or is killed from the task manager.

namespace Common {

namespace {

class UniqueHandle final {
public:
	UniqueHandle() = default;
	explicit UniqueHandle(HANDLE h): m_h(h) {}
	~UniqueHandle() { Reset(); }
	KYTY_CLASS_NO_COPY(UniqueHandle);

	[[nodiscard]] HANDLE Get() const { return m_h; }
	[[nodiscard]] bool   Valid() const { return m_h != nullptr && m_h != INVALID_HANDLE_VALUE; }
	// For out-parameters of Win32 calls.
	HANDLE* Receive() {
		Reset();
		return &m_h;
	}

	[[nodiscard]] HANDLE Release() { return std::exchange(m_h, nullptr); }

	void Reset(HANDLE h = nullptr) {
		if (Valid()) {
			CloseHandle(m_h);
		}
		m_h = h;
	}

private:
	HANDLE m_h = nullptr;
};

class ProcThreadAttributeList final {
public:
	ProcThreadAttributeList() {
		SIZE_T size = 0;
		InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
		m_storage.resize(size);
		if (InitializeProcThreadAttributeList(List(), 1, 0, &size) != FALSE) {
			m_initialized = true;
		}
	}
	~ProcThreadAttributeList() {
		if (m_initialized) {
			DeleteProcThreadAttributeList(List());
		}
	}
	KYTY_CLASS_NO_COPY(ProcThreadAttributeList);

	[[nodiscard]] bool Initialized() const { return m_initialized; }

	LPPROC_THREAD_ATTRIBUTE_LIST List() {
		return reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(m_storage.data());
	}

private:
	std::vector<char> m_storage;
	bool              m_initialized = false;
};

std::string LastErrorMessage(const char* what) {
	return std::string(what) + " failed (error " + std::to_string(GetLastError()) + ")";
}

std::wstring Utf8ToWide(std::string_view s) {
	if (s.empty()) {
		return {};
	}
	const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring out(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
	return out;
}

// Quotes one argument so that CommandLineToArgvW / the MSVCRT parser reproduces it exactly:
// backslashes are literal except in front of a quote, where they must be doubled.
void AppendQuotedArg(std::wstring* cmdline, const std::wstring& arg) {
	if (!cmdline->empty()) {
		cmdline->push_back(L' ');
	}
	if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
		cmdline->append(arg);
		return;
	}
	cmdline->push_back(L'"');
	for (auto it = arg.begin();; ++it) {
		size_t backslashes = 0;
		while (it != arg.end() && *it == L'\\') {
			++it;
			++backslashes;
		}
		if (it == arg.end()) {
			cmdline->append(backslashes * 2, L'\\'); // the closing quote follows
			break;
		}
		if (*it == L'"') {
			cmdline->append(backslashes * 2 + 1, L'\\');
		} else {
			cmdline->append(backslashes, L'\\');
		}
		cmdline->push_back(*it);
	}
	cmdline->push_back(L'"');
}

// Returns an owned job handle, or nullptr.
HANDLE CreateKillOnCloseJob() {
	UniqueHandle job(CreateJobObjectW(nullptr, nullptr));
	if (!job.Valid()) {
		return nullptr;
	}
	JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
	limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
	if (SetInformationJobObject(job.Get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits)) == FALSE) {
		return nullptr;
	}
	return job.Release();
}

} // namespace

struct ChildProcess::Impl {
	UniqueHandle       process;
	UniqueHandle       job; // closing it kills the child (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
	UniqueHandle       stdin_write;
	std::optional<int> exit_code;

	bool TryReap(DWORD timeout_ms) {
		if (exit_code.has_value()) {
			return true;
		}
		if (WaitForSingleObject(process.Get(), timeout_ms) != WAIT_OBJECT_0) {
			return false;
		}
		DWORD code = 0;
		exit_code  = GetExitCodeProcess(process.Get(), &code) != FALSE ? static_cast<int>(code) : -1;
		return true;
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

	SECURITY_ATTRIBUTES inheritable {};
	inheritable.nLength        = sizeof(inheritable);
	inheritable.bInheritHandle = TRUE;

	UniqueHandle stdin_read;
	if (CreatePipe(stdin_read.Receive(), impl->stdin_write.Receive(), &inheritable, 0) == FALSE) {
		*error = LastErrorMessage("CreatePipe");
		return nullptr;
	}
	SetHandleInformation(impl->stdin_write.Get(), HANDLE_FLAG_INHERIT, 0);

	const std::wstring log_name = log_path.empty() ? std::wstring(L"NUL") : log_path.wstring();
	UniqueHandle       log(CreateFileW(log_name.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
	                                   &inheritable, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
	if (!log.Valid()) {
		*error = LastErrorMessage("CreateFileW(log)");
		return nullptr;
	}

	ProcThreadAttributeList attributes;
	HANDLE                  inherited[] = {stdin_read.Get(), log.Get()};
	if (!attributes.Initialized() ||
	    UpdateProcThreadAttribute(attributes.List(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
	                              sizeof(inherited), nullptr, nullptr) == FALSE) {
		*error = LastErrorMessage("UpdateProcThreadAttribute");
		return nullptr;
	}

	STARTUPINFOEXW startup {};
	startup.StartupInfo.cb         = sizeof(startup);
	startup.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
	startup.StartupInfo.hStdInput  = stdin_read.Get();
	startup.StartupInfo.hStdOutput = log.Get();
	startup.StartupInfo.hStdError  = log.Get();
	startup.lpAttributeList        = attributes.List();

	const std::wstring application = Utf8ToWide(argv[0]);
	std::wstring       cmdline;
	for (const auto& arg: argv) {
		AppendQuotedArg(&cmdline, Utf8ToWide(arg));
	}

	// Suspended until it sits in the job, so not even a fast-failing child escapes it.
	PROCESS_INFORMATION info {};
	if (CreateProcessW(application.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
	                   EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr,
	                   &startup.StartupInfo, &info) == FALSE) {
		*error = LastErrorMessage(("CreateProcessW(" + argv[0] + ")").c_str());
		return nullptr;
	}
	impl->process.Reset(info.hProcess);
	UniqueHandle thread(info.hThread);

	// Best effort: without the job the child merely loses the die-with-the-emulator guarantee
	// (the recorder still stops it explicitly), which is no reason to refuse to record.
	impl->job.Reset(CreateKillOnCloseJob());
	if (impl->job.Valid() && AssignProcessToJobObject(impl->job.Get(), impl->process.Get()) == FALSE) {
		impl->job.Reset();
	}
	ResumeThread(thread.Get());
	return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

bool ChildProcess::WriteStdin(std::string_view data) {
	if (!m_impl->stdin_write.Valid()) {
		return false;
	}
	while (!data.empty()) {
		DWORD written = 0;
		if (WriteFile(m_impl->stdin_write.Get(), data.data(), static_cast<DWORD>(data.size()), &written, nullptr) ==
		    FALSE) {
			return false; // ERROR_BROKEN_PIPE / ERROR_NO_DATA once the child has exited
		}
		data.remove_prefix(written);
	}
	return true;
}

void ChildProcess::CloseStdin() {
	m_impl->stdin_write.Reset();
}

std::optional<int> ChildProcess::Wait(uint32_t timeout_ms) {
	m_impl->TryReap(timeout_ms);
	return m_impl->exit_code;
}

bool ChildProcess::IsRunning() {
	return !m_impl->TryReap(0);
}

void ChildProcess::Kill() {
	if (m_impl->TryReap(0)) {
		return;
	}
	TerminateProcess(m_impl->process.Get(), 1);
	if (!m_impl->TryReap(INFINITE)) {
		m_impl->exit_code = -1;
	}
}

std::filesystem::path GetExecutableDirectory() {
	std::wstring buffer(MAX_PATH, L'\0');
	for (;;) {
		const DWORD n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
		if (n == 0) {
			std::error_code ec;
			return std::filesystem::current_path(ec);
		}
		if (n < buffer.size()) {
			buffer.resize(n);
			return std::filesystem::path(buffer).parent_path();
		}
		buffer.resize(buffer.size() * 2); // truncated long path
	}
}

} // namespace Common

#endif
