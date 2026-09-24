#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// #error "KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS"
#else

#include "common/perf/hostStats.h"
#include "common/systemInfo.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#include <tlhelp32.h>

namespace Common::Perf {

namespace {

struct HandleCloser {
	void operator()(HANDLE handle) const { ::CloseHandle(handle); }
};
using UniqueHandle = std::unique_ptr<void, HandleCloser>;

// FILETIME durations are in 100 ns units.
constexpr double kFileTimeUnitsPerSec = 10'000'000.0;

uint64_t FileTimeToU64(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}

float Percent(double part, double whole) {
	return whole > 0.0 ? static_cast<float>(std::clamp(part / whole * 100.0, 0.0, 100.0)) : 0.0f;
}

std::string WideToUtf8(const wchar_t* text) {
	const int size = ::WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
	if (size <= 1) {
		return {};
	}
	std::string utf8(static_cast<size_t>(size), '\0');
	::WideCharToMultiByte(CP_UTF8, 0, text, -1, utf8.data(), size, nullptr, nullptr);
	utf8.resize(static_cast<size_t>(size - 1)); // drop the terminator
	return utf8;
}

using GetThreadDescriptionFunc = HRESULT(WINAPI*)(HANDLE, PWSTR*);

// GetThreadDescription exists since Windows 10 1607; resolved at runtime so older systems simply
// report unnamed threads.
GetThreadDescriptionFunc ResolveGetThreadDescription() {
	HMODULE kernel32 = ::GetModuleHandleW(L"kernel32.dll");
	if (kernel32 == nullptr) {
		return nullptr;
	}
	return reinterpret_cast<GetThreadDescriptionFunc>(
	    reinterpret_cast<void*>(::GetProcAddress(kernel32, "GetThreadDescription")));
}

std::string ThreadName(HANDLE thread) {
	static const auto get_description = ResolveGetThreadDescription();
	if (get_description == nullptr) {
		return {};
	}
	PWSTR description = nullptr;
	if (FAILED(get_description(thread, &description)) || description == nullptr) {
		return {};
	}
	std::string name = WideToUtf8(description);
	::LocalFree(description);
	return name;
}

using RtlGetVersionFunc = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);

// GetVersionEx lies to unmanifested processes (reports 6.2), RtlGetVersion does not.
std::string OsName() {
	HMODULE ntdll = ::GetModuleHandleW(L"ntdll.dll");
	auto    rtl_get_version =
	    ntdll != nullptr ? reinterpret_cast<RtlGetVersionFunc>(
	                           reinterpret_cast<void*>(::GetProcAddress(ntdll, "RtlGetVersion")))
	                     : nullptr;
	RTL_OSVERSIONINFOW version {};
	version.dwOSVersionInfoSize = sizeof(version);
	if (rtl_get_version == nullptr || rtl_get_version(&version) != 0) {
		return "Windows";
	}

	const auto build = std::to_string(version.dwBuildNumber);
	if (version.dwMajorVersion == 10) {
		// Windows 11 still reports 10.0; build 22000 is the first Windows 11 release.
		return std::string(version.dwBuildNumber >= 22000 ? "Windows 11" : "Windows 10") +
		       " (build " + build + ")";
	}
	return "Windows " + std::to_string(version.dwMajorVersion) + "." +
	       std::to_string(version.dwMinorVersion) + " (build " + build + ")";
}

} // namespace

HostSpecs CollectHostSpecs(GpuIdentity gpu) {
	const auto info = Common::GetSystemInfo();

	HostSpecs specs;
	specs.cpu_model       = info.ProcessorName;
	specs.physical_cores  = info.PhysicalCores;
	specs.logical_threads = info.LogicalThreads;
	specs.os              = OsName();
	specs.gpu             = std::move(gpu);

	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (::GlobalMemoryStatusEx(&status) != FALSE) {
		specs.ram_bytes = status.ullTotalPhys;
	}
	return specs;
}

MemoryUsage SampleMemoryUsage() {
	MemoryUsage usage;

	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (::GlobalMemoryStatusEx(&status) != FALSE) {
		usage.system_total_bytes = status.ullTotalPhys;
		usage.system_used_bytes  = status.ullTotalPhys - status.ullAvailPhys;
	}

	// The K32 variant lives in kernel32 (Windows 7+), so psapi.lib is not needed.
	PROCESS_MEMORY_COUNTERS counters {};
	if (::K32GetProcessMemoryInfo(::GetCurrentProcess(), &counters, sizeof(counters)) != FALSE) {
		usage.process_bytes = counters.WorkingSetSize;
	}
	return usage;
}

struct CpuUsageSampler::Impl {
	struct ThreadEntry {
		UniqueHandle handle;
		uint64_t     prev_ticks = 0;
		bool         seen       = false;
	};

	bool                                  primed = false;
	std::chrono::steady_clock::time_point prev_time;
	uint64_t                              prev_process      = 0;
	uint64_t                              prev_system_total = 0;
	uint64_t                              prev_system_busy  = 0;
	// Thread handles are kept open across samples (opening them is the expensive part); an open
	// handle also keeps Windows from reusing the thread id for a new thread.
	std::unordered_map<DWORD, ThreadEntry> threads;
	uint32_t logical_cpus = (std::max)(Common::GetSystemInfo().LogicalThreads, 1u);

	void SampleThreads(double interval_units, CpuUsage* usage);
};

void CpuUsageSampler::Impl::SampleThreads(double interval_units, CpuUsage* usage) {
	UniqueHandle snapshot;
	{
		HANDLE raw = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (raw == INVALID_HANDLE_VALUE) {
			return;
		}
		snapshot.reset(raw);
	}

	for (auto& [tid, entry]: threads) {
		entry.seen = false;
	}

	const DWORD   pid           = ::GetCurrentProcessId();
	uint64_t      busiest_delta = 0;
	bool          have_busiest  = false;
	THREADENTRY32 thread_entry {};
	thread_entry.dwSize = sizeof(thread_entry);
	for (BOOL ok = ::Thread32First(snapshot.get(), &thread_entry); ok != FALSE;
	     ok      = ::Thread32Next(snapshot.get(), &thread_entry)) {
		if (thread_entry.th32OwnerProcessID != pid) {
			continue;
		}
		const DWORD tid = thread_entry.th32ThreadID;

		auto it = threads.find(tid);
		if (it == threads.end()) {
			HANDLE handle = ::OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
			if (handle == nullptr) {
				continue; // exited since the snapshot
			}
			it = threads.emplace(tid, ThreadEntry {UniqueHandle(handle)}).first;
		}
		ThreadEntry& entry = it->second;
		entry.seen         = true;

		FILETIME creation {};
		FILETIME exit_time {};
		FILETIME kernel {};
		FILETIME user {};
		if (::GetThreadTimes(entry.handle.get(), &creation, &exit_time, &kernel, &user) == FALSE) {
			continue;
		}
		const uint64_t ticks = FileTimeToU64(kernel) + FileTimeToU64(user);
		// A thread first seen now was created during the interval (prev_ticks is 0), so all of
		// its time belongs to it.
		const uint64_t delta = ticks > entry.prev_ticks ? ticks - entry.prev_ticks : 0;
		entry.prev_ticks     = ticks;
		if (!primed) {
			continue;
		}

		ThreadUsage thread;
		thread.id           = tid;
		thread.name         = ThreadName(entry.handle.get());
		thread.core_percent = Percent(static_cast<double>(delta), interval_units);
		const bool is_gpu   = thread.name == "Thread_Gpu";
		if (!have_busiest || delta > busiest_delta) {
			busiest_delta  = delta;
			have_busiest   = true;
			usage->busiest = thread;
		}
		if (is_gpu) {
			usage->gpu_thread = std::move(thread);
		}
	}

	std::erase_if(threads, [](const auto& item) { return !item.second.seen; });
}

CpuUsageSampler::CpuUsageSampler(): m_impl(std::make_unique<Impl>()) {}

CpuUsageSampler::~CpuUsageSampler() = default;

CpuUsage CpuUsageSampler::Sample() {
	Impl&      impl = *m_impl;
	const auto now  = std::chrono::steady_clock::now();

	FILETIME creation {};
	FILETIME exit_time {};
	FILETIME kernel {};
	FILETIME user {};
	uint64_t process = impl.prev_process;
	if (::GetProcessTimes(::GetCurrentProcess(), &creation, &exit_time, &kernel, &user) != FALSE) {
		process = FileTimeToU64(kernel) + FileTimeToU64(user);
	}

	// System kernel time includes idle time.
	FILETIME sys_idle {};
	FILETIME sys_kernel {};
	FILETIME sys_user {};
	uint64_t system_total = impl.prev_system_total;
	uint64_t system_busy  = impl.prev_system_busy;
	if (::GetSystemTimes(&sys_idle, &sys_kernel, &sys_user) != FALSE) {
		system_total = FileTimeToU64(sys_kernel) + FileTimeToU64(sys_user);
		system_busy  = system_total - FileTimeToU64(sys_idle);
	}

	CpuUsage     usage;
	const double interval_units =
	    std::chrono::duration<double>(now - impl.prev_time).count() * kFileTimeUnitsPerSec;
	impl.SampleThreads(interval_units, &usage);

	if (impl.primed) {
		const double process_units =
		    static_cast<double>(process > impl.prev_process ? process - impl.prev_process : 0);
		usage.process_core_percent =
		    interval_units > 0.0 ? static_cast<float>(process_units / interval_units * 100.0)
		                         : 0.0f;
		usage.process_system_percent =
		    (std::min)(usage.process_core_percent / static_cast<float>(impl.logical_cpus), 100.0f);
		if (system_total > impl.prev_system_total && system_busy >= impl.prev_system_busy) {
			usage.system_percent =
			    Percent(static_cast<double>(system_busy - impl.prev_system_busy),
			            static_cast<double>(system_total - impl.prev_system_total));
		}
	}

	impl.primed            = true;
	impl.prev_time         = now;
	impl.prev_process      = process;
	impl.prev_system_total = system_total;
	impl.prev_system_busy  = system_busy;
	return usage;
}

} // namespace Common::Perf

#endif
