#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_LINUX
// #error "KYTY_PLATFORM != KYTY_PLATFORM_LINUX"
#else

#include "common/perf/hostStats.h"
#include "common/systemInfo.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <sys/utsname.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <span>
#include <sys/sysinfo.h>
#endif

// KYTY_PLATFORM_LINUX also covers macOS builds (see sysLinuxDbg.cpp): Linux reads /proc, macOS
// asks Mach. Both feed the same CPU delta logic in CpuUsageSampler::Sample().

namespace Common::Perf {

namespace {

struct SystemTicks {
	uint64_t total = 0;
	uint64_t busy  = 0;
};

float Percent(double part, double whole) {
	return whole > 0.0 ? static_cast<float>(std::clamp(part / whole * 100.0, 0.0, 100.0)) : 0.0f;
}

#if defined(__APPLE__)

// Process and thread CPU times are reported in nanoseconds.
double CpuUnitsPerSec() {
	return 1e9;
}

uint64_t SysctlU64(const char* name) {
	uint64_t value = 0;
	size_t   size  = sizeof(value);
	return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
}

std::string SysctlString(const char* name) {
	std::array<char, 256> value {};
	size_t                size = value.size() - 1;
	return ::sysctlbyname(name, value.data(), &size, nullptr, 0) == 0 ? std::string(value.data())
	                                                                  : std::string();
}

uint64_t RamBytes() {
	return SysctlU64("hw.memsize");
}

std::string OsName() {
	const std::string version = SysctlString("kern.osproductversion");
	utsname           uts {};
	const std::string kernel = ::uname(&uts) == 0 ? std::string("Darwin ") + uts.release : "Darwin";
	return (version.empty() ? std::string("macOS") : "macOS " + version) + " (" + kernel + ")";
}

// Owns a Mach port send right (mach_host_self() and task_threads() hand out new rights).
class MachPort final {
public:
	explicit MachPort(mach_port_t port): m_port(port) {}
	~MachPort() {
		if (m_port != MACH_PORT_NULL) {
			::mach_port_deallocate(::mach_task_self(), m_port);
		}
	}
	KYTY_CLASS_NO_COPY(MachPort);

	[[nodiscard]] mach_port_t Get() const { return m_port; }

private:
	mach_port_t m_port;
};

uint64_t ReadProcessTime() {
	rusage usage {};
	if (::getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0;
	}
	const auto to_ns = [](const timeval& time) {
		return static_cast<uint64_t>(time.tv_sec) * 1'000'000'000u +
		       static_cast<uint64_t>(time.tv_usec) * 1'000u;
	};
	return to_ns(usage.ru_utime) + to_ns(usage.ru_stime);
}

SystemTicks ReadSystemTicks() {
	MachPort                  host(::mach_host_self());
	host_cpu_load_info_data_t load {};
	mach_msg_type_number_t    count = HOST_CPU_LOAD_INFO_COUNT;
	if (::host_statistics(host.Get(), HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&load),
	                      &count) != KERN_SUCCESS) {
		return {};
	}
	SystemTicks ticks;
	for (const auto state_ticks: load.cpu_ticks) {
		ticks.total += state_ticks;
	}
	ticks.busy = ticks.total - load.cpu_ticks[CPU_STATE_IDLE];
	return ticks;
}

// Calls func(tid, name, cpu_time_ns) for every thread of this process.
template <typename Func>
void ForEachThread(Func&& func) {
	thread_act_array_t     threads = nullptr;
	mach_msg_type_number_t count   = 0;
	if (::task_threads(::mach_task_self(), &threads, &count) != KERN_SUCCESS) {
		return;
	}
	for (mach_msg_type_number_t i = 0; i < count; i++) {
		MachPort thread(threads[i]);

		thread_identifier_info_data_t identifier {};
		mach_msg_type_number_t        identifier_count = THREAD_IDENTIFIER_INFO_COUNT;
		thread_extended_info_data_t   extended {};
		mach_msg_type_number_t        extended_count = THREAD_EXTENDED_INFO_COUNT;
		if (::thread_info(thread.Get(), THREAD_IDENTIFIER_INFO,
		                  reinterpret_cast<thread_info_t>(&identifier),
		                  &identifier_count) != KERN_SUCCESS ||
		    ::thread_info(thread.Get(), THREAD_EXTENDED_INFO,
		                  reinterpret_cast<thread_info_t>(&extended),
		                  &extended_count) != KERN_SUCCESS) {
			continue;
		}
		extended.pth_name[sizeof(extended.pth_name) - 1] = '\0';
		func(identifier.thread_id, std::string_view(extended.pth_name),
		     extended.pth_user_time + extended.pth_system_time);
	}
	::vm_deallocate(::mach_task_self(), reinterpret_cast<vm_address_t>(threads),
	                count * sizeof(thread_act_t));
}

#else

// Process and thread CPU times are reported in clock ticks (USER_HZ).
double CpuUnitsPerSec() {
	static const double ticks_per_sec = static_cast<double>(::sysconf(_SC_CLK_TCK));
	return ticks_per_sec;
}

class FileDescriptor final {
public:
	explicit FileDescriptor(int fd): m_fd(fd) {}
	~FileDescriptor() {
		if (m_fd >= 0) {
			::close(m_fd);
		}
	}
	KYTY_CLASS_NO_COPY(FileDescriptor);

	[[nodiscard]] int Get() const { return m_fd; }

private:
	int m_fd;
};

// Reads up to buf.size() - 1 bytes of a (proc) file into buf, NUL-terminated. Fixed buffers keep
// the per-sample cost free of allocations. Returns the text, empty on failure.
std::string_view ReadFileHead(const char* path, std::span<char> buf) {
	FileDescriptor fd(::open(path, O_RDONLY | O_CLOEXEC));
	if (fd.Get() < 0) {
		return {};
	}
	size_t size = 0;
	while (size + 1 < buf.size()) {
		const ssize_t n = ::read(fd.Get(), buf.data() + size, buf.size() - 1 - size);
		if (n <= 0) {
			break;
		}
		size += static_cast<size_t>(n);
	}
	buf[size] = '\0';
	return {buf.data(), size};
}

uint64_t ParseU64(std::string_view* text) {
	size_t pos = 0;
	while (pos < text->size() && (*text)[pos] == ' ') {
		pos++;
	}
	uint64_t value = 0;
	while (pos < text->size() && (*text)[pos] >= '0' && (*text)[pos] <= '9') {
		value = value * 10 + static_cast<uint64_t>((*text)[pos] - '0');
		pos++;
	}
	text->remove_prefix(pos);
	return value;
}

// Skips `count` space-separated fields.
void SkipFields(std::string_view* text, int count) {
	for (int i = 0; i < count; i++) {
		const size_t start = text->find_first_not_of(' ');
		if (start == std::string_view::npos) {
			*text = {};
			return;
		}
		const size_t end = text->find(' ', start);
		text->remove_prefix(end == std::string_view::npos ? text->size() : end);
	}
}

struct TaskStat {
	std::string_view comm;
	uint64_t         ticks = 0; // utime + stime
};

// /proc/<pid>[/task/<tid>]/stat: "pid (comm) state ... utime(14) stime(15) ...". comm may contain
// spaces and parentheses, so fields are counted from the last ')'. comm is the same (<= 15 char)
// name that /comm reports, so no second file read per thread is needed.
bool ParseTaskStat(std::string_view stat, TaskStat* out) {
	const size_t open  = stat.find('(');
	const size_t close = stat.rfind(')');
	if (open == std::string_view::npos || close == std::string_view::npos || close < open) {
		return false;
	}
	out->comm = stat.substr(open + 1, close - open - 1);

	std::string_view rest = stat.substr(close + 1);
	SkipFields(&rest, 11); // state(3) .. cmajflt(13)
	const uint64_t utime = ParseU64(&rest);
	const uint64_t stime = ParseU64(&rest);
	out->ticks           = utime + stime;
	return true;
}

// Value of "Key:   123 kB" in /proc/meminfo, in bytes.
uint64_t MemInfoBytes(std::string_view meminfo, std::string_view key) {
	size_t pos = 0;
	while (pos < meminfo.size()) {
		const size_t     eol = meminfo.find('\n', pos);
		std::string_view line =
		    meminfo.substr(pos, eol == std::string_view::npos ? eol : eol - pos);
		if (line.size() > key.size() && line.substr(0, key.size()) == key &&
		    line[key.size()] == ':') {
			line.remove_prefix(key.size() + 1);
			return ParseU64(&line) * 1024;
		}
		if (eol == std::string_view::npos) {
			break;
		}
		pos = eol + 1;
	}
	return 0;
}

uint64_t RamBytes() {
	struct sysinfo info {};
	return ::sysinfo(&info) == 0 ? static_cast<uint64_t>(info.totalram) * info.mem_unit : 0;
}

std::string OsName() {
	std::string pretty;
	if (FILE* file = std::fopen("/etc/os-release", "re")) {
		std::array<char, 512>      line {};
		constexpr std::string_view kKey = "PRETTY_NAME=";
		while (std::fgets(line.data(), static_cast<int>(line.size()), file) != nullptr) {
			std::string_view text(line.data());
			if (text.substr(0, kKey.size()) != kKey) {
				continue;
			}
			text.remove_prefix(kKey.size());
			while (!text.empty() && (text.back() == '\n' || text.back() == '"')) {
				text.remove_suffix(1);
			}
			if (!text.empty() && text.front() == '"') {
				text.remove_prefix(1);
			}
			pretty = text;
			break;
		}
		std::fclose(file);
	}

	utsname           uts {};
	const std::string kernel = ::uname(&uts) == 0 ? std::string("Linux ") + uts.release : "Linux";
	return pretty.empty() ? kernel : pretty + " (" + kernel + ")";
}

uint64_t ReadProcessTime() {
	std::array<char, 1024> buf {};
	TaskStat               stat;
	return ParseTaskStat(ReadFileHead("/proc/self/stat", buf), &stat) ? stat.ticks : 0;
}

// First line of /proc/stat: "cpu  user nice system idle iowait irq softirq steal guest ...".
// guest time is already included in user, so only the first eight fields are summed. Busy time
// excludes iowait, which (unlike the other fields) is not monotonic.
SystemTicks ReadSystemTicks() {
	std::array<char, 1024> buf {};
	std::string_view       text = ReadFileHead("/proc/stat", buf);
	if (text.substr(0, 4) != "cpu ") {
		return {};
	}
	text.remove_prefix(4);

	std::array<uint64_t, 8> fields {};
	for (auto& field: fields) {
		field = ParseU64(&text);
	}
	SystemTicks ticks;
	for (const auto field: fields) {
		ticks.total += field;
	}
	ticks.busy = ticks.total - fields[3] - fields[4];
	return ticks;
}

// Calls func(tid, name, cpu_time_ticks) for every thread of this process.
template <typename Func>
void ForEachThread(Func&& func) {
	std::unique_ptr<DIR, decltype(&::closedir)> dir(::opendir("/proc/self/task"), &::closedir);
	if (dir == nullptr) {
		return;
	}
	while (const dirent* entry = ::readdir(dir.get())) {
		if (entry->d_name[0] < '0' || entry->d_name[0] > '9') {
			continue;
		}
		std::array<char, 64> path {};
		std::snprintf(path.data(), path.size(), "/proc/self/task/%s/stat", entry->d_name);
		std::array<char, 1024> buf {};
		TaskStat               stat;
		if (!ParseTaskStat(ReadFileHead(path.data(), buf), &stat)) {
			continue; // the thread exited between readdir and open
		}
		func(std::strtoull(entry->d_name, nullptr, 10), stat.comm, stat.ticks);
	}
}

#endif

} // namespace

HostSpecs CollectHostSpecs(GpuIdentity gpu) {
	const auto info = Common::GetSystemInfo();

	HostSpecs specs;
	specs.cpu_model       = info.ProcessorName;
	specs.physical_cores  = info.PhysicalCores;
	specs.logical_threads = info.LogicalThreads;
	specs.ram_bytes       = RamBytes();
	specs.os              = OsName();
	specs.gpu             = std::move(gpu);
	return specs;
}

MemoryUsage SampleMemoryUsage() {
	MemoryUsage usage;
#if defined(__APPLE__)
	usage.system_total_bytes = RamBytes();

	// "Memory Used" as Activity Monitor computes it: app (anonymous, non-purgeable) + wired +
	// compressed pages.
	MachPort               host(::mach_host_self());
	vm_statistics64_data_t vm {};
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	if (::host_statistics64(host.Get(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
	                        &count) == KERN_SUCCESS) {
		const uint64_t pages = static_cast<uint64_t>(vm.internal_page_count) - vm.purgeable_count +
		                       vm.wire_count + vm.compressor_page_count;
		usage.system_used_bytes = std::min(pages * vm_kernel_page_size, usage.system_total_bytes);
	}

	mach_task_basic_info_data_t task {};
	mach_msg_type_number_t      task_count = MACH_TASK_BASIC_INFO_COUNT;
	if (::task_info(::mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&task),
	                &task_count) == KERN_SUCCESS) {
		usage.process_bytes = task.resident_size;
	}
#else
	std::array<char, 4096> buf {};
	const auto             meminfo = ReadFileHead("/proc/meminfo", buf);
	usage.system_total_bytes       = MemInfoBytes(meminfo, "MemTotal");
	const uint64_t available       = MemInfoBytes(meminfo, "MemAvailable");
	usage.system_used_bytes =
	    usage.system_total_bytes > available ? usage.system_total_bytes - available : 0;

	// statm: "size resident shared ..." in pages.
	std::array<char, 256> statm_buf {};
	std::string_view      statm = ReadFileHead("/proc/self/statm", statm_buf);
	(void)ParseU64(&statm);
	usage.process_bytes = ParseU64(&statm) * static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
#endif
	return usage;
}

struct CpuUsageSampler::Impl {
	bool                                   primed = false;
	std::chrono::steady_clock::time_point  prev_time;
	uint64_t                               prev_process = 0;
	SystemTicks                            prev_system;
	std::unordered_map<uint64_t, uint64_t> prev_threads; // tid -> CPU time
	std::unordered_map<uint64_t, uint64_t> threads;      // scratch, swapped with prev_threads
	uint32_t logical_cpus = static_cast<uint32_t>(std::max(::sysconf(_SC_NPROCESSORS_ONLN), 1L));

	void SampleThreads(double interval_units, CpuUsage* usage);
};

void CpuUsageSampler::Impl::SampleThreads(double interval_units, CpuUsage* usage) {
	threads.clear();

	uint64_t busiest_delta = 0;
	bool     have_busiest  = false;
	ForEachThread([&](uint64_t tid, std::string_view name, uint64_t cpu_time) {
		threads[tid] = cpu_time;
		if (!primed) {
			return;
		}

		// A thread absent from the previous pass was created during the interval, so all of its
		// time belongs to it.
		const auto     prev  = prev_threads.find(tid);
		const uint64_t base  = prev != prev_threads.end() ? prev->second : 0;
		const uint64_t delta = cpu_time > base ? cpu_time - base : 0;

		ThreadUsage thread;
		thread.id           = tid;
		thread.name         = name;
		thread.core_percent = Percent(static_cast<double>(delta), interval_units);
		if (!have_busiest || delta > busiest_delta) {
			busiest_delta  = delta;
			have_busiest   = true;
			usage->busiest = thread;
		}
		if (name == "Thread_Gpu") {
			usage->gpu_thread = std::move(thread);
		}
	});
	prev_threads.swap(threads);
}

CpuUsageSampler::CpuUsageSampler(): m_impl(std::make_unique<Impl>()) {}

CpuUsageSampler::~CpuUsageSampler() = default;

CpuUsage CpuUsageSampler::Sample() {
	Impl&          impl    = *m_impl;
	const auto     now     = std::chrono::steady_clock::now();
	const uint64_t process = ReadProcessTime();
	const auto     system  = ReadSystemTicks();

	CpuUsage     usage;
	const double interval_units =
	    std::chrono::duration<double>(now - impl.prev_time).count() * CpuUnitsPerSec();
	impl.SampleThreads(interval_units, &usage);

	if (impl.primed) {
		const double process_units =
		    static_cast<double>(process > impl.prev_process ? process - impl.prev_process : 0);
		usage.process_core_percent =
		    interval_units > 0.0 ? static_cast<float>(process_units / interval_units * 100.0)
		                         : 0.0f;
		usage.process_system_percent =
		    std::min(usage.process_core_percent / static_cast<float>(impl.logical_cpus), 100.0f);
		if (system.total > impl.prev_system.total && system.busy >= impl.prev_system.busy) {
			usage.system_percent =
			    Percent(static_cast<double>(system.busy - impl.prev_system.busy),
			            static_cast<double>(system.total - impl.prev_system.total));
		}
	}

	impl.primed       = true;
	impl.prev_time    = now;
	impl.prev_process = process;
	impl.prev_system  = system;
	return usage;
}

} // namespace Common::Perf

#endif
