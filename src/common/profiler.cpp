#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <cstdio>

#ifdef TRACY_ENABLE

#include <algorithm>
#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <tracy/Tracy.hpp>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <cstring>
#include <pthread.h>
#include <unistd.h>
#endif

namespace {

thread_local std::vector<Profiler::ScopedBlock*> g_block_stack;

void RemoveBlock(Profiler::ScopedBlock* block) {
	auto block_it = std::find(g_block_stack.rbegin(), g_block_stack.rend(), block);
	if (block_it != g_block_stack.rend()) {
		g_block_stack.erase(std::next(block_it).base());
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
using SetThreadDescriptionFunc = HRESULT(WINAPI*)(HANDLE, PCWSTR);

// SetThreadDescription exists since Windows 10 1607; older systems keep unnamed threads.
SetThreadDescriptionFunc ResolveSetThreadDescription() {
	HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
	if (kernel32 == nullptr) {
		return nullptr;
	}
	return reinterpret_cast<SetThreadDescriptionFunc>(
	    reinterpret_cast<void*>(GetProcAddress(kernel32, "SetThreadDescription")));
}
#endif

// The OS-level name is what debuggers, top/Task Manager and the performance overlay's CPU sampler
// (which looks for "Thread_Gpu") see, so it is set even when tracy is not running.
void SetOsThreadName(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const auto set_description = ResolveSetThreadDescription();
	if (set_description == nullptr) {
		return;
	}
	wchar_t   wide[64] {};
	const int len = MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, 64);
	if (len > 0) {
		set_description(GetCurrentThread(), wide);
	}
#elif defined(__APPLE__)
	pthread_setname_np(name); // current thread only; longer names are rejected, not truncated
#else
	// The main thread's name is the process name that pgrep/pkill/killall and ps match on, so
	// renaming it would make "kyty_emulator" invisible to every tool that looks for it.
	if (gettid() == getpid()) {
		return;
	}
	// Linux limits thread names to 15 characters plus the terminator.
	char truncated[16] {};
	std::strncpy(truncated, name, sizeof(truncated) - 1);
	pthread_setname_np(pthread_self(), truncated);
#endif
}

} // namespace

namespace Profiler {

ScopedBlock::ScopedBlock(const tracy::SourceLocationData* source_location) {
	if (tracy::ProfilerAvailable()) {
		m_zone.emplace(source_location, TRACY_CALLSTACK, true);
		g_block_stack.push_back(this);
	}
}

ScopedBlock::~ScopedBlock() {
	End();
}

void ScopedBlock::End() {
	if (m_zone.has_value()) {
		m_zone.reset();
		RemoveBlock(this);
	}
}

void EndBlock() {
	if (!g_block_stack.empty()) {
		g_block_stack.back()->End();
	}
}

void SetThreadName(const char* name) {
	if (name == nullptr) {
		return;
	}
	SetOsThreadName(name);
	if (tracy::ProfilerAvailable()) {
		tracy::SetThreadName(name);
	}
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler

#else // TRACY_ENABLE

namespace Profiler {

void EndBlock() {}

void SetThreadName(const char* /*name*/) {}

void Initialize() {
	if (Config::ProfilerEnabled()) {
		::printf("--profile ignored: this build has Tracy compiled out (configure with "
		         "-DKYTY_TRACY=ON)\n");
	}
}

void Shutdown() {}

} // namespace Profiler

#endif // TRACY_ENABLE
