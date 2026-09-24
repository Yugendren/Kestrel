#include "common/perf/hostStats.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <dlfcn.h>
#endif

// NVML is loaded at runtime; the handful of types and entry points used here are declared from
// the documented NVML ABI (nvml.h) instead of depending on the CUDA toolkit headers.

namespace Common::Perf {

namespace {

using NvmlReturn = int;
struct NvmlDeviceOpaque;
using NvmlDevice = NvmlDeviceOpaque*;

struct NvmlUtilization {
	unsigned int gpu;
	unsigned int memory;
};

struct NvmlMemory {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
};

constexpr NvmlReturn   kNvmlSuccess           = 0;
constexpr int          kNvmlTemperatureGpu    = 0;  // NVML_TEMPERATURE_GPU
constexpr unsigned int kNvmlDeviceNameBufSize = 96; // NVML_DEVICE_NAME_V2_BUFFER_SIZE

using NvmlInitFunc           = NvmlReturn (*)();
using NvmlShutdownFunc       = NvmlReturn (*)();
using NvmlGetCountFunc       = NvmlReturn (*)(unsigned int*);
using NvmlGetHandleFunc      = NvmlReturn (*)(unsigned int, NvmlDevice*);
using NvmlGetNameFunc        = NvmlReturn (*)(NvmlDevice, char*, unsigned int);
using NvmlGetUtilFunc        = NvmlReturn (*)(NvmlDevice, NvmlUtilization*);
using NvmlGetMemoryFunc      = NvmlReturn (*)(NvmlDevice, NvmlMemory*);
using NvmlGetTemperatureFunc = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
using NvmlGetPowerFunc       = NvmlReturn (*)(NvmlDevice, unsigned int*);

// Owns the loaded NVML shared library.
class NvmlLibrary final {
public:
	NvmlLibrary() = default;
	~NvmlLibrary() { Close(); }
	KYTY_CLASS_NO_COPY(NvmlLibrary);

	bool Load() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Modern drivers install nvml.dll into System32; older ones only under NVSMI.
		m_module = LoadLibraryW(L"nvml.dll");
		if (m_module == nullptr) {
			std::array<wchar_t, MAX_PATH> program_files {};
			const DWORD len = GetEnvironmentVariableW(L"ProgramFiles", program_files.data(),
			                                          static_cast<DWORD>(program_files.size()));
			if (len > 0 && len < program_files.size()) {
				std::wstring path(program_files.data(), len);
				path += L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
				m_module = LoadLibraryW(path.c_str());
			}
		}
#else
		m_module = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
		return m_module != nullptr;
	}

	template <typename Func>
	[[nodiscard]] Func Resolve(const char* name) const {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return reinterpret_cast<Func>(reinterpret_cast<void*>(GetProcAddress(m_module, name)));
#else
		return reinterpret_cast<Func>(::dlsym(m_module, name));
#endif
	}

private:
	void Close() {
		if (m_module == nullptr) {
			return;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		FreeLibrary(m_module);
#else
		::dlclose(m_module);
#endif
		m_module = nullptr;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HMODULE m_module = nullptr;
#else
	void* m_module = nullptr;
#endif
};

} // namespace

struct NvmlGpuMonitor::Impl {
	NvmlLibrary            library;
	bool                   initialized     = false;
	NvmlShutdownFunc       shutdown        = nullptr;
	NvmlGetUtilFunc        get_utilization = nullptr;
	NvmlGetMemoryFunc      get_memory      = nullptr;
	NvmlGetTemperatureFunc get_temperature = nullptr;
	NvmlGetPowerFunc       get_power       = nullptr;
	NvmlDevice             device          = nullptr;

	Impl() = default;
	~Impl() {
		if (initialized) {
			shutdown();
		}
	}
	KYTY_CLASS_NO_COPY(Impl);
};

namespace {

// Returns the device named `device_name`, else device 0, else nullptr.
NvmlDevice FindDevice(const NvmlLibrary& library, std::string_view device_name) {
	auto get_count  = library.Resolve<NvmlGetCountFunc>("nvmlDeviceGetCount_v2");
	auto get_handle = library.Resolve<NvmlGetHandleFunc>("nvmlDeviceGetHandleByIndex_v2");
	auto get_name   = library.Resolve<NvmlGetNameFunc>("nvmlDeviceGetName");
	if (get_count == nullptr || get_handle == nullptr || get_name == nullptr) {
		return nullptr;
	}

	unsigned int count = 0;
	if (get_count(&count) != kNvmlSuccess || count == 0) {
		return nullptr;
	}

	NvmlDevice first = nullptr;
	for (unsigned int index = 0; index < count; index++) {
		NvmlDevice device = nullptr;
		if (get_handle(index, &device) != kNvmlSuccess) {
			continue;
		}
		if (first == nullptr) {
			first = device;
		}
		std::array<char, kNvmlDeviceNameBufSize> name {};
		if (!device_name.empty() &&
		    get_name(device, name.data(), static_cast<unsigned int>(name.size())) == kNvmlSuccess &&
		    device_name == name.data()) {
			return device;
		}
	}
	return first;
}

} // namespace

NvmlGpuMonitor::NvmlGpuMonitor(std::unique_ptr<Impl> impl): m_impl(std::move(impl)) {}

NvmlGpuMonitor::~NvmlGpuMonitor() = default;

std::unique_ptr<NvmlGpuMonitor> NvmlGpuMonitor::Open(std::string_view device_name) {
	auto impl = std::make_unique<Impl>();
	if (!impl->library.Load()) {
		return nullptr;
	}

	auto init             = impl->library.Resolve<NvmlInitFunc>("nvmlInit_v2");
	impl->shutdown        = impl->library.Resolve<NvmlShutdownFunc>("nvmlShutdown");
	impl->get_utilization = impl->library.Resolve<NvmlGetUtilFunc>("nvmlDeviceGetUtilizationRates");
	impl->get_memory      = impl->library.Resolve<NvmlGetMemoryFunc>("nvmlDeviceGetMemoryInfo");
	impl->get_temperature =
	    impl->library.Resolve<NvmlGetTemperatureFunc>("nvmlDeviceGetTemperature");
	impl->get_power = impl->library.Resolve<NvmlGetPowerFunc>("nvmlDeviceGetPowerUsage");
	if (init == nullptr || impl->shutdown == nullptr || impl->get_utilization == nullptr ||
	    impl->get_memory == nullptr || init() != kNvmlSuccess) {
		return nullptr;
	}
	impl->initialized = true;

	impl->device = FindDevice(impl->library, device_name);
	if (impl->device == nullptr) {
		return nullptr;
	}
	return std::unique_ptr<NvmlGpuMonitor>(new NvmlGpuMonitor(std::move(impl)));
}

std::optional<GpuUsage> NvmlGpuMonitor::Sample() {
	NvmlUtilization utilization {};
	NvmlMemory      memory {};
	if (m_impl->get_utilization(m_impl->device, &utilization) != kNvmlSuccess ||
	    m_impl->get_memory(m_impl->device, &memory) != kNvmlSuccess) {
		return std::nullopt;
	}

	GpuUsage usage;
	usage.utilization_percent = utilization.gpu;
	usage.vram_used_bytes     = memory.used;
	usage.vram_total_bytes    = memory.total;

	// Temperature and power are optional: not every board or driver exposes them.
	unsigned int temperature_c = 0;
	if (m_impl->get_temperature != nullptr &&
	    m_impl->get_temperature(m_impl->device, kNvmlTemperatureGpu, &temperature_c) ==
	        kNvmlSuccess) {
		usage.temperature_c = temperature_c;
	}
	unsigned int power_mw = 0;
	if (m_impl->get_power != nullptr &&
	    m_impl->get_power(m_impl->device, &power_mw) == kNvmlSuccess) {
		usage.power_w = (power_mw + 500) / 1000;
	}
	return usage;
}

} // namespace Common::Perf
