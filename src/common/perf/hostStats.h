#ifndef KYTY_COMMON_PERF_HOSTSTATS_H_
#define KYTY_COMMON_PERF_HOSTSTATS_H_

#include "common/common.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// Host machine specs and live usage counters for the performance overlay. Everything here is
// sampled from a background thread (see PerfSampler); none of it is safe or cheap enough for
// the render, present or command-processor threads. Platform code lives in
// common/platform/sys{Linux,Windows}HostStats.cpp; <windows.h> never leaks into this header.

namespace Common::Perf {

// Filled by the graphics layer from the Vulkan physical device, since common/ has no Vulkan.
struct GpuIdentity {
	std::string name;           // VkPhysicalDeviceProperties::deviceName
	std::string driver_version; // human-readable, e.g. "535.183.01" (VkPhysicalDeviceDriverProperties)
	uint64_t    vram_bytes = 0; // sum of DEVICE_LOCAL heaps
	uint32_t    vendor_id  = 0; // PCI vendor (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel)
};

struct HostSpecs {
	std::string cpu_model;            // e.g. "AMD Ryzen 5 3600 6-Core Processor"
	uint32_t    physical_cores   = 0;
	uint32_t    logical_threads  = 0;
	uint64_t    ram_bytes        = 0;
	std::string os;                   // e.g. "Ubuntu 24.04.1 LTS (Linux 6.8.0-45)", "Windows 10 (build 19045)"
	GpuIdentity gpu;
};

// Collected once at start (reads /proc, uname, registry...). Cheap enough for startup only.
[[nodiscard]] HostSpecs CollectHostSpecs(GpuIdentity gpu);

struct ThreadUsage {
	uint64_t    id          = 0;   // OS thread id (Linux tid / Windows thread id)
	std::string name;              // OS thread name, empty when unnamed
	float       core_percent = 0;  // percent of one logical core, 0..100
};

struct CpuUsage {
	float process_core_percent   = 0; // top-style: 250 = 2.5 logical cores busy
	float process_system_percent = 0; // share of the whole machine, 0..100 (Task Manager style)
	float system_percent         = 0; // whole machine busy, 0..100
	ThreadUsage busiest;               // busiest thread of this process in the interval
	// The command-processor thread (named "Thread_Gpu" via KYTY_PROFILER_THREAD), when found.
	std::optional<ThreadUsage> gpu_thread;
};

// Per-process and per-thread CPU usage as deltas between consecutive Sample() calls.
// The first call primes the counters and returns zeros.
class CpuUsageSampler final {
public:
	CpuUsageSampler();
	~CpuUsageSampler();
	KYTY_CLASS_NO_COPY(CpuUsageSampler);

	[[nodiscard]] CpuUsage Sample();

private:
	struct Impl; // platform state (previous tick counters, cached thread handles)
	std::unique_ptr<Impl> m_impl;
};

struct MemoryUsage {
	uint64_t system_total_bytes = 0;
	uint64_t system_used_bytes  = 0; // total - available
	uint64_t process_bytes      = 0; // resident set (Linux RSS / Windows working set)
};

[[nodiscard]] MemoryUsage SampleMemoryUsage();

struct GpuUsage {
	uint32_t                utilization_percent = 0;
	uint64_t                vram_used_bytes     = 0;
	uint64_t                vram_total_bytes    = 0;
	std::optional<uint32_t> temperature_c;
	std::optional<uint32_t> power_w;
};

// NVIDIA GPU counters through NVML, loaded at runtime (libnvidia-ml.so.1 / nvml.dll) so the
// emulator never links against or requires it. Open() returns nullptr when the library, its
// init, or a matching device is unavailable (non-NVIDIA GPU, container without the driver
// libraries...), and the overlay shows "n/a".
class NvmlGpuMonitor final {
public:
	// Picks the NVML device whose name matches `device_name` (the Vulkan device name), else
	// device 0.
	[[nodiscard]] static std::unique_ptr<NvmlGpuMonitor> Open(std::string_view device_name);
	~NvmlGpuMonitor();
	KYTY_CLASS_NO_COPY(NvmlGpuMonitor);

	// nullopt if the query fails (e.g. GPU fell off the bus); callers keep showing "n/a".
	[[nodiscard]] std::optional<GpuUsage> Sample();

private:
	struct Impl; // library handle, resolved entry points, device handle
	explicit NvmlGpuMonitor(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> m_impl;
};

} // namespace Common::Perf

#endif /* KYTY_COMMON_PERF_HOSTSTATS_H_ */
