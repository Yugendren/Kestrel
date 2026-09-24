#ifndef KYTY_COMMON_PERF_PERFSAMPLER_H_
#define KYTY_COMMON_PERF_PERFSAMPLER_H_

#include "common/common.h"
#include "common/perf/frameTimeStats.h"
#include "common/perf/hostStats.h"
#include "common/threads.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace Common::Perf {

// One published set of live counters. Copied out whole by the overlay each frame (a few
// hundred bytes), so readers never hold the sampler lock while drawing.
struct PerfSnapshot {
	uint64_t         sequence = 0;    // 0 = nothing sampled yet
	FrameTimeSummary last_second;     // fps / current frame time
	FrameTimeSummary window;          // rolling window (10 s): average, 1% low, max
	bool             stalled = false; // nothing presented for over a second (loading, hang)
	CpuUsage         cpu;
	MemoryUsage      memory;
	std::optional<GpuUsage> gpu;      // nullopt: NVML unavailable
	float            sample_cost_ms = 0; // wall time of the last sampling pass
};

// Background thread that samples frame statistics, CPU, memory and GPU counters at a fixed
// rate and publishes a PerfSnapshot. It only runs while active (overlay visible): an inactive
// sampler blocks on a condition variable and costs nothing.
class PerfSampler final {
public:
	PerfSampler(const FrameTimeHistory& frames, std::unique_ptr<NvmlGpuMonitor> gpu,
	            uint32_t interval_ms = 250);
	~PerfSampler(); // stops and joins the thread
	KYTY_CLASS_NO_COPY(PerfSampler);

	// Thread-safe. Activating wakes the thread and takes a sample immediately.
	void SetActive(bool active);

	// Thread-safe; returns the latest published snapshot.
	[[nodiscard]] PerfSnapshot Latest() const;

private:
	static void ThreadMain(void* arg);
	void        Run();
	void        SampleOnce(); // sampler thread only

	const FrameTimeHistory&          m_frames;
	std::unique_ptr<NvmlGpuMonitor>  m_gpu;     // sampler thread only
	CpuUsageSampler                  m_cpu;     // sampler thread only
	std::vector<float>               m_scratch; // sampler thread only
	uint32_t                         m_interval_ms;

	mutable Common::Mutex            m_mutex;
	Common::CondVar                  m_wake;
	bool                             m_active = false; // guarded by m_mutex
	bool                             m_stop   = false; // guarded by m_mutex
	PerfSnapshot                     m_latest;         // guarded by m_mutex
	std::unique_ptr<Common::Thread>  m_thread;
};

} // namespace Common::Perf

#endif /* KYTY_COMMON_PERF_PERFSAMPLER_H_ */
