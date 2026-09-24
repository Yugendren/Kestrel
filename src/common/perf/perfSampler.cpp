#include "common/perf/perfSampler.h"

#include "common/profiler.h"

#include <chrono>
#include <utility>

namespace Common::Perf {

namespace {

constexpr uint64_t kLastSecondUs = 1'000'000;
constexpr uint64_t kStallUs      = 1'000'000;

// After (re)activation the CPU counters are re-primed and the first real sample is taken this
// much later, so it measures current load instead of averaging over the whole paused period.
constexpr uint32_t kPrimeUs = 100'000;

uint64_t NowUs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

PerfSampler::PerfSampler(const FrameTimeHistory& frames, std::unique_ptr<NvmlGpuMonitor> gpu,
                         uint32_t interval_ms)
    : m_frames(frames), m_gpu(std::move(gpu)), m_interval_ms(interval_ms) {
	m_thread = std::make_unique<Common::Thread>(ThreadMain, this);
}

PerfSampler::~PerfSampler() {
	m_mutex.Lock();
	m_stop = true;
	m_wake.Signal();
	m_mutex.Unlock();

	m_thread->Join();
}

void PerfSampler::SetActive(bool active) {
	Common::LockGuard lock(m_mutex);
	if (m_active != active) {
		m_active = active;
		m_wake.Signal();
	}
}

PerfSnapshot PerfSampler::Latest() const {
	Common::LockGuard lock(m_mutex);
	return m_latest;
}

void PerfSampler::ThreadMain(void* arg) {
	static_cast<PerfSampler*>(arg)->Run();
}

void PerfSampler::Run() {
	Profiler::SetThreadName("Thread_Perf");

	bool primed = false;

	m_mutex.Lock();
	while (!m_stop) {
		if (!m_active) {
			primed = false;
			m_wake.Wait(&m_mutex);
			continue;
		}

		// Sampling runs unlocked so SetActive()/Latest() never wait on /proc or NVML reads.
		m_mutex.Unlock();
		uint32_t wait_us = m_interval_ms * 1000;
		if (primed) {
			SampleOnce();
		} else {
			(void)m_cpu.Sample();
			primed  = true;
			wait_us = kPrimeUs;
		}
		m_mutex.Lock();

		if (!m_stop && m_active) {
			m_wake.WaitFor(&m_mutex, wait_us);
		}
	}
	m_mutex.Unlock();
}

void PerfSampler::SampleOnce() {
	const auto     start  = std::chrono::steady_clock::now();
	const uint64_t now_us = NowUs();

	PerfSnapshot snapshot;

	m_frames.CopyFrameTimes(now_us, kLastSecondUs, &m_scratch);
	snapshot.last_second = SummarizeFrameTimes(m_scratch);
	m_frames.CopyFrameTimes(now_us, m_frames.WindowUs(), &m_scratch);
	snapshot.window = SummarizeFrameTimes(m_scratch);

	const uint64_t last_present_us = m_frames.LastPresentUs();
	snapshot.stalled =
	    last_present_us == 0 || (now_us > last_present_us && now_us - last_present_us > kStallUs);

	snapshot.cpu    = m_cpu.Sample();
	snapshot.memory = SampleMemoryUsage();
	if (m_gpu != nullptr) {
		snapshot.gpu = m_gpu->Sample();
	}

	snapshot.sample_cost_ms =
	    std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

	Common::LockGuard lock(m_mutex);
	snapshot.sequence = m_latest.sequence + 1;
	m_latest          = std::move(snapshot);
}

} // namespace Common::Perf
