#ifndef KYTY_COMMON_PERF_FRAMETIMESTATS_H_
#define KYTY_COMMON_PERF_FRAMETIMESTATS_H_

#include "common/common.h"
#include "common/threads.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Common::Perf {

// Statistics over a sequence of frame times (milliseconds, oldest first).
struct FrameTimeSummary {
	uint32_t frames  = 0;
	double   fps     = 0.0; // frames / total time of the sequence
	double   avg_ms  = 0.0;
	double   last_ms = 0.0; // newest frame
	double   max_ms  = 0.0;
	// "1% low" in the PresentMon / CapFrameX sense: the fps of the 99th-percentile frame time
	// (nearest-rank), i.e. only 1% of frames in the sequence were slower than this. 0 when the
	// sequence is empty.
	double low1_fps = 0.0;
};

// Pure function so the math is unit-testable offline. Does not modify or keep the input.
[[nodiscard]] FrameTimeSummary SummarizeFrameTimes(std::span<const float> frame_ms);

// Present timestamps of the last `window_us` microseconds. RecordPresent() is called on the
// present thread for every presented frame and costs an uncontended mutex plus a ring write
// (no allocation once the ring has grown to the steady-state frame rate); readers copy out
// under the same mutex and do all math on their own thread.
class FrameTimeHistory final {
public:
	explicit FrameTimeHistory(uint64_t window_us = 10'000'000);
	~FrameTimeHistory() = default;
	KYTY_CLASS_NO_COPY(FrameTimeHistory);

	// time_us: monotonic (steady_clock) microseconds, non-decreasing across calls.
	void RecordPresent(uint64_t time_us);

	// Frame times (ms, oldest first) of frames presented in (now_us - span_us, now_us], where a
	// frame's time is the interval since the previous present. Replaces *out's contents.
	void CopyFrameTimes(uint64_t now_us, uint64_t span_us, std::vector<float>* out) const;

	// The newest out.size() frame times (oldest first), for the frame-time graph. Returns how
	// many were written (fewer while the history is still filling).
	size_t CopyLatest(std::span<float> out) const;

	// 0 when nothing has been presented yet.
	[[nodiscard]] uint64_t LastPresentUs() const;

	[[nodiscard]] uint64_t WindowUs() const { return m_window_us; }

private:
	// (implementation-defined ring of timestamps guarded by m_mutex)
	mutable Common::Mutex m_mutex;
	uint64_t              m_window_us;
	std::vector<uint64_t> m_ring;      // guarded by m_mutex
	size_t                m_head  = 0; // guarded by m_mutex
	size_t                m_count = 0; // guarded by m_mutex
};

} // namespace Common::Perf

#endif /* KYTY_COMMON_PERF_FRAMETIMESTATS_H_ */
