#include "common/perf/frameTimeStats.h"

#include <algorithm>

namespace Common::Perf {

namespace {

constexpr size_t kInitialRingCapacity = 1024;

float IntervalMs(uint64_t from_us, uint64_t to_us) {
	return static_cast<float>(static_cast<double>(to_us - from_us) / 1000.0);
}

} // namespace

FrameTimeSummary SummarizeFrameTimes(std::span<const float> frame_ms) {
	FrameTimeSummary summary;
	if (frame_ms.empty()) {
		return summary;
	}

	double total_ms = 0.0;
	for (const float ms: frame_ms) {
		total_ms += ms;
		summary.max_ms = std::max(summary.max_ms, static_cast<double>(ms));
	}

	const size_t count = frame_ms.size();
	summary.frames     = static_cast<uint32_t>(count);
	summary.avg_ms     = total_ms / static_cast<double>(count);
	summary.last_ms    = frame_ms.back();
	summary.fps        = total_ms > 0.0 ? static_cast<double>(count) * 1000.0 / total_ms : 0.0;

	// Nearest-rank 99th percentile: the ceil(0.99 * n)-th smallest frame time. Integer math so
	// that e.g. n = 200 gives rank 198 exactly instead of depending on 0.99 * 200 rounding.
	const size_t       rank = (99 * count + 99) / 100;
	std::vector<float> sorted(frame_ms.begin(), frame_ms.end());
	auto               nth = sorted.begin() + static_cast<std::ptrdiff_t>(rank - 1);
	std::nth_element(sorted.begin(), nth, sorted.end());
	summary.low1_fps = *nth > 0.0f ? 1000.0 / static_cast<double>(*nth) : 0.0;

	return summary;
}

FrameTimeHistory::FrameTimeHistory(uint64_t window_us)
    : m_window_us(window_us), m_ring(kInitialRingCapacity) {}

void FrameTimeHistory::RecordPresent(uint64_t time_us) {
	Common::LockGuard lock(m_mutex);

	// Evict first so a full ring only grows when the whole window genuinely needs more slots
	// than it has (higher frame rate than seen before); at a steady rate no allocation happens.
	while (m_count > 0 && time_us - m_ring[m_head] > m_window_us) {
		m_head = (m_head + 1) % m_ring.size();
		m_count--;
	}

	if (m_count == m_ring.size()) {
		std::vector<uint64_t> grown(m_ring.size() * 2);
		for (size_t i = 0; i < m_count; i++) {
			grown[i] = m_ring[(m_head + i) % m_ring.size()];
		}
		m_ring.swap(grown);
		m_head = 0;
	}

	m_ring[(m_head + m_count) % m_ring.size()] = time_us;
	m_count++;
}

void FrameTimeHistory::CopyFrameTimes(uint64_t now_us, uint64_t span_us,
                                      std::vector<float>* out) const {
	out->clear();

	Common::LockGuard lock(m_mutex);

	const size_t capacity = m_ring.size();
	for (size_t i = 1; i < m_count; i++) {
		const uint64_t time_us = m_ring[(m_head + i) % capacity];
		// (now - span, now], written without the subtraction so span > now cannot underflow.
		if (time_us > now_us || time_us + span_us <= now_us) {
			continue;
		}
		out->push_back(IntervalMs(m_ring[(m_head + i - 1) % capacity], time_us));
	}
}

size_t FrameTimeHistory::CopyLatest(std::span<float> out) const {
	Common::LockGuard lock(m_mutex);

	const size_t available = m_count > 0 ? m_count - 1 : 0;
	const size_t written   = std::min(out.size(), available);
	const size_t capacity  = m_ring.size();
	// Timestamp index (relative to the oldest) of the first frame to copy.
	const size_t first = m_count - written;
	for (size_t i = 0; i < written; i++) {
		const size_t index = m_head + first + i;
		out[i]             = IntervalMs(m_ring[(index - 1) % capacity], m_ring[index % capacity]);
	}
	return written;
}

uint64_t FrameTimeHistory::LastPresentUs() const {
	Common::LockGuard lock(m_mutex);
	return m_count > 0 ? m_ring[(m_head + m_count - 1) % m_ring.size()] : 0;
}

} // namespace Common::Perf
