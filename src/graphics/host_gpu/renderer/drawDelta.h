#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWDELTA_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWDELTA_H_

#include <array>
#include <chrono>
#include <cstdint>

// Delta draw: recording only what changed since the previous draw.
//
// Consecutive draws mostly repeat the draw before them with a fresh per-draw constant block and
// little else. On top of draw-state reuse (drawReuse.h), which keeps the previous draw's render
// state and pipeline, the draw path keeps more of what provably did not change: the
// register-only entry checks and the topology, when DrawStateTracker vouches for the registers
// (RenderExecutor::RunsDrawEntryChecks(), renderDraw.cpp). Every piece is guarded by what makes
// it equivalent to the full path, never by title knowledge. KYTY_DELTA_ORACLE=1 runs the full
// path beside every reuse and exits on the first difference.

namespace Libs::Graphics {

// KYTY_DELTA_ORACLE=1: everything a draw keeps from an earlier draw is also resolved the full
// way and compared, and the first difference EXITs. The counters double as a once-per-second
// summary of how much was kept. The checks themselves live next to the code they verify
// (renderDraw.cpp); this is the bookkeeping.
class DeltaDrawOracle {
public:
	enum class Counter : uint32_t {
		Checked,          // reuses compared against the full path
		EntriesKept,      // draws whose register checks and topology were the previous draw's
		Count,
	};

	[[nodiscard]] static bool Enabled();

	void Note(Counter counter, uint64_t amount = 1) noexcept {
		m_counters[static_cast<uint32_t>(counter)] += amount;
	}

	// Logs the running totals at most once per second.
	void LogSummaryIfDue();

	[[nodiscard]] uint64_t Count(Counter counter) const noexcept {
		return m_counters[static_cast<uint32_t>(counter)];
	}

private:
	std::array<uint64_t, static_cast<uint32_t>(Counter::Count)> m_counters {};
	std::chrono::steady_clock::time_point m_last_summary = std::chrono::steady_clock::now();
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWDELTA_H_ */
