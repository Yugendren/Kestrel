#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FRAMETIMELOG_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FRAMETIMELOG_H_

#include "common/common.h"
#include "common/threads.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

namespace Libs::Graphics {

// --frame-time-log: one "<frame> <monotonic_us>" line per presented frame, so smoothness (1% /
// 0.1% lows, frame-time percentiles, hitches) can be measured instead of only the per-second
// average of --fps-log. Record() runs on the present path and only appends 16 bytes to an
// in-memory batch under an uncontended mutex; a background thread swaps the batch out and writes
// it once a second, so file I/O never lands on the render path. Timestamps are CLOCK_MONOTONIC
// (std::chrono::steady_clock), so external tools can correlate them with their own monotonic
// clock (e.g. python time.monotonic_ns()).
class FrameTimeLog final {
public:
	// Returns nullptr when the path cannot be opened (logged), so callers simply skip recording.
	[[nodiscard]] static std::unique_ptr<FrameTimeLog> Open(const std::filesystem::path& path);

	~FrameTimeLog();
	KYTY_CLASS_NO_COPY(FrameTimeLog);

	void Record();

private:
	struct Sample {
		uint64_t frame;
		uint64_t time_us;
	};

	explicit FrameTimeLog(std::FILE* file);
	static void WriterMain(void* arg);
	void        WriteBatch(const std::vector<Sample>& batch);

	std::FILE*                    m_file;
	uint64_t                      m_frame = 0; // present thread only
	Common::Mutex                 m_mutex;
	Common::CondVar               m_wake;
	std::vector<Sample>           m_pending;          // guarded by m_mutex
	bool                          m_stop    = false;  // guarded by m_mutex
	std::unique_ptr<Common::Thread> m_writer;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FRAMETIMELOG_H_
