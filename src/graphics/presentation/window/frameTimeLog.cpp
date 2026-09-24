#include "graphics/presentation/window/frameTimeLog.h"

#include "common/logging/log.h"

#include <chrono>
#include <cinttypes>
#include <utility>

namespace Libs::Graphics {

// How often the writer thread flushes; also the most frames lost if the process is killed.
static constexpr uint32_t FLUSH_INTERVAL_US = 1000000;
// A few seconds of frames, so the present thread never reallocates in steady state.
static constexpr size_t   BATCH_RESERVE     = 1024;

std::unique_ptr<FrameTimeLog> FrameTimeLog::Open(const std::filesystem::path& path) {
	std::FILE* file = std::fopen(path.string().c_str(), "a");
	if (file == nullptr) {
		LOGF("frame-time-log: cannot open %s, frame timing disabled\n", path.string().c_str());
		return nullptr;
	}
	return std::unique_ptr<FrameTimeLog>(new FrameTimeLog(file));
}

FrameTimeLog::FrameTimeLog(std::FILE* file): m_file(file) {
	m_pending.reserve(BATCH_RESERVE);
	m_writer = std::make_unique<Common::Thread>(WriterMain, this);
}

FrameTimeLog::~FrameTimeLog() {
	{
		Common::LockGuard lock(m_mutex);
		m_stop = true;
	}
	m_wake.Signal();
	m_writer->Join();
	std::fclose(m_file);
}

void FrameTimeLog::Record() {
	const auto time_us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count());
	Common::LockGuard lock(m_mutex);
	m_pending.push_back({m_frame++, time_us});
}

void FrameTimeLog::WriterMain(void* arg) {
	auto* self = static_cast<FrameTimeLog*>(arg);
	std::vector<Sample> batch;
	batch.reserve(BATCH_RESERVE);
	for (;;) {
		bool stop = false;
		{
			Common::LockGuard lock(self->m_mutex);
			if (!self->m_stop) {
				self->m_wake.WaitFor(&self->m_mutex, FLUSH_INTERVAL_US);
			}
			stop = self->m_stop;
			std::swap(batch, self->m_pending);
		}
		self->WriteBatch(batch);
		batch.clear();
		if (stop) {
			return;
		}
	}
}

void FrameTimeLog::WriteBatch(const std::vector<Sample>& batch) {
	if (batch.empty()) {
		return;
	}
	for (const auto& s: batch) {
		std::fprintf(m_file, "%" PRIu64 " %" PRIu64 "\n", s.frame, s.time_us);
	}
	std::fflush(m_file);
}

} // namespace Libs::Graphics
