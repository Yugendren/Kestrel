#pragma once

#include <cstdint>

namespace Libs::Graphics {

// A RELEASE_MEM packet either writes a label into guest memory or raises an end-of-pipe
// interrupt once the work before it has finished. Both are carried by the command buffer the
// packet was recorded into and take effect when that buffer completes, so they only need the
// buffer to be submitted eventually, not immediately. Submitting per packet costs a queue
// submission -- and the GPU pipeline bubble that comes with it -- for every one of the hundreds
// of these a title emits per frame.
//
// Batching them delays when the guest observes a label or an event, never the other way round,
// so the guest can still not see work reported as finished before it is. The count is bounded
// so a guest thread polling a label or waiting on the event queue cannot be held up for long,
// and the command processor flushes the batch whenever it waits, submits, or finishes a
// command buffer.
class ReleaseMemBatch {
public:
	static constexpr uint32_t MAX_PACKETS = 32;

	// Returns true when the caller may leave its flush to a later packet.
	bool Defer() noexcept {
		if (m_pending + 1 >= MAX_PACKETS) {
			return false;
		}
		++m_pending;
		return true;
	}

	[[nodiscard]] bool Pending() const noexcept { return m_pending != 0; }
	void               Reset() noexcept { m_pending = 0; }

private:
	uint32_t m_pending = 0;
};

} // namespace Libs::Graphics
