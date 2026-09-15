#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERHAZARDTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERHAZARDTRACKER_H_

#include "common/slotVector.h"
#include "graphics/host_gpu/rangeSet.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The guest memory and images one shader item touches, as the resource materialisation resolved
// them. `everything` marks an access that could not be bounded -- DMA, GDS, atomics, indirect
// arguments, a descriptor that named no guest range -- and forces the conservative barrier.
struct ShaderHazardAccess {
	std::vector<std::pair<uint64_t, uint64_t>> buffers;
	std::vector<Common::SlotId>                images;
	bool                                       everything = false;

	void Clear() {
		buffers.clear();
		images.clear();
		everything = false;
	}
};

// Remembers what the shader work recorded since the last memory barrier read and wrote, so that
// a barrier is emitted only where a later item actually depends on it. Dispatches that touch
// disjoint resources then overlap on the GPU instead of each draining the pipeline.
//
// The tracker is a recording-time structure: it is only ever used from the thread that records
// commands, and it is emptied whenever the deferred barrier is emitted.
class ShaderHazardTracker {
public:
	// True when a memory dependency must separate the recorded work from this item.
	[[nodiscard]] bool NeedsBarrier(const ShaderHazardAccess& reads,
	                                const ShaderHazardAccess& writes) const {
		if (!m_pending) {
			return false;
		}
		if (m_everything || reads.everything || writes.everything) {
			return true;
		}
		for (const auto& [address, size]: reads.buffers) {
			if (m_written_buffers.Intersects(address, size)) {
				return true; // read after write
			}
		}
		for (const auto& [address, size]: writes.buffers) {
			if (m_written_buffers.Intersects(address, size) ||
			    m_read_buffers.Intersects(address, size)) {
				return true; // write after write, write after read
			}
		}
		for (const auto id: reads.images) {
			if (Listed(m_written_images, id)) {
				return true;
			}
		}
		for (const auto id: writes.images) {
			if (Listed(m_written_images, id) || Listed(m_read_images, id)) {
				return true;
			}
		}
		return false;
	}

	void Record(const ShaderHazardAccess& reads, const ShaderHazardAccess& writes) {
		m_pending = true;
		// The sets exist only to refine the barrier. Once they grow past this the tracker stops
		// refining and every following item takes the barrier, which is what the recorder did
		// unconditionally before.
		if (++m_items > MaxTrackedItems || reads.everything || writes.everything) {
			m_everything = true;
		}
		for (const auto& [address, size]: reads.buffers) {
			m_read_buffers.Add(address, size);
		}
		for (const auto& [address, size]: writes.buffers) {
			m_written_buffers.Add(address, size);
		}
		Append(m_read_images, reads.images);
		Append(m_written_images, writes.images);
	}

	// True when work has been recorded that a later command may depend on.
	[[nodiscard]] bool HasPendingWork() const noexcept { return m_pending; }

	void Clear() {
		m_read_buffers.Clear();
		m_written_buffers.Clear();
		m_read_images.clear();
		m_written_images.clear();
		m_everything = false;
		m_pending    = false;
		m_items      = 0;
	}

private:
	static constexpr size_t MaxTrackedItems = 64;

	[[nodiscard]] static bool Listed(const std::vector<Common::SlotId>& list, Common::SlotId id) {
		return std::find(list.begin(), list.end(), id) != list.end();
	}

	static void Append(std::vector<Common::SlotId>& list, const std::vector<Common::SlotId>& ids) {
		for (const auto id: ids) {
			if (!Listed(list, id)) {
				list.push_back(id);
			}
		}
	}

	RangeSet                    m_read_buffers;
	RangeSet                    m_written_buffers;
	std::vector<Common::SlotId> m_read_images;
	std::vector<Common::SlotId> m_written_images;
	bool                        m_everything = false;
	bool                        m_pending    = false;
	size_t                      m_items      = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERHAZARDTRACKER_H_
