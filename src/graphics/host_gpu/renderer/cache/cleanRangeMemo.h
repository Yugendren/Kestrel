#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANRANGEMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANRANGEMEMO_H_

#include "common/slotVector.h"

#include <array>
#include <cstdint>
#include <optional>

namespace Libs::Graphics {

// Guest ranges BufferCache::ObtainBuffer() has already synchronised for reading, each with the
// buffer that holds it and the memory tracker's CPU-modification epoch it was synchronised under
// (MemoryTracker::CpuModificationEpoch). While that epoch is unchanged no byte of the range can
// have become CPU-modified, so there is nothing to upload and the same buffer still answers: the
// tracker queries and the synchronisation can be skipped. Draws re-read the same vertex, index
// and storage ranges thousands of times a frame, and this is what makes that repetition cheap.
//
// Direct-mapped: a colliding range simply replaces the older entry. Only the caller can decide
// whether the buffer is still alive, so Find() returns the id for it to check.
class CleanRangeMemo final {
public:
	[[nodiscard]] std::optional<Common::SlotId> Find(uint64_t vaddr, uint64_t size,
	                                                 uint64_t epoch) const noexcept {
		if (epoch == 0) {
			return std::nullopt;
		}
		const auto& entry = m_entries[Index(vaddr, size)];
		if (entry.epoch != epoch || entry.vaddr != vaddr || entry.size != size) {
			return std::nullopt;
		}
		return entry.id;
	}

	// `epoch` must have been read before the tracker was queried for this range, so a
	// modification racing the synchronisation always leaves the entry stale rather than trusted.
	void Remember(uint64_t vaddr, uint64_t size, uint64_t epoch, Common::SlotId id) noexcept {
		if (epoch == 0) {
			return;
		}
		m_entries[Index(vaddr, size)] = {vaddr, size, epoch, id};
	}

private:
	static constexpr size_t EntryCount = 1024;

	struct Entry {
		uint64_t       vaddr = 0;
		uint64_t       size  = 0;
		uint64_t       epoch = 0;
		Common::SlotId id;
	};

	[[nodiscard]] static size_t Index(uint64_t vaddr, uint64_t size) noexcept {
		const auto hash = (vaddr ^ (size * 0x9E3779B97F4A7C15ull)) * 0xD6E8FEB86659FD93ull;
		return static_cast<size_t>(hash >> 54) & (EntryCount - 1);
	}

	std::array<Entry, EntryCount> m_entries {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANRANGEMEMO_H_
