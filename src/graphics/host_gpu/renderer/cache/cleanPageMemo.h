#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANPAGEMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANPAGEMEMO_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

// Guest pages already proven readable straight through the guest mapping: nothing the GPU wrote
// is pending for any byte, so a load returns the current value and does not fault. Resource
// materialisation reads shader resource tables a dword at a time, over a hundred thousand times a
// frame, from a small set of pages; proving a page once replaces the buffer-cache, texture-cache
// and tracker queries every one of those reads needed.
//
// A proof holds until some range may have become GPU-dirty or the guest mappings changed. The
// owner passes that epoch (LibKernel::Memory::CurrentMappingEpoch) to Begin() before it reads,
// and the memo forgets every page when the epoch has moved. Pages can also become clean
// meanwhile (a download publishing its bytes), which leaves a proof true; that is why a page that
// fails the proof is not remembered: a later read may find it clean. Epoch 0 means the caller
// cannot know, and nothing is answered.
//
// A read that straddles pages, or touches page 0, is never answered here: the caller checks it
// exactly, as it would without the memo. Direct-mapped: a colliding page replaces the older one,
// which then only has to be proven again.
class CleanPageMemo final {
public:
	static constexpr uint64_t PageSize = TRACKER_PAGE_SIZE;

	void Begin(uint64_t epoch) noexcept {
		if (epoch != m_epoch) {
			m_pages.fill(0);
			m_epoch = epoch;
		}
	}

	// True when [vaddr, vaddr + size) lies in one page that is remembered, or that
	// `prove(page_address, PageSize)` now proves (it is then remembered).
	template <typename Prove>
	[[nodiscard]] bool Contains(uint64_t vaddr, uint64_t size, Prove&& prove) {
		const auto page = vaddr & ~(PageSize - 1u);
		if (m_epoch == 0 || size == 0 || page == 0 || size > PageSize ||
		    vaddr - page > PageSize - size) {
			return false;
		}
		auto& entry = m_pages[(page / PageSize) % EntryCount];
		if (entry == page) {
			return true;
		}
		if (!prove(page, PageSize)) {
			return false;
		}
		entry = page;
		return true;
	}

private:
	// Enough for the pages a frame's draws read between two epochs, streamed constants
	// included; Begin() clears it a few dozen times a frame.
	static constexpr size_t EntryCount = 256;

	std::array<uint64_t, EntryCount> m_pages {};
	uint64_t                         m_epoch = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_CLEANPAGEMEMO_H_
