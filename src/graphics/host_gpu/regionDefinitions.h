#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONDEFINITIONS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONDEFINITIONS_H_

#include "common/bitArray.h"
#include "common/common.h"

#include <compare>

namespace Libs::Graphics {

constexpr uint64_t TRACKER_PAGE_SIZE    = 4ull * 1024ull;
constexpr uint64_t TRACKER_REGION_SIZE  = 4ull * 1024ull * 1024ull;
constexpr uint64_t TRACKER_ADDRESS_SIZE = 1ull << 40u;
constexpr size_t   TRACKER_REGION_PAGES = TRACKER_REGION_SIZE / TRACKER_PAGE_SIZE;

// Guest write faults are widened to the coarse block they belong to (see
// RegionManager::MarkCpuModifiedFromFault). One fault then covers a whole block of sequential
// guest writes instead of one fault, one mprotect and one TLB shootdown per 4 KB page.
constexpr size_t   TRACKER_FAULT_BLOCK_PAGES = 16; // 64 KB
constexpr size_t   TRACKER_FAULT_BLOCKS      = TRACKER_REGION_PAGES / TRACKER_FAULT_BLOCK_PAGES;
// Widen only once a block has been seen to hold several separately written pages, so that a
// lone hot page in an otherwise cold block is still tracked at page granularity.
constexpr int      TRACKER_FAULT_WIDEN_MIN_PAGES = 3;
static_assert(TRACKER_FAULT_BLOCK_PAGES <= 16, "the per-block page set is stored in a uint16_t");
static_assert(TRACKER_REGION_PAGES % TRACKER_FAULT_BLOCK_PAGES == 0);

struct GuestRange {
	uint64_t address = 0;
	uint64_t size    = 0;

	[[nodiscard]] constexpr bool Empty() const noexcept { return address == 0 && size == 0; }
	[[nodiscard]] constexpr bool Valid() const noexcept {
		return address != 0 && size != 0 && address < TRACKER_ADDRESS_SIZE &&
		       size <= TRACKER_ADDRESS_SIZE - address;
	}
	[[nodiscard]] constexpr bool     ValidOrEmpty() const noexcept { return Empty() || Valid(); }
	[[nodiscard]] constexpr uint64_t End() const noexcept { return address + size; }
	auto                             operator<=>(const GuestRange&) const = default;
};

enum class DirtySource { Cpu, Gpu };
using RegionBits = Common::BitArray<TRACKER_REGION_PAGES>;
static_assert(sizeof(RegionBits) == TRACKER_REGION_PAGES / 8);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONDEFINITIONS_H_
