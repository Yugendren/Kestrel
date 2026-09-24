#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions    = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
	m_cpu_epochs = std::make_unique<std::atomic<uint64_t>[]>(REGION_COUNT);
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	// A new region starts out entirely CPU-dirty, so every buffer already cached inside it needs
	// re-uploading.
	ScopedDirtyPublish published(*this, index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE);
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

void MemoryTracker::MarkRangeDirty(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
	{
		std::scoped_lock lock(m_dirty_mutex);
		m_dirty.Add(vaddr, size);
	}
	m_has_dirty.store(true, std::memory_order_release);
	const auto first = vaddr / TRACKER_REGION_SIZE;
	if (first >= REGION_COUNT) {
		return;
	}
	const auto last = std::min<uint64_t>((vaddr + (size - 1)) / TRACKER_REGION_SIZE,
	                                     REGION_COUNT - 1);
	for (auto index = first; index <= last; index++) {
		m_cpu_epochs[index].fetch_add(1, std::memory_order_acq_rel);
	}
}

uint64_t MemoryTracker::CpuModificationEpoch(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	// Every counter only grows, so their sum changes whenever any of them does.
	constexpr uint64_t MaxRegions = 16;
	const auto         first      = vaddr / TRACKER_REGION_SIZE;
	const auto         last       = (vaddr + size - 1) / TRACKER_REGION_SIZE;
	if (last - first >= MaxRegions) {
		return 0;
	}
	uint64_t epoch = 0;
	for (auto index = first; index <= last; index++) {
		if (m_regions[index].load(std::memory_order_acquire) == nullptr) {
			return 0;
		}
		epoch += m_cpu_epochs[index].load(std::memory_order_acquire);
	}
	return epoch;
}

MemoryTracker::Modification MemoryTracker::QueryModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Modification result;
	Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		result.cpu = result.cpu || manager->IsModified<DirtySource::Cpu>(offset, bytes);
		result.gpu = result.gpu || manager->IsModified<DirtySource::Gpu>(offset, bytes);
		return result.cpu && result.gpu;
	});
	return result;
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	ScopedDirtyPublish published(*this, vaddr, size);
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UntrackMemory(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	ScopedDirtyPublish published(*this, vaddr, size);
	std::vector<RegionManager*> managers;
	managers.reserve((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) /
	                 TRACKER_REGION_SIZE);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
		managers.push_back(manager);
	});

	std::vector<std::unique_lock<TrackingSpinLock>> locks;
	locks.reserve(managers.size());
	for (auto* manager: managers) {
		locks.emplace_back(manager->lock);
	}
	if (Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	    })) {
		EXIT("cannot untrack GPU-dirty memory\n");
	}
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

} // namespace Libs::Graphics
