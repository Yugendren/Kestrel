#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);

	struct Modification {
		bool cpu = false;
		bool gpu = false;
	};
	// Both dirty states of a range in one pass, taking each region lock once. Creates missing
	// regions exactly like IsRegionCpuModified().
	[[nodiscard]] Modification QueryModified(uint64_t vaddr, uint64_t size);

	// Advances whenever bytes of the range's tracker regions may have become CPU-modified: every
	// such change is published through MarkRangeDirty(), which advances the per-region counters
	// this sums. A caller that saw the range CPU-clean at some epoch may keep relying on that
	// while the epoch is unchanged, without asking the tracker again. 0 -- never trusted -- when
	// one of its regions does not exist yet or the range spans more than a few regions.
	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const noexcept;
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Records that the guest bytes in this range may now need re-uploading into whatever buffer
	// caches them. Thread-safe: guest write faults publish from the thread that faulted.
	void MarkRangeDirty(uint64_t vaddr, uint64_t size) noexcept;
	[[nodiscard]] bool HasDirtyRanges() const noexcept {
		return m_has_dirty.load(std::memory_order_acquire);
	}
	// Moves the published set out and reports each range as (begin, end). Ranges published while
	// the drain runs stay for the next one.
	template <typename Func>
	void DrainDirtyRanges(Func&& func) {
		RangeSet taken;
		{
			std::scoped_lock lock(m_dirty_mutex);
			if (m_dirty.Empty()) {
				return;
			}
			taken = std::move(m_dirty);
			m_dirty.Clear();
			m_has_dirty.store(false, std::memory_order_relaxed);
		}
		taken.ForEach(func);
	}

	// Publication has to bracket the state change it describes. A drain running concurrently then
	// either takes the range before the pages are marked -- and the publication on the way out
	// arms the next drain -- or takes it after them. Publishing only on the way out would leave a
	// freshly written page waiting for an extra preparation, because unprotecting it costs an
	// mprotect and its TLB shootdown; publishing only on the way in would let a drain consume the
	// range while the marks did not exist yet, and lose it.
	class ScopedDirtyPublish final {
	public:
		ScopedDirtyPublish(MemoryTracker& tracker, uint64_t vaddr, uint64_t size) noexcept
		    : m_tracker(tracker), m_vaddr(vaddr), m_size(size) {
			m_tracker.MarkRangeDirty(m_vaddr, m_size);
		}
		~ScopedDirtyPublish() { m_tracker.MarkRangeDirty(m_vaddr, m_size); }
		KYTY_CLASS_NO_COPY(ScopedDirtyPublish);

	private:
		MemoryTracker& m_tracker;
		uint64_t       m_vaddr;
		uint64_t       m_size;
	};
	// Removes protection from a range and flushes GPU-owned data when required.
	// With from_fault the range is a single faulting guest page and the CPU-dirty mark is
	// widened over its coarse block, which is what keeps a sequential guest write from costing
	// one fault per page.
	template <bool from_fault = false, typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();
		// A write fault marks the whole coarse block its page belongs to (see
		// RegionManager::MarkCpuModifiedFromFault), so that is what has to be published.
		constexpr uint64_t block_bytes = TRACKER_FAULT_BLOCK_PAGES * TRACKER_PAGE_SIZE;
		ScopedDirtyPublish published(*this, from_fault ? (vaddr & ~(block_bytes - 1)) : vaddr,
		                             from_fault ? block_bytes : size);

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				if constexpr (from_fault) {
					manager->MarkCpuModifiedFromFault(manager->GetCpuAddr() + offset);
				} else {
					manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
					                                             bytes);
				}
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	// Per region, see CpuModificationEpoch().
	std::unique_ptr<std::atomic<uint64_t>[]>       m_cpu_epochs;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;

	// The guest ranges whose bytes have changed since the set was last drained.
	std::mutex        m_dirty_mutex;
	RangeSet          m_dirty;
	std::atomic<bool> m_has_dirty {false};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
