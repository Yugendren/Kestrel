#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/cleanRangeMemo.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size, bool from_fault = false);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);

	// Opens the reuse scope for the descriptors of one draw or dispatch. Consecutive slots, and
	// the two stages of a draw, routinely ask ObtainBuffer() for the same guest range: half of all
	// requests repeat one of the last sixteen. Everything ObtainBuffer() does for a range -- the
	// stream re-upload, the LRU touch, the synchronisation, the GPU-modified bookkeeping -- is
	// idempotent within one scope, so repeats are answered from a small table instead.
	void BeginBufferScope();
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	// Newest scheduler tick at which any cached buffer overlapping this range was used, or 0 when
	// the range is not cached at all. Lets a caller that must know the GPU has finished with a
	// range wait for that submission instead of for everything queued.
	[[nodiscard]] uint64_t LastUseTick(uint64_t vaddr, uint64_t size);
	// True when ObtainBuffer() can serve this range without newly registering guest memory for
	// write tracking: either a cached buffer already covers it, or it is small and host-resident
	// enough for the streaming shortcut. Callers that only read a few bytes use this to stay off
	// the write-fault path for memory the cache has no other reason to track.
	[[nodiscard]] bool IsRegionObtainableWithoutRegistering(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Appends the copies for guest memory the CPU reads every frame to the command buffer that is
	// about to be submitted, so the read waits on that submission instead of draining everything
	// recorded after it. Called by CommandScheduler just before it closes the buffer.
	void               RecordPendingReadbacks();
	// The stream ring's host writes of the recording submission, carried to device memory by a
	// command buffer submitted ahead of it. See StreamBuffer::RecordUploads().
	[[nodiscard]] bool HasPendingStreamUploads() const noexcept {
		return m_stream_buffer.HasPendingUploads();
	}
	void RecordStreamUploads(vk::CommandBuffer command) { m_stream_buffer.RecordUploads(command); }
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();

	// The guest ranges whose bytes have changed since the set was last drained. A draw whose
	// program reaches memory through the page table synchronises those ranges and nothing else.
	void MarkRangeDirty(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.MarkRangeDirty(vaddr, size);
	}
	[[nodiscard]] bool HasDirtyRanges() const noexcept { return m_memory_tracker.HasDirtyRanges(); }
	template <typename Func>
	void DrainDirtyRanges(Func&& func) {
		m_memory_tracker.DrainDirtyRanges(std::forward<Func>(func));
	}

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));

	// A guest range downloaded as one unit. Nearby reads share a window so they share a drain.
	struct ReadbackWindow {
		uint64_t begin = 0;
		uint64_t end   = 0;

		[[nodiscard]] uint64_t Size() const { return end - begin; }
	};
	static constexpr uint64_t READBACK_WINDOW_SIZE   = 512 * 1024;
	static constexpr uint64_t READBACK_WINDOW_BUDGET = 4 * 1024 * 1024;
	// A download is recorded per submission for every hot window, so the pending set grows
	// whenever the GPU falls behind the command processor. Nothing needs it settled promptly;
	// this bound only keeps the bookkeeping, and the linear scans over it, from growing.
	static constexpr size_t   MAX_PENDING_DOWNLOADS  = 64;

	// A download recorded before the guest asked for it. Between recording and retirement the
	// bytes are in flight: the range has left m_gpu_modified_ranges, but the tracker still marks
	// its pages and the guest mapping still holds the stale values, so nothing may read it until
	// RetireCompletedDownloads() has published it. HasGpuDirtyBytes() reports such a range as
	// dirty and DownloadReadbackWindows() leaves it alone, which keeps the cache's "tracker pages
	// and dirty bytes agree" invariant intact for everything that is not in flight.
	struct PendingDownload {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};

	[[nodiscard]] ReadbackWindow ReadbackWindowFor(const Buffer& buffer, uint64_t vaddr,
	                                              uint64_t size) const;
	void RememberReadbackWindow(ReadbackWindow window);
	void DownloadReadbackWindows(ReadbackWindow current, std::vector<ReadbackWindow>& downloaded);
	[[nodiscard]] bool HasPendingDownload(uint64_t vaddr, uint64_t size) const;
	// Waits for the submission that already carries this range, which is normally long finished,
	// and publishes every download that has completed since.
	void ResolvePendingDownloads(uint64_t vaddr, uint64_t size);
	void RetireCompletedDownloads();
	// Settles the oldest downloads when too many are outstanding. Never waits for the newest
	// submission: that is the one the command processor has just recorded.
	void BoundPendingDownloads();
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	// See BeginBufferScope(). A scope only survives while the command buffer it was opened on is
	// still recording, because a stream-buffer offset from an earlier one must never be reused.
	struct BufferScopeEntry {
		uint64_t vaddr   = 0;
		uint64_t size    = 0;
		uint64_t offset  = 0;
		BufferId id;
		bool     stream  = false;
		bool     written = false;
		bool     texel   = false;
	};

	static constexpr size_t MaxBufferScopeEntries = 16;

	[[nodiscard]] const BufferScopeEntry* FindInScope(uint64_t vaddr, uint64_t size,
	                                                  bool is_written, bool is_texel_buffer);
	void RememberInScope(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer,
	                     BufferId id, uint64_t offset, bool stream);

	std::array<BufferScopeEntry, MaxBufferScopeEntries> m_buffer_scope {};
	CleanRangeMemo                                      m_clean_ranges;
	size_t                                            m_buffer_scope_count = 0;
	uint64_t                                          m_buffer_scope_tick  = 0;

	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Windows the guest has read back at least once. Draining the GPU, not copying, is what a
	// readback costs, so every drain also flushes these: the next read of a known-hot window
	// then finds clean memory and needs no drain of its own.
	RangeSet                                          m_readback_windows;
	uint64_t                                          m_readback_window_bytes = 0;
	std::vector<PendingDownload>                      m_pending_downloads;
	// Recording a readback can wrap the download buffer, which submits; one level is enough.
	bool                                              m_recording_readbacks = false;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
