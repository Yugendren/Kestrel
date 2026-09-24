#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct StreamBufferTestAccess;
struct GraphicContext;

enum class MemoryUsage : uint8_t {
	DeviceLocal,
	Upload,
	Download,
	Stream,
};

inline constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

inline constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

class Buffer {
public:
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size)
	    : Buffer(graphics, scheduler, usage, usage, cpu_address, flags, size) {}
	~Buffer();
	KYTY_CLASS_NO_COPY(Buffer);

	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] std::span<uint8_t> Mapped() const noexcept { return m_mapped; }
	[[nodiscard]] bool               IsCoherent() const noexcept { return m_coherent; }
	[[nodiscard]] MemoryUsage        Usage() const noexcept { return m_usage; }
	[[nodiscard]] uint64_t           CpuAddress() const noexcept { return m_cpu_address; }
	[[nodiscard]] vk::DeviceAddress BufferDeviceAddress() const noexcept;
	[[nodiscard]] uint64_t           Offset(uint64_t address) const noexcept {
		return address - m_cpu_address;
	}
	[[nodiscard]] bool IsInBounds(uint64_t address, uint64_t size) const noexcept;
	void               IncreaseStreamScore(int score) noexcept { stream_score += score; }
	[[nodiscard]] int  StreamScore() const noexcept { return stream_score; }
	void               Flush(uint64_t offset, uint64_t size);
	void               Invalidate(uint64_t offset, uint64_t size);
	void CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
	              uint64_t destination_offset, uint64_t size,
	              vk::AccessFlags source_before      = vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_before = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags source_after       = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_after  = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite);
	void Fill(uint64_t offset, uint64_t size, uint32_t value);

	// BufferCache state lives directly on the resource.
	bool   is_deleted   = false;
	int    stream_score = 0;
	// Scheduler tick at which this buffer was last handed to a command. Work that references it
	// was recorded at or before that tick, so waiting for it is enough to know the buffer is no
	// longer in use -- far less than waiting for everything the scheduler has queued.
	uint64_t last_use_tick = 0;
	size_t lru_id       = 0;

protected:
	// `placement` chooses the memory the buffer lives in; `usage` is what it is used for.
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       MemoryUsage placement, uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size);

	[[nodiscard]] GraphicContext&   Graphics() const noexcept { return *m_graphics; }
	[[nodiscard]] CommandScheduler& Scheduler() const noexcept { return *m_scheduler; }

private:
	[[nodiscard]] vk::BufferMemoryBarrier Barrier(uint64_t offset, uint64_t size,
	                                              vk::AccessFlags source,
	                                              vk::AccessFlags destination) const;

	GraphicContext*               m_graphics    = nullptr;
	CommandScheduler*             m_scheduler   = nullptr;
	MemoryUsage                   m_usage       = MemoryUsage::DeviceLocal;
	uint64_t                      m_cpu_address = 0;
	vk::DeviceAddress             m_device_address = 0;
	vk::Buffer                    m_buffer     = nullptr;
	VmaAllocation                 m_allocation = nullptr;
	uint64_t                      m_size;
	bool                          m_coherent = false;
	std::span<uint8_t>            m_mapped;
};

// How far StreamBuffer::Map() may go to free ring space that earlier work still reads.
enum class RingWait : uint8_t {
	// Fail instead of waiting.
	Never,
	// Wait for submissions the GPU already has, but fail rather than submit the command buffer
	// that is still recording. Submitting runs the scheduler's end-of-submission hooks, which
	// callers holding memory-tracker state (an upload callback, for one) must not re-enter.
	Submitted,
	// Wait for anything, submitting the recording command buffer if the range is its own.
	Any,
};

class StreamBuffer final: public Buffer {
public:
	StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	             uint64_t size);

	[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0,
	                                                RingWait wait = RingWait::Any);
	void                                        Commit();
	[[nodiscard]] uint64_t Copy(const void* source, uint64_t size, uint64_t alignment = 0);
	// Reserves a range without producing a CPU pointer, for a usage a shader writes into directly
	// (MemoryUsage::DeviceLocal has no host mapping to copy through). Shares Map()'s ring offset and
	// in-flight watch bookkeeping, so a later reservation still waits on whatever last read this
	// range instead of racing it.
	[[nodiscard]] uint64_t Reserve(uint64_t size, uint64_t alignment = 0);

	// A download that landed either in the shared ring, or -- when it does not fit the ring at
	// all -- in a dedicated buffer sized just for it. `overflow` is null in the former case; in
	// the latter it owns the buffer `destination` points at, so the caller must move it into
	// whatever deferred operation reads `mapped`/`destination` afterwards. That operation running
	// is what releases it: nothing else keeps it alive.
	struct DownloadAllocation {
		uint8_t*                 mapped      = nullptr;
		uint64_t                 offset      = 0;
		Buffer*                  destination = nullptr;
		std::unique_ptr<Buffer>  overflow;
	};

	// Usage() must be MemoryUsage::Download. Behaves like Map() for a download that fits the
	// ring. A download too large for the ring itself (an oversize render target readback, for
	// example) would never fit no matter how the ring rotates, so this allocates a one-shot
	// buffer for it instead of failing the whole download.
	[[nodiscard]] DownloadAllocation AcquireDownload(uint64_t size, uint64_t alignment = 0);

	// A MemoryUsage::Stream ring lives in device memory, where the GPU reads it at full speed, and
	// the CPU writes a host-memory twin at the same offsets instead (writes into mapped device
	// memory cross the bus uncached, and on some hosts crawl). The bytes the recording submission
	// wrote reach device memory through RecordUploads(), which the scheduler records into a
	// command buffer it submits ahead of the recording one.
	[[nodiscard]] bool HasPendingUploads() const noexcept { return !m_pending_uploads.empty(); }
	void               RecordUploads(vk::CommandBuffer command);

private:
	friend struct StreamBufferTestAccess;

	struct Watch {
		uint64_t tick        = 0;
		uint64_t upper_bound = 0;
	};

	[[nodiscard]] static bool NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
	                                               uint64_t& alignment);
	// Whether a range still read by work recorded at `tick` may be waited for under `wait`, given
	// the tick the scheduler is currently recording.
	[[nodiscard]] static bool MayWaitFor(RingWait wait, uint64_t tick,
	                                     uint64_t current_tick) noexcept {
		return wait == RingWait::Any || (wait == RingWait::Submitted && tick < current_tick);
	}
	[[nodiscard]] bool        WaitPendingOperations(const std::vector<Watch>& watches,
	                                                std::optional<size_t>     invalidation_mark,
	                                                uint64_t requested_upper_bound, RingWait wait,
	                                                size_t& wait_cursor, uint64_t& wait_bound);

	[[nodiscard]] std::span<uint8_t> HostView() const noexcept {
		return m_host_twin != nullptr ? m_host_twin->Mapped() : Mapped();
	}
	[[nodiscard]] bool HostCoherent() const noexcept {
		return m_host_twin != nullptr ? m_host_twin->IsCoherent() : IsCoherent();
	}
	void CommitRange(bool upload);
	// Adds a ring range written on the host to the copies RecordUploads() will record.
	static void AppendUpload(std::vector<vk::BufferCopy>& uploads, uint64_t offset,
	                         uint64_t size) {
		// Allocations are packed in ring order, so a range usually continues the previous one up
		// to its alignment padding; copying the padding too keeps the writes one region.
		constexpr uint64_t MaxMergedGap = 4096;
		if (!uploads.empty()) {
			auto&      last     = uploads.back();
			const auto last_end = last.srcOffset + last.size;
			if (offset >= last_end && offset - last_end <= MaxMergedGap) {
				last.size = offset + size - last.srcOffset;
				return;
			}
		}
		uploads.push_back({offset, offset, size});
	}

	std::unique_ptr<Buffer>     m_host_twin;
	// Ring ranges written by the CPU since the last RecordUploads(), in ring order.
	std::vector<vk::BufferCopy> m_pending_uploads;
	uint64_t              m_offset      = 0;
	uint64_t              m_mapped_size = 0;
	std::vector<Watch>    m_current_watches;
	size_t                m_current_watch_cursor = 0;
	std::optional<size_t> m_invalidation_mark;
	std::vector<Watch>    m_previous_watches;
	size_t                m_wait_cursor = 0;
	uint64_t              m_wait_bound  = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
