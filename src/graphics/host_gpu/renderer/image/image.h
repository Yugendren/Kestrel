#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <compare>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class BlitHelper;
class Buffer;
class CommandScheduler;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
	vk::ImageAspectFlags pixel_sampled_aspects;
	vk::ImageAspectFlags other_sampled_aspects;
};

class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	// The transfer that resamples this image is not usable for depth/stencil surfaces, so the
	// cache hands every image the helper that redraws them instead. Set once, at creation.
	void SetResampler(BlitHelper* resampler) noexcept { m_resampler = resampler; }

	// Resamples a compatible image across a resolution-scale difference. The subresource
	// ranges select matching source and destination levels/layers; the whole-image overload
	// covers every level and layer the two images have in common.
	void BlitScaled(Image& source, const ImageSubresourceRange& source_range,
	                const ImageSubresourceRange& destination_range);
	void BlitScaled(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
		}
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept { m_buffer_modified = true; }
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool IsStencilModified() const noexcept { return m_stencil_modified; }
	void               MarkStencilModified() noexcept { m_stencil_modified = true; }
	void               ClearStencilModified() noexcept { m_stencil_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;
	[[nodiscard]] bool     IsScaled() const noexcept { return info.IsScaled(); }
	[[nodiscard]] float    ScaleFactor() const noexcept { return info.ScaleFactor(); }

	// Internal resolution scaling pairs a scaled image with one native-resolution companion, the
	// scale twin. Compute shaders address storage images in absolute guest texels, so a storage
	// binding of this range is served by the twin while rasterisation keeps the scaled image.
	// The twin doubles as the staging alias for guest-memory transfers, so a range never carries
	// two native allocations.
	void                  AdoptScaleTwin(Image& twin, ImageId twin_id);
	void                  DropScaleTwin() noexcept;
	[[nodiscard]] ImageId ScaleTwinId() const noexcept { return m_scale_twin_id; }
	// Set on the twin itself; empty on every other image.
	[[nodiscard]] ImageId ScaleTwinOwner() const noexcept { return m_scale_twin_owner; }
	void SetScaleTwinOwner(ImageId owner) noexcept { m_scale_twin_owner = owner; }
	// Which half of the pair the other one is missing contents from. Only the two halves can be
	// out of date with respect to each other; guest memory keeps its own dirty flags.
	[[nodiscard]] bool IsTwinNewest() const noexcept { return m_twin_newest; }
	[[nodiscard]] bool IsTwinStale() const noexcept { return m_twin_stale; }
	// The scaled half is about to be written, so the twin stops matching it.
	void MarkScaledNewest() noexcept {
		m_twin_newest = false;
		m_twin_stale  = true;
	}
	// A guest-texel binding is about to write the twin, so the scaled half stops matching it.
	void MarkTwinNewest() noexcept {
		m_twin_newest = true;
		m_twin_stale  = false;
	}
	// A resample copied one half onto the other; they now hold the same contents.
	void MarkTwinSynced() noexcept {
		m_twin_newest = false;
		m_twin_stale  = false;
	}

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);
	// Guest-resolution companion allocation used to stage guest-memory transfers of a
	// resolution-scaled image. Created on first use; most scaled targets never need one.
	[[nodiscard]] Image& GuestAlias();

	std::unique_ptr<Image> m_guest_alias;
	BlitHelper*            m_resampler = nullptr;
	// Borrowed from the texture cache, which owns the twin's slot and frees it with this image.
	Image*            m_scale_twin = nullptr;
	ImageId           m_scale_twin_id {};
	ImageId           m_scale_twin_owner {};
	bool              m_twin_newest = false;
	bool              m_twin_stale  = false;
	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash   = 0;
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_stencil_modified  = false;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
