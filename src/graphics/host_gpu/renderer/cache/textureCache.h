#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <map>
#include <set>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };
	// Coordinate space an absolute texel address computed by a shader lives in. A rasterising
	// stage derives it from its own fragment position, so it is expressed in the resolution of
	// the render targets of the draw and moves with internal resolution scaling. A compute
	// dispatch is launched with guest-sized workgroup counts, so it stays in guest texels. A
	// dispatch of a tile-rescaled program (renderer/computeRescale.h) shifts its texel addresses
	// at runtime: they are guest texels >> s, s = log2 of 1/render scale, which is the space of
	// an image scaled at exactly the render scale.
	enum class TexelSpace : uint8_t { Guest, RenderTarget, ScaledGuest };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		// The shader reaches this image without a sampler, so it addresses it in absolute guest
		// texels. True for every storage binding and for sampled bindings that are texel-fetched
		// or queried for their extent.
		bool          texel_addressed = false;
		// Which space those texel addresses are measured in; only meaningful when the binding
		// is texel-addressed. Filled in from the shader stage that owns the binding.
		TexelSpace    texel_space     = TexelSpace::Guest;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	// Replays the per-lookup side effects FindImage() has on `id` beyond returning it -- the LRU
	// touch and the access tick -- without walking the page table again. For a caller that has
	// memoized a previous FindImage() result against TextureCache::Generation() and observed no
	// change, FindImage() would resolve to this exact id again; this is the cheap remainder of
	// that lookup instead of repeating it.
	void                        NoteImageReuse(ImageId id);
	// NoteImageReuse() for every id in `ids` under one acquisition of the cache lock.
	void                        NoteImageReuse(std::span<const ImageId> ids);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	// One render or depth target of a draw, as FindRenderTarget()/FindDepthTarget() receive it.
	struct TargetAcquisition {
		ImageId          id;
		const ImageDesc* desc = nullptr;
	};
	// Whether FindRenderTarget()/FindDepthTarget() for every target in `targets`, each with the
	// same desc as the previous call for that target, would find every step already done, apart
	// from the LRU touch (NoteImageReuse()). Takes the cache lock once for all of them.
	[[nodiscard]] bool TargetAcquisitionsRepeat(std::span<const TargetAcquisition> targets);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	// Permanently drops internal resolution scaling for the guest range of an image and returns
	// the native replacement (the old id is freed). Used when a render pass would otherwise mix
	// scaled and native attachments, which a single Vulkan render area cannot express.
	[[nodiscard]] ImageId DenyImageScale(ImageId id);

	// A private native-resolution image with the shape of `shape` and no guest range: no lookup,
	// invalidation or garbage collection ever reaches it, exactly like a scale twin. The
	// tile-rescale dual-run check (renderer/computeRescale.h) replays a dispatch into one.
	[[nodiscard]] ImageId AcquireScratchImage(const ImageInfo& shape);
	// Frees an image from AcquireScratchImage() once the work already recorded against it has
	// completed.
	void                  ReleaseScratchImage(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	// Advances whenever an image may have become GPU-modified, so a range IsRegionGpuModified()
	// cleared stays clear while this is unchanged.
	[[nodiscard]] uint64_t GpuModifiedAdditions() const noexcept {
		return m_gpu_modified_page_cover.Additions();
	}

	[[nodiscard]] bool IsMeta(uint64_t address);
	// `fill_value` receives the dword the last recognised metadata fill wrote, when known.
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice,
	                                 uint32_t* fill_value = nullptr, bool* fill_known = nullptr);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool ClearMeta(uint64_t address, uint32_t fill_value);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	void RunGarbageCollector();

	// Bumped whenever an image is created, destroyed, (un)registered or flagged for rebind, i.e.
	// whenever a descriptor that resolved to some image on an earlier draw could now resolve
	// somewhere else. Also bumped when a state FindImage()/ResolveScaleBinding() consult besides
	// the image set itself changes in a way that can change a lookup's answer without touching
	// any image: m_scale_denied gaining a new address (it feeds the scale FindImage() assigns a
	// range) and m_display_extent growing (it anchors ResolveImageScale()'s fidelity classes). A
	// caller that cached a resolution only has to compare this.
	[[nodiscard]] uint64_t Generation() const { return m_generation; }

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
		// The dword a recognised uniform fill last wrote (e.g. the HTILE clear value).
		uint32_t fill_value = 0xffffffffu;
		bool     fill_known = false;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 40, 10>;
	// Image cover counts use the owner index grid: a zero count stands for "the owner index
	// holds nothing on this page".
	using ImagePageCounters =
	    PageCounters<ImagePageTable::kPageBits, ImagePageTable::kAddressSpaceBits>;
	void ConfigureGarbageCollectionBudget(uint64_t available_budget);

	// Callers have validated the nonempty 40-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	// Coarse, lock-free "is any image registered here" filter, one counter per image page-table
	// page. A guest write fault on memory that holds no image can then skip the texture-cache
	// lock instead of contending with the command-processor thread, which holds that lock for
	// the duration of a draw. The counters use the same page granularity as the owner index, so
	// a zero counter means FindImagesInRegion() would have returned nothing.
	[[nodiscard]] bool MayCoverImages(uint64_t address, uint64_t size) const noexcept;
	void               UpdateImageCover(uint64_t address, uint64_t size, bool add);

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	// Host allocation factor for one image, taken from its fidelity class (see the policy in
	// textureCache.cpp). Must be called with m_lock held: it reads the learned display extent.
	[[nodiscard]] float ResolveImageScale(const ImageInfo& info, BindingType binding) const;
	[[nodiscard]] bool               SafeToDownload(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	// Internal resolution scaling: see ResolveScaleBinding for how the two halves of a scaled
	// range divide the work between rasterisation and compute.
	[[nodiscard]] bool        ReportImageScale(const ImageInfo& info, BindingType binding);
	[[nodiscard]] bool        CanTwinScale(const ImageInfo& info) const;
	[[nodiscard]] ImageId     ResolveScaleBinding(const ImageDesc& desc, ImageId id);
	void                      PrepareScaledStorageBinding(ImageId id);
	[[nodiscard]] ImageId     AcquireScaleTwin(ImageId owner_id);
	void                      PrepareScaleTwinBinding(ImageId owner_id);
	void                      PrepareScaleTwinRead(ImageId owner_id);
	void                      RefreshTwinContents(ImageId owner_id);
	void                      SyncScaledContents(ImageId owner_id);
	void                      FreeScaleTwin(Image& owner);
	void                        RefreshImage(ImageId id);
	void                        MaterializeDccClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	// Caller holds m_lock. See TargetAcquisitionsRepeat().
	[[nodiscard]] bool    TargetAcquisitionRepeats(ImageId id, const ImageDesc& desc);
	[[nodiscard]] bool    IsSampledViewCurrent(const Image& image, const ImageDesc& desc) const;
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	ImagePageCounters                                 m_image_page_cover;
	// Pages holding at least one registered GPU-modified image (see IsRegionGpuModified).
	GpuModifiedPageCounters                           m_gpu_modified_page_cover;
	std::unordered_map<vk::Format, ImageId>           m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	// Guest addresses whose images must stay at native resolution: a render pass mixed scaled
	// and native attachments, or compute bound a range that cannot carry a native twin.
	std::unordered_set<uint64_t>                      m_scale_denied;
	// One scaling report per guest range and binding class, so the log describes the policy
	// and not the traffic.
	std::set<std::pair<uint64_t, BindingType>>        m_scale_logged;
	// Guest video-out extent, learned from the scanout surface and only ever grown. The
	// fidelity classes are defined relative to it; until it is known every target is primary.
	vk::Extent2D                                      m_display_extent {};
	using SurfaceMetas = std::map<uint64_t, MetaDataInfo>;
	// Every removal from m_surface_metas goes through EraseSurfaceMeta(), which counts it in
	// m_surface_meta_removals: entries are otherwise only ever added, so an address seen present
	// stays present while that count is unchanged (see SurfaceMetaPresent()).
	SurfaceMetas::iterator                            EraseSurfaceMeta(SurfaceMetas::iterator entry);
	// m_surface_metas.contains(address), remembering the last address found present. Caller
	// holds m_lock.
	[[nodiscard]] bool                                SurfaceMetaPresent(uint64_t address);
	SurfaceMetas                                      m_surface_metas;
	uint64_t                                          m_surface_meta_removals = 0;
	GenerationMemo<uint64_t, bool>                    m_surface_meta_present_memo;
	// Which image AssociateStencil() would pick for a depth target's stencil range, as last found
	// by TargetAcquisitionRepeats() (a null id when none matches). The pick is a function of the
	// registered image set over that range and the depth image's extent, which only change with
	// Generation(), so a repeated draw's target check skips the page-table walk while it holds.
	struct StencilLookupKey {
		ImageId    depth;
		GuestRange stencil;
		bool       operator==(const StencilLookupKey&) const = default;
	};
	GenerationMemo<StencilLookupKey, ImageId>         m_stencil_lookup_memo;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;
	// See Generation(). Starts at 1 so a default-constructed cached value of 0 always compares
	// as stale.
	uint64_t         m_generation             = 1;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
