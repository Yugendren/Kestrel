#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

struct RenderColorInfo {
	// Discovery keeps guest image information but can remap the view into a larger cache image.
	TextureCache::ImageDesc         desc;
	ImageId                         image_id;
	uint32_t                        target_slot      = 0;
	uint32_t                        guest_mip_level   = 0;
	uint32_t                        guest_array_layer = 0;
	Prospero::ColorComponentMapping export_mapping;

	// Guest-visible attachment size, as the game programmed it.
	[[nodiscard]] vk::Extent2D GuestExtent() const {
		return {std::max(desc.info.extent.width >> guest_mip_level, 1u),
		        std::max(desc.info.extent.height >> guest_mip_level, 1u)};
	}
	// Host attachment size: the guest size times the internal render scale.
	[[nodiscard]] vk::Extent2D Extent() const { return desc.info.HostExtent2D(guest_mip_level); }
	[[nodiscard]] bool IsScaled() const { return desc.info.IsScaled(); }
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
