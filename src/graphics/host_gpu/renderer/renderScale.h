#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERSCALE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERSCALE_H_

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Libs::Graphics::RenderScale {

// Cached once; Config is immutable after startup.
[[nodiscard]] inline float Factor() {
	static const float factor = Config::GetRenderScale();
	return factor;
}

[[nodiscard]] inline bool Enabled() { return Factor() != 1.0F; }

// Scales one dimension, never to zero. Rounds to nearest.
[[nodiscard]] inline uint32_t Apply(uint32_t value) {
	if (!Enabled() || value == 0) {
		return value;
	}
	const auto scaled = static_cast<uint32_t>(
	    std::lround(static_cast<double>(value) * static_cast<double>(Factor())));
	return std::max(1U, scaled);
}

// Inverse of Apply for mapping host coordinates back to guest space.
[[nodiscard]] inline uint32_t Unapply(uint32_t value) {
	if (!Enabled() || value == 0) {
		return value;
	}
	const auto scaled = static_cast<uint32_t>(
	    std::lround(static_cast<double>(value) / static_cast<double>(Factor())));
	return std::max(1U, scaled);
}

[[nodiscard]] inline float ApplyF(float value) {
	return Enabled() ? value * Factor() : value;
}

[[nodiscard]] inline vk::Extent2D Apply(vk::Extent2D e) {
	return {Apply(e.width), Apply(e.height)};
}

// Depth (3D slice count / array layers) is never scaled.
[[nodiscard]] inline vk::Extent3D Apply(vk::Extent3D e) {
	return {Apply(e.width), Apply(e.height), e.depth};
}

} // namespace Libs::Graphics::RenderScale

#endif
