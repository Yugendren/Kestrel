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

// True when any fidelity setting can move a host allocation off its guest footprint. While
// this is false the renderer skips all scale bookkeeping. Cached once; Config is immutable
// after startup.
[[nodiscard]] inline bool Enabled() {
	static const bool enabled =
	    Factor() != 1.0F || Config::GetPostScale() != 1.0F || Config::GetShadowMax() != 0;
	return enabled;
}

// Scales one dimension by factor, never to zero. Rounds to nearest.
[[nodiscard]] inline uint32_t Apply(uint32_t value, float factor) {
	if (factor == 1.0F || value == 0) {
		return value;
	}
	const auto scaled = static_cast<uint32_t>(
	    std::lround(static_cast<double>(value) * static_cast<double>(factor)));
	return std::max(1U, scaled);
}

// Inverse of Apply for mapping host coordinates back to guest space.
[[nodiscard]] inline uint32_t Unapply(uint32_t value, float factor) {
	if (factor == 1.0F || value == 0) {
		return value;
	}
	const auto scaled = static_cast<uint32_t>(
	    std::lround(static_cast<double>(value) / static_cast<double>(factor)));
	return std::max(1U, scaled);
}

[[nodiscard]] inline float ApplyF(float value, float factor) {
	return factor == 1.0F ? value : value * factor;
}

[[nodiscard]] inline vk::Extent2D Apply(vk::Extent2D e, float factor) {
	return {Apply(e.width, factor), Apply(e.height, factor)};
}

// Depth (3D slice count / array layers) is never scaled.
[[nodiscard]] inline vk::Extent3D Apply(vk::Extent3D e, float factor) {
	return {Apply(e.width, factor), Apply(e.height, factor), e.depth};
}

// Maps guest-resolution geometry onto the attachments of one render pass. Every attachment of
// a pass shares a single factor, coming from the pass's attachments rather than from a global,
// so a pass has exactly one mapping. Call sites use this instead of multiplying by the factor
// themselves.
class Mapping {
public:
	explicit Mapping(float factor) noexcept : m_factor(factor) {}

	[[nodiscard]] bool IsIdentity() const noexcept { return m_factor == 1.0F; }

	[[nodiscard]] vk::Extent2D Extent(vk::Extent2D extent) const noexcept {
		return IsIdentity() ? extent : Apply(extent, m_factor);
	}

	// Stretches a guest window transform onto the host attachment. The depth range describes
	// clip space, not the framebuffer, so it is never touched.
	void Viewport(vk::Viewport& viewport) const noexcept {
		if (IsIdentity()) {
			return;
		}
		viewport.x      = viewport.x * m_factor;
		viewport.y      = viewport.y * m_factor;
		viewport.width  = viewport.width * m_factor;
		viewport.height = viewport.height * m_factor;
	}

	// Maps a guest-space rectangle onto the host attachment. Edges round outward so a rectangle
	// covering the whole guest target still covers the whole host target, then clamp to it.
	[[nodiscard]] vk::Rect2D Rect(int32_t left, int32_t top, int32_t right, int32_t bottom,
	                              vk::Extent2D guest_extent) const noexcept {
		if (!IsIdentity()) {
			const auto factor = static_cast<double>(m_factor);
			const auto host   = Apply(guest_extent, m_factor);
			left   = static_cast<int32_t>(std::floor(left * factor));
			top    = static_cast<int32_t>(std::floor(top * factor));
			right  = std::min(static_cast<int32_t>(std::ceil(right * factor)),
			                  static_cast<int32_t>(host.width));
			bottom = std::min(static_cast<int32_t>(std::ceil(bottom * factor)),
			                  static_cast<int32_t>(host.height));
			left   = std::min(left, right);
			top    = std::min(top, bottom);
		}
		return {{left, top},
		        {static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)}};
	}

private:
	float m_factor;
};

} // namespace Libs::Graphics::RenderScale

#endif
