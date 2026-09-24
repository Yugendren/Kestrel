#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGET_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

static constexpr uint32_t RENDER_COLOR_ATTACHMENTS_MAX = 8;

// A single-entry memo keyed on (Key, cache generation). Render-target discovery re-derives the
// same TextureCache lookup from the same guest registers on almost every draw; a generation that
// hasn't moved since the last resolution is a cheap, always-correct proof that the cache would
// answer identically again, so a caller can skip straight to the stored Value instead of redoing
// the work that produced it. One entry is enough for every caller here: each memoizes a single
// render-target slot (or, for depth, the single depth-stencil attachment) between consecutive
// draws, not a history of slots.
//
// Header-only and Vulkan-free by design, so it can be reused anywhere a "does this still match
// what I last computed" cache of exactly one entry is useful.
template <typename Key, typename Value>
class GenerationMemo {
public:
	// Returns the memoized value when both `key` and `generation` match what Store() last saw;
	// nullptr otherwise. A generation mismatch alone invalidates the entry even when `key` is
	// identical: the whole point of the generation is that the same key can now resolve to a
	// different value (e.g. TextureCache::FindImage() returning a different image id), so it must
	// be checked before the key is trusted at all.
	[[nodiscard]] const Value* Find(const Key& key, uint64_t generation) const {
		if (!m_valid || m_generation != generation || !(m_key == key)) {
			return nullptr;
		}
		return &m_value;
	}

	// Overwrites whatever entry was stored before, if any.
	void Store(const Key& key, uint64_t generation, const Value& value) {
		m_key        = key;
		m_generation = generation;
		m_value      = value;
		m_valid      = true;
	}

	// Forces the next Find() to miss regardless of key or generation.
	void Invalidate() { m_valid = false; }

private:
	Key      m_key {};
	uint64_t m_generation = 0;
	Value    m_value {};
	bool     m_valid = false;
};

struct RenderAttachment {
	vk::ImageView           image_view    = nullptr;
	vk::ImageLayout         image_layout  = vk::ImageLayout::eUndefined;
	std::array<uint32_t, 4> clear_value   = {};
	bool                    is_clear      = false;
	bool                    has_depth     = false;
	bool                    depth_clear   = false;
	bool                    has_stencil   = false;
	bool                    stencil_clear = false;

	bool operator==(const RenderAttachment&) const = default;
};

struct RenderState {
	std::array<RenderAttachment, RENDER_COLOR_ATTACHMENTS_MAX> color_attachments;
	RenderAttachment                                           depth_stencil_attachment;
	uint32_t                                                   width                 = 0;
	uint32_t                                                   height                = 0;
	// Guest-space size of the render area and the render scale factor its attachments share;
	// guest scissor rectangles are clamped against this size and then mapped onto the host
	// attachments.
	uint32_t                                                   guest_width           = 0;
	uint32_t                                                   guest_height          = 0;
	float                                                      scale                 = 1.0F;
	uint32_t                                                   num_layers            = 1;
	uint32_t                                                   num_color_attachments = 0;

	bool operator==(const RenderState&) const = default;
};

[[nodiscard]] inline constexpr uint32_t render_sample_count(uint32_t encoded_samples) {
	return encoded_samples <= 3 ? 1u << encoded_samples : 0;
}

[[nodiscard]] inline constexpr vk::SampleCountFlagBits vulkan_sample_count(uint32_t samples) {
	switch (samples) {
		case 1: return vk::SampleCountFlagBits::e1;
		case 2: return vk::SampleCountFlagBits::e2;
		case 4: return vk::SampleCountFlagBits::e4;
		case 8: return vk::SampleCountFlagBits::e8;
		default: return {};
	}
}

enum class TargetViewType : uint8_t { Image2D, Image2DArray, Unsupported };

struct TargetViewInfo {
	TargetViewType type         = TargetViewType::Unsupported;
	uint32_t       base_layer   = 0;
	uint32_t       layer_count  = 0;
	uint32_t       image_layers = 0;
};

inline constexpr TargetViewInfo ResolveTargetViewInfo(uint32_t base_layer, uint32_t last_layer,
                                                      uint32_t draw_layer_offset = 0) {
	if (base_layer > last_layer || draw_layer_offset != 0) {
		return {};
	}
	return {base_layer == last_layer ? TargetViewType::Image2D : TargetViewType::Image2DArray,
	        base_layer, last_layer - base_layer + 1u, last_layer + 1u};
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGET_H_
