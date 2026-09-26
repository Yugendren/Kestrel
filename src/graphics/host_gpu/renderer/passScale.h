#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSCALE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSCALE_H_

#include <cstdint>
#include <optional>
#include <span>

namespace Libs::Graphics::PassScale {

// One colour attachment of a draw as the pass-scale decision sees it: the resolution factor its
// image was allocated at, and whether the draw can write it at all.
struct ColorAttachment {
	float scale   = 1.0F;
	bool  written = false;
};

// Outcome for one draw. The colour attachments kept in the pass are always a prefix of the ones
// discovery produced, because an attachment's index is its colour output location: dropping one
// from the middle would move every later attachment onto the wrong fragment output.
struct Decision {
	uint32_t color_count = 0;     // attachments that stay in the pass
	bool     uniform     = true;  // false: what the draw writes itself mixes scales
};

// A Vulkan render pass has one render area and the draw one viewport transform, so everything the
// pass holds has to share one scale. Only the attachments the draw writes -- and the depth
// target, which it tests against -- decide that scale. A colour slot the draw cannot write is
// usually a stale binding left over from an earlier pass (Astro's Playroom renders a 256x256
// reflection-cube face with the four full-screen G-buffer targets still enabled in
// CB_TARGET_MASK, but the pixel shader exports MRT0 only); letting such a slot vote would turn a
// harmless leftover into a scale conflict. Unwritten trailing attachments at another scale are
// dropped from the pass instead: nothing reaches them, so the pass loses nothing.
//
// The result is not uniform only when attachments the draw really writes (or the depth target)
// disagree, or an unwritten attachment at another scale sits in front of a kept one.
[[nodiscard]] inline Decision Decide(std::span<const ColorAttachment> colors,
                                     std::optional<float>              depth_scale) {
	Decision decision {static_cast<uint32_t>(colors.size()), true};

	std::optional<float> pass_scale = depth_scale;
	for (const auto& color: colors) {
		if (!color.written) {
			continue;
		}
		if (!pass_scale) {
			pass_scale = color.scale;
		} else if (*pass_scale != color.scale) {
			decision.uniform = false;
			return decision;
		}
	}
	// Nothing written and no depth: the first attachment anchors the pass, as it anchors the
	// viewport and scissor.
	if (!pass_scale) {
		if (colors.empty()) {
			return decision;
		}
		pass_scale = colors.front().scale;
	}

	uint32_t kept = decision.color_count;
	while (kept > 0 && !colors[kept - 1].written && colors[kept - 1].scale != *pass_scale) {
		kept--;
	}
	for (uint32_t i = 0; i < kept; i++) {
		if (colors[i].scale != *pass_scale) {
			decision.uniform = false;
			return decision;
		}
	}
	decision.color_count = kept;
	return decision;
}

} // namespace Libs::Graphics::PassScale

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSCALE_H_ */
