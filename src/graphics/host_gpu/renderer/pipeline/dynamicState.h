#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DYNAMICSTATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DYNAMICSTATE_H_

#include "common/assert.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <bitset>

namespace Libs::Graphics {

// Remembers, per command buffer, the last value SetGraphicsDynamicParams() (renderDraw.cpp)
// recorded for each piece of per-draw Vulkan dynamic state, and skips the vkCmdSet* call when the
// incoming value already matches it. Every graphics pipeline PipelineCache creates (see
// shaders.cpp) declares the states below dynamic unconditionally, apart from the two described
// next, so a cached value stays meaningful across our own vkCmdBindPipeline calls. The cache is
// reset whenever a command buffer begins, so every state is still set at least once per buffer.
//
// Binding a pipeline that does not declare a state dynamic replaces it with the pipeline's static
// value, so a state shaders.cpp declares only conditionally needs care. ePrimitiveTopology and
// ePrimitiveRestartEnable (absent from mesh pipelines) are not cached; renderDraw.cpp sets them on
// every non-mesh draw. eColorWriteEnableEXT is declared only by pipelines with colour attachments,
// so a draw without any calls ForgetColorWriteEnable() for the pipeline it binds.
// eAttachmentFeedbackLoopEnableEXT is declared by every draw pipeline whenever the device feature
// is on, which is also the only time it is set.
class DynamicStateCache {
public:
	// Mirrors HW::ScreenViewport::viewports, which this header deliberately does not include so it
	// stays light. renderDraw.cpp static_asserts its own slot count against this, so the two
	// cannot drift apart without the build failing.
	static constexpr uint32_t MaxViewports = 16;
	// Mirrors RENDER_COLOR_ATTACHMENTS_MAX (renderTarget.h), checked the same way.
	static constexpr uint32_t MaxColorAttachments = 8;

	// Bound to a fresh vk::CommandBuffer recording (CommandBuffer::Begin()) or to any point where
	// something outside SetGraphicsDynamicParams() touched dynamic state on the buffer (blitHelper
	// binding its own graphics pipelines) -- nothing is known about the buffer's state anymore.
	void Reset() noexcept { m_known.reset(); }

	void SetViewportWithCount(vk::CommandBuffer buffer, uint32_t count,
	                          const vk::Viewport* viewports) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::ViewportWithCount);
		if (m_known.test(idx) && m_viewport_count == count &&
		    std::equal(viewports, viewports + count, m_viewports.begin())) {
			return;
		}
		buffer.setViewportWithCount(count, viewports);
		std::copy(viewports, viewports + count, m_viewports.begin());
		m_viewport_count = count;
		m_known.set(idx);
	}

	void SetScissorWithCount(vk::CommandBuffer buffer, uint32_t count,
	                         const vk::Rect2D* scissors) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::ScissorWithCount);
		if (m_known.test(idx) && m_scissor_count == count &&
		    std::equal(scissors, scissors + count, m_scissors.begin())) {
			return;
		}
		buffer.setScissorWithCount(count, scissors);
		std::copy(scissors, scissors + count, m_scissors.begin());
		m_scissor_count = count;
		m_known.set(idx);
	}

	void SetLineWidth(vk::CommandBuffer buffer, float line_width) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::LineWidth);
		if (m_known.test(idx) && m_line_width == line_width) {
			return;
		}
		buffer.setLineWidth(line_width);
		m_line_width = line_width;
		m_known.set(idx);
	}

	void SetBlendConstants(vk::CommandBuffer buffer, const float blend_constants[4]) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::BlendConstants);
		const std::array<float, 4> value {blend_constants[0], blend_constants[1],
		                                  blend_constants[2], blend_constants[3]};
		if (m_known.test(idx) && m_blend_constants == value) {
			return;
		}
		buffer.setBlendConstants(blend_constants);
		m_blend_constants = value;
		m_known.set(idx);
	}

	void SetDepthTestEnable(vk::CommandBuffer buffer, vk::Bool32 enable) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthTestEnable);
		if (m_known.test(idx) && m_depth_test_enable == enable) {
			return;
		}
		buffer.setDepthTestEnable(enable);
		m_depth_test_enable = enable;
		m_known.set(idx);
	}

	void SetDepthWriteEnable(vk::CommandBuffer buffer, vk::Bool32 enable) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthWriteEnable);
		if (m_known.test(idx) && m_depth_write_enable == enable) {
			return;
		}
		buffer.setDepthWriteEnable(enable);
		m_depth_write_enable = enable;
		m_known.set(idx);
	}

	void SetDepthCompareOp(vk::CommandBuffer buffer, vk::CompareOp compare_op) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthCompareOp);
		if (m_known.test(idx) && m_depth_compare_op == compare_op) {
			return;
		}
		buffer.setDepthCompareOp(compare_op);
		m_depth_compare_op = compare_op;
		m_known.set(idx);
	}

	void SetDepthBoundsTestEnable(vk::CommandBuffer buffer, vk::Bool32 enable) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthBoundsTestEnable);
		if (m_known.test(idx) && m_depth_bounds_test_enable == enable) {
			return;
		}
		buffer.setDepthBoundsTestEnable(enable);
		m_depth_bounds_test_enable = enable;
		m_known.set(idx);
	}

	void SetDepthBounds(vk::CommandBuffer buffer, float min_depth_bounds,
	                    float max_depth_bounds) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthBounds);
		const std::array<float, 2> value {min_depth_bounds, max_depth_bounds};
		if (m_known.test(idx) && m_depth_bounds == value) {
			return;
		}
		buffer.setDepthBounds(min_depth_bounds, max_depth_bounds);
		m_depth_bounds = value;
		m_known.set(idx);
	}

	void SetStencilTestEnable(vk::CommandBuffer buffer, vk::Bool32 enable) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::StencilTestEnable);
		if (m_known.test(idx) && m_stencil_test_enable == enable) {
			return;
		}
		buffer.setStencilTestEnable(enable);
		m_stencil_test_enable = enable;
		m_known.set(idx);
	}

	void SetStencilOp(vk::CommandBuffer buffer, vk::StencilFaceFlagBits face,
	                  vk::StencilOp fail_op, vk::StencilOp pass_op, vk::StencilOp depth_fail_op,
	                  vk::CompareOp compare_op) noexcept {
		const auto      face_index = FaceIndex(face);
		const auto      idx        = static_cast<size_t>(Slot::StencilOpFront) + face_index;
		const StencilOp value {fail_op, pass_op, depth_fail_op, compare_op};
		if (m_known.test(idx) && m_stencil_op[face_index] == value) {
			return;
		}
		buffer.setStencilOp(face, fail_op, pass_op, depth_fail_op, compare_op);
		m_stencil_op[face_index] = value;
		m_known.set(idx);
	}

	void SetCullMode(vk::CommandBuffer buffer, vk::CullModeFlags cull_mode) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::CullMode);
		if (m_known.test(idx) && m_cull_mode == cull_mode) {
			return;
		}
		buffer.setCullMode(cull_mode);
		m_cull_mode = cull_mode;
		m_known.set(idx);
	}

	void SetFrontFace(vk::CommandBuffer buffer, vk::FrontFace front_face) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::FrontFace);
		if (m_known.test(idx) && m_front_face == front_face) {
			return;
		}
		buffer.setFrontFace(front_face);
		m_front_face = front_face;
		m_known.set(idx);
	}

	void SetDepthBiasEnable(vk::CommandBuffer buffer, vk::Bool32 enable) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthBiasEnable);
		if (m_known.test(idx) && m_depth_bias_enable == enable) {
			return;
		}
		buffer.setDepthBiasEnable(enable);
		m_depth_bias_enable = enable;
		m_known.set(idx);
	}

	void SetDepthBias(vk::CommandBuffer buffer, float constant_factor, float clamp,
	                  float slope_factor) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::DepthBias);
		const std::array<float, 3> value {constant_factor, clamp, slope_factor};
		if (m_known.test(idx) && m_depth_bias == value) {
			return;
		}
		buffer.setDepthBias(constant_factor, clamp, slope_factor);
		m_depth_bias = value;
		m_known.set(idx);
	}

	void SetStencilCompareMask(vk::CommandBuffer buffer, vk::StencilFaceFlagBits face,
	                           uint32_t compare_mask) noexcept {
		const auto face_index = FaceIndex(face);
		const auto idx        = static_cast<size_t>(Slot::StencilCompareMaskFront) + face_index;
		if (m_known.test(idx) && m_stencil_compare_mask[face_index] == compare_mask) {
			return;
		}
		buffer.setStencilCompareMask(face, compare_mask);
		m_stencil_compare_mask[face_index] = compare_mask;
		m_known.set(idx);
	}

	void SetStencilWriteMask(vk::CommandBuffer buffer, vk::StencilFaceFlagBits face,
	                         uint32_t write_mask) noexcept {
		const auto face_index = FaceIndex(face);
		const auto idx        = static_cast<size_t>(Slot::StencilWriteMaskFront) + face_index;
		if (m_known.test(idx) && m_stencil_write_mask[face_index] == write_mask) {
			return;
		}
		buffer.setStencilWriteMask(face, write_mask);
		m_stencil_write_mask[face_index] = write_mask;
		m_known.set(idx);
	}

	void SetStencilReference(vk::CommandBuffer buffer, vk::StencilFaceFlagBits face,
	                         uint32_t reference) noexcept {
		const auto face_index = FaceIndex(face);
		const auto idx        = static_cast<size_t>(Slot::StencilReferenceFront) + face_index;
		if (m_known.test(idx) && m_stencil_reference[face_index] == reference) {
			return;
		}
		buffer.setStencilReference(face, reference);
		m_stencil_reference[face_index] = reference;
		m_known.set(idx);
	}

	void SetColorWriteEnable(vk::CommandBuffer buffer, uint32_t count,
	                         const vk::Bool32* enables) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::ColorWriteEnable);
		if (m_known.test(idx) && m_color_write_count == count &&
		    std::equal(enables, enables + count, m_color_write_enable.begin())) {
			return;
		}
		buffer.setColorWriteEnableEXT(count, enables);
		std::copy(enables, enables + count, m_color_write_enable.begin());
		m_color_write_count = count;
		m_known.set(idx);
	}

	// The draw about to bind a pipeline without colour attachments, which does not declare
	// eColorWriteEnableEXT dynamic and so replaces whatever value was recorded.
	void ForgetColorWriteEnable() noexcept {
		m_known.reset(static_cast<size_t>(Slot::ColorWriteEnable));
	}

	void SetAttachmentFeedbackLoopEnable(vk::CommandBuffer    buffer,
	                                     vk::ImageAspectFlags aspects) noexcept {
		constexpr auto idx = static_cast<size_t>(Slot::AttachmentFeedbackLoopEnable);
		if (m_known.test(idx) && m_feedback_loop_aspects == aspects) {
			return;
		}
		buffer.setAttachmentFeedbackLoopEnableEXT(aspects);
		m_feedback_loop_aspects = aspects;
		m_known.set(idx);
	}

private:
	// Bit index of each cached state within m_known. StencilOpFront/StencilCompareMaskFront/
	// StencilWriteMaskFront/StencilReferenceFront must stay immediately followed by their Back
	// counterpart -- FaceIndex() below is added onto the Front slot to reach the Back one.
	enum class Slot : size_t {
		ViewportWithCount,
		ScissorWithCount,
		LineWidth,
		BlendConstants,
		DepthTestEnable,
		DepthWriteEnable,
		DepthCompareOp,
		DepthBoundsTestEnable,
		DepthBounds,
		StencilTestEnable,
		StencilOpFront,
		StencilOpBack,
		CullMode,
		FrontFace,
		DepthBiasEnable,
		DepthBias,
		StencilCompareMaskFront,
		StencilCompareMaskBack,
		StencilWriteMaskFront,
		StencilWriteMaskBack,
		StencilReferenceFront,
		StencilReferenceBack,
		ColorWriteEnable,
		AttachmentFeedbackLoopEnable,
		Count,
	};
	static_assert(static_cast<size_t>(Slot::Count) <= 32);

	// The two stencil faces are the only pieces of dynamic state SetGraphicsDynamicParams() (and
	// blitHelper.cpp) ever name individually; a combined eFrontAndBack mask is never passed to a
	// method of this class.
	static size_t FaceIndex(vk::StencilFaceFlagBits face) {
		// One slot per face. vk::StencilFaceFlagBits also spells eFrontAndBack, which would have
		// to update both slots; the draw path only ever names a single face, so reject the
		// combined mask rather than silently recording it against the back face alone.
		EXIT_IF(face != vk::StencilFaceFlagBits::eFront && face != vk::StencilFaceFlagBits::eBack);
		return face == vk::StencilFaceFlagBits::eFront ? 0 : 1;
	}

	struct StencilOp {
		vk::StencilOp fail_op;
		vk::StencilOp pass_op;
		vk::StencilOp depth_fail_op;
		vk::CompareOp compare_op;

		bool operator==(const StencilOp&) const noexcept = default;
	};

	std::bitset<32> m_known;

	std::array<vk::Viewport, MaxViewports> m_viewports {};
	uint32_t                               m_viewport_count = 0;
	std::array<vk::Rect2D, MaxViewports>   m_scissors {};
	uint32_t                               m_scissor_count = 0;
	float                                  m_line_width = 0.0f;
	std::array<float, 4>                   m_blend_constants {};
	vk::Bool32                             m_depth_test_enable = VK_FALSE;
	vk::Bool32                             m_depth_write_enable = VK_FALSE;
	vk::CompareOp                          m_depth_compare_op = vk::CompareOp::eNever;
	vk::Bool32                             m_depth_bounds_test_enable = VK_FALSE;
	std::array<float, 2>                   m_depth_bounds {};
	vk::Bool32                             m_stencil_test_enable = VK_FALSE;
	std::array<StencilOp, 2>               m_stencil_op {};
	vk::CullModeFlags                      m_cull_mode {};
	vk::FrontFace                          m_front_face = vk::FrontFace::eCounterClockwise;
	vk::Bool32                             m_depth_bias_enable = VK_FALSE;
	std::array<float, 3>                   m_depth_bias {};
	std::array<uint32_t, 2>                m_stencil_compare_mask {};
	std::array<uint32_t, 2>                m_stencil_write_mask {};
	std::array<uint32_t, 2>                m_stencil_reference {};
	std::array<vk::Bool32, MaxColorAttachments> m_color_write_enable {};
	uint32_t                               m_color_write_count = 0;
	vk::ImageAspectFlags                   m_feedback_loop_aspects {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DYNAMICSTATE_H_
