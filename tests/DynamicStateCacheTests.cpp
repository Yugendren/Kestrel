#include "graphics/host_gpu/renderer/pipeline/dynamicState.h"

#include <cstdio>
#include <cstdlib>

// dynamicState.h calls through vk::CommandBuffer's dynamic-state setters, which resolve their
// vkCmdSet* entry point via VULKAN_HPP_DEFAULT_DISPATCHER (see vulkanCommon.h). This test defines
// the dispatcher's storage itself, rather than linking vulkanCommon.cpp, and points every entry
// point this header touches at a fake that counts calls and records arguments, so the cache's
// skip/emit decisions can be checked without a real Vulkan device or command buffer.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using Libs::Graphics::DynamicStateCache;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "DynamicStateCacheTests: failed: %s\n", text);
		std::abort();
	}
}

// Number of times each fake vkCmdSet* entry point was invoked, and the arguments of the most
// recent call, so a test case can assert both "did it call through" and "what did it pass".
struct Recorder {
	int      viewport_with_count = 0;
	uint32_t last_viewport_count = 0;
	vk::Viewport last_viewports[DynamicStateCache::MaxViewports] {};

	int      scissor_with_count = 0;
	uint32_t last_scissor_count = 0;
	vk::Rect2D last_scissors[DynamicStateCache::MaxViewports] {};

	int   line_width = 0;
	float last_line_width = 0.0f;

	int   blend_constants = 0;
	float last_blend_constants[4] {};

	int      depth_test_enable = 0;
	VkBool32 last_depth_test_enable = VK_FALSE;

	int      depth_write_enable = 0;
	VkBool32 last_depth_write_enable = VK_FALSE;

	int         depth_compare_op = 0;
	VkCompareOp last_depth_compare_op = VK_COMPARE_OP_NEVER;

	int      depth_bounds_test_enable = 0;
	VkBool32 last_depth_bounds_test_enable = VK_FALSE;

	int   depth_bounds = 0;
	float last_min_depth_bounds = 0.0f;
	float last_max_depth_bounds = 0.0f;

	int      stencil_test_enable = 0;
	VkBool32 last_stencil_test_enable = VK_FALSE;

	int                 stencil_op = 0;
	VkStencilFaceFlags  last_stencil_op_face = 0;
	VkStencilOp         last_stencil_fail_op = VK_STENCIL_OP_KEEP;
	VkStencilOp         last_stencil_pass_op = VK_STENCIL_OP_KEEP;
	VkStencilOp         last_stencil_depth_fail_op = VK_STENCIL_OP_KEEP;
	VkCompareOp         last_stencil_compare_op = VK_COMPARE_OP_NEVER;

	int             cull_mode = 0;
	VkCullModeFlags last_cull_mode = 0;

	int         front_face = 0;
	VkFrontFace last_front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	int      depth_bias_enable = 0;
	VkBool32 last_depth_bias_enable = VK_FALSE;

	int   depth_bias = 0;
	float last_depth_bias_constant = 0.0f;
	float last_depth_bias_clamp = 0.0f;
	float last_depth_bias_slope = 0.0f;

	int                stencil_compare_mask = 0;
	VkStencilFaceFlags last_stencil_compare_mask_face = 0;
	uint32_t           last_stencil_compare_mask = 0;

	int                stencil_write_mask = 0;
	VkStencilFaceFlags last_stencil_write_mask_face = 0;
	uint32_t           last_stencil_write_mask = 0;

	int                stencil_reference = 0;
	VkStencilFaceFlags last_stencil_reference_face = 0;
	uint32_t           last_stencil_reference = 0;

	int      color_write_enable = 0;
	uint32_t last_color_write_count = 0;
	VkBool32 last_color_write_enable[DynamicStateCache::MaxColorAttachments] {};

	int                feedback_loop_enable = 0;
	VkImageAspectFlags last_feedback_loop_aspects = 0;
};

Recorder g_rec;

void ResetRecorder() { g_rec = Recorder {}; }

VkCommandBuffer FakeHandle() {
	// Never dereferenced by anything in dynamicState.h or by the fakes below -- only used as a
	// distinguishable non-null handle to pass through vk::CommandBuffer.
	static int dummy_command_buffer = 0;
	return reinterpret_cast<VkCommandBuffer>(&dummy_command_buffer);
}

VKAPI_ATTR void VKAPI_CALL FakeSetViewportWithCount(VkCommandBuffer /*commandBuffer*/, uint32_t viewportCount,
                                                    const VkViewport* pViewports) {
	++g_rec.viewport_with_count;
	g_rec.last_viewport_count = viewportCount;
	for (uint32_t i = 0; i < viewportCount && i < DynamicStateCache::MaxViewports; ++i) {
		g_rec.last_viewports[i] = pViewports[i];
	}
}

VKAPI_ATTR void VKAPI_CALL FakeSetScissorWithCount(VkCommandBuffer /*commandBuffer*/, uint32_t scissorCount,
                                                   const VkRect2D* pScissors) {
	++g_rec.scissor_with_count;
	g_rec.last_scissor_count = scissorCount;
	for (uint32_t i = 0; i < scissorCount && i < DynamicStateCache::MaxViewports; ++i) {
		g_rec.last_scissors[i] = pScissors[i];
	}
}

VKAPI_ATTR void VKAPI_CALL FakeSetLineWidth(VkCommandBuffer /*commandBuffer*/, float lineWidth) {
	++g_rec.line_width;
	g_rec.last_line_width = lineWidth;
}

VKAPI_ATTR void VKAPI_CALL FakeSetBlendConstants(VkCommandBuffer /*commandBuffer*/, const float blendConstants[4]) {
	++g_rec.blend_constants;
	for (int i = 0; i < 4; ++i) {
		g_rec.last_blend_constants[i] = blendConstants[i];
	}
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthTestEnable(VkCommandBuffer /*commandBuffer*/, VkBool32 depthTestEnable) {
	++g_rec.depth_test_enable;
	g_rec.last_depth_test_enable = depthTestEnable;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthWriteEnable(VkCommandBuffer /*commandBuffer*/, VkBool32 depthWriteEnable) {
	++g_rec.depth_write_enable;
	g_rec.last_depth_write_enable = depthWriteEnable;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthCompareOp(VkCommandBuffer /*commandBuffer*/, VkCompareOp depthCompareOp) {
	++g_rec.depth_compare_op;
	g_rec.last_depth_compare_op = depthCompareOp;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthBoundsTestEnable(VkCommandBuffer /*commandBuffer*/,
                                                        VkBool32 depthBoundsTestEnable) {
	++g_rec.depth_bounds_test_enable;
	g_rec.last_depth_bounds_test_enable = depthBoundsTestEnable;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthBounds(VkCommandBuffer /*commandBuffer*/, float minDepthBounds,
                                              float maxDepthBounds) {
	++g_rec.depth_bounds;
	g_rec.last_min_depth_bounds = minDepthBounds;
	g_rec.last_max_depth_bounds = maxDepthBounds;
}

VKAPI_ATTR void VKAPI_CALL FakeSetStencilTestEnable(VkCommandBuffer /*commandBuffer*/, VkBool32 stencilTestEnable) {
	++g_rec.stencil_test_enable;
	g_rec.last_stencil_test_enable = stencilTestEnable;
}

VKAPI_ATTR void VKAPI_CALL FakeSetStencilOp(VkCommandBuffer /*commandBuffer*/, VkStencilFaceFlags faceMask,
                                            VkStencilOp failOp, VkStencilOp passOp, VkStencilOp depthFailOp,
                                            VkCompareOp compareOp) {
	++g_rec.stencil_op;
	g_rec.last_stencil_op_face = faceMask;
	g_rec.last_stencil_fail_op = failOp;
	g_rec.last_stencil_pass_op = passOp;
	g_rec.last_stencil_depth_fail_op = depthFailOp;
	g_rec.last_stencil_compare_op = compareOp;
}

VKAPI_ATTR void VKAPI_CALL FakeSetCullMode(VkCommandBuffer /*commandBuffer*/, VkCullModeFlags cullMode) {
	++g_rec.cull_mode;
	g_rec.last_cull_mode = cullMode;
}

VKAPI_ATTR void VKAPI_CALL FakeSetFrontFace(VkCommandBuffer /*commandBuffer*/, VkFrontFace frontFace) {
	++g_rec.front_face;
	g_rec.last_front_face = frontFace;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthBiasEnable(VkCommandBuffer /*commandBuffer*/, VkBool32 depthBiasEnable) {
	++g_rec.depth_bias_enable;
	g_rec.last_depth_bias_enable = depthBiasEnable;
}

VKAPI_ATTR void VKAPI_CALL FakeSetDepthBias(VkCommandBuffer /*commandBuffer*/, float depthBiasConstantFactor,
                                            float depthBiasClamp, float depthBiasSlopeFactor) {
	++g_rec.depth_bias;
	g_rec.last_depth_bias_constant = depthBiasConstantFactor;
	g_rec.last_depth_bias_clamp = depthBiasClamp;
	g_rec.last_depth_bias_slope = depthBiasSlopeFactor;
}

VKAPI_ATTR void VKAPI_CALL FakeSetStencilCompareMask(VkCommandBuffer /*commandBuffer*/, VkStencilFaceFlags faceMask,
                                                     uint32_t compareMask) {
	++g_rec.stencil_compare_mask;
	g_rec.last_stencil_compare_mask_face = faceMask;
	g_rec.last_stencil_compare_mask = compareMask;
}

VKAPI_ATTR void VKAPI_CALL FakeSetStencilWriteMask(VkCommandBuffer /*commandBuffer*/, VkStencilFaceFlags faceMask,
                                                   uint32_t writeMask) {
	++g_rec.stencil_write_mask;
	g_rec.last_stencil_write_mask_face = faceMask;
	g_rec.last_stencil_write_mask = writeMask;
}

VKAPI_ATTR void VKAPI_CALL FakeSetStencilReference(VkCommandBuffer /*commandBuffer*/, VkStencilFaceFlags faceMask,
                                                   uint32_t reference) {
	++g_rec.stencil_reference;
	g_rec.last_stencil_reference_face = faceMask;
	g_rec.last_stencil_reference = reference;
}

VKAPI_ATTR void VKAPI_CALL FakeSetColorWriteEnable(VkCommandBuffer /*commandBuffer*/, uint32_t attachmentCount,
                                                   const VkBool32* pColorWriteEnables) {
	++g_rec.color_write_enable;
	g_rec.last_color_write_count = attachmentCount;
	for (uint32_t i = 0; i < attachmentCount && i < DynamicStateCache::MaxColorAttachments; ++i) {
		g_rec.last_color_write_enable[i] = pColorWriteEnables[i];
	}
}

VKAPI_ATTR void VKAPI_CALL FakeSetAttachmentFeedbackLoopEnable(VkCommandBuffer /*commandBuffer*/,
                                                               VkImageAspectFlags aspectMask) {
	++g_rec.feedback_loop_enable;
	g_rec.last_feedback_loop_aspects = aspectMask;
}

void InstallFakeDispatcher() {
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	d.vkCmdSetViewportWithCount = FakeSetViewportWithCount;
	d.vkCmdSetScissorWithCount = FakeSetScissorWithCount;
	d.vkCmdSetLineWidth = FakeSetLineWidth;
	d.vkCmdSetBlendConstants = FakeSetBlendConstants;
	d.vkCmdSetDepthTestEnable = FakeSetDepthTestEnable;
	d.vkCmdSetDepthWriteEnable = FakeSetDepthWriteEnable;
	d.vkCmdSetDepthCompareOp = FakeSetDepthCompareOp;
	d.vkCmdSetDepthBoundsTestEnable = FakeSetDepthBoundsTestEnable;
	d.vkCmdSetDepthBounds = FakeSetDepthBounds;
	d.vkCmdSetStencilTestEnable = FakeSetStencilTestEnable;
	d.vkCmdSetStencilOp = FakeSetStencilOp;
	d.vkCmdSetCullMode = FakeSetCullMode;
	d.vkCmdSetFrontFace = FakeSetFrontFace;
	d.vkCmdSetDepthBiasEnable = FakeSetDepthBiasEnable;
	d.vkCmdSetDepthBias = FakeSetDepthBias;
	d.vkCmdSetStencilCompareMask = FakeSetStencilCompareMask;
	d.vkCmdSetStencilWriteMask = FakeSetStencilWriteMask;
	d.vkCmdSetStencilReference = FakeSetStencilReference;
	d.vkCmdSetColorWriteEnableEXT = FakeSetColorWriteEnable;
	d.vkCmdSetAttachmentFeedbackLoopEnableEXT = FakeSetAttachmentFeedbackLoopEnable;
}

vk::Viewport MakeViewport(float x, float width = 100.0f) {
	vk::Viewport viewport {};
	viewport.x = x;
	viewport.y = 0.0f;
	viewport.width = width;
	viewport.height = 100.0f;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	return viewport;
}

vk::Rect2D MakeScissor(int32_t x, uint32_t width = 100) {
	vk::Rect2D scissor {};
	scissor.offset.x = x;
	scissor.offset.y = 0;
	scissor.extent.width = width;
	scissor.extent.height = 100;
	return scissor;
}

void TestLineWidth() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetLineWidth(buffer, 2.0f);
	Check(g_rec.line_width == 1 && g_rec.last_line_width == 2.0f,
	      "first SetLineWidth emits exactly one vkCmdSetLineWidth call");

	cache.SetLineWidth(buffer, 2.0f);
	Check(g_rec.line_width == 1, "identical SetLineWidth repeat is skipped");

	cache.SetLineWidth(buffer, 3.0f);
	Check(g_rec.line_width == 2 && g_rec.last_line_width == 3.0f,
	      "a changed SetLineWidth value emits and becomes current");

	cache.SetLineWidth(buffer, 3.0f);
	Check(g_rec.line_width == 2, "repeat of the new SetLineWidth value is skipped");

	cache.Reset();
	cache.SetLineWidth(buffer, 3.0f);
	Check(g_rec.line_width == 3, "Reset() makes the next identical SetLineWidth emit again");
}

void TestBlendConstantsOneChannelChange() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	float first[4] = {0.1f, 0.2f, 0.3f, 0.4f};
	cache.SetBlendConstants(buffer, first);
	Check(g_rec.blend_constants == 1 && g_rec.last_blend_constants[0] == 0.1f &&
	          g_rec.last_blend_constants[3] == 0.4f,
	      "first SetBlendConstants emits exactly one vkCmdSetBlendConstants call with all four channels");

	cache.SetBlendConstants(buffer, first);
	Check(g_rec.blend_constants == 1, "identical SetBlendConstants repeat is skipped");

	float one_channel_changed[4] = {0.1f, 0.2f, 0.3f, 0.9f};
	cache.SetBlendConstants(buffer, one_channel_changed);
	Check(g_rec.blend_constants == 2 && g_rec.last_blend_constants[3] == 0.9f,
	      "a change in a single blend-constant channel emits");

	cache.SetBlendConstants(buffer, one_channel_changed);
	Check(g_rec.blend_constants == 2, "repeat of the new blend constants is skipped");

	cache.Reset();
	cache.SetBlendConstants(buffer, one_channel_changed);
	Check(g_rec.blend_constants == 3, "Reset() makes the next identical SetBlendConstants emit again");
}

void TestDepthTestEnableToggle() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_test_enable == 1 && g_rec.last_depth_test_enable == VK_TRUE,
	      "first SetDepthTestEnable emits exactly one vkCmdSetDepthTestEnable call");

	cache.SetDepthTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_test_enable == 1, "identical SetDepthTestEnable repeat is skipped");

	cache.SetDepthTestEnable(buffer, VK_FALSE);
	Check(g_rec.depth_test_enable == 2 && g_rec.last_depth_test_enable == VK_FALSE,
	      "toggling SetDepthTestEnable off emits");

	cache.SetDepthTestEnable(buffer, VK_FALSE);
	Check(g_rec.depth_test_enable == 2, "repeat of the toggled-off value is skipped");

	cache.SetDepthTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_test_enable == 3, "toggling SetDepthTestEnable back on emits");

	cache.Reset();
	cache.SetDepthTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_test_enable == 4, "Reset() makes the next identical SetDepthTestEnable emit again");
}

void TestDepthWriteEnableToggle() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthWriteEnable(buffer, VK_TRUE);
	Check(g_rec.depth_write_enable == 1 && g_rec.last_depth_write_enable == VK_TRUE,
	      "first SetDepthWriteEnable emits exactly one vkCmdSetDepthWriteEnable call");

	cache.SetDepthWriteEnable(buffer, VK_TRUE);
	Check(g_rec.depth_write_enable == 1, "identical SetDepthWriteEnable repeat is skipped");

	cache.SetDepthWriteEnable(buffer, VK_FALSE);
	Check(g_rec.depth_write_enable == 2 && g_rec.last_depth_write_enable == VK_FALSE,
	      "toggling SetDepthWriteEnable off emits");

	cache.SetDepthWriteEnable(buffer, VK_FALSE);
	Check(g_rec.depth_write_enable == 2, "repeat of the toggled-off value is skipped");

	cache.Reset();
	cache.SetDepthWriteEnable(buffer, VK_FALSE);
	Check(g_rec.depth_write_enable == 3, "Reset() makes the next identical SetDepthWriteEnable emit again");
}

void TestDepthCompareOp() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthCompareOp(buffer, vk::CompareOp::eLess);
	Check(g_rec.depth_compare_op == 1 && g_rec.last_depth_compare_op == VK_COMPARE_OP_LESS,
	      "first SetDepthCompareOp emits exactly one vkCmdSetDepthCompareOp call");

	cache.SetDepthCompareOp(buffer, vk::CompareOp::eLess);
	Check(g_rec.depth_compare_op == 1, "identical SetDepthCompareOp repeat is skipped");

	cache.SetDepthCompareOp(buffer, vk::CompareOp::eGreaterOrEqual);
	Check(g_rec.depth_compare_op == 2 && g_rec.last_depth_compare_op == VK_COMPARE_OP_GREATER_OR_EQUAL,
	      "a changed SetDepthCompareOp value emits and becomes current");

	cache.SetDepthCompareOp(buffer, vk::CompareOp::eGreaterOrEqual);
	Check(g_rec.depth_compare_op == 2, "repeat of the new SetDepthCompareOp value is skipped");

	cache.Reset();
	cache.SetDepthCompareOp(buffer, vk::CompareOp::eGreaterOrEqual);
	Check(g_rec.depth_compare_op == 3, "Reset() makes the next identical SetDepthCompareOp emit again");
}

void TestDepthBoundsTestEnableToggle() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthBoundsTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_bounds_test_enable == 1 && g_rec.last_depth_bounds_test_enable == VK_TRUE,
	      "first SetDepthBoundsTestEnable emits exactly one vkCmdSetDepthBoundsTestEnable call");

	cache.SetDepthBoundsTestEnable(buffer, VK_TRUE);
	Check(g_rec.depth_bounds_test_enable == 1, "identical SetDepthBoundsTestEnable repeat is skipped");

	cache.SetDepthBoundsTestEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bounds_test_enable == 2 && g_rec.last_depth_bounds_test_enable == VK_FALSE,
	      "toggling SetDepthBoundsTestEnable off emits");

	cache.SetDepthBoundsTestEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bounds_test_enable == 2, "repeat of the toggled-off value is skipped");

	cache.Reset();
	cache.SetDepthBoundsTestEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bounds_test_enable == 3,
	      "Reset() makes the next identical SetDepthBoundsTestEnable emit again");
}

void TestDepthBounds() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthBounds(buffer, 0.0f, 1.0f);
	Check(g_rec.depth_bounds == 1 && g_rec.last_min_depth_bounds == 0.0f && g_rec.last_max_depth_bounds == 1.0f,
	      "first SetDepthBounds emits exactly one vkCmdSetDepthBounds call");

	cache.SetDepthBounds(buffer, 0.0f, 1.0f);
	Check(g_rec.depth_bounds == 1, "identical SetDepthBounds repeat is skipped");

	cache.SetDepthBounds(buffer, 0.0f, 0.5f); // only the max bound changes
	Check(g_rec.depth_bounds == 2 && g_rec.last_max_depth_bounds == 0.5f,
	      "a change in only the max depth bound emits");

	cache.SetDepthBounds(buffer, 0.0f, 0.5f);
	Check(g_rec.depth_bounds == 2, "repeat of the new depth bounds is skipped");

	cache.Reset();
	cache.SetDepthBounds(buffer, 0.0f, 0.5f);
	Check(g_rec.depth_bounds == 3, "Reset() makes the next identical SetDepthBounds emit again");
}

void TestStencilTestEnableToggle() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetStencilTestEnable(buffer, VK_TRUE);
	Check(g_rec.stencil_test_enable == 1 && g_rec.last_stencil_test_enable == VK_TRUE,
	      "first SetStencilTestEnable emits exactly one vkCmdSetStencilTestEnable call");

	cache.SetStencilTestEnable(buffer, VK_TRUE);
	Check(g_rec.stencil_test_enable == 1, "identical SetStencilTestEnable repeat is skipped");

	cache.SetStencilTestEnable(buffer, VK_FALSE);
	Check(g_rec.stencil_test_enable == 2 && g_rec.last_stencil_test_enable == VK_FALSE,
	      "toggling SetStencilTestEnable off emits");

	cache.SetStencilTestEnable(buffer, VK_FALSE);
	Check(g_rec.stencil_test_enable == 2, "repeat of the toggled-off value is skipped");

	cache.Reset();
	cache.SetStencilTestEnable(buffer, VK_FALSE);
	Check(g_rec.stencil_test_enable == 3, "Reset() makes the next identical SetStencilTestEnable emit again");
}

void TestStencilOpFrontBack() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eFront, vk::StencilOp::eKeep, vk::StencilOp::eReplace,
	                   vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	Check(g_rec.stencil_op == 1 && g_rec.last_stencil_op_face == VK_STENCIL_FACE_FRONT_BIT,
	      "first SetStencilOp(front) emits exactly one vkCmdSetStencilOp call");

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eFront, vk::StencilOp::eKeep, vk::StencilOp::eReplace,
	                   vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	Check(g_rec.stencil_op == 1, "identical SetStencilOp(front) repeat is skipped");

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eBack, vk::StencilOp::eZero,
	                   vk::StencilOp::eIncrementAndClamp, vk::StencilOp::eZero, vk::CompareOp::eNever);
	Check(g_rec.stencil_op == 2 && g_rec.last_stencil_op_face == VK_STENCIL_FACE_BACK_BIT,
	      "front and back stencil-op faces are cached independently: setting back emits despite front already "
	      "being current");

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eFront, vk::StencilOp::eKeep, vk::StencilOp::eReplace,
	                   vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	Check(g_rec.stencil_op == 2, "the front repeat stays skipped after an intervening back update");

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eBack, vk::StencilOp::eZero,
	                   vk::StencilOp::eIncrementAndClamp, vk::StencilOp::eZero, vk::CompareOp::eNever);
	Check(g_rec.stencil_op == 2, "repeat of the current back value is skipped");

	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eBack, vk::StencilOp::eZero,
	                   vk::StencilOp::eIncrementAndClamp, vk::StencilOp::eZero, vk::CompareOp::eAlways);
	Check(g_rec.stencil_op == 3 && g_rec.last_stencil_compare_op == VK_COMPARE_OP_ALWAYS,
	      "a change in a single stencil-op component (compare op) emits");

	cache.Reset();
	cache.SetStencilOp(buffer, vk::StencilFaceFlagBits::eFront, vk::StencilOp::eKeep, vk::StencilOp::eReplace,
	                   vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	Check(g_rec.stencil_op == 4, "Reset() makes the next identical SetStencilOp(front) emit again");
}

void TestCullMode() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetCullMode(buffer, vk::CullModeFlagBits::eBack);
	Check(g_rec.cull_mode == 1 && g_rec.last_cull_mode == VK_CULL_MODE_BACK_BIT,
	      "first SetCullMode emits exactly one vkCmdSetCullMode call");

	cache.SetCullMode(buffer, vk::CullModeFlagBits::eBack);
	Check(g_rec.cull_mode == 1, "identical SetCullMode repeat is skipped");

	cache.SetCullMode(buffer, vk::CullModeFlagBits::eFront);
	Check(g_rec.cull_mode == 2 && g_rec.last_cull_mode == VK_CULL_MODE_FRONT_BIT,
	      "a changed SetCullMode value emits and becomes current");

	cache.SetCullMode(buffer, vk::CullModeFlagBits::eFront);
	Check(g_rec.cull_mode == 2, "repeat of the new SetCullMode value is skipped");

	cache.Reset();
	cache.SetCullMode(buffer, vk::CullModeFlagBits::eFront);
	Check(g_rec.cull_mode == 3, "Reset() makes the next identical SetCullMode emit again");
}

void TestFrontFace() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetFrontFace(buffer, vk::FrontFace::eClockwise);
	Check(g_rec.front_face == 1 && g_rec.last_front_face == VK_FRONT_FACE_CLOCKWISE,
	      "first SetFrontFace emits exactly one vkCmdSetFrontFace call");

	cache.SetFrontFace(buffer, vk::FrontFace::eClockwise);
	Check(g_rec.front_face == 1, "identical SetFrontFace repeat is skipped");

	cache.SetFrontFace(buffer, vk::FrontFace::eCounterClockwise);
	Check(g_rec.front_face == 2 && g_rec.last_front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE,
	      "a changed SetFrontFace value emits and becomes current");

	cache.SetFrontFace(buffer, vk::FrontFace::eCounterClockwise);
	Check(g_rec.front_face == 2, "repeat of the new SetFrontFace value is skipped");

	cache.Reset();
	cache.SetFrontFace(buffer, vk::FrontFace::eCounterClockwise);
	Check(g_rec.front_face == 3, "Reset() makes the next identical SetFrontFace emit again");
}

void TestDepthBiasEnableToggle() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthBiasEnable(buffer, VK_TRUE);
	Check(g_rec.depth_bias_enable == 1 && g_rec.last_depth_bias_enable == VK_TRUE,
	      "first SetDepthBiasEnable emits exactly one vkCmdSetDepthBiasEnable call");

	cache.SetDepthBiasEnable(buffer, VK_TRUE);
	Check(g_rec.depth_bias_enable == 1, "identical SetDepthBiasEnable repeat is skipped");

	cache.SetDepthBiasEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bias_enable == 2 && g_rec.last_depth_bias_enable == VK_FALSE,
	      "toggling SetDepthBiasEnable off emits");

	cache.SetDepthBiasEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bias_enable == 2, "repeat of the toggled-off value is skipped");

	cache.Reset();
	cache.SetDepthBiasEnable(buffer, VK_FALSE);
	Check(g_rec.depth_bias_enable == 3, "Reset() makes the next identical SetDepthBiasEnable emit again");
}

void TestDepthBiasOneComponentChange() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetDepthBias(buffer, 1.0f, 2.0f, 3.0f);
	Check(g_rec.depth_bias == 1, "first SetDepthBias emits exactly one vkCmdSetDepthBias call");

	cache.SetDepthBias(buffer, 1.0f, 2.0f, 3.0f);
	Check(g_rec.depth_bias == 1, "identical SetDepthBias repeat is skipped");

	cache.SetDepthBias(buffer, 1.0f, 2.0f, 9.0f); // only the slope factor changes
	Check(g_rec.depth_bias == 2 && g_rec.last_depth_bias_slope == 9.0f,
	      "a change in only the slope factor emits");

	cache.SetDepthBias(buffer, 1.0f, 9.0f, 9.0f); // only the clamp changes
	Check(g_rec.depth_bias == 3 && g_rec.last_depth_bias_clamp == 9.0f,
	      "a change in only the clamp component emits");

	cache.SetDepthBias(buffer, 5.0f, 9.0f, 9.0f); // only the constant factor changes
	Check(g_rec.depth_bias == 4 && g_rec.last_depth_bias_constant == 5.0f,
	      "a change in only the constant factor emits");

	cache.SetDepthBias(buffer, 5.0f, 9.0f, 9.0f);
	Check(g_rec.depth_bias == 4, "repeat of the current depth bias is skipped");

	cache.Reset();
	cache.SetDepthBias(buffer, 5.0f, 9.0f, 9.0f);
	Check(g_rec.depth_bias == 5, "Reset() makes the next identical SetDepthBias emit again");
}

void TestStencilCompareMaskFrontBack() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetStencilCompareMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_compare_mask == 1 && g_rec.last_stencil_compare_mask_face == VK_STENCIL_FACE_FRONT_BIT &&
	          g_rec.last_stencil_compare_mask == 0x0f,
	      "first SetStencilCompareMask(front) emits exactly one vkCmdSetStencilCompareMask call");

	cache.SetStencilCompareMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_compare_mask == 1, "identical SetStencilCompareMask(front) repeat is skipped");

	cache.SetStencilCompareMask(buffer, vk::StencilFaceFlagBits::eBack, 0xf0);
	Check(g_rec.stencil_compare_mask == 2 && g_rec.last_stencil_compare_mask_face == VK_STENCIL_FACE_BACK_BIT &&
	          g_rec.last_stencil_compare_mask == 0xf0,
	      "front and back stencil compare masks are cached independently: setting back emits despite front "
	      "already being current");

	cache.SetStencilCompareMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_compare_mask == 2, "the front repeat stays skipped after an intervening back update");

	cache.Reset();
	cache.SetStencilCompareMask(buffer, vk::StencilFaceFlagBits::eBack, 0xf0);
	Check(g_rec.stencil_compare_mask == 3, "Reset() makes the next identical SetStencilCompareMask(back) emit again");
}

void TestStencilWriteMaskFrontBack() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetStencilWriteMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_write_mask == 1 && g_rec.last_stencil_write_mask_face == VK_STENCIL_FACE_FRONT_BIT &&
	          g_rec.last_stencil_write_mask == 0x0f,
	      "first SetStencilWriteMask(front) emits exactly one vkCmdSetStencilWriteMask call");

	cache.SetStencilWriteMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_write_mask == 1, "identical SetStencilWriteMask(front) repeat is skipped");

	cache.SetStencilWriteMask(buffer, vk::StencilFaceFlagBits::eBack, 0xf0);
	Check(g_rec.stencil_write_mask == 2 && g_rec.last_stencil_write_mask_face == VK_STENCIL_FACE_BACK_BIT &&
	          g_rec.last_stencil_write_mask == 0xf0,
	      "front and back stencil write masks are cached independently: setting back emits despite front already "
	      "being current");

	cache.SetStencilWriteMask(buffer, vk::StencilFaceFlagBits::eFront, 0x0f);
	Check(g_rec.stencil_write_mask == 2, "the front repeat stays skipped after an intervening back update");

	cache.Reset();
	cache.SetStencilWriteMask(buffer, vk::StencilFaceFlagBits::eBack, 0xf0);
	Check(g_rec.stencil_write_mask == 3, "Reset() makes the next identical SetStencilWriteMask(back) emit again");
}

void TestStencilReferenceFrontBackIndependence() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eFront, 0x10);
	Check(g_rec.stencil_reference == 1 && g_rec.last_stencil_reference_face == VK_STENCIL_FACE_FRONT_BIT &&
	          g_rec.last_stencil_reference == 0x10,
	      "first SetStencilReference(front) emits exactly one vkCmdSetStencilReference call");

	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eBack, 0x20);
	Check(g_rec.stencil_reference == 2 && g_rec.last_stencil_reference_face == VK_STENCIL_FACE_BACK_BIT &&
	          g_rec.last_stencil_reference == 0x20,
	      "first SetStencilReference(back) emits exactly one vkCmdSetStencilReference call");

	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eFront, 0x10);
	Check(g_rec.stencil_reference == 2, "repeat of the still-current front reference is skipped");

	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eBack, 0x99);
	Check(g_rec.stencil_reference == 3 && g_rec.last_stencil_reference_face == VK_STENCIL_FACE_BACK_BIT &&
	          g_rec.last_stencil_reference == 0x99,
	      "changing only the back reference emits once with the back face flag");

	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eFront, 0x10);
	Check(g_rec.stencil_reference == 3, "the front repeat stays skipped after the back-only change");

	cache.Reset();
	cache.SetStencilReference(buffer, vk::StencilFaceFlagBits::eFront, 0x10);
	Check(g_rec.stencil_reference == 4, "Reset() makes the next identical SetStencilReference emit again");
}

void TestViewportWithCount() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	vk::Viewport one[1] = {MakeViewport(1.0f)};
	cache.SetViewportWithCount(buffer, 1, one);
	Check(g_rec.viewport_with_count == 1 && g_rec.last_viewport_count == 1,
	      "first SetViewportWithCount emits exactly one vkCmdSetViewportWithCount call");

	cache.SetViewportWithCount(buffer, 1, one);
	Check(g_rec.viewport_with_count == 1, "identical SetViewportWithCount repeat is skipped");

	vk::Viewport two[2] = {MakeViewport(1.0f), MakeViewport(2.0f)};
	cache.SetViewportWithCount(buffer, 2, two);
	Check(g_rec.viewport_with_count == 2 && g_rec.last_viewport_count == 2,
	      "a count change emits even though the common prefix (viewport 0) matches");

	cache.SetViewportWithCount(buffer, 2, two);
	Check(g_rec.viewport_with_count == 2, "identical two-viewport repeat is skipped");

	vk::Viewport second_changed[2] = {MakeViewport(1.0f), MakeViewport(9.0f)};
	cache.SetViewportWithCount(buffer, 2, second_changed);
	Check(g_rec.viewport_with_count == 3 && g_rec.last_viewports[1].x == 9.0f,
	      "a change in only the second of two viewports emits");

	cache.SetViewportWithCount(buffer, 2, second_changed);
	Check(g_rec.viewport_with_count == 3, "repeat of the new two-viewport list is skipped");

	cache.Reset();
	cache.SetViewportWithCount(buffer, 2, second_changed);
	Check(g_rec.viewport_with_count == 4, "Reset() makes the next identical SetViewportWithCount emit again");
}

void TestScissorWithCount() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	vk::Rect2D one[1] = {MakeScissor(1)};
	cache.SetScissorWithCount(buffer, 1, one);
	Check(g_rec.scissor_with_count == 1 && g_rec.last_scissor_count == 1,
	      "first SetScissorWithCount emits exactly one vkCmdSetScissorWithCount call");

	cache.SetScissorWithCount(buffer, 1, one);
	Check(g_rec.scissor_with_count == 1, "identical SetScissorWithCount repeat is skipped");

	vk::Rect2D two[2] = {MakeScissor(1), MakeScissor(2)};
	cache.SetScissorWithCount(buffer, 2, two);
	Check(g_rec.scissor_with_count == 2 && g_rec.last_scissor_count == 2,
	      "a count change emits even though the common prefix (scissor 0) matches");

	cache.SetScissorWithCount(buffer, 2, two);
	Check(g_rec.scissor_with_count == 2, "identical two-scissor repeat is skipped");

	vk::Rect2D second_changed[2] = {MakeScissor(1), MakeScissor(9)};
	cache.SetScissorWithCount(buffer, 2, second_changed);
	Check(g_rec.scissor_with_count == 3 && g_rec.last_scissors[1].offset.x == 9,
	      "a change in only the second of two scissors emits");

	cache.SetScissorWithCount(buffer, 2, second_changed);
	Check(g_rec.scissor_with_count == 3, "repeat of the new two-scissor list is skipped");

	cache.Reset();
	cache.SetScissorWithCount(buffer, 2, second_changed);
	Check(g_rec.scissor_with_count == 4, "Reset() makes the next identical SetScissorWithCount emit again");
}

void TestColorWriteEnable() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	const vk::Bool32 two[2] = {VK_TRUE, VK_FALSE};
	cache.SetColorWriteEnable(buffer, 2, two);
	Check(g_rec.color_write_enable == 1 && g_rec.last_color_write_count == 2,
	      "first SetColorWriteEnable emits exactly one vkCmdSetColorWriteEnableEXT call");

	cache.SetColorWriteEnable(buffer, 2, two);
	Check(g_rec.color_write_enable == 1, "identical SetColorWriteEnable repeat is skipped");

	const vk::Bool32 flipped[2] = {VK_TRUE, VK_TRUE};
	cache.SetColorWriteEnable(buffer, 2, flipped);
	Check(g_rec.color_write_enable == 2 && g_rec.last_color_write_enable[1] == VK_TRUE,
	      "a change in one attachment emits");

	cache.SetColorWriteEnable(buffer, 1, flipped);
	Check(g_rec.color_write_enable == 3 && g_rec.last_color_write_count == 1,
	      "an attachment count change emits even though the common prefix matches");

	// A pipeline without colour attachments replaces the recorded value when it is bound.
	cache.ForgetColorWriteEnable();
	cache.SetColorWriteEnable(buffer, 1, flipped);
	Check(g_rec.color_write_enable == 4,
	      "ForgetColorWriteEnable() makes the next identical SetColorWriteEnable emit again");

	cache.SetLineWidth(buffer, 1.0f);
	cache.ForgetColorWriteEnable();
	cache.SetLineWidth(buffer, 1.0f);
	Check(g_rec.line_width == 1, "ForgetColorWriteEnable() leaves every other state known");

	cache.Reset();
	cache.SetColorWriteEnable(buffer, 1, flipped);
	Check(g_rec.color_write_enable == 5, "Reset() makes the next identical SetColorWriteEnable emit again");
}

void TestAttachmentFeedbackLoopEnable() {
	ResetRecorder();
	DynamicStateCache cache;
	vk::CommandBuffer buffer(FakeHandle());

	cache.SetAttachmentFeedbackLoopEnable(buffer, {});
	Check(g_rec.feedback_loop_enable == 1,
	      "first SetAttachmentFeedbackLoopEnable emits even for no aspects");

	cache.SetAttachmentFeedbackLoopEnable(buffer, {});
	Check(g_rec.feedback_loop_enable == 1, "identical SetAttachmentFeedbackLoopEnable repeat is skipped");

	cache.SetAttachmentFeedbackLoopEnable(buffer, vk::ImageAspectFlagBits::eDepth);
	Check(g_rec.feedback_loop_enable == 2 &&
	          g_rec.last_feedback_loop_aspects == VK_IMAGE_ASPECT_DEPTH_BIT,
	      "a changed aspect mask emits and becomes current");

	cache.Reset();
	cache.SetAttachmentFeedbackLoopEnable(buffer, vk::ImageAspectFlagBits::eDepth);
	Check(g_rec.feedback_loop_enable == 3,
	      "Reset() makes the next identical SetAttachmentFeedbackLoopEnable emit again");
}

} // namespace

int main() {
	InstallFakeDispatcher();

	TestLineWidth();
	TestBlendConstantsOneChannelChange();
	TestDepthTestEnableToggle();
	TestDepthWriteEnableToggle();
	TestDepthCompareOp();
	TestDepthBoundsTestEnableToggle();
	TestDepthBounds();
	TestStencilTestEnableToggle();
	TestStencilOpFrontBack();
	TestCullMode();
	TestFrontFace();
	TestDepthBiasEnableToggle();
	TestDepthBiasOneComponentChange();
	TestStencilCompareMaskFrontBack();
	TestStencilWriteMaskFrontBack();
	TestStencilReferenceFrontBackIndependence();
	TestViewportWithCount();
	TestScissorWithCount();
	TestColorWriteEnable();
	TestAttachmentFeedbackLoopEnable();

	std::printf("DynamicStateCacheTests: all cases passed\n");
	return 0;
}
