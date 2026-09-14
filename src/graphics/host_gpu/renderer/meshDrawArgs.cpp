#include "graphics/host_gpu/renderer/meshDrawArgs.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/mesh_draw_args_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

// Mirrors the layout of the push-constant block in mesh_draw_args.comp.
struct PushConstants {
	uint32_t args_dword_offset     = 0;
	uint32_t params_dword_offset   = 0;
	uint32_t dispatch_dword_offset = 0;
	uint32_t primitive_size        = 0;
	uint32_t primitive_step        = 0;
	uint32_t primitives_per_group  = 0;
	uint32_t element_size          = 0;
	uint32_t index_base_low        = 0;
	uint32_t index_base_high       = 0;
	uint32_t max_groups_x          = 0;
	uint32_t max_groups_y          = 0;
};

} // namespace

MeshDrawArgsBuilder::MeshDrawArgsBuilder(GraphicContext& graphics): m_graphics(graphics) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_desc_layout),
	    "create mesh-draw-args descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	    "create mesh-draw-args pipeline layout");

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = std::size(MESH_DRAW_ARGS_SPV) * sizeof(uint32_t);
	module_info.pCode    = MESH_DRAW_ARGS_SPV;
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(m_graphics.device.createShaderModule(&module_info, nullptr, &module),
	                     "create mesh-draw-args shader module");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr,
	                                                              &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-draw-args pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Draw Args Builder");
}

MeshDrawArgsBuilder::~MeshDrawArgsBuilder() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
}

void MeshDrawArgsBuilder::Record(vk::CommandBuffer command, const Args& args) const {
	// The guest args offset comes from the buffer cache and is not necessarily a legal storage-
	// buffer descriptor offset, unlike vkCmdDrawIndexedIndirect's offset, which only has to be
	// 4-byte aligned. Bind the nearest legal offset below it and carry the remainder, in dwords, as
	// a push constant instead.
	const auto alignment =
	    std::max<vk::DeviceSize>(m_graphics.StorageMinAlignment(), sizeof(uint32_t));
	const auto aligned_args_offset = args.guest_args_offset & ~(alignment - 1);
	const auto args_remainder      = args.guest_args_offset - aligned_args_offset;
	const auto args_dword_offset   = static_cast<uint32_t>(args_remainder / sizeof(uint32_t));
	const auto args_range          = args_remainder + sizeof(vk::DrawIndexedIndirectCommand);

	const vk::DescriptorBufferInfo infos[] {
	    {args.guest_args_buffer, aligned_args_offset, args_range},
	    {args.params_buffer, args.params_offset, ParamsDwordCount * sizeof(uint32_t)},
	    {args.dispatch_buffer, args.dispatch_offset, DispatchDwordCount * sizeof(uint32_t)},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	PushConstants push {};
	push.args_dword_offset   = args_dword_offset;
	push.primitive_size      = args.primitive_size;
	push.primitive_step      = args.primitive_step;
	push.primitives_per_group = args.primitives_per_group;
	push.element_size        = args.element_size;
	push.index_base_low      = static_cast<uint32_t>(args.index_base);
	push.index_base_high     = static_cast<uint32_t>(args.index_base >> 32u);
	push.max_groups_x        = args.max_groups_x;
	push.max_groups_y        = args.max_groups_y;

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);
}

} // namespace Libs::Graphics
