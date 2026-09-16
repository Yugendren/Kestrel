#include "graphics/host_gpu/renderer/image/blitHelper.h"

#include "common/assert.h"
#include "gpu_blit_shaders/gpu_blit_color_to_ms_depth_spv.h"
#include "gpu_blit_shaders/gpu_blit_depth_resample_spv.h"
#include "gpu_blit_shaders/gpu_blit_stencil_bit_resample_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <algorithm>
#include <array>
#include <iterator>

namespace Libs::Graphics {

namespace {

[[nodiscard]] bool FormatHasStencil(vk::Format format) {
	switch (format) {
		case vk::Format::eS8Uint:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint: return true;
		default: return false;
	}
}

[[nodiscard]] uint32_t MipExtent(uint32_t extent, uint32_t level) {
	return std::max(extent >> level, 1u);
}

} // namespace

BlitHelper::BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	vk::DescriptorSetLayoutBinding texture_binding {};
	texture_binding.binding         = 0;
	texture_binding.descriptorType  = vk::DescriptorType::eSampledImage;
	texture_binding.descriptorCount = 1;
	texture_binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = 1;
	descriptor_info.pBindings    = &texture_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create BlitHelper descriptor layout");

	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &m_descriptor_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create BlitHelper pipeline layout");

	const vk::PushConstantRange bit_constant {vk::ShaderStageFlagBits::eFragment, 0,
	                                          sizeof(uint32_t)};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &bit_constant;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_stencil_pipeline_layout),
	    "create BlitHelper stencil pipeline layout");

	m_vertex_shader = CreateShader(GPU_BLIT_FS_TRIANGLE_SPV, std::size(GPU_BLIT_FS_TRIANGLE_SPV));
	m_fragment_shader =
	    CreateShader(GPU_BLIT_COLOR_TO_MS_DEPTH_SPV, std::size(GPU_BLIT_COLOR_TO_MS_DEPTH_SPV));
	m_depth_shader =
	    CreateShader(GPU_BLIT_DEPTH_RESAMPLE_SPV, std::size(GPU_BLIT_DEPTH_RESAMPLE_SPV));
	m_stencil_shader = CreateShader(GPU_BLIT_STENCIL_BIT_RESAMPLE_SPV,
	                                std::size(GPU_BLIT_STENCIL_BIT_RESAMPLE_SPV));
}

BlitHelper::~BlitHelper() {
	for (const auto& pipeline: m_pipelines) {
		m_graphics.device.destroyPipeline(pipeline.handle, nullptr);
	}
	for (const auto module: {m_stencil_shader, m_depth_shader, m_fragment_shader}) {
		if (module != nullptr) {
			m_graphics.device.destroyShaderModule(module, nullptr);
		}
	}
	if (m_vertex_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_vertex_shader, nullptr);
	}
	for (const auto layout: {m_stencil_pipeline_layout, m_pipeline_layout}) {
		if (layout != nullptr) {
			m_graphics.device.destroyPipelineLayout(layout, nullptr);
		}
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

vk::ShaderModule BlitHelper::CreateShader(const uint32_t* code, size_t words) const {
	EXIT_IF(code == nullptr || words == 0);
	vk::ShaderModuleCreateInfo create {};
	create.codeSize         = words * sizeof(uint32_t);
	create.pCode            = code;
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(m_graphics.device.createShaderModule(&create, nullptr, &module),
	                     "create BlitHelper shader module");
	return module;
}

vk::Pipeline BlitHelper::GetPipeline(PipelineKey key) {
	const auto cached = std::ranges::find(m_pipelines, key, &Pipeline::key);
	if (cached != m_pipelines.end()) {
		return cached->handle;
	}

	const auto samples = vulkan_sample_count(key.samples);
	EXIT_IF(samples == vk::SampleCountFlagBits {} || key.format == vk::Format::eUndefined);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_vertex_shader;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = key.kind == PipelineKind::ColorToMsDepth ? m_fragment_shader
	                   : key.kind == PipelineKind::Depth        ? m_depth_shader
	                                                            : m_stencil_shader;
	stages[1].pName  = "main";
	EXIT_IF(stages[1].module == nullptr);

	vk::PipelineVertexInputStateCreateInfo vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = samples;
	vk::PipelineDepthStencilStateCreateInfo depth {};
	depth.depthTestEnable  = VK_TRUE;
	depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp   = vk::CompareOp::eAlways;
	if (key.kind == PipelineKind::StencilBit) {
		// Only the surviving fragments reach the stencil op, and the write mask limits it to the
		// bit plane this pass carries, so the other planes keep whatever earlier passes wrote.
		vk::StencilOpState stencil {};
		stencil.failOp      = vk::StencilOp::eKeep;
		stencil.passOp      = vk::StencilOp::eReplace;
		stencil.depthFailOp = vk::StencilOp::eReplace;
		stencil.compareOp   = vk::CompareOp::eAlways;
		stencil.compareMask = 0xffu;
		depth.depthTestEnable   = VK_FALSE;
		depth.depthWriteEnable  = VK_FALSE;
		depth.stencilTestEnable = VK_TRUE;
		depth.front             = stencil;
		depth.back              = stencil;
	}
	vk::PipelineColorBlendStateCreateInfo color_blend {};
	const std::array base_dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	const std::array stencil_dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor,
	                                         vk::DynamicState::eStencilWriteMask,
	                                         vk::DynamicState::eStencilReference};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	if (key.kind == PipelineKind::StencilBit) {
		dynamic.dynamicStateCount = static_cast<uint32_t>(stencil_dynamic_states.size());
		dynamic.pDynamicStates    = stencil_dynamic_states.data();
	} else {
		dynamic.dynamicStateCount = static_cast<uint32_t>(base_dynamic_states.size());
		dynamic.pDynamicStates    = base_dynamic_states.data();
	}

	vk::PipelineRenderingCreateInfo rendering {};
	if (key.kind == PipelineKind::StencilBit) {
		rendering.stencilAttachmentFormat = key.format;
	} else {
		rendering.depthAttachmentFormat = key.format;
	}

	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth;
	create.pColorBlendState    = &color_blend;
	create.pDynamicState       = &dynamic;
	create.layout = key.kind == PipelineKind::StencilBit ? m_stencil_pipeline_layout
	                                                     : m_pipeline_layout;

	vk::Pipeline pipeline = nullptr;
	RequireVulkanSuccess(
	    m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	    "create BlitHelper depth pipeline");
	m_pipelines.push_back({key, pipeline});
	return pipeline;
}

void BlitHelper::ReinterpretColorAsMsDepth(Image& source, Image& destination) {
	const auto& source_info      = source.info;
	const auto& destination_info = destination.info;
	EXIT_IF(DepthAspectTransferFormat(source_info.pixel_format) != vk::Format::eUndefined ||
	        DepthAspectTransferFormat(destination_info.pixel_format) == vk::Format::eUndefined ||
	        source_info.samples != 1 || destination_info.samples <= 1 ||
	        destination_info.samples > 4 || source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        source_info.extent.width != destination_info.extent.width ||
	        source_info.extent.height != destination_info.extent.height ||
	        source_info.extent.depth != 1 || destination_info.extent.depth != 1 ||
	        source.backing.image == nullptr || destination.backing.image == nullptr);
	m_scheduler.EndRendering();

	ImageViewInfo source_view_info {};
	source_view_info.format = source_info.pixel_format;
	source_view_info.type   = vk::ImageViewType::e2D;
	source_view_info.aspect = vk::ImageAspectFlagBits::eColor;
	source_view_info.usage  = vk::ImageUsageFlagBits::eSampled;
	const auto source_view  = source.FindView(source_view_info);

	ImageViewInfo destination_view_info {};
	destination_view_info.format = destination_info.pixel_format;
	destination_view_info.type   = vk::ImageViewType::e2D;
	destination_view_info.aspect = vk::ImageAspectFlagBits::eDepth;
	destination_view_info.usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	const auto destination_view  = destination.FindView(destination_view_info);

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command);
	destination.Transit(ColorToMsDepthLayout, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {},
	                    command);

	vk::RenderingAttachmentInfo depth_attachment {};
	depth_attachment.imageView               = destination_view;
	depth_attachment.imageLayout             = ColorToMsDepthLayout;
	depth_attachment.loadOp                  = vk::AttachmentLoadOp::eClear;
	depth_attachment.storeOp                 = vk::AttachmentStoreOp::eStore;
	depth_attachment.clearValue.depthStencil = {0.0f, 0};

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent = {destination_info.extent.width, destination_info.extent.height};
	rendering.layerCount        = 1;
	rendering.pDepthAttachment  = &depth_attachment;
	command.beginRendering(&rendering);

	vk::DescriptorImageInfo descriptor_image {};
	descriptor_image.imageView   = source_view;
	descriptor_image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet descriptor_write {};
	descriptor_write.dstBinding      = 0;
	descriptor_write.descriptorCount = 1;
	descriptor_write.descriptorType  = vk::DescriptorType::eSampledImage;
	descriptor_write.pImageInfo      = &descriptor_image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1,
	                             &descriptor_write);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics,
	                     GetPipeline({PipelineKind::ColorToMsDepth, destination_info.samples,
	                                  destination_info.pixel_format}));

	const vk::Viewport viewport {0.0f,
	                             0.0f,
	                             static_cast<float>(destination_info.extent.width),
	                             static_cast<float>(destination_info.extent.height),
	                             0.0f,
	                             1.0f};
	const vk::Rect2D   scissor {{0, 0},
	                            {destination_info.extent.width, destination_info.extent.height}};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

bool BlitHelper::CanResampleDepth(const ImageInfo& info) {
	return DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined;
}

void BlitHelper::ResampleDepth(Image& source, Image& destination,
                               const ImageSubresourceRange& source_range,
                               const ImageSubresourceRange& destination_range) {
	const auto format = destination.backing.format;
	EXIT_IF(source.backing.image == nullptr || destination.backing.image == nullptr);
	EXIT_IF(source.backing.format != format || !CanResampleDepth(destination.info));
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1);
	EXIT_IF(source_range.level_count != 1 || destination_range.level_count != 1);
	EXIT_IF(source_range.layer_count != destination_range.layer_count ||
	        source_range.layer_count == 0);
	// A resample writes an attachment, so it cannot be recorded inside someone else's pass.
	m_scheduler.EndRendering();

	// The stencil aspect only has to be carried when the guest keeps something in it.
	const bool stencil = FormatHasStencil(format) && destination.info.HasStencil();

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
	               source_range, command);
	destination.Transit(ColorToMsDepthLayout, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	                    destination_range, command);

	const auto width  = MipExtent(destination.backing.extent.width, destination_range.base_level);
	const auto height = MipExtent(destination.backing.extent.height, destination_range.base_level);
	const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(width),
	                             static_cast<float>(height), 0.0f, 1.0f};
	const vk::Rect2D   scissor {{0, 0}, {width, height}};

	const auto view_of = [](const ImageSubresourceRange& range, uint32_t layer, vk::Format fmt,
	                        vk::ImageAspectFlagBits aspect, vk::ImageUsageFlagBits usage) {
		ImageViewInfo info {};
		info.format      = fmt;
		info.type        = vk::ImageViewType::e2D;
		info.aspect      = aspect;
		info.usage       = usage;
		info.base_level  = range.base_level;
		info.level_count = 1;
		info.base_layer  = range.base_layer + layer;
		info.layer_count = 1;
		return info;
	};
	const auto bind_source = [&](vk::ImageView view, vk::PipelineLayout layout) {
		vk::DescriptorImageInfo image {};
		image.imageView   = view;
		image.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
		vk::WriteDescriptorSet write {};
		write.dstBinding      = 0;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eSampledImage;
		write.pImageInfo      = &image;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &write);
	};

	for (uint32_t layer = 0; layer < source_range.layer_count; layer++) {
		vk::RenderingAttachmentInfo depth_attachment {};
		depth_attachment.imageView =
		    destination.FindView(view_of(destination_range, layer, format,
		                                 vk::ImageAspectFlagBits::eDepth,
		                                 vk::ImageUsageFlagBits::eDepthStencilAttachment));
		depth_attachment.imageLayout = ColorToMsDepthLayout;
		depth_attachment.loadOp      = vk::AttachmentLoadOp::eDontCare;
		depth_attachment.storeOp     = vk::AttachmentStoreOp::eStore;

		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {width, height};
		rendering.layerCount        = 1;
		rendering.pDepthAttachment  = &depth_attachment;
		command.beginRendering(&rendering);
		bind_source(source.FindView(view_of(source_range, layer, format,
		                                    vk::ImageAspectFlagBits::eDepth,
		                                    vk::ImageUsageFlagBits::eSampled)),
		            m_pipeline_layout);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics,
		                     GetPipeline({PipelineKind::Depth, 1, format}));
		command.setViewport(0, 1, &viewport);
		command.setScissor(0, 1, &scissor);
		command.draw(3, 1, 0, 0);
		command.endRendering();

		if (!stencil) {
			continue;
		}
		// One pass per bit plane, over a destination cleared to zero: a pass only ever sets its
		// own bit, so together they reproduce the source value exactly.
		vk::RenderingAttachmentInfo stencil_attachment {};
		stencil_attachment.imageView =
		    destination.FindView(view_of(destination_range, layer, format,
		                                 vk::ImageAspectFlagBits::eStencil,
		                                 vk::ImageUsageFlagBits::eDepthStencilAttachment));
		stencil_attachment.imageLayout             = ColorToMsDepthLayout;
		stencil_attachment.storeOp                 = vk::AttachmentStoreOp::eStore;
		stencil_attachment.clearValue.depthStencil = {0.0f, 0};
		const auto stencil_view =
		    source.FindView(view_of(source_range, layer, format,
		                            vk::ImageAspectFlagBits::eStencil,
		                            vk::ImageUsageFlagBits::eSampled));
		for (uint32_t bit = 0; bit < 8; bit++) {
			stencil_attachment.loadOp =
			    bit == 0 ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
			vk::RenderingInfo stencil_rendering {};
			stencil_rendering.renderArea.extent  = {width, height};
			stencil_rendering.layerCount         = 1;
			stencil_rendering.pStencilAttachment = &stencil_attachment;
			command.beginRendering(&stencil_rendering);
			bind_source(stencil_view, m_stencil_pipeline_layout);
			command.bindPipeline(vk::PipelineBindPoint::eGraphics,
			                     GetPipeline({PipelineKind::StencilBit, 1, format}));
			command.setViewport(0, 1, &viewport);
			command.setScissor(0, 1, &scissor);
			const uint32_t plane = 1u << bit;
			command.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, plane);
			command.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, plane);
			command.pushConstants(m_stencil_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
			                      sizeof(bit), &bit);
			command.draw(3, 1, 0, 0);
			command.endRendering();
		}
	}

	destination.Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal,
	                    vk::AccessFlagBits2::eShaderRead, destination_range, command);
}

} // namespace Libs::Graphics
