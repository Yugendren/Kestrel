#include "graphics/host_gpu/renderer/computeRescale.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/tile_rescale_compare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageView.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <string>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

// Checks can be in flight for a frame or two before their command buffers complete; a Verify
// run checks every dispatch of every rescalable program, so leave room for a busy frame.
constexpr uint32_t ResultSlots = 64;
// Per slot: texels that differ, texels compared (tile_rescale_compare.comp).
constexpr uint32_t SlotDwords = 2;
constexpr uint64_t SlotBytes  = SlotDwords * sizeof(uint32_t);
// Scratch images are the size of a full-screen target, so they are freed once no check has been
// recorded for this many rescalable dispatches -- the steady state of Auto once every program is
// verified or blacklisted.
constexpr uint64_t ScratchIdlePolls = 16384;
// The compare shader's workgroup edge.
constexpr uint32_t CompareGroupSize = 8;

// Mirrors the push-constant block of tile_rescale_compare.comp.
struct ComparePushConstants {
	uint32_t width      = 0;
	uint32_t height     = 0;
	uint32_t scale_log2 = 0;
	uint32_t slot_dword = 0;
};

} // namespace

TileRescaleController::TileRescaleController(): m_slot_busy(ResultSlots, false) {}

TileRescaleController::~TileRescaleController() {
	// Scratch images belong to the texture cache, which frees every image it holds itself.
	if (m_graphics == nullptr) {
		return;
	}
	m_graphics->device.destroyPipeline(m_pipeline, nullptr);
	m_graphics->device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics->device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
	m_graphics->device.destroySampler(m_sampler, nullptr);
}

Config::ComputeRescale TileRescaleController::Mode() {
	if (!m_mode) {
		m_mode = Config::GetComputeRescale();
	}
	return *m_mode;
}

TileRescale::ProgramState& TileRescaleController::Track(uint64_t hash, const void* identity) {
	return m_programs[ProgramKey {hash, identity}];
}

void TileRescaleController::Off(TileRescale::ProgramState& state, uint64_t hash,
                                TileRescale::Reason reason) {
	if (state.CountOff(reason)) {
		LOGF("tile rescale: hash=0x%016" PRIx64 " dispatch=%" PRIu64 " off: %s\n", hash,
		     state.dispatches, TileRescale::ReasonName(reason));
	}
}

void TileRescaleController::NoteDispatched(const TileRescale::ProgramState& state,
                                           uint64_t hash) const {
	if (!state.SummaryDue()) {
		return;
	}
	std::string off;
	for (size_t i = 0; i < TileRescale::ReasonCount; i++) {
		if (state.off[i] != 0) {
			const auto reason = static_cast<TileRescale::Reason>(i);
			off += fmt::format(" {}={}", TileRescale::ReasonName(reason), state.off[i]);
		}
	}
	LOGF("tile rescale: hash=0x%016" PRIx64 " dispatches=%" PRIu64 " rescaled=%" PRIu64
	     " checks=%u/%u verified=%d blacklisted=%d off:%s\n",
	     hash, state.dispatches, state.rescaled, state.checks_passed, state.checks_submitted,
	     state.verified ? 1 : 0, state.blacklisted ? 1 : 0, off.empty() ? " none" : off.c_str());
}

void TileRescaleController::Poll(CommandScheduler& scheduler, TextureCache& cache) {
	if (m_pending.empty()) {
		if (!m_scratch.empty() && ++m_idle_polls > ScratchIdlePolls) {
			for (const auto& scratch: m_scratch) {
				cache.ReleaseScratchImage(scratch.image);
			}
			m_scratch.clear();
		}
		return;
	}
	const auto mode = Mode();
	std::erase_if(m_pending, [&](const PendingCheck& check) {
		if (!scheduler.IsFree(check.tick)) {
			return false;
		}
		const uint64_t offset = check.slot * SlotBytes;
		m_results->Invalidate(offset, SlotBytes);
		std::array<uint32_t, SlotDwords> words {};
		std::memcpy(words.data(), m_results->Mapped().data() + offset, SlotBytes);
		m_slot_busy[check.slot] = false;

		const uint32_t mismatches = words[0];
		const uint32_t compared   = words[1];
		LOGF("tile rescale check: hash=0x%016" PRIx64 " dispatch=%" PRIu64
		     " mismatches=%u compared=%u\n",
		     check.hash, check.dispatch, mismatches, compared);
		switch (check.state->NoteCheckResult(mode, mismatches, compared)) {
			case TileRescale::CheckOutcome::Blacklisted:
				LOGF("tile rescale: hash=0x%016" PRIx64 " blacklisted for the session: the "
				     "remap-only replay differs from the native dispatch\n",
				     check.hash);
				break;
			case TileRescale::CheckOutcome::Verified:
				LOGF("tile rescale: hash=0x%016" PRIx64 " verified after %u checks, rescaling\n",
				     check.hash, check.state->checks_passed);
				break;
			case TileRescale::CheckOutcome::Inconclusive:
				LOGF("tile rescale: hash=0x%016" PRIx64 " check compared no pixel\n",
				     check.hash);
				break;
			case TileRescale::CheckOutcome::Passed:
			case TileRescale::CheckOutcome::Ignored: break;
		}
		return true;
	});
}

vk::Format TileRescaleController::RawViewFormat(vk::Format format) {
	vk::Format raw = vk::Format::eUndefined;
	switch (vk::blockSize(format)) {
		case 1: raw = vk::Format::eR8Uint; break;
		case 2: raw = vk::Format::eR16Uint; break;
		case 4: raw = vk::Format::eR32Uint; break;
		case 8: raw = vk::Format::eR32G32Uint; break;
		case 16: raw = vk::Format::eR32G32B32A32Uint; break;
		default: return vk::Format::eUndefined;
	}
	// A view in another format of the same size is only legal within the image's compatibility
	// class, which rules out depth and block formats.
	return ImageViewOps::FormatsCompatible(format, raw) ? raw : vk::Format::eUndefined;
}

bool TileRescaleController::CanCompare(const Image& image) {
	return image.backing.image != nullptr &&
	       static_cast<bool>(image.backing.usage & vk::ImageUsageFlagBits::eSampled) &&
	       static_cast<bool>(image.backing.flags & vk::ImageCreateFlagBits::eMutableFormat) &&
	       RawViewFormat(image.backing.format) != vk::Format::eUndefined;
}

bool TileRescaleController::HasFreeSlot() const {
	return std::ranges::find(m_slot_busy, false) != m_slot_busy.end();
}

ImageId TileRescaleController::AcquireScratch(TextureCache& cache, const Image& native) {
	const auto& info = native.info;
	for (const auto& scratch: m_scratch) {
		if (scratch.format == info.pixel_format && scratch.width == info.extent.width &&
		    scratch.height == info.extent.height && scratch.layers == info.resources.layers) {
			return scratch.image;
		}
	}
	const auto id = cache.AcquireScratchImage(info);
	m_scratch.push_back({info.pixel_format, info.extent.width, info.extent.height,
	                     info.resources.layers, id});
	return id;
}

void TileRescaleController::InitializeGpu(CommandScheduler& scheduler) {
	auto& graphics = scheduler.Graphics();
	m_results      = std::make_unique<Buffer>(
        graphics, scheduler, MemoryUsage::Download, 0,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        ResultSlots * SlotBytes);
	EXIT_IF(m_results->Mapped().size() < ResultSlots * SlotBytes);

	// texelFetch ignores every sampler state; a combined sampler is only what lets the compare
	// shader read any sampled format without declaring a storage format.
	vk::SamplerCreateInfo sampler_info {};
	sampler_info.magFilter    = vk::Filter::eNearest;
	sampler_info.minFilter    = vk::Filter::eNearest;
	sampler_info.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	RequireVulkanSuccess(graphics.device.createSampler(&sampler_info, nullptr, &m_sampler),
	                     "create tile-rescale compare sampler");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute,
	     nullptr},
	    {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute,
	     nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_desc_layout),
	    "create tile-rescale compare descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(ComparePushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	    "create tile-rescale compare pipeline layout");

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = std::size(TILE_RESCALE_COMPARE_SPV) * sizeof(uint32_t);
	module_info.pCode       = TILE_RESCALE_COMPARE_SPV;
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(graphics.device.createShaderModule(&module_info, nullptr, &module),
	                     "create tile-rescale compare shader module");
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create tile-rescale compare pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Tile Rescale Compare");
	// Set last: the destructor takes a non-null m_graphics to mean every object above exists.
	m_graphics = &graphics;
}

void TileRescaleController::RecordCompare(CommandScheduler& scheduler, Image& native,
                                          Image& scratch, uint32_t scale_log2,
                                          TileRescale::ProgramState& state, uint64_t hash) {
	if (m_graphics == nullptr) {
		InitializeGpu(scheduler);
	}
	const auto slot_it = std::ranges::find(m_slot_busy, false);
	EXIT_IF(slot_it == m_slot_busy.end());
	const auto slot = static_cast<uint32_t>(slot_it - m_slot_busy.begin());
	*slot_it        = true;

	auto command = scheduler.Current().Handle();
	// Both images are read through the raw view of their (identical) format, so the compare is
	// of stored bits: float formats keep their NaN payloads and signed zeros, unorm formats their
	// exact codes.
	ImageViewInfo view {};
	view.format = RawViewFormat(native.backing.format);
	view.usage  = vk::ImageUsageFlagBits::eSampled;
	native.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {}, command);
	scratch.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {}, command);
	const vk::DescriptorImageInfo native_info {m_sampler, native.FindView(view),
	                                           vk::ImageLayout::eGeneral};
	const vk::DescriptorImageInfo replay_info {m_sampler, scratch.FindView(view),
	                                           vk::ImageLayout::eGeneral};

	const uint64_t offset = slot * SlotBytes;
	command.fillBuffer(m_results->Handle(), offset, SlotBytes, 0);
	vk::BufferMemoryBarrier clear_barrier {};
	clear_barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	clear_barrier.dstAccessMask =
	    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clear_barrier.buffer              = m_results->Handle();
	clear_barrier.offset              = offset;
	clear_barrier.size                = SlotBytes;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags {}, 0,
	                        nullptr, 1, &clear_barrier, 0, nullptr);

	const vk::DescriptorBufferInfo result_info {m_results->Handle(), 0, ResultSlots * SlotBytes};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
	}
	writes[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
	writes[0].pImageInfo     = &native_info;
	writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
	writes[1].pImageInfo     = &replay_info;
	writes[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	writes[2].pBufferInfo    = &result_info;

	ComparePushConstants push {};
	push.width      = native.info.extent.width;
	push.height     = native.info.extent.height;
	push.scale_log2 = scale_log2;
	push.slot_dword = slot * SlotDwords;
	// One invocation per representative pixel: the guest pixels at multiples of k on both axes.
	const uint32_t k       = 1u << scale_log2;
	const uint32_t columns = (push.width + k - 1u) >> scale_log2;
	const uint32_t rows    = (push.height + k - 1u) >> scale_log2;

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch((columns + CompareGroupSize - 1u) / CompareGroupSize,
	                 (rows + CompareGroupSize - 1u) / CompareGroupSize, 1);

	// The host reads the slot once the command buffer's tick has passed (Poll()).
	vk::BufferMemoryBarrier host_barrier = clear_barrier;
	host_barrier.srcAccessMask           = vk::AccessFlagBits::eShaderWrite;
	host_barrier.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eHost, vk::DependencyFlags {}, 0, nullptr,
	                        1, &host_barrier, 0, nullptr);

	state.checks_submitted++;
	m_pending.push_back({&state, hash, state.dispatches, scheduler.CurrentTick(), slot});
	m_idle_polls = 0;
}

} // namespace Libs::Graphics
