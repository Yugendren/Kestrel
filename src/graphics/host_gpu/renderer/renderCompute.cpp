#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderScale.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Whether a remapped dispatch shades the centre of each scaled texel
// (IR::RescaleControl::CentreBit) rather than the top-left guest pixel of its k x k block. Off
// until an image comparison against the native reference shows the correction helps
// (research/remap-detector-2026-09-26.md, section 3).
static constexpr bool kShadeTexelCentre = false;

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size);

// Either shape of the guest's buffer-fill kernels: every dword of the resolved range is written
// with the resolved value and nothing else is written.
static bool ResolveComputeFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                               uint32_t group_y, uint32_t group_z, uint32_t mode,
                               ShaderBufferResource& descriptor, uint32_t& value, uint64_t& size) {
	return ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, value,
	                                size) ||
	       ResolveComputePatternFill(input, group_x, group_y, group_z, mode, descriptor, value,
	                                 size);
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer& buffer, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z,
                                                uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		ShaderBufferResource fill_descriptor;
		uint32_t             fill_value   = 0;
		uint64_t             fill_size    = 0;
		const bool           uniform_fill = ResolveComputeFill(
		    input, group_x, group_y, group_z, mode, fill_descriptor, fill_value, fill_size);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				const bool known =
				    uniform_fill && descriptor.Base48() == fill_descriptor.Base48();
				if (known ? cache.ClearMeta(descriptor.Base48(), fill_value)
				          : cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& user_data = resources.user_data;
	if (program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.images.empty() || !program.info.samplers.empty() ||
	    program.info.uses_dma || input.dispatch_thread_dimensions || mode != 0x41u ||
	    user_data.size() != 10 || program.user_data_base != 0) {
		return false;
	}
	const auto& resource   = program.info.buffers.front();
	const auto& raw        = resources.buffers.front();
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
	if (!resource.written || resource.read || resource.atomic || resource.scalar ||
	    resource.max_byte_extent != 4 ||
	    (resource.formatted && descriptor.Format() != Prospero::BufferFormat::k32UInt) ||
	    (descriptor.Stride() != 4 && descriptor.Stride() != 0) || descriptor.SwizzleEnabled() ||
	    descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    resource.packed_stride != descriptor.PackedStride() || raw.dword_count != 4 ||
	    descriptor.Base48() == 0) {
		return false;
	}
	for (uint32_t i = 0; i < raw.dword_count; i++) {
		if (raw.dwords[i] != user_data[i]) {
			return false;
		}
	}
	const uint32_t clear  = user_data[4];
	const uint32_t period = user_data[9];
	const uint32_t slots  = period == 0u ? 4u : std::min(period, 4u);
	for (uint32_t slot = 1; slot < slots; slot++) {
		if (user_data[4 + slot] != clear) {
			return false;
		}
	}
	if (input.threads_num[0] != 64 || input.threads_num[1] != 1 || input.threads_num[2] != 1 ||
	    group_x == 0 || group_y != 1 || group_z != 1 || !input.group_id[0] || input.group_id[1] ||
	    input.group_id[2] || input.thread_ids_num != 1 || input.wave_size != 64 ||
	    input.tg_size_en) {
		return false;
	}
	const uint64_t count = user_data[8];
	const auto     size  = descriptor.GetSize();
	if (count == 0 || size == 0 || size > UINT32_MAX || count * sizeof(uint32_t) != size ||
	    group_x != (count + input.threads_num[0] - 1) / input.threads_num[0]) {
		return false;
	}
	resolved_descriptor = descriptor;
	resolved_clear      = clear;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding =
		    ResolveTexture(resource, resources.images[0], ShaderType::Compute);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                        size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	const auto compute_program =
	    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	if (!compute_program) {
		// Soft ladder (PPSA21564): the shader's resources could not be materialised (a
		// descriptor it builds from a runtime-dynamic / loop-carried SRT pointer, which
		// KytyPS5 has no bindless path for). Drop the dispatch rather than abort the title --
		// same handling as a null / invalid CS address.
		LOGF("GraphicsRenderDispatchDirect: skipping dispatch, shader resources not "
		     "materialisable, shader=0x%016" PRIx64 "\n",
		     sh_ctx.GetCs().cs_regs.data_addr);
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (TryConsumeComputeMetaClear(input_info, buffer, thread_group_x, thread_group_y,
	                               thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	// A fill kernel that still has to run leaves a value the cache can know without reading it
	// back (see BufferCache::RecordGpuFill); the epoch is taken before its binding is obtained.
	ShaderBufferResource fill_descriptor;
	uint32_t             fill_value = 0;
	uint64_t             fill_size  = 0;
	const bool           is_fill =
	    !program.info.has_bitwise_xor &&
	    ResolveComputeFill(input_info, thread_group_x, thread_group_y, thread_group_z, mode,
	                       fill_descriptor, fill_value, fill_size);
	const uint64_t fill_epoch =
	    is_fill ? m_context.GetBufferCache().CpuModificationEpoch(fill_descriptor.Base48(), fill_size)
	            : 0;
	const bool large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	if ((large_workgroup || has_sampler) &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	// A program the recompiler rescaled (computeRescale.h) runs natively, remapped, or natively
	// with a dual-run check, as its state and the configuration decide.
	auto  rescale  = BeginTileRescale(program);
	auto& bindings = m_compute_bindings;
	m_context.GetBufferCache().BeginBufferScope();
	PrepareBindings(input_info.stage, bindings, rescale.Space());
	GateTileRescale(program, bindings, rescale);
	FindBuffers(bindings);
	if (program.info.uses_dma) {
		m_context.PrepareBda(program.info.writes_through_addresses);
	}
	RebindImages(bindings);
	RebindBuffers(bindings);
	if (is_fill) {
		m_context.GetBufferCache().RecordGpuFill(fill_descriptor.Base48(), fill_size, fill_value,
		                                         fill_epoch);
	}
	ConfirmTileRescale(program, bindings, rescale);

	PreparedBindings* descriptor_stage = &bindings;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	// Classify what this dispatch touches so the scheduler can tell whether it depends on the
	// work recorded before it. Anything the materialisation could not resolve to a guest range,
	// and every access the host cannot reason about -- atomics, GDS, DMA, indirect arguments
	// (see DispatchIndirect) -- is reported as unbounded and takes the conservative barrier.
	ShaderHazardAccess reads;
	ShaderHazardAccess writes;
	for (uint32_t i = 0; i < bindings.buffer_sources.size(); i++) {
		const auto& resource = program.info.buffers[i];
		const auto& source   = bindings.buffer_sources[i];
		auto&       access   = resource.written ? writes : reads;
		if (resource.atomic || source.address == 0 || source.size == 0) {
			access.everything = true;
			continue;
		}
		access.buffers.emplace_back(source.address, source.size);
	}
	for (uint32_t i = 0; i < bindings.images.size(); i++) {
		if (i >= program.info.images.size()) {
			reads.everything = true;
			break;
		}
		const auto& resource = program.info.images[i];
		auto&       access   = resource.written ? writes : reads;
		if (resource.atomic) {
			access.everything = true;
		}
		access.images.push_back(bindings.images[i].image_id);
	}
	if (program.info.uses_dma || bindings.gds.buffer != nullptr) {
		reads.everything  = true;
		writes.everything = true;
	}
	auto& hazards = m_context.GetCommandScheduler().ShaderHazards();
	if (hazards.NeedsBarrier(reads, writes)) {
		ShaderHazardBarrier(buffer.Handle());
		hazards.Clear();
	}
	const ComputeDispatchSize size {thread_group_x, thread_group_y, thread_group_z};
	RecordComputeDispatch(buffer, pipeline, size);

	// The dependency this dispatch creates is published by the next item that needs it, or by
	// the scheduler when anything else may observe it.
	hazards.Record(reads, writes);
	FinishTileRescale(buffer, pipeline, program, bindings, rescale, size);
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	if (!compute_program) {
		// Resources could not be materialised; drop the dispatch as DispatchDirect does.
		LOGF("GraphicsRenderDispatchIndirect: skipping dispatch, shader resources not "
		     "materialisable, shader=0x%016" PRIx64 "\n",
		     cs_regs.cs_regs.data_addr);
		ResetBindings();
		return;
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	const auto& program = *input_info.stage.program;
	// Tile rescale runs here exactly as for DispatchDirect(): the control word and the texel
	// space do not depend on where the workgroup counts come from.
	auto  rescale  = BeginTileRescale(program);
	auto& bindings = m_compute_bindings;
	m_context.GetBufferCache().BeginBufferScope();
	PrepareBindings(input_info.stage, bindings, rescale.Space());
	GateTileRescale(program, bindings, rescale);
	FindBuffers(bindings);
	if (program.info.uses_dma) {
		m_context.PrepareBda(program.info.writes_through_addresses);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	ConfirmTileRescale(program, bindings, rescale);
	PreparedBindings* descriptor_stage = &bindings;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const auto vk_buffer = buffer.Handle();
	// The dispatch size comes out of guest memory, so this dispatch cannot be bounded: it takes
	// the conservative barrier against pending shader work and is recorded as touching everything.
	ShaderHazardAccess unbounded;
	unbounded.everything = true;
	auto& hazards        = m_context.GetCommandScheduler().ShaderHazards();
	if (hazards.NeedsBarrier(unbounded, unbounded)) {
		ShaderHazardBarrier(vk_buffer);
		hazards.Clear();
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	// A dual-run replay reads the same arguments: the full barriers around it order it after
	// every write this one waits for, so the block is not obtained or fenced a second time.
	ComputeDispatchSize size;
	size.args_buffer = args_buffer->Handle();
	size.args_offset = args_offset;
	RecordComputeDispatch(buffer, pipeline, size);
	// Published by the next item that needs it, or by the scheduler, as for DispatchDirect.
	hazards.Record(unbounded, unbounded);
	FinishTileRescale(buffer, pipeline, program, bindings, rescale, size);
	ResetBindings();
}

void RenderExecutor::RecordComputeDispatch(CommandBuffer&                 buffer,
                                           const PipelineCache::Pipeline& pipeline,
                                           const ComputeDispatchSize&     size) {
	auto vk_buffer = buffer.Handle();
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (size.args_buffer != nullptr) {
		vk_buffer.dispatchIndirect(size.args_buffer, size.args_offset);
	} else {
		vk_buffer.dispatch(size.thread_group_x, size.thread_group_y, size.thread_group_z);
	}
}

RenderExecutor::TileRescaleDispatch
RenderExecutor::BeginTileRescale(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	TileRescaleDispatch rescale;
	if (!program.tile_rescale.Enabled() || !program.bindings.HasRescaleControl() ||
	    m_tile_rescale.Mode() == Config::ComputeRescale::Off) {
		return rescale;
	}
	m_tile_rescale.Poll(m_context.GetCommandScheduler(), m_context.GetTextureCache());
	rescale.state = &m_tile_rescale.Track(program.shader_hash, &program);
	rescale.plan  = rescale.state->Decide(m_tile_rescale.Mode(),
	                                      program.tile_rescale.NeedsRuntimeCheck());
	if (rescale.plan == TileRescale::Plan::Native) {
		m_tile_rescale.Off(*rescale.state, program.shader_hash,
		                   rescale.state->blacklisted ? TileRescale::Reason::Blacklisted
		                                              : TileRescale::Reason::ChecksPending);
	}
	return rescale;
}

void RenderExecutor::GateTileRescale(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                     PreparedBindings& bindings, TileRescaleDispatch& rescale) {
	if (rescale.plan != TileRescale::Plan::Rescale) {
		return;
	}
	const auto rescale_log2 = program.tile_rescale.scale_log2;
	DescribeRescaleBindings(program, bindings);
	const auto gate =
	    TileRescale::EvaluateGate(RenderScale::Factor(), rescale_log2, m_rescale_images);
	if (gate.on) {
		rescale.rescaled = true;
		rescale.mask     = gate.image_mask;
	} else {
		m_tile_rescale.Off(*rescale.state, program.shader_hash, gate.reason);
		// Nothing has been acquired yet -- RebindImages() does that -- so resolving the same
		// descriptors again in guest texels leaves the scaled images untouched.
		ResetBindings();
		ResolveImages(bindings, TextureCache::TexelSpace::Guest);
	}
	// A clear word runs the program exactly as translated.
	SetShaderDataWord(bindings, program.bindings.rescale_control_dword,
	                  rescale.rescaled ? ShaderRecompiler::IR::RescaleControl::Encode(
	                                         rescale.mask, rescale_log2, true, kShadeTexelCentre)
	                                   : 0u);
}

void RenderExecutor::ConfirmTileRescale(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                        PreparedBindings& bindings, TileRescaleDispatch& rescale) {
	if (rescale.rescaled) {
		// RebindImages() re-resolves an entry whose image went away while earlier entries were
		// acquired. The word has to describe what is bound, so check again; on a change fall back
		// to guest texels (the scaled images already acquired just resample back to their twins).
		DescribeRescaleBindings(program, bindings);
		const auto gate = TileRescale::EvaluateGate(
		    RenderScale::Factor(), program.tile_rescale.scale_log2, m_rescale_images);
		if (gate.on && gate.image_mask == rescale.mask) {
			rescale.state->rescaled++;
		} else {
			m_tile_rescale.Off(*rescale.state, program.shader_hash, TileRescale::Reason::Rebound);
			ResetBindings();
			ResolveImages(bindings, TextureCache::TexelSpace::Guest);
			RebindImages(bindings);
			SetShaderDataWord(bindings, program.bindings.rescale_control_dword, 0);
			rescale.rescaled = false;
		}
	}
	if (rescale.plan == TileRescale::Plan::NativeWithCheck) {
		rescale.check = PrepareTileRescaleCheck(program, bindings, *rescale.state);
	}
}

void RenderExecutor::FinishTileRescale(CommandBuffer&                                  buffer,
                                       const PipelineCache::Pipeline&                  pipeline,
                                       const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                       const PreparedBindings&                         bindings,
                                       const TileRescaleDispatch&                      rescale,
                                       const ComputeDispatchSize&                      size) {
	if (rescale.check) {
		RecordTileRescaleReplay(buffer, pipeline, program, bindings, *rescale.check,
		                        *rescale.state, size);
	}
	if (rescale.state != nullptr) {
		m_tile_rescale.NoteDispatched(*rescale.state, program.shader_hash);
	}
}

void RenderExecutor::DescribeRescaleBindings(
    const ShaderRecompiler::IR::CompiledShaderInfo& program, const PreparedBindings& bindings) {
	auto& images = m_context.GetTextureCache().m_slot_images;
	m_rescale_images.clear();
	for (uint32_t i = 0; i < bindings.images.size(); i++) {
		const auto&            binding = bindings.images[i];
		TileRescale::BoundImage bound;
		bound.image           = (static_cast<uint64_t>(binding.image_id.index) << 32u) |
		                        binding.image_id.generation;
		bound.guest_address   = binding.desc.info.data.address;
		bound.guest_size      = binding.desc.info.data.size;
		bound.storage         = binding.desc.type == TextureCache::BindingType::Storage;
		bound.texel_addressed = binding.desc.texel_addressed;
		bound.written = i < program.info.images.size() && program.info.images[i].written;
		const Image* shape = images.try_get(binding.image_id);
		if (shape != nullptr && shape->ScaleTwinOwner()) {
			// A twin stands for its owner: the gate asks about the scaled image behind it.
			bound.twin = true;
			shape      = images.try_get(shape->ScaleTwinOwner());
		}
		if (shape != nullptr) {
			const auto& info = shape->info;
			bound.width      = info.extent.width;
			bound.height     = info.extent.height;
			bound.scale      = info.scale;
			bound.samples    = info.samples;
			bound.levels     = info.resources.levels;
			bound.color_2d   = info.type == Prospero::ImageType::kColor2D;
			bound.depth      = info.IsDepth();
			bound.block      = info.IsBlock();
		}
		m_rescale_images.push_back(bound);
	}
}

std::optional<RenderExecutor::TileRescaleCheck>
RenderExecutor::PrepareTileRescaleCheck(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                        const PreparedBindings&    bindings,
                                        TileRescale::ProgramState& state) {
	DescribeRescaleBindings(program, bindings);
	const auto target = TileRescale::EvaluateCheck(
	    RenderScale::Factor(), program.tile_rescale.scale_log2, m_rescale_images);
	if (!target.eligible) {
		m_tile_rescale.Off(state, program.shader_hash, target.reason);
		return std::nullopt;
	}
	auto&      cache     = m_context.GetTextureCache();
	const auto native_id = bindings.images[target.written].image_id;
	auto&      native    = cache.GetImage(native_id);
	if (!TileRescaleController::CanCompare(native)) {
		m_tile_rescale.Off(state, program.shader_hash, TileRescale::Reason::CheckFormat);
		return std::nullopt;
	}
	if (!m_tile_rescale.HasFreeSlot()) {
		m_tile_rescale.Off(state, program.shader_hash, TileRescale::Reason::CheckNoSlot);
		return std::nullopt;
	}
	const auto scratch_id = m_tile_rescale.AcquireScratch(cache, native);
	// Both runs have to start from the pixels the native dispatch is about to overwrite, so a
	// read-modify-write replays exactly. The hazard tracker knows nothing of the scratch image,
	// so the copy is fenced by full barriers rather than by it.
	m_context.GetCommandScheduler().RequestFullBarrier();
	cache.GetImage(scratch_id).CopyImage(native);
	m_context.GetCommandScheduler().RequestFullBarrier();
	return TileRescaleCheck {native_id, scratch_id};
}

void RenderExecutor::RecordTileRescaleReplay(
    CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
    const ShaderRecompiler::IR::CompiledShaderInfo& program, const PreparedBindings& bindings,
    const TileRescaleCheck& check, TileRescale::ProgramState& state,
    const ComputeDispatchSize& size) {
	auto& scheduler = m_context.GetCommandScheduler();
	auto& cache     = m_context.GetTextureCache();
	auto& native    = cache.GetImage(check.native);
	auto& scratch   = cache.GetImage(check.scratch);
	// The replay reads what the native dispatch read and must not overlap it.
	scheduler.RequestFullBarrier();

	// Same descriptors, with every binding of the written image -- the storage binding and any
	// texel-addressed read of it -- moved to the scratch copy through the same view description.
	// A copy, so the stage's persistent resolution (and its reuse memo) never sees the scratch.
	PreparedBindings replay    = bindings;
	const auto       swap_view = [&](vk::ImageView view) {
		const auto cached = std::ranges::find(native.views, view, &CachedImageView::view);
		EXIT_IF(cached == native.views.end());
		return scratch.FindView(cached->info);
	};
	for (auto& binding: replay.images) {
		if (binding.image_id != check.native) {
			continue;
		}
		binding.image_id   = check.scratch;
		binding.image_view = swap_view(binding.image_view);
		for (auto& mip_view: binding.mip_views) {
			mip_view = swap_view(mip_view);
		}
	}
	// Remap with no texel shift: each representative invocation writes its own guest pixel,
	// so the scratch holds, at every pixel with x and y multiples of k, what a remapped dispatch
	// would compute for the scaled texel there.
	SetShaderDataWord(replay, program.bindings.rescale_control_dword,
	                  ShaderRecompiler::IR::RescaleControl::Encode(
	                      0, program.tile_rescale.scale_log2, true, false));
	PreparedBindings* descriptor_stage = &replay;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	RecordComputeDispatch(buffer, pipeline, size);

	scheduler.RequestFullBarrier();
	m_tile_rescale.RecordCompare(scheduler, native, scratch, program.tile_rescale.scale_log2,
	                             state, program.shader_hash);
	// Whatever is recorded next must not overlap the compare's reads either.
	scheduler.RequestFullBarrier();
}

} // namespace Libs::Graphics
