#include "graphics/host_gpu/renderer/renderDraw.h"

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
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/passScale.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderScale.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Out of line because MeshDrawArgsBuilder's constructor needs RenderContext::GetGraphics(), and
// RenderContext is only forward-declared in render.h to avoid an include cycle.
RenderExecutor::RenderExecutor(RenderContext& context)
    : m_context(context), m_mesh_draw_args_builder(context.GetGraphics()) {}

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	// A program that fetches its own vertices runs the guest's index arithmetic, which adds the
	// user-data vertex and instance offsets itself; feeding them through the draw as well would
	// apply them twice. Only the packet's index offset stays with the draw, as on the hardware.
	if (program.info.gpu_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u attr_num=%d\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index, b.attr_num);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < b.attr_num; ai++) {
					const auto  res_index = b.attr_indices[ai];
					const auto& r         = vs_input_info.resources[res_index];
					const auto& rd        = vs_input_info.resources_dst[res_index];
					const auto  offset    = b.attr_offsets[ai];
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < b.attr_num; ai++) {
			const auto  res_index = b.attr_indices[ai];
			const auto& r         = vs_input_info.resources[res_index];
			const auto& rd        = vs_input_info.resources_dst[res_index];
			LOGF("DrawInputState[%u]: attr[%d] res=%d offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, res_index, b.attr_offsets[ai], rd.register_start, rd.registers_num,
			     rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

// Resolves the per-draw Vulkan dynamic state from the guest registers, the draw's last vertex
// stage, its depth target and the render pass it records into. SetGraphicsDynamicParams() records
// it; the draw-reuse oracle compares it against what is already recorded (see
// OracleConfirmsReusedPipeline()).
static void ResolveGraphicsDynamicState(const CommandBuffer&         buffer,
                                        const ShaderVertexInputInfo& vs_input_info,
                                        const RenderDepthInfo& depth, const RenderState& rendering,
                                        vk::ImageAspectFlags  feedback_aspects,
                                        vk::PrimitiveTopology topology, bool primitive_restart_enable,
                                        bool mesh_active, GraphicsDynamicState& out) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto& vp = ctx.GetScreenViewport();
	// Guest-space framebuffer size: guest scissor rectangles are clamped against it, and the
	// result is converted to host space afterwards.
	const vk::Extent2D framebuffer_extent {rendering.guest_width, rendering.guest_height};
	const RenderScale::Mapping mapping {rendering.scale};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	static_assert(viewport_slots <= DynamicStateCache::MaxViewports,
	              "DynamicStateCache stores a full viewport array per command buffer");
	static_assert(viewport_slots <= GraphicsDynamicState::MaxViewports);
	auto&          viewports      = out.viewports;
	auto&          scissors       = out.scissors;
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	out.viewport_count            = viewport_count;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			// The guest emits window coordinates and the vertex shader normalises them against
			// a fixed half-extent (see PipelineCache clip_space). Window position therefore
			// comes out as viewport_size / (2 * half_extent) times the guest coordinate, so
			// stretching this viewport by the render scale is what moves those draws onto a
			// scaled attachment.
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			const auto  base_x = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			const auto  base_y = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
			viewport.width     = base_x;
			viewport.height    = base_y;
			mapping.Viewport(viewport);
			// Supersampling beyond the device viewport limit cannot preserve the identity.
			viewport.width =
			    std::min(viewport.width, static_cast<float>(limits.maxViewportDimensions[0]));
			viewport.height =
			    std::min(viewport.height, static_cast<float>(limits.maxViewportDimensions[1]));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
			mapping.Viewport(viewport);
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		scissors[i] = mapping.Rect(final_scissor.left, final_scissor.top, final_scissor.right,
		                           final_scissor.bottom, framebuffer_extent);
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width      = 1.0f;
			scissors[i].extent  = {0, 0};
		}
	}

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	out.line_width         = line_width;
	const auto& blend      = ctx.GetBlendColor();
	out.blend_constants    = {blend.red, blend.green, blend.blue, blend.alpha};
	out.depth_test_enable  = depth.depth_test_enable ? VK_TRUE : VK_FALSE;
	out.depth_write_enable = depth.depth_write_enable ? VK_TRUE : VK_FALSE;
	out.depth_compare_op   = depth.depth_compare_op;
#if defined(__APPLE__)
	// MoltenVK lacks the depthBounds feature; depth-bounds testing stays disabled and the bounds
	// themselves are never set (see the matching guard in CreatePipelineInternal).
	out.depth_bounds_test_enable = VK_FALSE;
#else
	out.depth_bounds_test_enable = depth.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
	// vkCmdSetDepthBounds requires both bounds in [0, 1] without VK_EXT_depth_range_unrestricted
	// (VUID-vkCmdSetDepthBounds-minDepthBounds-02508/-02509), but the guest registers are raw
	// floats. Depth values are in [0, 1] too, so clamping leaves the test's outcome unchanged
	// (fmax/fmin also turn a NaN bound into the matching limit). With the test off the bounds
	// are unused: record the neutral full range so stale guest values neither break validity nor
	// defeat the dynamic-state cache.
	const auto unit          = [](float value) { return std::fmin(std::fmax(value, 0.0f), 1.0f); };
	out.depth_bounds         = depth.depth_bounds_test_enable
	                               ? std::array {unit(depth.depth_min_bounds),
	                                             unit(depth.depth_max_bounds)}
	                               : std::array {0.0f, 1.0f};
#endif

	const auto& mode = ctx.GetModeControl();

	// Cull mode, front face, topology and primitive restart are dynamic per draw;
	// PipelineStaticParameters no longer bakes them into the pipeline (see pipelineCache.cpp).
	// Tessellation draws use a patch list too, so only the guest primitive type identifies a
	// rect list (whose host triangles must never be culled).
	const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (!rect_list && mode.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (!rect_list && mode.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}
	out.cull_mode  = cull_mode;
	out.front_face = mode.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise;
	// A mesh pipeline has no input-assembly state, matching CreatePipelineInternal's !mesh guard on
	// the topology and primitive-restart dynamic states.
	out.input_assembly = !mesh_active;
	if (out.input_assembly) {
		out.topology          = topology;
		out.primitive_restart = primitive_restart_enable;
	}

	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	out.depth_bias_enable         = depth_bias_enable ? VK_TRUE : VK_FALSE;
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		out.depth_bias = {constant_factor, poly_offset.clamp, slope_factor};
	}

	// Every pipeline declares the stencil op, masks and reference dynamic, and Vulkan requires
	// each declared dynamic state to be set in the command buffer before a draw even when the
	// stencil test is off. With the test disabled the depth info keeps value-initialised faces
	// (KEEP ops, zero masks), which are inert.
	out.stencil_test_enable = depth.stencil_test_enable ? VK_TRUE : VK_FALSE;
	const auto stencil_face = [](const vk::StencilOpState& state) {
		return GraphicsDynamicState::StencilFace {
		    .fail_op       = state.failOp,
		    .pass_op       = state.passOp,
		    .depth_fail_op = state.depthFailOp,
		    .compare_op    = state.compareOp,
		    .compare_mask  = state.compareMask,
		    .write_mask    = state.writeMask,
		    .reference     = state.reference,
		};
	};
	out.stencil = {stencil_face(depth.stencil_front), stencil_face(depth.stencil_back)};

#if !defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
	static_assert(RENDER_COLOR_ATTACHMENTS_MAX <= DynamicStateCache::MaxColorAttachments,
	              "DynamicStateCache stores a full colour-write-enable array per command buffer");
	out.color_write_count = rendering.num_color_attachments;
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		out.color_write_enable[slot] =
		    rendering.color_attachments[slot].image_view != nullptr ? VK_TRUE : VK_FALSE;
	}
#endif
	out.feedback_loop_dynamic = buffer.GetGraphics().attachment_feedback_loop_enabled;
	if (out.feedback_loop_dynamic) {
		out.feedback_aspects = feedback_aspects;
	}
}

// Records `state` on `vk_buffer`, going through the buffer's DynamicStateCache for every piece
// of state it caches.
static void SetGraphicsDynamicParams(const CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                                     const GraphicsDynamicState& state) {
	KYTY_PROFILER_FUNCTION();

	auto& cache = buffer.DynamicState();
	cache.SetViewportWithCount(vk_buffer, state.viewport_count, state.viewports.data());
	cache.SetScissorWithCount(vk_buffer, state.viewport_count, state.scissors.data());
	cache.SetLineWidth(vk_buffer, state.line_width);
	cache.SetBlendConstants(vk_buffer, state.blend_constants.data());
	cache.SetDepthTestEnable(vk_buffer, state.depth_test_enable);
	cache.SetDepthWriteEnable(vk_buffer, state.depth_write_enable);
	cache.SetDepthCompareOp(vk_buffer, state.depth_compare_op);
	cache.SetDepthBoundsTestEnable(vk_buffer, state.depth_bounds_test_enable);
#if !defined(__APPLE__)
	cache.SetDepthBounds(vk_buffer, state.depth_bounds[0], state.depth_bounds[1]);
#endif
	cache.SetCullMode(vk_buffer, state.cull_mode);
	cache.SetFrontFace(vk_buffer, state.front_face);
	if (state.input_assembly) {
		vk_buffer.setPrimitiveTopology(state.topology);
		vk_buffer.setPrimitiveRestartEnable(state.primitive_restart ? VK_TRUE : VK_FALSE);
	}
	cache.SetDepthBiasEnable(vk_buffer, state.depth_bias_enable);
	if (state.depth_bias_enable != VK_FALSE) {
		cache.SetDepthBias(vk_buffer, state.depth_bias[0], state.depth_bias[1],
		                   state.depth_bias[2]);
	}
	cache.SetStencilTestEnable(vk_buffer, state.stencil_test_enable);
	const auto faces = std::array {vk::StencilFaceFlagBits::eFront, vk::StencilFaceFlagBits::eBack};
	for (size_t face = 0; face < faces.size(); face++) {
		const auto& stencil = state.stencil[face];
		cache.SetStencilOp(vk_buffer, faces[face], stencil.fail_op, stencil.pass_op,
		                   stencil.depth_fail_op, stencil.compare_op);
		cache.SetStencilCompareMask(vk_buffer, faces[face], stencil.compare_mask);
		cache.SetStencilWriteMask(vk_buffer, faces[face], stencil.write_mask);
		cache.SetStencilReference(vk_buffer, faces[face], stencil.reference);
	}
#if !defined(__APPLE__)
	if (state.color_write_count != 0) {
		cache.SetColorWriteEnable(vk_buffer, state.color_write_count,
		                          state.color_write_enable.data());
	} else {
		cache.ForgetColorWriteEnable();
	}
#endif
	if (state.feedback_loop_dynamic) {
		cache.SetAttachmentFeedbackLoopEnable(vk_buffer, state.feedback_aspects);
	}
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

// Debug aid for a lost device. With VK_NV_device_diagnostic_checkpoints the driver keeps the
// markers the GPU last passed, so tagging every draw with its pixel shader hash tells us which
// draw was executing when the device died -- something neither GPU-assisted validation nor
// VK_EXT_device_fault reports on NVIDIA. No-op unless KYTY_NV_DIAGNOSTICS enabled the extension.
static void SetGpuCheckpoint(CommandBuffer& buffer, uint64_t marker) {
	const auto& graphics = buffer.GetGraphics();
	if (!graphics.nv_diagnostics_enabled || buffer.IsInvalid()) {
		return;
	}
	static auto* set_checkpoint = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(
	    graphics.device.getProcAddr("vkCmdSetCheckpointNV"));
	if (set_checkpoint == nullptr) {
		return;
	}
	set_checkpoint(buffer.Handle(), reinterpret_cast<const void*>(static_cast<uintptr_t>(marker)));
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;

	// Restores everything a draw reads before writing it. Left as the previous draw wrote them:
	// the RES_MAX arrays of each vertex stage's input info (every reader stops at resources_num,
	// buffers_num and attr_num) and the color_info slots (ResolveRenderColorTarget clears a slot
	// before it is counted, and nothing reads past color_count except slot 0, which is always
	// resolved). All three vertex stages are reset: a draw after a tessellation draw must not
	// see its LS/HS/TES state.
	void Reset() {
		depth_info  = {};
		color_count = 0;
		ps_active   = true;
		for (auto& stage: vertex_info) {
			stage.ResetKeepingArrays();
		}
		ps_input_info = {};
		programs      = {};
	}
};

// A DrawRenderState is tens of kilobytes, nearly all of it arrays that only matter up to their
// counts; value-initialising one per draw showed up as memset in the command processor profile.
// Each thread keeps one and resets only what the next draw reads. Draws never nest, so one per
// thread is enough. Between draws it holds the previous draw's state, which a draw that reuses
// that draw's render state keeps (KeepDrawRenderState()).
static DrawRenderState& DrawRenderStateStorage() {
	thread_local auto state = std::make_unique<DrawRenderState>();
	return *state;
}

static DrawRenderState& AcquireDrawRenderState() {
	auto& state = DrawRenderStateStorage();
	state.Reset();
	return state;
}

// The previous draw's state with only the render-state discovery result kept: the colour and
// depth targets and pixel-stage activity, which depend on registers the draw-state tracker
// vouched for. The stage inputs and programs are recomputed by RefreshShaders(); every vertex
// stage is reset, as in Reset().
static DrawRenderState& KeepDrawRenderState() {
	auto& state = DrawRenderStateStorage();
	for (auto& stage: state.vertex_info) {
		stage.ResetKeepingArrays();
	}
	state.ps_input_info = {};
	state.programs      = {};
	return state;
}

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;
	// Non-null when the GPU reads the counts and offsets above out of guest memory itself.
	const DrawIndirectSource* indirect = nullptr;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// Draw-state reuse (drawReuse.h) is on unless KYTY_DRAW_REUSE=0. Read once.
static bool DrawReuseEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_REUSE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_DRAW_REUSE_ORACLE=1: every draw that reuses state also runs the steps it skipped and
// compares the results (OracleConfirmsRenderState(), OracleConfirmsReusedPipeline()). Read once.
static bool DrawReuseOracleEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_REUSE_ORACLE");
		return DrawReuseEnabled() && value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// A Vulkan handle as an integer, for identity comparisons.
template <typename Handle>
static uint64_t HandleBits(Handle handle) {
	using Native      = typename Handle::CType;
	const auto native = static_cast<Native>(handle);
	if constexpr (std::is_pointer_v<Native>) {
		return reinterpret_cast<uint64_t>(native);
	} else {
		return static_cast<uint64_t>(native);
	}
}

static DrawRecordingPosition CurrentRecordingPosition(const CommandBuffer& buffer) {
	const auto& scheduler = buffer.GetContext().GetCommandScheduler();
	return {
	    .tick                        = scheduler.CurrentTick(),
	    .command_buffer              = HandleBits(buffer.PeekHandle()),
	    .render_pass_epoch           = buffer.RenderPassEpoch(),
	    .rendering                   = buffer.IsRendering(),
	    .dynamic_state_invalidations = buffer.DynamicStateInvalidations(),
	    .deferred_operations         = scheduler.DeferredOperationsRun(),
	};
}

// Read before render-state discovery runs, so `texture_generation` is the generation discovery
// looked images up at.
static DrawReuseInputs CurrentDrawReuseInputs(const CommandBuffer& buffer, bool indexed,
                                              uint32_t index_type_and_size,
                                              uint32_t render_target_slice_offset) {
	const auto& shaders = buffer.GetShaders();
	return {
	    .position                   = CurrentRecordingPosition(buffer),
	    .kept_state                 = &DrawRenderStateStorage(),
	    .texture_generation         = buffer.GetContext().GetTextureCache().Generation(),
	    .es_address                 = shaders.GetVs().es_regs.data_addr,
	    .gs_address                 = shaders.GetVs().gs_regs.data_addr,
	    .ls_address                 = shaders.GetVs().ls_regs.data_addr,
	    .hs_address                 = shaders.GetVs().hs_regs.data_addr,
	    .ps_address                 = shaders.GetPs().ps_regs.data_addr,
	    .render_target_slice_offset = render_target_slice_offset,
	    .indexed                    = indexed,
	    .index_type_and_size        = indexed ? index_type_and_size : 0u,
	};
}

// Draw-state reuse for one draw: what the next draw compares against once this one is fully
// recorded, and the previous draw when this one kept its render state.
struct DrawReuseRequest {
	DrawReuseInputs        inputs;
	const DrawReuseRecord* previous = nullptr;
};

// The programs RefreshShaders() selected for `state`: every vertex stage the draw runs (three for
// a tessellation draw) and the pixel stage when it is active.
static DrawPrograms CurrentDrawPrograms(const DrawRenderState& state) {
	DrawPrograms programs {};
	static_assert(DrawPrograms::MaxVertexStages == std::tuple_size_v<decltype(state.vertex_info)>);
	for (uint32_t i = 0; i < state.programs.VertexStageCount(); i++) {
		programs.vertex_programs[i] = state.vertex_info[i].stage.program;
		programs.vertex_ids[i]      = state.programs.vertex[i].id;
		programs.vertex_modules[i]  = HandleBits(state.programs.vertex[i].module);
	}
	programs.ps_active = state.ps_active;
	if (state.ps_active) {
		programs.ps_program = state.ps_input_info.stage.program;
		programs.ps_id      = state.programs.pixel.id;
		programs.ps_module  = HandleBits(state.programs.pixel.module);
	}
	return programs;
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.guest_width           = std::numeric_limits<uint32_t>::max();
	state.guest_height          = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		const auto guest_extent = target.GuestExtent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.guest_width       = std::min(state.guest_width, guest_extent.width);
		state.guest_height      = std::min(state.guest_height, guest_extent.height);
		if (i == 0) {
			state.scale = target.ScaleFactor();
		}
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	// A stale depth target from an earlier, smaller pass must not shrink the render area for a
	// larger colour pass. Astro Bot's title screen keeps a 1920x1080 depth attachment bound on
	// a 3840x2160 composite pass; the render area would clamp to the 1080p corner and leave the
	// rest of the frame holding whatever was there before ("only a square is cleared"). The
	// guest cannot actually pair mismatched attachment sizes, so treat the depth as unbound.
	// The same applies when the depth target is the larger one: hardware clips each attachment
	// to its own bounds, but a Vulkan pass has a single render area, so pairing a 1024x1024
	// colour target with a 1920x1080 depth attachment writes depth only in the 1024x1024 corner
	// and leaves the rest of it stale. Astro Bot does exactly that -- two colour slots covering
	// one 1024x1024 surface with complementary channel masks, over a full-size depth buffer.
	if (depth.image_id && color_count > 0 &&
	    (depth.Extent().width != state.width || depth.Extent().height != state.height)) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("RenderState: depth target %ux%u does not match colour %ux%u -- unbinding it "
			     "for the draw\n",
			     depth.Extent().width, depth.Extent().height, state.width, state.height);
		}
		depth.image_id = {};
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		uint32_t   htile_fill       = 0;
		bool       htile_fill_known = false;
		const bool meta_cleared =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer,
		                        &htile_fill, &htile_fill_known);
		const bool stencil_compressed = depth.desc.info.metadata.stencil_compressed;
		const bool depth_uniform =
		    meta_cleared && htile_fill_known && !htile_fill_clears_depth(htile_fill) &&
		    htile_fill_depth_uniform(htile_fill, stencil_compressed);
		depth.depth_meta_clear_enable =
		    meta_cleared &&
		    (!htile_fill_known || htile_fill_clears_depth(htile_fill) || depth_uniform);
		depth.stencil_meta_clear_enable = meta_cleared && htile_fill_known && stencil_compressed &&
		                                  htile_fill_clears_stencil(htile_fill);
		if (depth_uniform) {
			depth.depth_clear_value = htile_fill_depth_value(htile_fill, stencil_compressed);
		}
		depth.depth_load_clear_enable = depth.depth_clear_enable || depth.depth_meta_clear_enable;
		if (meta_cleared &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.Extent().width);
		state.height              = std::min(state.height, depth.Extent().height);
		state.guest_width         = std::min(state.guest_width, depth.GuestExtent().width);
		state.guest_height        = std::min(state.guest_height, depth.GuestExtent().height);
		if (color_count == 0) {
			state.scale = depth.ScaleFactor();
		}
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable || depth.stencil_meta_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
		state.guest_width  = state.width;
		state.guest_height = state.height;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static bool DrawHasActivePixelShader(const CommandBuffer& buffer) {
	const auto& ctx              = buffer.GetRegisters();
	const auto& sh_regs          = ctx.GetShaderRegisters();
	const bool  has_color_output = (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	return buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	       (has_color_output || PixelShaderHasDepthOrCoverageSideEffects(sh_regs));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

// Programs that fetch vertices through the page table bound their reads by NUM_RECORDS in the
// shader and take no fixed-function vertex input, so nothing is bound or clamped for them.
static bool UsesGpuVertexFetch(const ShaderVertexInputInfo& info) {
	return info.stage.program != nullptr && info.stage.program->info.gpu_vertex_fetch;
}

// A GCN/RDNA vertex fetch goes through a V#: with OOB_SELECT 0 or 1 an index at or past
// NUM_RECORDS is out of range and reads as zero, so a primitive built from such vertices
// collapses and the geometry engine discards it. Host fixed-function vertex input has no
// range check, so any vertex the guest buffers cannot supply would be fetched from whatever
// guest memory follows the buffer and become a real primitive (a screen-covering wedge when
// w happens to be 0). Bound non-indexed draws to the vertices the strided per-vertex buffers
// actually hold.
// The highest vertex index the bound V#s can supply, or UINT64_MAX when none of them bounds it.
static uint64_t AutoVertexLimit(const ShaderVertexInputInfo& info) {
	uint64_t limit = UINT64_MAX;
	for (int i = 0; i < info.buffers_num; i++) {
		const auto& b = info.buffers[i];
		if (b.fetch_index != 0 || b.stride == 0 || b.num_records == 0 || b.attr_num == 0) {
			continue;
		}
		// OOB_SELECT == 2 only checks NumRecords != 0, so the index itself is unbounded.
		if (info.resources[b.attr_indices[0]].OutOfBounds() == 2) {
			continue;
		}
		limit = std::min<uint64_t>(limit, b.num_records);
	}
	return limit;
}

static uint32_t ClampAutoVertexCount(const ShaderVertexInputInfo& info, uint32_t first_vertex,
                                     uint32_t vertex_count) {
	const uint64_t limit = AutoVertexLimit(info);
	if (limit == UINT64_MAX) {
		return vertex_count;
	}
	const uint64_t available = first_vertex < limit ? limit - first_vertex : 0;
	if (available < vertex_count) {
		static std::atomic<uint32_t> clamp_logs {0};
		if (clamp_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("draw: clamping vertex_count %u to %" PRIu64 " (first_vertex %u, V# num_records %" PRIu64 ")\n",
			     vertex_count, available, first_vertex, limit);
		}
	}
	return static_cast<uint32_t>(std::min<uint64_t>(vertex_count, available));
}

static uint64_t VertexBufferDescriptorSize(const ShaderVertexInputBuffer& buffer,
                                           const ShaderVertexInputInfo& info) {
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < buffer.attr_num; i++) {
		const auto& resource = info.resources[buffer.attr_indices[i]];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? static_cast<uint64_t>(buffer.attr_offsets[i]) +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, range.binding.first->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

enum class PrimitiveRestartMode : uint8_t {
	Disabled,    // no reset index is active for the bound topology and index encoding
	Native,      // the reset index is the maximum value: Vulkan's own primitive restart
	CustomIndex, // a non-maximum reset index, only decidable by scanning the index data
};

// Classifies the guest's primitive-reset state without looking at the index data itself.
static PrimitiveRestartMode ResolvePrimitiveRestartMode(const CommandBuffer& buffer,
                                                        uint32_t guest_element_size) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return PrimitiveRestartMode::Disabled;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return PrimitiveRestartMode::Disabled;
	}

	const auto index_mask  = UINT32_MAX >> ((4 - guest_element_size) * 8);
	const auto reset_index = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return PrimitiveRestartMode::Disabled;
	}
	if ((reset_index & index_mask) == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return PrimitiveRestartMode::Native;
	}
	return PrimitiveRestartMode::CustomIndex;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	switch (ResolvePrimitiveRestartMode(buffer, source.guest_element_size)) {
		case PrimitiveRestartMode::Disabled: return false;
		case PrimitiveRestartMode::Native: return true;
		case PrimitiveRestartMode::CustomIndex: break;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	const auto element_size  = source.guest_element_size;
	const auto index_mask    = UINT32_MAX >> ((4 - element_size) * 8);
	const auto restart_index = buffer.GetRegisters().GetPrimitiveResetIndex() & index_mask;
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// Whether the draw can write colour attachment `target` at all. The colour block writes a slot
// only where CB_TARGET_MASK enables it and the pixel shader exports it: CB_SHADER_MASK carries the
// exported channels and SPI_SHADER_COL_FORMAT a non-zero export format. Without an active pixel
// shader nothing reaches a colour attachment.
static bool DrawWritesColorTarget(const CommandBuffer& buffer, const RenderColorInfo& target) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& sh   = ctx.GetShaderRegisters();
	const auto  slot = target.target_slot;
	return render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
	       render_target_mask_slot(sh.m_cbShaderMask, slot) != 0 &&
	       sh.target_output_mode[slot] != 0;
}

// Shrinks the pass to its first `kept` colour attachments. Discovery already marked every slot a
// target of this draw (BindRenderTarget()); a dropped image that is not also a kept attachment
// must lose that mark, or a texture binding of it in the same draw would be treated as a
// feedback read of an attachment that AcquireRenderTargets() never sets up.
static void DropColorTargets(TextureCache& cache, DrawRenderState& state, uint32_t kept) {
	for (uint32_t i = kept; i < state.color_count; i++) {
		const auto id   = state.color_info[i].image_id;
		bool       live = state.depth_info.image_id == id;
		for (uint32_t k = 0; k < kept && !live; k++) {
			live = state.color_info[k].image_id == id;
		}
		if (!live && id) {
			cache.GetImage(id).binding.is_target = false;
		}
	}
	state.color_count = kept;
}

// A Vulkan render pass has a single render area and the draw a single viewport transform, so
// every attachment of a pass must share one scale. Discovery can produce a mixed group in two ways:
//  - a colour slot the draw never writes is still enabled from an earlier pass (see
//    PassScale::Decide). It is dropped from this pass; its image and its scale are left alone.
//    Denying scale here instead used to make the transient leftover permanent and spread it:
//    Astro's Playroom's reflection-cube pass pushed all five G-buffer targets and the scene depth
//    back to native within a frame, about 6 ms of GPU time per frame from then on.
//  - attachments the draw really writes disagree, for example because one range is not eligible
//    for scaling (smaller than the threshold, or its format cannot be blitted). The whole pass
//    converges onto the native resolution; that denial is permanent, so later frames discover a
//    uniform group.
// DenyImageScale() below bumps TextureCache::Generation() when it denies a new address, so the
// discovery memo's stale color/depth entries simply miss on the next draw; nothing else here
// needs to invalidate them. Dropping a slot depends only on registers and image scales, which the
// draw-reuse inputs and that generation already cover.
void RenderExecutor::UnifyRenderTargetScale(CommandBuffer& buffer, DrawRenderState& state) {
	if (!RenderScale::Enabled()) {
		return;
	}
	bool  mixed         = false;
	bool  have_factor   = false;
	float common_factor = 1.0F;
	const auto observe = [&](float factor) {
		if (!have_factor) {
			have_factor   = true;
			common_factor = factor;
		} else if (factor != common_factor) {
			mixed = true;
		}
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		observe(state.color_info[i].ScaleFactor());
	}
	if (state.depth_info.image_id) {
		observe(state.depth_info.ScaleFactor());
	}
	if (!mixed) {
		return;
	}

	const bool ps_active = DrawHasActivePixelShader(buffer);
	std::array<PassScale::ColorAttachment, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.color_count; i++) {
		colors[i].scale   = state.color_info[i].ScaleFactor();
		colors[i].written = ps_active && DrawWritesColorTarget(buffer, state.color_info[i]);
	}
	std::optional<float> depth_scale;
	if (state.depth_info.image_id) {
		depth_scale = state.depth_info.ScaleFactor();
	}
	const auto decision =
	    PassScale::Decide(std::span {colors.data(), state.color_count}, depth_scale);
	auto& cache = buffer.GetContext().GetTextureCache();
	if (decision.uniform) {
		DropColorTargets(cache, state, decision.color_count);
		return;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		auto& target = state.color_info[i];
		if (target.image_id) {
			target.image_id        = cache.DenyImageScale(target.image_id);
			target.desc.info.scale = 1.0F;
		}
	}
	if (state.depth_info.image_id) {
		state.depth_info.image_id        = cache.DenyImageScale(state.depth_info.image_id);
		state.depth_info.desc.info.scale = 1.0F;
	}
}

// Soft ladder (PPSA21564 only). A companion to the oversized-shader drop in the pipeline
// cache: when Astro Bot's oversized GI pixel / mesh kernel is handed back as a null module,
// its draw has no usable program. Skip that draw for this title so the frame still presents
// (a missing GI contribution renders wrong; a device loss ends the title). Gated on TITLE_ID
// so no other title's draws are ever skipped. Real fix: bind-time dynamic-SRT resolution.
static bool DrawHasDroppedProgram(const DrawRenderState& state) {
	static const bool enabled = [] {
		std::string id;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &id) && id == "PPSA21564";
	}();
	if (!enabled) {
		return false;
	}
	const auto vertex_stages =
	    std::span(state.programs.vertex).first(state.programs.VertexStageCount());
	return std::ranges::any_of(vertex_stages, [](const auto& program) { return !program; }) ||
	       (state.ps_active && !state.programs.pixel);
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if (rt.base.addr != 0 && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	state.ps_active = DrawHasActivePixelShader(buffer);
	RefreshShaders(buffer, draw, state);
	if (DrawHasDroppedProgram(state)) {
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);
	UnifyRenderTargetScale(buffer, state);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=transient size=0x{:x} type={}]", source.size,
		                     static_cast<uint32_t>(source.type));
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

struct PreparedIndirectArgs {
	vk::Buffer     args_buffer  = nullptr;
	vk::DeviceSize args_offset  = 0;
	vk::DeviceSize args_size    = 0;
	vk::Buffer     count_buffer = nullptr;
	vk::DeviceSize count_offset = 0;
	uint32_t       draw_count   = 0;
	uint32_t       stride       = 0;

	[[nodiscard]] bool IsValid() const { return args_buffer != nullptr; }
};

// Number of guest bytes an indirect draw reads for its arguments.
static uint64_t IndirectArgsSize(const DrawIndirectSource& source, bool indexed) {
	const uint64_t record =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	return static_cast<uint64_t>(source.draw_count - 1u) * source.stride + record;
}

// Resolves the guest argument (and optional draw-count) addresses to cached device buffers. Like
// PrepareIndexBuffer this runs before the command buffer is written: obtaining a buffer touches
// guest memory and can finish and restart the scheduler.
static PreparedIndirectArgs PrepareIndirectArgs(CommandBuffer&            buffer,
                                                const DrawIndirectSource& source, bool indexed) {
	PreparedIndirectArgs prepared;
	auto&                cache = buffer.GetContext().GetBufferCache();
	prepared.args_size         = IndirectArgsSize(source, indexed);
	prepared.draw_count        = source.draw_count;
	prepared.stride            = source.stride;
	auto [args_buffer, args_offset] =
	    cache.ObtainBuffer(source.args_addr, prepared.args_size, false);
	prepared.args_buffer = args_buffer->Handle();
	prepared.args_offset = args_offset;
	SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.args_buffer,
	                     "Kyty.IndirectArgs[guest=0x{:016x} size=0x{:x}]", source.args_addr,
	                     prepared.args_size);
	if (source.count_addr != 0) {
		auto [count_buffer, count_offset] =
		    cache.ObtainBuffer(source.count_addr, sizeof(uint32_t), false);
		prepared.count_buffer = count_buffer->Handle();
		prepared.count_offset = count_offset;
	}
	return prepared;
}

// The arguments are produced by earlier GPU work (typically a compute pass writing through a
// buffer device address) or by a cache upload, and the command processor fetches them at
// VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT. Must be recorded outside a render pass instance because
// the source scope covers all commands.
static void EmitIndirectArgsBarrier(vk::CommandBuffer           vk_buffer,
                                    const PreparedIndirectArgs& indirect) {
	std::array<vk::BufferMemoryBarrier, 2> barriers {};
	uint32_t                               count = 0;
	const auto add = [&barriers, &count](vk::Buffer handle, vk::DeviceSize offset,
	                                     vk::DeviceSize size) {
		auto& barrier               = barriers[count++];
		barrier.sType               = vk::StructureType::eBufferMemoryBarrier;
		barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite |
		                        vk::AccessFlagBits::eTransferWrite |
		                        vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = handle;
		barrier.offset              = offset;
		barrier.size                = size;
	};
	add(indirect.args_buffer, indirect.args_offset, indirect.args_size);
	if (indirect.count_buffer != nullptr) {
		add(indirect.count_buffer, indirect.count_offset, sizeof(uint32_t));
	}
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                          vk::PipelineStageFlagBits::eDrawIndirect, vk::DependencyFlags {}, 0,
	                          nullptr, count, barriers.data(), 0, nullptr);
}

// Turns a mesh draw's guest indirect-args block into its mesh draw parameter block and
// VkDrawMeshTasksIndirectCommandEXT entirely on the GPU. Unlike EmitIndirectArgsBarrier(), the
// guest block itself is read by mesh_draw_args.comp rather than fetched by fixed-function indirect
// draw logic, so it needs a shader-read barrier rather than an indirect-command-read one; the two
// blocks the shader writes get an indirect-command-read / shader-read barrier of their own before
// the mesh draw that consumes them. Must be recorded outside a render pass instance, like
// EmitIndirectArgsBarrier().
static void EmitMeshIndirectArgsConversion(vk::CommandBuffer vk_buffer,
                                           const MeshDrawArgsBuilder&   builder,
                                           const PreparedIndirectArgs&  indirect,
                                           const ShaderMeshInputInfo&   mesh,
                                           const DrawIndexBufferSource& index_source,
                                           const vk::PhysicalDeviceMeshShaderPropertiesEXT& limits,
                                           vk::Buffer params_buffer, uint64_t params_offset,
                                           vk::Buffer dispatch_buffer, uint64_t dispatch_offset) {
	vk::BufferMemoryBarrier args_barrier {};
	args_barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite |
	                              vk::AccessFlagBits::eTransferWrite |
	                              vk::AccessFlagBits::eMemoryWrite;
	args_barrier.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
	args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	args_barrier.buffer              = indirect.args_buffer;
	args_barrier.offset              = indirect.args_offset;
	args_barrier.size                = indirect.args_size;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                          vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags {}, 0,
	                          nullptr, 1, &args_barrier, 0, nullptr);

	MeshDrawArgsBuilder::Args args {};
	args.guest_args_buffer    = indirect.args_buffer;
	args.guest_args_offset    = indirect.args_offset;
	args.params_buffer        = params_buffer;
	args.params_offset        = params_offset;
	args.dispatch_buffer      = dispatch_buffer;
	args.dispatch_offset      = dispatch_offset;
	args.primitive_size       = mesh.InputPrimitiveSize();
	args.primitive_step       = mesh.InputPrimitiveStep();
	args.primitives_per_group = mesh.primitives_per_group;
	args.element_size         = index_source.guest_element_size;
	args.index_base           = index_source.address;
	args.max_groups_x         = limits.maxMeshWorkGroupCount[0];
	args.max_groups_y         = limits.maxMeshWorkGroupCount[1];
	args.max_groups_total     = limits.maxMeshWorkGroupTotalCount;
	builder.Record(vk_buffer, args);

	std::array<vk::BufferMemoryBarrier, 2> out_barriers {};
	const auto add_out_barrier = [&out_barriers](uint32_t index, vk::Buffer buffer_handle,
	                                             uint64_t offset, uint64_t size) {
		auto& barrier               = out_barriers[index];
		barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead |
		                        vk::AccessFlagBits::eShaderRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = buffer_handle;
		barrier.offset              = offset;
		barrier.size                = size;
	};
	add_out_barrier(0, params_buffer, params_offset,
	                MeshDrawArgsBuilder::ParamsDwordCount * sizeof(uint32_t));
	add_out_barrier(1, dispatch_buffer, dispatch_offset,
	                MeshDrawArgsBuilder::DispatchDwordCount * sizeof(uint32_t));
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                          vk::PipelineStageFlagBits::eDrawIndirect |
	                              vk::PipelineStageFlagBits::eMeshShaderEXT,
	                          vk::DependencyFlags {}, 0, nullptr,
	                          static_cast<uint32_t>(out_barriers.size()), out_barriers.data(), 0,
	                          nullptr);
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

// The primitive types EmitDrawPrimitives() records as one host draw. The others expand into
// several host draws sized by counts the GPU has not read yet, so they cannot be issued
// indirectly.
static bool IsSingleHostDrawPrimitive(Prospero::PrimitiveType type) {
	switch (type) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch: return true;
		default: return false;
	}
}

// Records the draw once its arguments have been resolved to device buffers. Nothing here depends
// on the draw counts: the guest's argument block is consumed in place.
static void EmitIndirectPrimitives(vk::CommandBuffer vk_buffer, const DrawCallInfo& draw,
                                   const PreparedIndirectArgs& indirect) {
	if (indirect.count_buffer != nullptr) {
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirectCount(indirect.args_buffer, indirect.args_offset,
			                                   indirect.count_buffer, indirect.count_offset,
			                                   indirect.draw_count, indirect.stride);
		} else {
			vk_buffer.drawIndirectCount(indirect.args_buffer, indirect.args_offset,
			                            indirect.count_buffer, indirect.count_offset,
			                            indirect.draw_count, indirect.stride);
		}
		return;
	}
	if (draw.IsIndexed()) {
		vk_buffer.drawIndexedIndirect(indirect.args_buffer, indirect.args_offset,
		                              indirect.draw_count, indirect.stride);
	} else {
		vk_buffer.drawIndirect(indirect.args_buffer, indirect.args_offset, indirect.draw_count,
		                       indirect.stride);
	}
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const ShaderVertexInputInfo& vs_input_info, const DrawCallInfo& draw,
                               const DrawEmitInfo& emit, const PreparedIndirectArgs& indirect) {
	// Every other topology expands into several host draws using counts the GPU has not read
	// yet, so SupportsIndirectDraw() never lets an indirect draw reach them.
	EXIT_IF(indirect.IsValid() && !IsSingleHostDrawPrimitive(ucfg.GetPrimType()));
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (indirect.IsValid()) {
				EmitIndirectPrimitives(vk_buffer, draw, indirect);
			} else if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				const auto vertex_count =
				    UsesGpuVertexFetch(vs_input_info)
				        ? draw.index_count
				        : ClampAutoVertexCount(vs_input_info, emit.first_vertex, draw.index_count);
				if (vertex_count == 0) {
					break;
				}
				vk_buffer.draw(vertex_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

// What PipelineCache::GetGraphicsPipeline() and ResolveGraphicsDynamicState() take from the draw
// rather than from the registers or the render targets; see DrawPipelineFeed. The vertex-input
// half is built exactly as GetGraphicsPipeline() builds GraphicsPipelineKey::vertex_input, from
// the first vertex stage.
static void BuildDrawPipelineFeed(const DrawRenderState& state, vk::PrimitiveTopology topology,
                                  bool primitive_restart_enable, DrawPipelineFeed& feed) {
	const auto& vs         = state.vertex_info[0];
	feed.programs          = CurrentDrawPrograms(state);
	feed.topology          = static_cast<uint32_t>(topology);
	feed.primitive_restart = primitive_restart_enable;
	if (state.ps_active) {
		feed.ps_sample_shading = state.ps_input_info.ps_sample_shading;
	}
	if (vs.stage.program == nullptr || vs.stage.program->stage == ShaderType::Mesh ||
	    vs.stage.program->info.gpu_vertex_fetch || vs.buffers_num < 0 ||
	    vs.buffers_num > ShaderVertexInputInfo::RES_MAX || vs.resources_num < 0 ||
	    vs.resources_num > ShaderVertexInputInfo::RES_MAX) {
		// No fixed-function vertex input, or one GetGraphicsPipeline() rejects outright.
		return;
	}
	auto& input           = feed.vertex_input;
	input.binding_count   = static_cast<uint8_t>(vs.buffers_num);
	input.attribute_count = static_cast<uint8_t>(vs.resources_num);
	for (int binding = 0; binding < vs.buffers_num; binding++) {
		const auto& buffer      = vs.buffers[binding];
		input.bindings[binding] = {.stride = buffer.stride, .instance = buffer.fetch_index != 0};
		for (int attribute = 0;
		     attribute < buffer.attr_num && attribute < ShaderVertexInputBuffer::ATTR_MAX;
		     attribute++) {
			const auto index = buffer.attr_indices[attribute];
			if (index >= 0 && index < vs.resources_num) {
				input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
	}
}

// Whether a draw whose target acquisition resolves to `expected` may keep recording into the pass
// begun with `current`. They must agree on everything except that the open pass may have begun by
// clearing an attachment the draw would load, which leaves the same contents (see
// TextureCache::TargetAcquisitionRepeats()).
static bool RenderPassContinues(const RenderState& current, RenderState expected) {
	for (uint32_t i = 0; i < RENDER_COLOR_ATTACHMENTS_MAX; i++) {
		const auto& open  = current.color_attachments[i];
		auto&       wants = expected.color_attachments[i];
		if (open.is_clear && !wants.is_clear) {
			wants.is_clear    = true;
			wants.clear_value = open.clear_value;
		}
	}
	const auto& open  = current.depth_stencil_attachment;
	auto&       wants = expected.depth_stencil_attachment;
	if (open.depth_clear && !wants.depth_clear) {
		wants.depth_clear    = true;
		wants.clear_value[0] = open.clear_value[0];
	}
	if (open.stencil_clear && !wants.stencil_clear) {
		wants.stencil_clear  = true;
		wants.clear_value[1] = open.clear_value[1];
	}
	return current == expected;
}

bool RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
                                         bool primitive_restart_enable,
                                         const DrawReuseRequest* reuse) {
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	if (draw.indirect != nullptr && !draw.IsIndexed() && !UsesGpuVertexFetch(state.vertex_info[0]) &&
	    AutoVertexLimit(state.vertex_info[0]) != UINT64_MAX) {
		// The vertex count comes out of guest memory, so ClampAutoVertexCount() cannot bound it
		// to what the V#s hold. Rather than let the GPU fetch vertices past NUM_RECORDS, hand the
		// draw to the host path, which reads the count and clamps it. The bound depends on the
		// descriptors rather than the program, so it is re-checked for every draw.
		return false;
	}
	if (mesh_active && draw.indirect != nullptr && !draw.IsIndexed()) {
		// mesh_draw_args.comp assumes the guest argument record is a DrawIndexedIndirectArgs block
		// (index_count, instance_count, start_index, base_vertex, start_instance); a non-indexed mesh
		// draw's guest block is the smaller DrawIndirectArgs one, and converting it as though it were
		// the larger layout would read past it and misinterpret every field after the first two. A
		// mesh draw without real indices is rare enough that this just falls back to the host-read
		// path instead of teaching the shader a second layout.
		return false;
	}
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		// draw.index_count and draw.instance_count are only real counts for a direct draw; an
		// indirect draw's are read out of the guest argument block by mesh_draw_args.comp, on the
		// GPU, so neither the zero-count shortcut nor the host workgroup-limit clamp below can run
		// for it. The shader clamps to the same limits itself (see MeshDrawArgsBuilder::Args).
		if (draw.indirect == nullptr) {
			const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
			if (primitives == 0 || draw.instance_count == 0) {
				return true;
			}
			mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
			const auto& limits = m_context.GetGraphics().mesh_shader_properties;
			if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
			    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
			    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
			        limits.maxMeshWorkGroupTotalCount) {
				EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
				     draw.instance_count);
			}
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		if (!UsesGpuVertexFetch(state.vertex_info[0])) {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		}
		index_binding = PrepareIndexBuffer(buffer, index_source);
	}
	PreparedIndirectArgs indirect_binding;
	if (draw.indirect != nullptr) {
		indirect_binding = PrepareIndirectArgs(buffer, *draw.indirect, draw.IsIndexed());
	}
	static_assert(MeshDrawArgsBuilder::ParamsDwordCount ==
	              ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
	uint64_t mesh_draw_block_address = 0;
	uint64_t mesh_params_offset      = 0;
	uint64_t mesh_dispatch_offset    = 0;
	if (mesh_active) {
		if (draw.indirect != nullptr) {
			// mesh_draw_args.comp fills both blocks, not the host, so they come from the DeviceLocal
			// utility buffer rather than the Stream one below. Reserving still has to happen here,
			// alongside PrepareIndirectArgs, because it can wait on a previous reservation's GPU tick
			// and, like Stream's Copy(), that can finish and restart the scheduler.
			auto& device_buffer =
			    buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::DeviceLocal);
			const auto alignment =
			    std::max<uint64_t>(buffer.GetGraphics().StorageMinAlignment(), sizeof(uint32_t));
			mesh_params_offset = device_buffer.Reserve(
			    MeshDrawArgsBuilder::ParamsDwordCount * sizeof(uint32_t), alignment);
			mesh_dispatch_offset = device_buffer.Reserve(
			    MeshDrawArgsBuilder::DispatchDwordCount * sizeof(uint32_t), alignment);
			mesh_draw_block_address = device_buffer.BufferDeviceAddress() + mesh_params_offset;
		} else {
			const uint32_t draw_data[] {
			    draw.index_count,
			    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
			    emit.first_instance, index_source.guest_element_size,
			    static_cast<uint32_t>(index_source.address),
			    static_cast<uint32_t>(index_source.address >> 32u)};
			static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
			// Uploading the parameter block writes to host-visible memory and can finish and restart
			// the scheduler, so it has to happen alongside PrepareIndexBuffer/PrepareIndirectArgs,
			// before the point below where the command buffer may no longer touch guest memory. Only
			// the resulting device address -- not the block itself -- is pushed at that point.
			auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
			const auto offset = stream.Copy(draw_data, sizeof(draw_data), 16);
			mesh_draw_block_address = stream.BufferDeviceAddress() + offset;
		}
	}
	// Only a direct, non-mesh draw with guest-memory indices is recorded for the next draw to
	// reuse; mesh draws and host-expanded indices take the full path every time.
	const bool reuse_tracked = reuse != nullptr && !mesh_active && draw.indirect == nullptr &&
	                           index_source.host_data == nullptr;
	DrawPipelineFeed feed {};
	if (reuse_tracked) {
		BuildDrawPipelineFeed(state, topology, primitive_restart_enable, feed);
	}
	// Non-null when this draw keeps the previous draw's render targets, pipeline, dynamic state
	// and render pass, and records only its bindings and the draw itself.
	PipelineCache::Pipeline* reused_pipeline = nullptr;
	if (reuse_tracked && reuse->previous != nullptr) {
		reused_pipeline = ReusableDrawPipeline(buffer, state, *reuse->previous, feed);
	}
	RenderState              rendering {};
	vk::ImageAspectFlags     feedback_aspects;
	PipelineCache::Pipeline* pipeline_found = reused_pipeline;
	// AcquireRenderTargets() unbinds a depth target whose extent differs from the colour targets'
	// after the pipeline was looked up with it. The state then no longer holds what the lookup
	// saw, so such a draw is not recorded for the next one to keep.
	bool depth_dropped = false;
	if (reused_pipeline == nullptr || DrawReuseOracleEnabled()) {
		if (draw.IsIndexed()) {
			LogDrawPhase(draw.Name(), "CreatePipeline");
		}
		pipeline_found = &m_context.GetPipelineCache().GetGraphicsPipeline(
		    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages,
		    buffer, state.ps_active ? &state.ps_input_info : nullptr, topology,
		    primitive_restart_enable, state.programs);
		const bool had_depth = static_cast<bool>(state.depth_info.image_id);
		rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
		                                 state.depth_info, feedback_aspects, stages);
		depth_dropped = had_depth && !state.depth_info.image_id;
		if (reused_pipeline != nullptr &&
		    !OracleConfirmsReusedPipeline(buffer, state, *reuse->previous, rendering,
		                                  feedback_aspects, *pipeline_found, *reused_pipeline,
		                                  topology, primitive_restart_enable)) {
			reused_pipeline = nullptr;
		}
	}
	auto& pipeline = *pipeline_found;

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (mesh_active) {
		const std::array<uint32_t, ShaderRecompiler::IR::PushData::MeshDrawAddressDwordCount>
		    address_data {static_cast<uint32_t>(mesh_draw_block_address),
		                  static_cast<uint32_t>(mesh_draw_block_address >> 32u)};
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0,
		                        ShaderRecompiler::IR::PushData::MeshDrawAddressDwordCount *
		                            sizeof(uint32_t),
		                        address_data.data());
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	if (reused_pipeline == nullptr) {
		GraphicsDynamicState dynamic_state {};
		ResolveGraphicsDynamicState(buffer, vertex_stages.back(), state.depth_info, rendering,
		                            feedback_aspects, topology, primitive_restart_enable,
		                            mesh_active, dynamic_state);
		SetGraphicsDynamicParams(buffer, vk_buffer, dynamic_state);
		if (DrawReuseOracleEnabled()) {
			m_draw_reuse_oracle.recorded_dynamic_state       = dynamic_state;
			m_draw_reuse_oracle.recorded_dynamic_state_known = true;
		}
	}

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	if (indirect_binding.IsValid()) {
		// A buffer barrier may not name VK_PIPELINE_STAGE_ALL_COMMANDS_BIT inside a render pass
		// instance, so close the current one before making the arguments visible.
		m_context.GetCommandScheduler().EndRendering();
		if (mesh_active) {
			auto& device_buffer =
			    buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::DeviceLocal);
			EmitMeshIndirectArgsConversion(
			    vk_buffer, m_mesh_draw_args_builder, indirect_binding, state.vertex_info[0].mesh,
			    index_source, m_context.GetGraphics().mesh_shader_properties, device_buffer.Handle(),
			    mesh_params_offset, device_buffer.Handle(), mesh_dispatch_offset);
		} else {
			EmitIndirectArgsBarrier(vk_buffer, indirect_binding);
		}
	}
	if (reused_pipeline == nullptr) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	SetGpuCheckpoint(buffer, state.ps_input_info.stage.program != nullptr
	                             ? state.ps_input_info.stage.program->shader_hash
	                             : 0u);
	if (mesh_active) {
		if (draw.indirect != nullptr) {
			auto& device_buffer =
			    buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::DeviceLocal);
			vk_buffer.drawMeshTasksIndirectEXT(device_buffer.Handle(), mesh_dispatch_offset, 1, 0);
		} else {
			vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
		}
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, state.vertex_info[0], draw, emit, indirect_binding);
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
	// A draw whose shaders write memory ended the pass above; the next draw begins a new one.
	if (reuse_tracked && !depth_dropped && buffer.IsRendering()) {
		auto inputs     = reuse->inputs;
		inputs.position = CurrentRecordingPosition(buffer);
		m_draw_reuse.Store(inputs, feed, &pipeline);
	}
	return true;
}

// Resolves the guest index encoding to the host index type and the guest element size.
static void ResolveIndexEncoding(uint32_t index_type_and_size, DrawIndexBufferSource& source) {
	switch (static_cast<Prospero::IndexType>(index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			source.type               = vk::IndexType::eUint16;
			source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			source.type               = vk::IndexType::eUint32;
			source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			// Vulkan has no 8-bit index type here: the caller widens the data to 16 bits.
			source.type               = vk::IndexType::eUint16;
			source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", index_type_and_size);
	}
}

bool RenderExecutor::KeepDrawTargets(const DrawRenderState& state) {
	auto&      cache = m_context.GetTextureCache();
	const auto live  = [&cache](ImageId id) {
		const auto* image = cache.m_slot_images.try_get(id);
		return image != nullptr && image->registered && !image->binding.needs_rebind;
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!live(state.color_info[i].image_id)) {
			return false;
		}
	}
	if (state.depth_info.image_id && !live(state.depth_info.image_id)) {
		return false;
	}
	// What discovery still does per draw when its memo hits (ResolveRenderColorTarget(),
	// ResolveRenderDepthTarget()): the LRU touch and access tick, and marking the image a target
	// of this draw, so a texture-cache change during resource preparation flags it for rebinding.
	for (uint32_t i = 0; i < state.color_count; i++) {
		cache.NoteImageReuse(state.color_info[i].image_id);
		BindRenderTarget(state.color_info[i].image_id);
	}
	if (state.depth_info.image_id) {
		cache.NoteImageReuse(state.depth_info.image_id);
		BindRenderTarget(state.depth_info.image_id);
	}
	return true;
}

// Keeping the previous draw's state skips the target half of PrepareDrawRenderState(), which
// beyond resolving the targets from the registers can
//  - skip a draw with no target and no active pixel shader,
//  - look an image up afresh on a memo miss, or deny an image its render scale.
// None of these can happen to a kept draw: the registers are unchanged, so the previous draw --
// which was recorded, so was not skipped -- saw the same answers; an image lookup answers the same
// way while the texture-cache generation DrawReuseInputs compares is unchanged, and a scale denial
// moves that generation. The colour slots discovery resolves are the pixel program's MRT outputs,
// so the programs are selected again (RefreshShaders()) and must be the previous draw's; the same
// programs were not dropped for the previous draw either (DrawHasDroppedProgram()). What discovery
// does per draw on a memo hit is replayed by KeepDrawTargets().
DrawRenderState& RenderExecutor::SelectDrawRenderState(bool& keep, CommandBuffer& buffer,
                                                       const DrawCallInfo&    draw,
                                                       uint32_t               render_target_slice_offset,
                                                       const DrawReuseRecord& previous) {
	if (keep) {
		auto& state = KeepDrawRenderState();
		RefreshShaders(buffer, draw, state);
		if (previous.KeepsPrograms(CurrentDrawPrograms(state)) && KeepDrawTargets(state) &&
		    (!DrawReuseOracleEnabled() ||
		     OracleConfirmsRenderState(buffer, draw, render_target_slice_offset, state))) {
			return state;
		}
		keep = false;
	}
	return AcquireDrawRenderState();
}

PipelineCache::Pipeline* RenderExecutor::ReusableDrawPipeline(const CommandBuffer&    buffer,
                                                              const DrawRenderState&  state,
                                                              const DrawReuseRecord&  previous,
                                                              const DrawPipelineFeed& feed) {
	// Pixel-stage images resolved afresh this draw may include one of the targets, whose
	// attachment layout and feedback handling then have to be decided again.
	if (state.ps_active && !m_pixel_bindings_reused) {
		return nullptr;
	}
	auto& cache = m_context.GetTextureCache();
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color = state.color_info[i];
		if (!cache.TargetAcquisitionRepeats(color.image_id, color.desc)) {
			return nullptr;
		}
	}
	if (state.depth_info.image_id &&
	    !cache.TargetAcquisitionRepeats(state.depth_info.image_id, state.depth_info.desc)) {
		return nullptr;
	}
	// Resource preparation may have ended the pass (an upload, a CommandBuffer::RequestFullBarrier
	// from a queued flush) or restarted the command buffer since the draw started.
	return previous.ReusablePipeline(CurrentRecordingPosition(buffer), feed);
}

void RenderExecutor::LogDrawReuseOracleSummary() {
	if (m_draw_reuse_oracle.SummaryDue()) {
		LOGF("DRAWREUSE-ORACLE checked=%" PRIu64 " mismatches=%" PRIu64 "\n",
		     m_draw_reuse_oracle.Checked(), m_draw_reuse_oracle.Mismatches());
	}
}

bool RenderExecutor::OracleConfirmsRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                               uint32_t               render_target_slice_offset,
                                               const DrawRenderState& kept) {
	// A second per-thread state, like DrawRenderStateStorage(), so the kept one stays untouched.
	thread_local auto scratch_storage = std::make_unique<DrawRenderState>();
	auto&             scratch         = *scratch_storage;
	scratch.Reset();
	m_draw_reuse_oracle.NoteChecked();
	const bool prepared = PrepareDrawRenderState(buffer, draw, render_target_slice_offset, scratch);

	const auto same_view = [](const ImageViewInfo& a, const ImageViewInfo& b) {
		return a.base_level == b.base_level && a.level_count == b.level_count &&
		       a.base_layer == b.base_layer && a.layer_count == b.layer_count &&
		       a.format == b.format;
	};
	const auto same_extent = [](vk::Extent2D a, vk::Extent2D b) {
		return a.width == b.width && a.height == b.height;
	};
	char what[192] {};
	if (!prepared) {
		std::snprintf(what, sizeof(what), "discovery would skip the draw");
	} else if (scratch.ps_active != kept.ps_active) {
		std::snprintf(what, sizeof(what), "ps_active %d, kept %d", scratch.ps_active ? 1 : 0,
		              kept.ps_active ? 1 : 0);
	} else if (scratch.color_count != kept.color_count) {
		std::snprintf(what, sizeof(what), "color_count %u, kept %u", scratch.color_count,
		              kept.color_count);
	}
	for (uint32_t i = 0; what[0] == 0 && i < kept.color_count; i++) {
		const auto& a = scratch.color_info[i];
		const auto& b = kept.color_info[i];
		if (a.image_id != b.image_id || !same_view(a.desc.view_info, b.desc.view_info) ||
		    !same_extent(a.Extent(), b.Extent()) || a.target_slot != b.target_slot ||
		    a.export_mapping.packed != b.export_mapping.packed) {
			std::snprintf(what, sizeof(what),
			              "color %u: image %u/%u slot %u, kept image %u/%u slot %u", i,
			              a.image_id.index, a.image_id.generation, a.target_slot,
			              b.image_id.index, b.image_id.generation, b.target_slot);
		}
	}
	if (what[0] == 0) {
		const auto& a = scratch.depth_info;
		const auto& b = kept.depth_info;
		// A draw whose depth target AcquireRenderTargets() dropped is never kept (see
		// ExecutePreparedDraw()), so the kept depth target is discovery's.
		const bool same_image =
		    a.image_id == b.image_id &&
		    (!a.image_id ||
		     (same_view(a.desc.view_info, b.desc.view_info) && same_extent(a.Extent(), b.Extent())));
		if (!same_image) {
			std::snprintf(what, sizeof(what), "depth: image %u/%u, kept image %u/%u",
			              a.image_id.index, a.image_id.generation, b.image_id.index,
			              b.image_id.generation);
		} else if (a.depth_test_enable != b.depth_test_enable ||
		           a.depth_write_enable != b.depth_write_enable ||
		           a.depth_compare_op != b.depth_compare_op ||
		           a.depth_bounds_test_enable != b.depth_bounds_test_enable ||
		           a.depth_clear_enable != b.depth_clear_enable ||
		           a.stencil_test_enable != b.stencil_test_enable ||
		           a.stencil_clear_enable != b.stencil_clear_enable) {
			std::snprintf(what, sizeof(what), "depth/stencil register state differs");
		}
	}
	if (what[0] != 0 && m_draw_reuse_oracle.NoteMismatch()) {
		LOGF("DRAWREUSE-ORACLE mismatch: %s: kept render targets: %s\n", draw.Name(), what);
	}
	return what[0] == 0;
}

bool RenderExecutor::OracleConfirmsReusedPipeline(
    const CommandBuffer& buffer, const DrawRenderState& state, const DrawReuseRecord& previous,
    const RenderState& rendering, vk::ImageAspectFlags feedback_aspects,
    const PipelineCache::Pipeline& expected, const PipelineCache::Pipeline& reused,
    vk::PrimitiveTopology topology, bool primitive_restart_enable) {
	m_draw_reuse_oracle.NoteChecked();
	const auto& current = buffer.CurrentRenderState();
	char        what[192] {};
	if (!(CurrentRecordingPosition(buffer) == previous.Inputs().position)) {
		std::snprintf(what, sizeof(what),
		              "acquiring the targets ended the pass or restarted the buffer");
	} else if (!RenderPassContinues(current, rendering)) {
		std::snprintf(what, sizeof(what),
		              "render pass: %ux%ux%u colors=%u depth_layout=%u depth_clear=%d, open "
		              "%ux%ux%u colors=%u depth_layout=%u depth_clear=%d",
		              rendering.width, rendering.height, rendering.num_layers,
		              rendering.num_color_attachments,
		              static_cast<uint32_t>(rendering.depth_stencil_attachment.image_layout),
		              rendering.depth_stencil_attachment.depth_clear ? 1 : 0, current.width,
		              current.height, current.num_layers, current.num_color_attachments,
		              static_cast<uint32_t>(current.depth_stencil_attachment.image_layout),
		              current.depth_stencil_attachment.depth_clear ? 1 : 0);
	} else if (&expected != &reused) {
		std::snprintf(what, sizeof(what), "pipeline %p, kept %p (vs=%016" PRIx64 " ps=%016" PRIx64 ")",
		              static_cast<const void*>(&expected), static_cast<const void*>(&reused),
		              state.programs.vertex[0].id, state.ps_active ? state.programs.pixel.id : 0u);
	} else {
		GraphicsDynamicState dynamic_state {};
		const auto& last_vertex_stage = state.vertex_info[state.programs.VertexStageCount() - 1];
		ResolveGraphicsDynamicState(buffer, last_vertex_stage, state.depth_info, rendering,
		                            feedback_aspects, topology, primitive_restart_enable, false,
		                            dynamic_state);
		if (!m_draw_reuse_oracle.recorded_dynamic_state_known ||
		    !(dynamic_state == m_draw_reuse_oracle.recorded_dynamic_state)) {
			std::snprintf(what, sizeof(what), "dynamic state differs from the recorded one");
		}
	}
	if (what[0] != 0 && m_draw_reuse_oracle.NoteMismatch()) {
		LOGF("DRAWREUSE-ORACLE mismatch: kept pipeline and render pass: %s\n", what);
	}
	return what[0] == 0;
}

const char* IndirectDrawSupportName(IndirectDrawSupport support) {
	switch (support) {
		case IndirectDrawSupport::Supported: return "supported";
		case IndirectDrawSupport::DeviceFeature: return "device feature";
		case IndirectDrawSupport::Topology: return "topology";
		case IndirectDrawSupport::IndexEncoding: return "index encoding";
		case IndirectDrawSupport::PrimitiveRestart: return "primitive restart";
		case IndirectDrawSupport::ArgumentsNotCached: return "arguments not cached";
		case IndirectDrawSupport::IndexRangeUnknown: return "index range unknown";
		case IndirectDrawSupport::RendererRefused: return "renderer refused";
	}
	EXIT("unknown indirect-draw support value: %u\n", static_cast<uint32_t>(support));
}

IndirectDrawSupport RenderExecutor::SupportsIndirectDraw(CommandBuffer&            buffer,
                                                         const DrawIndirectSource& source,
                                                         bool                      indexed,
                                                         uint32_t index_type_and_size) {
	const auto& graphics = m_context.GetGraphics();
	// The argument blocks always carry a start-instance; a host that cannot honour it would
	// silently draw the wrong instances.
	if (source.draw_count == 0 || source.draw_count > graphics.max_draw_indirect_count ||
	    (source.draw_count > 1 && !graphics.multi_draw_indirect_enabled) ||
	    (source.count_addr != 0 && !graphics.draw_indirect_count_enabled) ||
	    !graphics.draw_indirect_first_instance_enabled) {
		return IndirectDrawSupport::DeviceFeature;
	}

	auto&                 ucfg     = buffer.GetUserConfig();
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology) || !IsSingleHostDrawPrimitive(ucfg.GetPrimType())) {
		return IndirectDrawSupport::Topology;
	}

	if (indexed) {
		DrawIndexBufferSource encoding {};
		ResolveIndexEncoding(index_type_and_size, encoding);
		if (encoding.guest_element_size == 1) {
			return IndirectDrawSupport::IndexEncoding;
		}
		if (ResolvePrimitiveRestartMode(buffer, encoding.guest_element_size) ==
		    PrimitiveRestartMode::CustomIndex) {
			return IndirectDrawSupport::PrimitiveRestart;
		}
	}

	// The arguments have to reach the GPU without pulling new guest memory into the cache:
	// registering the range would put a write fault on it every time the guest rebuilds the
	// block, and reading it back is exactly the stall this path removes.
	auto& cache = m_context.GetBufferCache();
	if (!cache.IsRegionObtainableWithoutRegistering(source.args_addr,
	                                                IndirectArgsSize(source, indexed)) ||
	    (source.count_addr != 0 &&
	     !cache.IsRegionObtainableWithoutRegistering(source.count_addr, sizeof(uint32_t)))) {
		return IndirectDrawSupport::ArgumentsNotCached;
	}
	return IndirectDrawSupport::Supported;
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	const bool pm4_state_unchanged = m_draw_state_tracker.TakeDrawVerdict();
	// An indirect draw carries no host-side counts: `index_count` is the bound index range and
	// `instance_count` is unused.
	if (args.indirect == nullptr && (args.index_count == 0 || args.instance_count == 0)) {
		return true;
	}
	// Whatever happens to this draw, the next one may only reuse what it fully recorded.
	const auto       previous_draw = m_draw_reuse.Take();
	if (DrawReuseOracleEnabled()) {
		LogDrawReuseOracleSummary();
	}
	DrawReuseRequest reuse {};
	const bool       reuse_tracked = DrawReuseEnabled() && args.indirect == nullptr &&
	                           args.offset_source == DrawOffsetSource::DrawState;
	if (reuse_tracked) {
		reuse.inputs = CurrentDrawReuseInputs(buffer, true, args.index_type_and_size,
		                                      args.render_target_slice_offset);
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	ResolveIndexEncoding(args.index_type_and_size, index_source);
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance, args.indirect};
	// 8-bit indices are widened on the host into a transient buffer every draw; such a draw is
	// never recorded for reuse, and never reuses.
	bool keep_render_state = reuse_tracked && pm4_state_unchanged &&
	                         index_source.guest_element_size != 1 &&
	                         previous_draw.KeepsRenderState(reuse.inputs);
	auto& state = SelectDrawRenderState(keep_render_state, buffer, draw,
	                                    args.render_target_slice_offset, previous_draw);
	if (!keep_render_state &&
	    !PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;

	reuse.previous      = keep_render_state ? &previous_draw : nullptr;
	const bool recorded = ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit,
	                                          index_source, primitive_restart,
	                                          reuse_tracked ? &reuse : nullptr);
	ResetBindings();
	return recorded;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	const bool pm4_state_unchanged = m_draw_state_tracker.TakeDrawVerdict();
	// An indirect draw carries no host-side counts; they come from the argument block.
	if (args.indirect == nullptr && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}
	// Whatever happens to this draw, the next one may only reuse what it fully recorded.
	const auto       previous_draw = m_draw_reuse.Take();
	if (DrawReuseOracleEnabled()) {
		LogDrawReuseOracleSummary();
	}
	DrawReuseRequest reuse {};
	const bool       reuse_tracked = DrawReuseEnabled() && args.indirect == nullptr &&
	                           args.offset_source == DrawOffsetSource::DrawState;
	if (reuse_tracked) {
		reuse.inputs = CurrentDrawReuseInputs(buffer, false, 0, args.render_target_slice_offset);
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto, args.vertex_count,
	                         args.instance_count, args.first_instance, args.indirect};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return true;
	}
	bool keep_render_state =
	    reuse_tracked && pm4_state_unchanged && previous_draw.KeepsRenderState(reuse.inputs);
	auto& state = SelectDrawRenderState(keep_render_state, buffer, draw,
	                                    args.render_target_slice_offset, previous_draw);
	if (!keep_render_state &&
	    !PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;

	DrawIndexBufferSource index_source {};
	reuse.previous      = keep_render_state ? &previous_draw : nullptr;
	const bool recorded = ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit,
	                                          index_source, false, reuse_tracked ? &reuse : nullptr);
	ResetBindings();
	return recorded;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
