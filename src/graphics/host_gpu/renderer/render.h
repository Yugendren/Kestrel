#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
// Discovery-memo key types below need the complete HW::RenderTarget / HW::DepthRenderTarget
// register structs (and their operator==), plus RenderColorInfo, as the memoized value types;
// a forward declaration is no longer enough once a GenerationMemo<...> of them is a member.
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/meshDrawArgs.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/dynamicState.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// A draw whose parameters the GPU fetches from guest memory instead of the command processor
// reading them back. `args_addr` points at tightly packed VkDrawIndirectCommand /
// VkDrawIndexedIndirectCommand records -- the guest's DrawIndirectArgs and DrawIndexedIndirectArgs
// blocks have exactly those layouts. When `count_addr` is non-zero the GPU reads the draw count
// from there as well and `draw_count` is only an upper bound.
struct DrawIndirectSource {
	uint64_t args_addr  = 0;
	uint64_t count_addr = 0;
	uint32_t draw_count = 0;
	uint32_t stride     = 0;
};

// Why a draw cannot keep its arguments on the GPU. Anything other than Supported sends the draw
// down the command processor's host-read path, which is correct but stalls on a GPU drain.
enum class IndirectDrawSupport : uint8_t {
	Supported,
	DeviceFeature,       // the host cannot express this draw count or a GPU-resident draw count
	Topology,            // legacy rect / quad lists expand into host draws sized by the counts
	IndexEncoding,       // 8-bit indices are widened on the host from a host-known count
	PrimitiveRestart,    // a custom reset index has to be looked for in the index data
	ArgumentsNotCached,  // the argument block is not backed by a cached device buffer
	IndexRangeUnknown,   // INDEX_BUFFER_SIZE gives no range for the GPU to index into
	// Only decidable once the shaders and their descriptors are resolved: a non-indexed draw whose
	// vertex count has to be clamped to what the V#s can supply.
	RendererRefused,
};

[[nodiscard]] const char* IndirectDrawSupportName(IndirectDrawSupport support);

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Non-null when the GPU reads the draw parameters itself. `index_addr` / `index_count` then
	// describe the whole index range to bind, and `instance_count`, `base_vertex` and
	// `first_instance` are unused: those come from the argument block.
	const DrawIndirectSource* indirect = nullptr;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Non-null when the GPU reads the draw parameters itself; every count and offset above is
	// then unused because they come from the argument block.
	const DrawIndirectSource* indirect = nullptr;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer  Handle() const;
	[[nodiscard]] GraphicContext&    GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&     GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&       GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&    GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&        GetShaders() const noexcept { return *m_shaders; }
	[[nodiscard]] bool               IsRendering() const noexcept { return m_rendering; }
	[[nodiscard]] DynamicStateCache& DynamicState() const noexcept { return m_dynamic_state; }
	// Called wherever something other than SetGraphicsDynamicParams() (renderDraw.cpp) records
	// dynamic state on this buffer -- currently blitHelper.cpp binding its own graphics pipelines
	// -- so the cache does not think a stale value is still current.
	void InvalidateDynamicState() const noexcept { m_dynamic_state.Reset(); }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	CommandScheduler&         m_scheduler;
	RenderContext&            m_context;
	GraphicContext&           m_graphics;
	vk::CommandBuffer         m_buffer          = nullptr;
	uint32_t                  m_debug_op        = 0;
	uint64_t                  m_debug_submit_id = 0;
	uint32_t                  m_debug_arg0      = 0;
	uint32_t                  m_debug_arg1      = 0;
	uint32_t                  m_debug_arg2      = 0;
	uint32_t                  m_debug_arg3      = 0;
	uint64_t                  m_debug_arg4      = 0;
	mutable RenderState       m_render_state;
	// Nothing is known about a command buffer's dynamic state until CommandBuffer::Begin() resets
	// this (see context.cpp); blitHelper.cpp additionally invalidates it after binding its own
	// graphics pipelines and issuing its own vkCmdSet* calls.
	mutable DynamicStateCache m_dynamic_state;
	mutable bool              m_rendering   = false;
	HW::Context*              m_registers   = nullptr;
	HW::UserConfig*           m_user_config = nullptr;
	HW::Shader*               m_shaders     = nullptr;

	friend class CommandScheduler;
};

// Key for RenderExecutor::m_color_target_memo (see ResolveRenderColorTarget() in
// colorRenderTarget.cpp). Covers exactly what that function reads out of the guest registers
// before it starts validating and building a TextureCache::ImageDesc: the slot's raw
// HW::RenderTarget block, its effective 4-bit mask after the ignore_target_mask adjustment
// (ResolveColorTargets() forces mask=0x0f for its copy-target lookups, which CB_TARGET_MASK
// alone would not show), the slice offset carried in from the draw, and exact_format
// (ResolveColorTargets() also asks for an exact pixel-format match, unlike the per-draw loop).
// Two lookups differing in any of these can legitimately resolve to different images.
struct ColorTargetKey {
	HW::RenderTarget rt;
	uint32_t         mask                       = 0;
	uint32_t         render_target_slice_offset = 0;
	bool             exact_format               = false;

	bool operator==(const ColorTargetKey&) const = default;
};

// Value for RenderExecutor::m_depth_target_memo: only the half of ResolveRenderDepthTarget()'s
// result that MakeDepthTargetDesc()/TextureCache::FindImage() derive from HW::DepthRenderTarget.
// Everything else RenderDepthInfo carries -- clear enables and values, the stencil face
// resolution, the depth/stencil-control enables -- is read from registers outside
// HW::DepthRenderTarget and is recomputed on every draw regardless of this memo.
struct DepthTargetDiscovery {
	TextureCache::ImageDesc desc;
	ImageId                 image_id;
};

class RenderExecutor {
public:
	// Defined out of line: building m_mesh_draw_args_builder needs RenderContext::GetGraphics(), and
	// RenderContext is only forward-declared here to avoid an include cycle with renderContext.h.
	explicit RenderExecutor(RenderContext& context);
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	// Records the draw. A draw with `args.indirect == nullptr` is always recorded and returns
	// true; an indirect draw returns false when the bound guest state cannot be expressed as a
	// GPU-side indirect draw, and the caller must retry it with host-read arguments.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// Whether DrawIndex()/DrawAuto() can record `source` as a GPU-side indirect draw with the
	// currently bound guest state. Evaluated before any draw state is touched, so a rejected draw
	// can take the host-read path without repeating work. What is only knowable once the shaders
	// and their descriptors are resolved is rejected later, by the draw itself.
	[[nodiscard]] IndirectDrawSupport SupportsIndirectDraw(CommandBuffer&            buffer,
	                                                       const DrawIndirectSource& source,
	                                                       bool                      indexed,
	                                                       uint32_t index_type_and_size);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// Consecutive draws on a stage present the same materialised image and sampler descriptors
	// about nine times out of ten, and resolving an image descriptor searches the texture cache
	// for every image of every draw. The previous resolution is kept per stage and reused while
	// the descriptors, the program and the texture cache generation are all unchanged; the
	// per-draw half of binding an image -- refreshing its contents, LRU, download tracking --
	// still runs, in RebindImages().
	struct StageTextures {
		const ShaderRecompiler::IR::CompiledShaderInfo*    program = nullptr;
		std::vector<ShaderRecompiler::IR::DescriptorValue> image_values;
		std::vector<ShaderRecompiler::IR::DescriptorValue> sampler_values;
		std::vector<TextureBinding>                        images;
		std::vector<vk::Sampler>                           samplers;
		uint64_t                                           texture_generation = 0;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            ShaderType                                   stage);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	static void        UnifyRenderTargetScale(CommandBuffer& buffer, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	// False when the prepared draw turned out to be an indirect draw that cannot be recorded; no
	// commands are written in that case.
	bool ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer& buffer, uint32_t group_x,
	                                                     uint32_t group_y, uint32_t group_z,
	                                                     uint32_t mode);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	// Converts a guest indirect-args block into a mesh draw's parameter and dispatch buffers on the
	// GPU; see meshDrawArgs.h. Owned here because ExecutePreparedDraw() is its only caller.
	MeshDrawArgsBuilder                   m_mesh_draw_args_builder;
	std::vector<ImageId>                  m_bound_images;
	// Indexed by ShaderType, which ends with the tessellation stages.
	std::array<StageTextures, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1>
	                                      m_stage_textures;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	std::unordered_set<uint64_t> m_unrepresentable_textures;
	std::unordered_set<uint64_t> m_depth_tiled_reports;
	// Per-slot / per-draw memo of render-target discovery, keyed on TextureCache::Generation() --
	// the same invariant StageTextures::texture_generation above already relies on for texture
	// descriptors. See ColorTargetKey / DepthTargetDiscovery and their use in
	// colorRenderTarget.cpp / depthRenderTarget.cpp.
	std::array<GenerationMemo<ColorTargetKey, RenderColorInfo>, RENDER_COLOR_ATTACHMENTS_MAX>
	                                                             m_color_target_memo;
	GenerationMemo<HW::DepthRenderTarget, DepthTargetDiscovery> m_depth_target_memo;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
