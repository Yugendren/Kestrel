#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
// Discovery-memo key types below need the complete HW::RenderTarget / HW::DepthRenderTarget
// register structs (and their operator==), plus RenderColorInfo, as the memoized value types;
// a forward declaration is no longer enough once a GenerationMemo<...> of them is a member.
#include "graphics/guest_gpu/command_processor/drawStateTracker.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawReuse.h"
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
struct DrawReuseRequest;
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
	// Orders everything recorded so far before everything recorded later with an ALL_COMMANDS
	// memory barrier (the command processor's cache flushes and partial flushes). The barrier is
	// not recorded here but by the next Handle() call, i.e. immediately before the next command,
	// so a run of requests with nothing recorded between them -- or following another full
	// barrier -- costs one barrier instead of one each. The recorded stream is the same as
	// recording every request on the spot, minus barriers that would sit back to back.
	void RequestFullBarrier() const;
	// Tells the buffer a full memory barrier was just recorded by other means (the shader hazard
	// barrier), so a request with nothing recorded since is already satisfied.
	void NoteFullBarrier() const noexcept { m_after_full_barrier = true; }

	// Every command recorded into the buffer goes through here, which is what makes it the place
	// to record a requested barrier. Code that only needs the buffer to be open checks IsInvalid()
	// instead, so it does not record the barrier early (the end-of-pipe writes in sync.cpp follow
	// most barrier requests and would otherwise defeat the coalescing).
	[[nodiscard]] vk::CommandBuffer  Handle() const;
	// The buffer being recorded, for identity comparisons only: unlike Handle() it records nothing.
	[[nodiscard]] vk::CommandBuffer  PeekHandle() const noexcept { return m_buffer; }
	[[nodiscard]] GraphicContext&    GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&     GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&       GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&    GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&        GetShaders() const noexcept { return *m_shaders; }
	[[nodiscard]] bool               IsRendering() const noexcept { return m_rendering; }
	// Moves with every render pass instance begun on this buffer: two draws that see the same
	// epoch while rendering are recorded into the same instance.
	[[nodiscard]] uint64_t           RenderPassEpoch() const noexcept { return m_render_pass_epoch; }
	// The render state the open render pass instance was begun with; empty when not rendering.
	[[nodiscard]] const RenderState& CurrentRenderState() const noexcept { return m_render_state; }
	[[nodiscard]] DynamicStateCache& DynamicState() const noexcept { return m_dynamic_state; }
	// Moves whenever nothing is known about the buffer's dynamic state anymore (a new recording,
	// InvalidateDynamicState()), so a draw can tell whether the state an earlier draw recorded is
	// still the current one.
	[[nodiscard]] uint64_t DynamicStateInvalidations() const noexcept {
		return m_dynamic_state_invalidations;
	}
	// Called wherever something other than SetGraphicsDynamicParams() (renderDraw.cpp) records
	// dynamic state on this buffer -- currently blitHelper.cpp binding its own graphics pipelines
	// -- so the cache does not think a stale value is still current.
	void InvalidateDynamicState() const noexcept {
		m_dynamic_state.Reset();
		m_dynamic_state_invalidations++;
	}

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;
	void RecordFullBarrier() const;

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
	mutable uint64_t          m_dynamic_state_invalidations = 0;
	mutable bool              m_rendering   = false;
	mutable uint64_t          m_render_pass_epoch = 0;
	// RequestFullBarrier() was called and the barrier has not been recorded yet.
	mutable bool              m_full_barrier_pending = false;
	// The last command recorded was a full memory barrier.
	mutable bool              m_after_full_barrier = false;
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

	// `compute_space` is the space a compute stage's texel addresses are resolved in; other
	// stages always address the resolution of their render targets.
	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                     TextureCache::TexelSpace compute_space = TextureCache::TexelSpace::Guest);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

	// The command processors report every packet they execute here, and DrawIndex()/DrawAuto()
	// read its verdict (drawStateTracker.h).
	[[nodiscard]] DrawStateTracker& GetDrawStateTracker() noexcept { return m_draw_state_tracker; }

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
	// the descriptors, the program, the texel space and the texture cache generation are all
	// unchanged; the per-draw half of binding an image -- refreshing its contents, LRU, download
	// tracking -- still runs, in RebindImages().
	struct StageTextures {
		const ShaderRecompiler::IR::CompiledShaderInfo*    program = nullptr;
		// The space a compute stage resolved in (PrepareBindings()); the same descriptors resolve
		// differently for a tile-rescaled dispatch.
		TextureCache::TexelSpace texel_space = TextureCache::TexelSpace::Guest;
		std::vector<ShaderRecompiler::IR::DescriptorValue> image_values;
		std::vector<ShaderRecompiler::IR::DescriptorValue> sampler_values;
		std::vector<TextureBinding>                        images;
		std::vector<vk::Sampler>                           samplers;
		uint64_t                                           texture_generation = 0;
		// Identifies this recorded resolution among every stage's (from m_texture_resolutions).
		uint64_t                                           resolution = 0;
	};

	[[nodiscard]] TextureBinding
	ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	               const ShaderRecompiler::IR::DescriptorValue& value, ShaderType stage,
	               TextureCache::TexelSpace compute_space = TextureCache::TexelSpace::Guest);
	// The image half of PrepareBindings(): resolves (or reuses) every image and sampler of the
	// stage `prepared` belongs to and binds the images. Called again, after ResetBindings(), to
	// resolve the same descriptors in another texel space.
	void ResolveImages(PreparedBindings& prepared, TextureCache::TexelSpace compute_space);
	// Sets one shader-data dword after RebindBuffers() may already have uploaded the block.
	void SetShaderDataWord(PreparedBindings& prepared, uint32_t dword, uint32_t value);
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
	// commands are written in that case. `reuse` is null for a draw that takes no part in
	// draw-state reuse (drawReuse.h); otherwise a fully recorded draw is stored for the next one,
	// and `reuse->previous`, when set, is the draw whose render state this one kept.
	bool ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, const DrawReuseRequest* reuse);
	// Draw-state reuse (drawReuse.h). The draw's render state: the previous draw's discovery
	// result, with this draw's programs, when `keep`, the programs are `previous`'s and its
	// targets are still live; else a reset state for PrepareDrawRenderState(), in which case
	// `keep` is cleared.
	[[nodiscard]] DrawRenderState& SelectDrawRenderState(bool& keep, CommandBuffer& buffer,
	                                                     const DrawCallInfo&    draw,
	                                                     uint32_t               render_target_slice_offset,
	                                                     const DrawReuseRecord& previous);
	// Re-applies the per-draw half of render-target discovery to the kept targets; false when one
	// of them is gone and discovery has to run.
	[[nodiscard]] bool KeepDrawTargets(const DrawRenderState& state);
	// The previous draw's pipeline when this draw, which kept its render state, may also keep its
	// render targets' acquisition, pipeline, dynamic state and render pass; null otherwise.
	[[nodiscard]] PipelineCache::Pipeline* ReusableDrawPipeline(const CommandBuffer&    buffer,
	                                                            const DrawRenderState&  state,
	                                                            const DrawReuseRecord&  previous,
	                                                            const DrawPipelineFeed& feed);
	// KYTY_DRAW_REUSE_ORACLE=1: run the skipped steps beside a reusing draw and compare; false on
	// a mismatch, which the draw then answers by taking the full path.
	[[nodiscard]] bool OracleConfirmsRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                             uint32_t               render_target_slice_offset,
	                                             const DrawRenderState& kept);
	[[nodiscard]] bool OracleConfirmsReusedPipeline(
	    const CommandBuffer& buffer, const DrawRenderState& state, const DrawReuseRecord& previous,
	    const RenderState& rendering, vk::ImageAspectFlags feedback_aspects,
	    const PipelineCache::Pipeline& expected, const PipelineCache::Pipeline& reused,
	    vk::PrimitiveTopology topology, bool primitive_restart_enable);
	void LogDrawReuseOracleSummary();
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
	uint64_t                              m_texture_resolutions = 0;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	// Whether the latest PrepareBindings() for the pixel stage kept the previous resolution of its
	// images (m_stage_textures); read by ReusableDrawPipeline().
	bool                                  m_pixel_bindings_reused = false;
	std::unordered_set<uint64_t> m_unrepresentable_textures;
	std::unordered_set<uint64_t> m_depth_tiled_reports;
	// Per-slot / per-draw memo of render-target discovery, keyed on TextureCache::Generation() --
	// the same invariant StageTextures::texture_generation above already relies on for texture
	// descriptors. See ColorTargetKey / DepthTargetDiscovery and their use in
	// colorRenderTarget.cpp / depthRenderTarget.cpp.
	std::array<GenerationMemo<ColorTargetKey, RenderColorInfo>, RENDER_COLOR_ATTACHMENTS_MAX>
	                                                             m_color_target_memo;
	GenerationMemo<HW::DepthRenderTarget, DepthTargetDiscovery> m_depth_target_memo;
	// Draw-state reuse (drawReuse.h). The tracker is owned here rather than by a command
	// processor: the graphics and compute command processors all run on the GPU thread and feed
	// this executor, so one tracker sees every packet executed before the executor's next draw,
	// whichever queue it came from.
	DrawStateTracker m_draw_state_tracker;
	DrawReuseRecord  m_draw_reuse;
	DrawReuseOracle  m_draw_reuse_oracle;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
