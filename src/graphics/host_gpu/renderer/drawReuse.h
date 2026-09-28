#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREUSE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREUSE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

// Reusing the previous draw's render state.
//
// Most draws of a frame repeat the draw before them with new user data or draw arguments only:
// the same programs, render targets, blend/depth state and viewport. When DrawStateTracker (PM4
// side) vouches that no register other than user data and draw arguments was written since the
// previous draw packet, RenderExecutor::DrawIndex()/DrawAuto() compare the draw against the
// DrawReuseRecord of the previous fully recorded draw:
//
//  - When DrawReuseInputs match and the draw's programs, looked up again, are the previous
//    draw's (DrawPrograms), render-state discovery is skipped and the previous draw's
//    colour/depth target resolution is kept: it depends only on registers, on the texture cache
//    generation, both unchanged, and on the pixel program's outputs, which pick the colour slots.
//  - When in addition the recording position (still the same render pass instance, nothing else
//    recorded dynamic state) and the DrawPipelineFeed match after the draw's resources were
//    prepared, the draw also skips render-target acquisition, the pipeline lookup, dynamic
//    state, the render pass begin and the pipeline bind, and records only its bindings and the
//    draw itself.
//
// Everything a draw compares is a named field here so the comparison is complete by inspection and
// testable without Vulkan objects: handles are carried as integers or opaque pointers.

namespace Libs::Graphics {

// Where in the command stream a draw is recorded. Two draws at the same position are recorded
// into the same render pass instance of the same command buffer with nothing but draws in
// between that could have touched the pass, the bound pipeline or dynamic state.
struct DrawRecordingPosition {
	uint64_t tick           = 0; // CommandScheduler::CurrentTick()
	uint64_t command_buffer = 0; // the VkCommandBuffer being recorded, as an integer
	// CommandBuffer::RenderPassEpoch(): moves with every render pass begun on the buffer.
	uint64_t render_pass_epoch = 0;
	bool     rendering         = false; // CommandBuffer::IsRendering()
	// CommandBuffer::DynamicStateInvalidations(): something other than the draw path recorded
	// dynamic state.
	uint64_t dynamic_state_invalidations = 0;
	// CommandScheduler::DeferredOperationsRun(): deferred callbacks ran on the recording thread.
	uint64_t deferred_operations = 0;

	bool operator==(const DrawRecordingPosition&) const = default;
};

// What render-state discovery and the draw's own classification depend on besides the registers
// DrawStateTracker vouches for.
struct DrawReuseInputs {
	DrawRecordingPosition position;
	// The DrawRenderState object that holds the previous draw's discovery result.
	const void* kept_state = nullptr;
	// TextureCache::Generation() when discovery ran: an unchanged generation means every image
	// lookup of that discovery would answer the same way again (see GenerationMemo).
	uint64_t texture_generation = 0;
	// ES/GS/LS/HS/PS program addresses (SPI_SHADER_PGM_*), which select the programs. LS/HS
	// come last, see below.
	uint64_t es_address                 = 0;
	uint64_t gs_address                 = 0;
	uint64_t ps_address                 = 0;
	uint32_t render_target_slice_offset = 0;
	bool     indexed                    = false;
	uint32_t index_type_and_size        = 0; // 0 for a non-indexed draw
	uint64_t ls_address                 = 0;
	uint64_t hs_address                 = 0;

	// Every draw compares these. Clang merges the defaulted comparison of each padding-free run of
	// fields into one memcmp and expands it inline only up to 64 bytes (x86-64-v3); a
	// longer run becomes a libc memcmp call per draw. The padding after `rendering` and `indexed`
	// splits the fields into runs, and LS/HS sit after `indexed` so the middle run stays short.
	bool operator==(const DrawReuseInputs&) const = default;
};
static_assert(offsetof(DrawReuseInputs, indexed) + sizeof(bool) -
                      offsetof(DrawRecordingPosition, dynamic_state_invalidations) <=
                  64,
              "the comparison run between `rendering` and `indexed` no longer expands inline");

// The programs a draw runs: one vertex stage, or three for a tessellation draw (LS/HS/TES, see
// PipelineCache::GraphicsPrograms), and the pixel stage when it is active. The registers only
// name program addresses; a program registered again at the same address is a different program,
// with possibly different outputs, so kept render targets are only valid for the same programs.
struct DrawPrograms {
	static constexpr uint32_t MaxVertexStages = 3;

	std::array<const void*, MaxVertexStages> vertex_programs {}; // ShaderStageRuntime::program
	std::array<uint64_t, MaxVertexStages>    vertex_ids {};
	std::array<uint64_t, MaxVertexStages>    vertex_modules {}; // VkShaderModule, as an integer
	const void* ps_program = nullptr; // null when the pixel stage is inactive
	uint64_t    ps_id      = 0;
	uint64_t    ps_module  = 0;
	bool        ps_active  = false;

	bool operator==(const DrawPrograms&) const = default;
};

// Everything the draw path hands PipelineCache::GetGraphicsPipeline() and
// SetGraphicsDynamicParams() that neither comes from registers nor from the kept render targets:
// the programs (every vertex stage id is part of GraphicsPipelineKey; the pixel program id also
// carries its dual-source blend verdict), pixel-stage sample shading, the vertex input layout (the
// same fields GraphicsPipelineKey takes from the first vertex stage's ShaderVertexInputInfo), the
// topology and primitive restart.
struct DrawPipelineFeed {
	DrawPrograms             programs;
	bool                     ps_sample_shading = false;
	uint32_t                 topology          = 0; // vk::PrimitiveTopology
	bool                     primitive_restart = false;
	PipelineVertexInputState vertex_input;

	bool operator==(const DrawPipelineFeed&) const = default;
};

// The previous fully recorded draw: direct, not a mesh draw, not using host-expanded indices, and
// leaving its render pass open. Invalid until such a draw is recorded; every draw takes it at its
// start (Take()), so a draw that returns early or records anything else leaves it invalid.
class DrawReuseRecord {
public:
	// The record as the previous draw left it; this one is invalid until Store().
	[[nodiscard]] DrawReuseRecord Take() noexcept {
		DrawReuseRecord previous = *this;
		m_valid                  = false;
		return previous;
	}

	void Store(const DrawReuseInputs& inputs, const DrawPipelineFeed& feed,
	           PipelineCache::Pipeline* pipeline) noexcept {
		m_valid    = pipeline != nullptr;
		m_inputs   = inputs;
		m_feed     = feed;
		m_pipeline = pipeline;
	}

	// Whether a draw with `inputs` may keep this draw's render-state discovery result.
	[[nodiscard]] bool KeepsRenderState(const DrawReuseInputs& inputs) const noexcept {
		return m_valid && m_inputs == inputs;
	}

	// Whether a draw that kept this draw's render state runs the same programs, so its kept
	// targets are the ones discovery would pick again.
	[[nodiscard]] bool KeepsPrograms(const DrawPrograms& programs) const noexcept {
		return m_valid && m_feed.programs == programs;
	}

	// This draw's pipeline when a draw that kept its render state, now at `position` with its
	// resources prepared and `feed` resolved, may also keep its pipeline, dynamic state and render
	// pass; null otherwise.
	[[nodiscard]] PipelineCache::Pipeline* ReusablePipeline(
	    const DrawRecordingPosition& position, const DrawPipelineFeed& feed) const noexcept {
		if (!m_valid || !(m_inputs.position == position) || !(m_feed == feed)) {
			return nullptr;
		}
		return m_pipeline;
	}

	[[nodiscard]] const DrawReuseInputs& Inputs() const noexcept { return m_inputs; }
	// The topology the draw was recorded with.
	[[nodiscard]] vk::PrimitiveTopology Topology() const noexcept {
		return static_cast<vk::PrimitiveTopology>(m_feed.topology);
	}

private:
	bool                     m_valid = false;
	DrawReuseInputs          m_inputs;
	DrawPipelineFeed         m_feed;
	PipelineCache::Pipeline* m_pipeline = nullptr;
};

// The per-draw dynamic state SetGraphicsDynamicParams() (renderDraw.cpp) records. Resolving it
// into a value first lets the reuse oracle check that a draw which skipped the call would have
// recorded exactly what is already current.
struct GraphicsDynamicState {
	static constexpr uint32_t MaxViewports = 16;

	// Every pipeline declares the stencil op, masks and reference dynamic, so each face is
	// recorded whole for every draw, stencil test or not.
	struct StencilFace {
		vk::StencilOp fail_op       = vk::StencilOp::eKeep;
		vk::StencilOp pass_op       = vk::StencilOp::eKeep;
		vk::StencilOp depth_fail_op = vk::StencilOp::eKeep;
		vk::CompareOp compare_op    = vk::CompareOp::eNever;
		uint32_t      compare_mask  = 0;
		uint32_t      write_mask    = 0;
		uint32_t      reference     = 0;

		bool operator==(const StencilFace&) const = default;
	};

	uint32_t                                viewport_count = 0;
	std::array<vk::Viewport, MaxViewports>  viewports {};
	std::array<vk::Rect2D, MaxViewports>    scissors {};
	float                                   line_width = 1.0f;
	std::array<float, 4>                    blend_constants {};
	vk::Bool32                              depth_test_enable        = VK_FALSE;
	vk::Bool32                              depth_write_enable       = VK_FALSE;
	vk::CompareOp                           depth_compare_op         = vk::CompareOp::eNever;
	vk::Bool32                              depth_bounds_test_enable = VK_FALSE;
	std::array<float, 2>                    depth_bounds {};
	vk::Bool32                              stencil_test_enable = VK_FALSE;
	std::array<StencilFace, 2>              stencil {}; // front, back
	vk::CullModeFlags                       cull_mode {};
	vk::FrontFace                           front_face = vk::FrontFace::eCounterClockwise;
	// Topology and primitive restart are only part of a pipeline with input assembly.
	bool                                    input_assembly    = false;
	vk::PrimitiveTopology                   topology          = vk::PrimitiveTopology::ePointList;
	bool                                    primitive_restart = false;
	vk::Bool32                              depth_bias_enable = VK_FALSE;
	std::array<float, 3>                    depth_bias {}; // constant, clamp, slope
	// One entry per colour attachment slot of the render pass; zero slots forgets the state.
	uint32_t                                color_write_count = 0;
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> color_write_enable {};
	// Only recorded when the host has dynamic attachment feedback loops.
	bool                                    feedback_loop_dynamic = false;
	vk::ImageAspectFlags                    feedback_aspects {};

	bool operator==(const GraphicsDynamicState&) const = default;
};

// KYTY_DRAW_REUSE_ORACLE=1: every draw that reuses state also runs the steps it skipped and
// compares. This is the oracle's bookkeeping; renderDraw.cpp does the checking and logging.
class DrawReuseOracle {
public:
	// Mismatches beyond this many are only counted.
	static constexpr uint32_t DetailedLogLimit = 20;

	void NoteChecked() noexcept { m_checked++; }
	// Counts a mismatch; true while the detailed log budget lasts.
	[[nodiscard]] bool NoteMismatch() noexcept {
		m_mismatches++;
		return m_detailed++ < DetailedLogLimit;
	}
	// True at most once per second, when the summary line is due.
	[[nodiscard]] bool SummaryDue() noexcept {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_last_summary < std::chrono::seconds(1)) {
			return false;
		}
		m_last_summary = now;
		return true;
	}
	[[nodiscard]] uint64_t Checked() const noexcept { return m_checked; }
	[[nodiscard]] uint64_t Mismatches() const noexcept { return m_mismatches; }

	// The dynamic state most recently recorded by the draw path, and whether it is known.
	GraphicsDynamicState recorded_dynamic_state;
	bool                 recorded_dynamic_state_known = false;

private:
	uint64_t                              m_checked    = 0;
	uint64_t                              m_mismatches = 0;
	uint32_t                              m_detailed   = 0;
	std::chrono::steady_clock::time_point m_last_summary = std::chrono::steady_clock::now();
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWREUSE_H_ */
