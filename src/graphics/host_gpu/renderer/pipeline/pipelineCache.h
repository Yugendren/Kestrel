#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <xxhash.h>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

// Vulkan only requires the pipeline's topology to match the dynamic one by class when
// dynamicPrimitiveTopologyUnrestricted is false, and a patch list additionally selects the
// tessellation stage, so the class is the only part of the topology the pipeline still needs.
enum class PipelineTopologyClass : uint8_t { Point, Line, Triangle, Patch };

PipelineTopologyClass TopologyClassOf(vk::PrimitiveTopology topology);
vk::PrimitiveTopology  RepresentativeTopology(PipelineTopologyClass topology_class);

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                   negative_one_to_one                                = false;
	bool                   depth_clip_enable                                  = true;
	PipelineTopologyClass  topology_class                                     = PipelineTopologyClass::Point;
	uint32_t               samples                                            = 1;
	bool                   sample_shading_enable                              = false;
	uint32_t               color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                   provoking_vtx_last                                 = false;
	vk::PolygonMode        polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                   separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                   blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 109);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

// Packed so that Binding/Attribute (and therefore the arrays below) have no padding bytes:
// GraphicsPipelineKeyHash hashes bindings/attributes via XXH3 over their raw byte range, and
// operator== below memcmp's the same ranges, both of which would be unsound if padding bytes
// were left indeterminate.
#pragma pack(push, 1)

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	// The key is value-initialised and only the entries below binding_count / attribute_count
	// are ever written, so unused slots are zero in every key. That invariant makes comparing
	// just the used prefixes equivalent to the old elementwise comparison of all RES_MAX
	// entries.
	bool operator==(const PipelineVertexInputState& other) const {
		return binding_count == other.binding_count && attribute_count == other.attribute_count &&
		       std::memcmp(bindings.data(), other.bindings.data(),
		                   sizeof(Binding) * binding_count) == 0 &&
		       std::memcmp(attributes.data(), other.attributes.data(),
		                   sizeof(Attribute) * attribute_count) == 0;
	}
};

#pragma pack(pop)

static_assert(sizeof(PipelineVertexInputState::Binding) == 5);
static_assert(sizeof(PipelineVertexInputState::Attribute) == 5);
static_assert(sizeof(PipelineVertexInputState) ==
              2 + (sizeof(PipelineVertexInputState::Binding) +
                   sizeof(PipelineVertexInputState::Attribute)) *
                      ShaderVertexInputInfo::RES_MAX);

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	[[nodiscard]] size_t PipelineCountLocked() const;

	// getPipelineCacheData() + the temp-file write, with none of m_mutex held: it touches only
	// state that is fixed once InitializeDriverCache() returns from the constructor
	// (m_graphics.device, m_driver_cache, m_driver_cache_path) and is never written again, so it
	// is safe to call from the background saver thread and from Save() alike. A VkPipelineCache
	// created without VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT (see
	// InitializeDriverCache()) is internally synchronised by the driver, so this may run
	// concurrently with vkCreateGraphicsPipelines/vkCreateComputePipelines against the same
	// cache on the command-processor thread. m_save_io_mutex only serialises this method against
	// itself, so a background save and a shutdown Save() never write the temp file at once.
	void SerializeAndWrite();
	std::mutex m_save_io_mutex;
	// Hash of the payload written by the most recent successful SerializeAndWrite(), guarded by
	// m_save_io_mutex like the rest of that method's state (never m_mutex). Lets a save whose
	// blob is byte-identical to what is already on disk -- e.g. a periodic save that landed right
	// after a shutdown save, or a save cycle where every new pipeline was a cache hit -- return
	// before touching the temp file, at the cost of one XXH3 pass over an in-memory buffer.
	uint64_t m_last_written_payload_hash = 0;

	// Background saver: MaybeSaveLocked() (command-processor thread, m_mutex held) only flips
	// m_save_requested and wakes this thread; the actual multi-megabyte serialise-and-write runs
	// here, off the draw thread and without m_mutex. The thread is started lazily by the first
	// RequestBackgroundSave() and is therefore never created while m_driver_cache == nullptr
	// (cache disabled). Guarded by m_saver_mutex/m_saver_cv, never by m_mutex -- this thread must
	// never block on m_mutex, since that would reintroduce the hitch this design removes.
	void SaverThreadLoop();
	void RequestBackgroundSave();
	std::mutex              m_saver_mutex;
	std::condition_variable m_saver_cv;
	bool                    m_save_requested = false;
	bool                    m_saver_stop     = false;
	std::thread             m_saver_thread;

	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	// Hashing used to mix the key byte-by-byte / field-by-field (109 Mix() calls just for
	// static_params, plus a Mix() per vertex-input field), which measured ~155 ns/op for a
	// real 696-byte key. GraphicsPipelineKeyHash below instead runs XXH3 once over each
	// contiguous byte range (static_params, and the used prefixes of bindings/attributes),
	// which is sound only because those ranges have no padding -- see the
	// #pragma pack(push, 1) on PipelineVertexInputState and PipelineStaticParameters above.
	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}

		static void MixRendering(std::size_t& hash, const PipelineRenderingState& rendering) {
			Mix(hash, rendering.color_count);
			for (uint32_t i = 0; i < rendering.color_count; i++) {
				Mix(hash, static_cast<uint32_t>(rendering.color_formats[i]));
			}
			Mix(hash, static_cast<uint32_t>(rendering.depth_format));
			Mix(hash, static_cast<uint32_t>(rendering.stencil_format));
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const {
			std::size_t hash = 0;
			PipelineKeyHash::MixRendering(hash, key.rendering);
			for (const auto id: key.vertex_shader_ids) {
				PipelineKeyHash::Mix(hash, id);
			}
			PipelineKeyHash::Mix(hash, key.ps_shader_id);
			PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
			if (key.vertex_input.binding_count != 0) {
				PipelineKeyHash::Mix(
				    hash, XXH3_64bits(key.vertex_input.bindings.data(),
				                       sizeof(PipelineVertexInputState::Binding) *
				                           key.vertex_input.binding_count));
			}
			PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
			if (key.vertex_input.attribute_count != 0) {
				PipelineKeyHash::Mix(
				    hash, XXH3_64bits(key.vertex_input.attributes.data(),
				                       sizeof(PipelineVertexInputState::Attribute) *
				                           key.vertex_input.attribute_count));
			}
			PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
			return hash;
		}
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	Common::Mutex m_mutex;

	// Pipeline count and wall-clock time as of the last background-save request, used by
	// MaybeSaveLocked() to decide whether a new save is worth its cost.
	size_t                                m_pipeline_count_at_last_save = 0;
	// Pipelines covered by the blob currently on disk (or, for a request still in flight on the
	// saver thread, optimistically assumed to be covered once it lands -- see MaybeSaveLocked()),
	// so a save with nothing new is skipped.
	size_t m_persisted_pipeline_count = 0;
	std::chrono::steady_clock::time_point m_last_save_time;

	void InitializeDriverCache();
	// Saves periodically as new pipelines are created; called with m_mutex already held by the
	// pipeline getters, so it must not take the lock itself. Only records bookkeeping under
	// m_mutex and hands the actual save off to the background saver thread.
	void MaybeSaveLocked();
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
