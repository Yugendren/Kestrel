#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderProgramMemo.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// The driver pipeline cache blob is tens of megabytes for a large title -- field data from a
// real single-HDD system (driver, game dump, and asset streaming all sharing one disk) measured
// 24.8-27.0 MB, written five times within a few minutes of play -- and serialising it
// (getPipelineCacheData + a temp-file write) costs tens of milliseconds, competing with asset
// streaming for the same disk and visibly worsening loading hitches (e.g. the boot splash). So
// MaybeSaveLocked() must not run it on every new pipeline, or even every few seconds. Saving too
// rarely, though, means a crash or force-quit loses more freshly-compiled pipelines and the next
// launch recompiles them from scratch, which is exactly what this whole mechanism exists to
// avoid. Vulkan has no incremental/delta pipeline-cache serialisation -- vkGetPipelineCacheData
// always returns the entire blob -- so the only lever here is cadence (this trio of thresholds)
// plus skipping a write whose payload turns out identical (see m_last_written_payload_hash).
//   - kMinSaveInterval is a hard floor between two writes, whatever else happens: without it, a
//     burst of new pipelines (e.g. loading into a new area) can still trigger back-to-back
//     multi-megabyte writes.
//   - kMinNewPipelinesToSave / kMaxSaveInterval are "enough new pipelines, or enough elapsed
//     time" -- whichever is reached first, same as before, just tuned far less eager.
constexpr std::chrono::seconds kMinSaveInterval {180};
constexpr size_t               kMinNewPipelinesToSave = 64;
constexpr std::chrono::seconds kMaxSaveInterval {600};

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
}

// Raw current guest memory, no GPU-clean gate. Shader resource tables (SRTs) are CPU-written
// and GPU-read-only, so a plain backing read is both correct and always available -- unlike the
// clean-gated reader above, which refuses while the GPU has pending writes to the range. Without
// this the SRT evaluator cannot resolve a descriptor the shader builds from SRT dwords
// (Astro Bot PPSA21564: a buffer V# base = CompositeExtractU64(... ReadConst(GetSrtResource))).
//
// Gated to PPSA21564: wiring this reader for every title changes descriptor resolution for
// shaders whose SRT raw-reads previously failed and were handled by a fallback -- it made
// Demon's Souls hang in the intro. PR #500 shipped this field NULL for that reason. Real fix:
// bind-time dynamic-SRT resolution, after which the reader is safe for all titles.
bool ReadShaderMappedMemory(void*, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}

ShaderRecompiler::IR::SrtMemoryReader ShaderMappedMemoryReaderForTitle() {
	static const bool enabled = [] {
		std::string id;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &id) && id == "PPSA21564";
	}();
	return enabled ? &ReadShaderMappedMemory : nullptr;
}

bool SyncShaderGuestMemory(void*, uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::SyncGpuCleanBacking(address, size);
}

// Returns false when the shader's resources could not be fully materialised. For a compute
// dispatch the caller then drops that dispatch -- a descriptor the shader assembles from a
// runtime-dynamic / loop-carried SRT pointer cannot be reconstructed ahead of the dispatch,
// and KytyPS5 has no bindless (srt_flatbuf / BDA) descriptor path yet.
// Soft ladder (PPSA21564): the real fix is a shadPS4-style flat-SRT bindless model; until then
// dropping the un-materialisable GI/lighting compute kernels keeps the title running. A
// graphics stage still aborts loudly.
bool ReportMaterialization(const char* label, ShaderType stage, uint64_t hash,
                           const ShaderRecompiler::IR::MaterializeReport& report, bool ok) {
	if (!ok) {
		if (stage == ShaderType::Compute) {
			LOGF("shader resource materialization incomplete: stage=%u hash=0x%016" PRIx64
			     " reason=%s -- dropping this dispatch\n",
			     static_cast<uint32_t>(stage), hash, report.reason.c_str());
			return false;
		}
		EXIT("shader resource materialization failed: stage=%u hash=0x%016" PRIx64 " reason=%s\n",
		     static_cast<uint32_t>(stage), hash, report.reason.c_str());
	}
	if (!report.dropped_summary.empty()) {
		LOGF("%s indirect image tables: hash=0x%016" PRIx64 " dropped=%" PRIu32 " shapes=%" PRIu32
		     "%s\n",
		     label, hash, report.dropped_candidates, report.dropped_shapes,
		     report.dropped_summary.c_str());
	}
	return true;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// shaders.cpp needs the inverse of this mapping (RepresentativeTopology) to build the pipeline's
// static input-assembly state, so both live here instead of the anonymous namespace above.
PipelineTopologyClass TopologyClassOf(vk::PrimitiveTopology topology) {
	switch (topology) {
		case vk::PrimitiveTopology::ePointList: return PipelineTopologyClass::Point;
		case vk::PrimitiveTopology::eLineList:
		case vk::PrimitiveTopology::eLineStrip:
		case vk::PrimitiveTopology::eLineListWithAdjacency:
		case vk::PrimitiveTopology::eLineStripWithAdjacency: return PipelineTopologyClass::Line;
		case vk::PrimitiveTopology::eTriangleList:
		case vk::PrimitiveTopology::eTriangleStrip:
		case vk::PrimitiveTopology::eTriangleFan:
		case vk::PrimitiveTopology::eTriangleListWithAdjacency:
		case vk::PrimitiveTopology::eTriangleStripWithAdjacency: return PipelineTopologyClass::Triangle;
		case vk::PrimitiveTopology::ePatchList: return PipelineTopologyClass::Patch;
		default: EXIT("Pipeline: unsupported topology %u\n", static_cast<uint32_t>(topology));
	}
}

vk::PrimitiveTopology RepresentativeTopology(PipelineTopologyClass topology_class) {
	switch (topology_class) {
		case PipelineTopologyClass::Point: return vk::PrimitiveTopology::ePointList;
		case PipelineTopologyClass::Line: return vk::PrimitiveTopology::eLineList;
		case PipelineTopologyClass::Triangle: return vk::PrimitiveTopology::eTriangleList;
		case PipelineTopologyClass::Patch: return vk::PrimitiveTopology::ePatchList;
	}
	EXIT("Pipeline: unsupported topology class %u\n", static_cast<uint32_t>(topology_class));
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)), compiled_plan(resource_plan) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		// Every draw of this program refreshes its resources; the plan's walk is lowered once here
		// instead of being interpreted per draw. Declared after resource_plan, which it refers to.
		ShaderRecompiler::IR::CompiledSrtPlan        compiled_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
	};

	// The entry and permutation the previous Get of one graphics stage resolved to. A caller that
	// knows its program key equals that call's (VertexProgramMemo / PixelProgramMemo) skips the
	// key build and lookup; materialisation and the permutation match still run per call.
	struct StageMemo {
		SourceEntry* entry       = nullptr;
		size_t       permutation = 0;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: stage_name = "vs"; break;
			case ShaderType::Mesh: stage_name = "ms"; break;
			case ShaderType::Local: stage_name = "ls"; break;
			case ShaderType::TessellationControl: stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: stage_name = "ds"; break;
			case ShaderType::Pixel: stage_name = "ps"; break;
			case ShaderType::Compute: stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		SetVulkanObjectNameF(device, module, "Kyty.Shader.{}[0x{:016x}]", stage_name,
		                     options.shader_hash);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                  StageMemo* memo = nullptr, bool same_program = false) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		const bool   from_memo        = memo != nullptr && same_program && memo->entry != nullptr;
		SourceEntry* source           = from_memo ? memo->entry : nullptr;
		const size_t memo_permutation = from_memo ? memo->permutation : 0;
		if (!from_memo) {
			lookup_key.stage           = stage;
			lookup_key.hash            = params.hash;
			lookup_key.user_data_count = params.user_data_count;
			lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
			BuildStageStaticKey(input_info, lookup_key.static_state);
			if (const auto found = programs.find(lookup_key); found != programs.end()) {
				source = &found->second;
			}
		}
		// Only a call that resolves a program may leave a memo behind.
		const auto remember = [&](size_t permutation) {
			if (memo != nullptr) {
				*memo = {.entry = source, .permutation = permutation};
			}
		};
		if (memo != nullptr) {
			*memo = {};
		}
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ShaderMappedMemoryReaderForTitle(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .sync_memory                = SyncShaderGuestMemory,
		};
		ShaderRecompiler::IR::MaterializeReport report;
		if (source != nullptr) {
			if (!ReportMaterialization(label, stage, params.hash, report,
			                           ShaderRecompiler::IR::MaterializeResources(
			                               source->resource_plan, &source->compiled_plan, runtime,
			                               source->resources, source->specialization, &report))) {
				return {};
			}
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                                    layout.ShaderDataDwords()) &&
				       candidate.specialization == source->specialization;
			};
			auto& permutations = source->permutations;
			// A repeated program nearly always materialises to the permutation it used last.
			const auto permutation =
			    from_memo && memo_permutation < permutations.size() &&
			            matches(permutations[memo_permutation])
			        ? permutations.begin() + static_cast<std::ptrdiff_t>(memo_permutation)
			        : std::ranges::find_if(permutations, matches);
			if (permutation != permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &source->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				remember(static_cast<size_t>(permutation - permutations.begin()));
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.bvh_always_miss    = Config::GetRtMode() == Config::RtMode::Off;
		options.bvh_reduced        = Config::GetRtMode() == Config::RtMode::Reduced;
		options.dump_label  = label;
		options.gpu_vertex_fetch =
		    stage == ShaderType::Vertex && Config::GetVertexFetchMode() == Config::VertexFetchMode::Gpu;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (source == nullptr) {
			source = &programs
			              .try_emplace(lookup_key,
			                           ShaderRecompiler::IR::ExtractResourcePlan(translated.program))
			              .first->second;
			if (!ReportMaterialization(label, stage, params.hash, report,
			                           ShaderRecompiler::IR::MaterializeResources(
			                               source->resource_plan, &source->compiled_plan, runtime,
			                               source->resources, source->specialization, &report))) {
				return {};
			}
		}
		source->permutations.push_back(CompilePermutation(
		    params, options, std::move(translated), source->specialization, push_data_cursor));
		const auto& permutation = source->permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &source->resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		remember(source->permutations.size() - 1);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, entry]: programs) {
			counts[static_cast<size_t>(key.stage)] += entry.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	// Graphics stages only; GetGraphicsPrograms pairs each with the matching PrepareProgram memo.
	StageMemo                                                   vertex_memo;
	StageMemo                                                   pixel_memo;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)),
      m_vertex_program_memo(std::make_unique<VertexProgramMemo>()),
      m_pixel_program_memo(std::make_unique<PixelProgramMemo>()),
      m_last_save_time(std::chrono::steady_clock::now()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	// Stop and join the saver thread before anything it touches goes away: SerializeAndWrite()
	// reads m_graphics.device and m_driver_cache, both of which are destroyed a few lines below,
	// so the thread must be fully stopped -- not just asked to stop -- before we proceed.
	if (m_saver_thread.joinable()) {
		{
			std::lock_guard<std::mutex> lock(m_saver_mutex);
			m_saver_stop = true;
			m_saver_cv.notify_one();
		}
		m_saver_thread.join();
	}
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

// Synchronous shutdown path (WindowRun() and ~PipelineCache()): unlike the periodic saves from
// MaybeSaveLocked(), this must have actually finished writing before the caller moves on, so it
// calls SerializeAndWrite() directly instead of going through the background saver thread.
void PipelineCache::Save() {
	{
		Common::LockGuard lock(m_mutex);
		if (m_driver_cache == nullptr) {
			return;
		}
		// The driver cache object stays alive for the whole session so it can be written again as
		// more pipelines appear; re-serialising a blob that already matches what is on disk would
		// only cost time, which is what a shutdown save right after a background one would
		// otherwise do.
		if (m_persisted_pipeline_count == PipelineCountLocked()) {
			return;
		}
	}
	SerializeAndWrite();
	Common::LockGuard lock(m_mutex);
	m_persisted_pipeline_count = PipelineCountLocked();
}

size_t PipelineCache::PipelineCountLocked() const {
	return m_graphics_pipelines.size() + m_compute_pipelines.size();
}

void PipelineCache::SerializeAndWrite() {
	if (m_driver_cache == nullptr) {
		return;
	}
	// Guards only against another SerializeAndWrite() call (background saver vs. a shutdown
	// Save()) racing to the same temp file -- see the declaration comment in the header for why
	// this needs no other lock.
	std::lock_guard<std::mutex> io_lock(m_save_io_mutex);

	// "serialise" below is both vkGetPipelineCacheData calls (the size query and the actual
	// copy, across every eIncomplete retry); "write" is the temp-file create+write+flush+rename.
	// Reported in the success log line so a slow HDD write is visible without a profiler attach.
	const auto serialize_start = std::chrono::steady_clock::now();

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	const auto payload_hash   = XXH3_64bits(payload.data(), payload.size());
	const auto serialize_end  = std::chrono::steady_clock::now();

	// The driver cache only grows as new pipelines compile in, but two consecutive periodic
	// saves can still see an identical blob -- e.g. every draw between them hit a pipeline the
	// previous save already covered. One XXH3 pass over the in-memory payload is negligible next
	// to a multi-megabyte HDD write, so skip the write (before the temp file is even touched)
	// rather than rewrite bytes already on disk.
	if (payload_hash == m_last_written_payload_hash) {
		return;
	}

	auto prefix = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	const auto write_end = std::chrono::steady_clock::now();
	m_last_written_payload_hash = payload_hash;
	const auto serialize_ms =
	    std::chrono::duration<double, std::milli>(serialize_end - serialize_start).count();
	const auto write_ms = std::chrono::duration<double, std::milli>(write_end - serialize_end).count();
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {} (serialise {:.1f} ms, write {:.1f} ms)",
	                 payload.size(), Common::PathToString(m_driver_cache_path), serialize_ms, write_ms);
	// m_persisted_pipeline_count is not updated here: it is a m_mutex-protected member and this
	// method must not touch m_mutex (see the header). MaybeSaveLocked() sets it optimistically at
	// request time, and Save() sets it itself after calling this method directly -- see both.
}

// Called from the pipeline getters right after a new pipeline is inserted, with m_mutex already
// held by that getter's own LockGuard, so this may only touch m_mutex-protected members directly
// and must hand the actual save off to the background saver thread instead of calling
// SerializeAndWrite() itself.
void PipelineCache::MaybeSaveLocked() {
	if (m_driver_cache == nullptr) {
		return;
	}
	const auto pipeline_count = PipelineCountLocked();
	if (pipeline_count <= m_pipeline_count_at_last_save) {
		return;
	}
	const auto now     = std::chrono::steady_clock::now();
	const auto elapsed = now - m_last_save_time;
	// Hard floor first, independent of how many pipelines just landed -- see kMinSaveInterval's
	// declaration comment for why a burst of new pipelines must not bypass it.
	if (elapsed < kMinSaveInterval) {
		return;
	}
	if (pipeline_count - m_pipeline_count_at_last_save < kMinNewPipelinesToSave &&
	    elapsed < kMaxSaveInterval) {
		return;
	}
	// Recorded optimistically here rather than once the background write actually completes:
	// the write happens asynchronously, off this thread and outside m_mutex, so there is nothing
	// to wait on. If the write later fails, the only consequence is that the retry waits for the
	// next threshold crossing -- the same failure policy a failed synchronous save already had.
	m_persisted_pipeline_count    = pipeline_count;
	m_pipeline_count_at_last_save = pipeline_count;
	m_last_save_time              = now;
	RequestBackgroundSave();
}

// Starts the saver thread lazily on first use (never while m_driver_cache == nullptr, since
// MaybeSaveLocked() -- the only caller -- already returns before reaching here in that case) and
// wakes it for one more pass. Several requests that land while a save is already in flight
// collapse into the same m_save_requested flag, so at most one further save follows -- never a
// queue.
void PipelineCache::RequestBackgroundSave() {
	std::lock_guard<std::mutex> lock(m_saver_mutex);
	if (!m_saver_thread.joinable()) {
		m_saver_thread = std::thread([this] { SaverThreadLoop(); });
	}
	m_save_requested = true;
	m_saver_cv.notify_one();
}

void PipelineCache::SaverThreadLoop() {
	for (;;) {
		std::unique_lock<std::mutex> lock(m_saver_mutex);
		m_saver_cv.wait(lock, [this] { return m_save_requested || m_saver_stop; });
		if (m_saver_stop) {
			return;
		}
		m_save_requested = false;
		// Unlocked before the (potentially multi-millisecond) serialise-and-write, so a request
		// that arrives while this is running only sets the flag above and is picked up by the
		// next loop iteration instead of blocking the requester.
		lock.unlock();
		SerializeAndWrite();
	}
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	// The program memos compare this draw with the previous one, so they share m_mutex with the
	// program cache they feed. Draws are already serialised by the render context lock, so
	// taking it before the prepare step (instead of after) costs no concurrency.
	Common::LockGuard lock(m_mutex);
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	// Tessellation draws (LS/HS/TES) are prepared without a memo: the memos cover the single
	// vertex/mesh stage and the pixel stage. A tessellation draw leaves both vertex memos alone,
	// and they stay paired, so the next non-tessellation draw still compares with the previous
	// non-tessellation one.
	bool same_vertex_program = false;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = m_vertex_program_memo->Prepare(vertex_regs, context, user_config,
		                                                  vertex_info[0], same_vertex_program);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	bool         same_pixel_program = false;
	if (pixel_active) {
		const auto& blend          = context.GetBlendControl(0);
		const auto  is_dual_source = [](uint8_t factor) {
			return factor >= static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color) &&
			       factor <= static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		};
		const bool dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (is_dual_source(blend.color_srcblend) || is_dual_source(blend.color_destblend) ||
		     (blend.separate_alpha_blend &&
		      (is_dual_source(blend.alpha_srcblend) || is_dual_source(blend.alpha_destblend))));
		// The verdict feeds the pixel program's static key, so the memo keys on it too.
		pixel_params = m_pixel_program_memo->Prepare(pixel_regs, sh, target_export_mapping,
		                                             dual_source_blending, pixel_info,
		                                             same_pixel_program);
		pixel_info.dual_source_blending = dual_source_blending;
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies a second blend source for the same render target as MRT0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t         push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawAddressDwordCount : 0;
	GraphicsPrograms result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor,
		                                    &m_program_cache->pixel_memo, same_pixel_program);
	}
	if (tess_active) {
		for (uint32_t i = 0; i < 3u; i++) {
			result.vertex[i] =
			    m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		}
	} else {
		result.vertex[0] = m_program_cache->Get(vertex_params[0], vertex_info[0], push_data_cursor,
		                                        &m_program_cache->vertex_memo, same_vertex_program);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		static_params.blend_enable[slot]         = bc.enable && !rt.info.blend_bypass;
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control          = ctx.GetClipControl();
	static_params.negative_one_to_one = !clip_control.dx_clip_space;
	static_params.depth_clip_enable   = clip_control.IsZClipEnabled();
	static_params.topology_class      = TopologyClassOf(topology);
	static_params.samples             = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	// Cull mode is dynamic per draw (see SetGraphicsDynamicParams), but the pipeline's fixed
	// polygon mode still depends on which face(s) are visible, so ResolvePolygonMode needs the
	// same flags as locals instead of static_params members.
	const bool rect_list  = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	const bool cull_back  = !rect_list && mc.cull_back;
	const bool cull_front = !rect_list && mc.cull_front;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode       = ResolvePolygonMode(mc, cull_front, cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh &&
	    !vs_input_info.stage.program->info.gpu_vertex_fetch) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	if (const auto* last = m_last_graphics_pipeline.Find(key, 0); last != nullptr) {
		return **last;
	}
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		m_last_graphics_pipeline.Store(key, 0, iter->second.get());
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(key, std::move(cached));
	EXIT_IF(!inserted);
	m_last_graphics_pipeline.Store(key, 0, iter->second.get());

	// m_mutex is already held by the LockGuard above, so MaybeSaveLocked() (not Save()) here.
	MaybeSaveLocked();

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	// m_mutex is already held by the LockGuard above, so MaybeSaveLocked() (not Save()) here.
	MaybeSaveLocked();

	return *iter->second;
}
} // namespace Libs::Graphics
