#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

using Keymap = std::vector<std::string>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

// Ray-tracing fidelity. Full runs the recompiled BVH traversal as the guest wrote it; Reduced
// keeps the traversal but lets the renderer run BVH-consuming work at a lower scale; Off makes
// every BVH intersection report a miss so RT-dependent passes take their no-hit branch.
enum class RtMode { Full, Reduced, Off };

// Resolution rescaling of per-pixel tile compute (shader recompiler pass TileRescale). Off runs
// every dispatch at guest resolution against native twins of scaled images. Auto runs a program
// the recompiler proved safe at the render resolution, after a dual-run check of its first
// dispatches. Verify never rescales but dual-run checks every dispatch of such a program (a test
// mode: frames look like Off, the log reports mismatches).
enum class ComputeRescale { Off, Auto, Verify };

// Host-side graphics fidelity knobs. Every default reproduces the guest's intent exactly, so a
// default-constructed GraphicsSettings leaves every code path unchanged.
struct GraphicsSettings {
	float  render_scale = 1.0F;
	RtMode rt_mode      = RtMode::Full;
	// Auto is inert at the default render scale 1.0: the pass only runs for a 2^-s downscale.
	ComputeRescale compute_rescale = ComputeRescale::Auto;

	// Sampler-translation overrides. max_anisotropy 0 follows the guest descriptor; any other
	// value is an upper bound on the anisotropic ratio. lod_bias is added to the guest's mip
	// LOD bias, so 0.0 is the guest's own choice.
	uint32_t max_anisotropy = 0;
	float    lod_bias       = 0.0F;

	// Per-pass fidelity. post_scale multiplies the render scale of auxiliary colour targets
	// (the reduced-resolution buffers a game allocates for its post chain); 1.0 leaves them at
	// the base render scale. shadow_max caps the longest host edge of an off-screen depth
	// target in pixels; 0 means no cap.
	float    post_scale     = 1.0F;
	uint32_t shadow_max     = 0;

	// Host frame pacing. 0 presents on every vblank. Any other value is the maximum number of
	// flips per second; guest-visible vblank timing is unaffected.
	uint32_t frame_cap      = 0;
};
// Game patches (loader/patches). A patch is applied to a matching module when it is named here,
// listed in the directory's patches.json, or marked isEnabled="true" in its file.
struct PatchSettings {
	// Empty selects the "patches" directory next to the emulator executable.
	std::filesystem::path              directory;
	std::vector<std::string>           enabled;
	// etaHEN/GoldHEN cheat JSON files (--game-patch), read in addition to the directory. Their
	// mods are enabled by the file's own "enabled" flags.
	std::vector<std::filesystem::path> files;
};

// Where the vertex attribute and V# tables are decoded. Cpu: the command processor reads
// them and binds fixed-function vertex input. Gpu: the vertex shader reads them itself
// through the buffer device address page table, so GPU-written tables never stall the CPU.
enum class VertexFetchMode { Cpu, Gpu };

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	PresentMode            present_mode                = PresentMode::Mailbox;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   vr_enabled                  = false;
	bool                   amd_cpu_enabled             = false;
	uint32_t               vblank_frequency            = 60;
	GraphicsSettings       graphics;
	PatchSettings          patches;
	VertexFetchMode        vertex_fetch                = VertexFetchMode::Cpu;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	// How often (in seconds) an fps/frame/time line is written to the log, using the same
	// counters as the window title. Lets a run be measured from the log when the window
	// title is not reachable, e.g. over SSH on Windows. 0 disables it.
	uint32_t               fps_log_seconds             = 0;
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   tessellation_enabled        = false;
	bool                   playgo_hack_enabled         = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
};

void Load(const ConfigOptions& cfg);

namespace Detail {
// Values the command processor reads per PM4 packet or per draw (the debug-dump switch alone is
// asked ~26k times per frame in a busy scene). Out-of-line getters cost a call and a pointer
// chase through g_config each time, so Load() copies these into a plain global the inline
// getters below read directly. Config is immutable after Load(), which runs on the main thread
// before the graphics threads are created, so a plain (non-atomic) copy is safe.
struct HotValues {
	bool         graphics_debug_dump_enabled = false;
	LogDirection printf_direction            = LogDirection::Silent;
	float        post_scale                  = 1.0F;
	uint32_t     shadow_max                  = 0;
};
inline HotValues g_hot_values;
} // namespace Detail

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     VrEnabled();
bool     AmdCpuEnabled();
uint32_t GetVblankFrequency();
float GetRenderScale();
RtMode GetRtMode();
ComputeRescale GetComputeRescale();
uint32_t GetMaxAnisotropy();
float    GetLodBias();
inline float    GetPostScale() { return Detail::g_hot_values.post_scale; }
inline uint32_t GetShadowMax() { return Detail::g_hot_values.shadow_max; }
uint32_t GetFrameCap();
std::filesystem::path           GetPatchDirectory();
const std::vector<std::string>& GetEnabledPatches();
const std::vector<std::filesystem::path>& GetPatchFiles();
VertexFetchMode GetVertexFetchMode();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

inline bool GraphicsDebugDumpEnabled() {
	return Detail::g_hot_values.graphics_debug_dump_enabled;
}

inline LogDirection   GetPrintfDirection() { return Detail::g_hot_values.printf_direction; }
std::filesystem::path GetPrintfOutputFile();
uint32_t              GetFpsLogSeconds();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool TessellationEnabled();
bool PlayGoHackEnabled();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
