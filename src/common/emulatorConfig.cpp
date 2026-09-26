#include "common/emulatorConfig.h"

#include "common/assert.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

static float ClampPostScale(float scale) {
	if (!std::isfinite(scale)) {
		return 1.0F;
	}
	return std::clamp(scale, 0.1F, 4.0F);
}

static uint32_t ClampShadowMax(uint32_t cap) {
	return cap == 0 ? 0 : std::clamp(cap, 64u, 16384u);
}

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	g_config.reset();
	Detail::g_hot_values = {};
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));

	*g_config = cfg;
	Detail::g_hot_values = {
	    .graphics_debug_dump_enabled = cfg.graphics_debug_dump_enabled,
	    .printf_direction            = cfg.printf_direction,
	    .post_scale                  = ClampPostScale(cfg.graphics.post_scale),
	    .shadow_max                  = ClampShadowMax(cfg.graphics.shadow_max),
	};
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

const std::string& GetAudioInputDevice() {
	return g_config->audio_input_device;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

int32_t GetGpuIndex() {
	return g_config->gpu_index;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

bool VrEnabled() {
	return g_config->vr_enabled;
}

bool AmdCpuEnabled() {
	return g_config->amd_cpu_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

VertexFetchMode GetVertexFetchMode() {
	return g_config->vertex_fetch;
}

float GetRenderScale() {
	const float scale = g_config->graphics.render_scale;
	if (!std::isfinite(scale)) {
		return 1.0F;
	}
	return std::clamp(scale, 0.1F, 4.0F);
}

RtMode GetRtMode() {
	return g_config->graphics.rt_mode;
}

ComputeRescale GetComputeRescale() {
	return g_config->graphics.compute_rescale;
}

uint32_t GetMaxAnisotropy() {
	const uint32_t cap = g_config->graphics.max_anisotropy;
	return cap == 0 ? 0 : std::clamp(cap, 1u, 16u);
}

float GetLodBias() {
	const float bias = g_config->graphics.lod_bias;
	if (!std::isfinite(bias)) {
		return 0.0F;
	}
	return std::clamp(bias, -4.0F, 4.0F);
}


uint32_t GetFrameCap() {
	const uint32_t cap = g_config->graphics.frame_cap;
	return cap == 0 ? 0 : std::clamp(cap, 1u, 480u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

ShaderOptimizationType GetShaderOptimizationType() {
	return g_config->shader_optimization_type;
}

LogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

uint32_t GetFpsLogSeconds() {
	return g_config->fps_log_seconds;
}

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

} // namespace Config
