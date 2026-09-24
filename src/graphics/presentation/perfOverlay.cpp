#include "graphics/presentation/perfOverlay.h"

#include "common/emulatorConfig.h"
#include "common/perf/hostStats.h"
#include "common/perf/perfSampler.h"
#include "graphics/host_gpu/renderer/renderScale.h"
#include "graphics/presentation/clipRecorder.h"
#include "imgui.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <fmt/format.h>

// Separator between HUD fields: U+00B7 MIDDLE DOT, inside the default font's Latin-1 range. A
// macro so it can be pasted into printf-style format literals.
#define KYTY_HUD_SEP " \xC2\xB7 "

namespace Libs::Graphics {

namespace {

constexpr float    BASE_FONT_SIZE     = 13.0f; // ImGui's built-in ProggyClean size
constexpr float    GRAPH_MAX_MS       = 100.0f;
constexpr float    GRAPH_TARGET_MS    = 1000.0f / 30.0f;
constexpr uint64_t CLIP_NOTICE_US     = 5'000'000;
constexpr double   BYTES_PER_GIB      = 1024.0 * 1024.0 * 1024.0;
constexpr ImU32    REC_COLOR          = IM_COL32(230, 40, 40, 255);
constexpr ImU32    GRAPH_TARGET_COLOR = IM_COL32(255, 200, 60, 160);

double Gib(uint64_t bytes) {
	return static_cast<double>(bytes) / BYTES_PER_GIB;
}

void RecDot(bool active) {
	const float  line   = ImGui::GetTextLineHeight();
	const float  radius = line * 0.3f;
	const ImVec2 pos    = ImGui::GetCursorScreenPos();
	ImGui::GetWindowDrawList()->AddCircleFilled({pos.x + radius, pos.y + line * 0.5f}, radius,
	                                            active ? REC_COLOR : IM_COL32(150, 150, 150, 255));
	ImGui::Dummy({radius * 2.0f, line});
	ImGui::SameLine();
}

} // namespace

void PerfOverlay::BuildSpecLines(const Common::Perf::HostSpecs& specs) {
	m_build_line = fmt::format("{}" KYTY_HUD_SEP "{}", KYTY_BUILD_LABEL, specs.os);
	m_cpu_line   = fmt::format("{}  {}C/{}T" KYTY_HUD_SEP "RAM {:.1f} GiB", specs.cpu_model,
	                           specs.physical_cores, specs.logical_threads, Gib(specs.ram_bytes));
	m_gpu_line   = fmt::format("{}" KYTY_HUD_SEP "{}" KYTY_HUD_SEP "VRAM {:.1f} GiB",
	                           specs.gpu.name, specs.gpu.driver_version,
	                           Gib(specs.gpu.vram_bytes));

	// Config is immutable after startup, so the fidelity summary is fixed too. Only settings
	// that differ from the guest's own choice are listed.
	switch (Config::GetRtMode()) {
		case Config::RtMode::Full: m_fidelity_line = "RT full"; break;
		case Config::RtMode::Reduced: m_fidelity_line = "RT reduced"; break;
		case Config::RtMode::Off: m_fidelity_line = "RT off"; break;
	}
	if (const auto post = Config::GetPostScale(); post != 1.0f) {
		m_fidelity_line += fmt::format(KYTY_HUD_SEP "post x{:.2f}", post);
	}
	if (const auto shadow = Config::GetShadowMax(); shadow != 0) {
		m_fidelity_line += fmt::format(KYTY_HUD_SEP "shadow {}", shadow);
	}
	if (const auto aniso = Config::GetMaxAnisotropy(); aniso != 0) {
		m_fidelity_line += fmt::format(KYTY_HUD_SEP "aniso {}x", aniso);
	}
	if (const auto bias = Config::GetLodBias(); bias != 0.0f) {
		m_fidelity_line += fmt::format(KYTY_HUD_SEP "lod {:+.2f}", bias);
	}
	if (const auto cap = Config::GetFrameCap(); cap != 0) {
		m_fidelity_line += fmt::format(KYTY_HUD_SEP "cap {} fps", cap);
	}
	if (Config::GetVertexFetchMode() == Config::VertexFetchMode::Gpu) {
		m_fidelity_line += KYTY_HUD_SEP "vertex fetch gpu";
	}
	m_specs_built = true;
}

void PerfOverlay::DrawSpecs(const PerfOverlayData& data) const {
	ImGui::TextUnformatted(m_build_line.c_str());
	ImGui::TextUnformatted(m_cpu_line.c_str());
	ImGui::TextUnformatted(m_gpu_line.c_str());

	const auto  guest  = data.guest_output;
	const float factor = RenderScale::Factor();
	if (factor == 1.0f) {
		ImGui::Text("out %ux%u native" KYTY_HUD_SEP "%s", guest.width, guest.height,
		            m_fidelity_line.c_str());
	} else {
		const auto internal = RenderScale::Apply(guest, factor);
		ImGui::Text("out %ux%u -> x%.2f (%ux%u)" KYTY_HUD_SEP "%s", guest.width, guest.height,
		            static_cast<double>(factor), internal.width, internal.height,
		            m_fidelity_line.c_str());
	}
}

void PerfOverlay::DrawFrameTimes(const PerfOverlayData& data, float scale) {
	const auto& snapshot = data.snapshot;
	if (snapshot.sequence == 0) {
		ImGui::TextDisabled("sampling...");
	} else {
		ImGui::Text("%5.1f fps  %5.1f ms%s", snapshot.last_second.fps, snapshot.last_second.last_ms,
		            snapshot.stalled ? "  (stalled)" : "");
		ImGui::Text("10s: avg %.1f ms" KYTY_HUD_SEP "1%% low %.1f fps" KYTY_HUD_SEP "max %.1f ms",
		            snapshot.window.avg_ms, snapshot.window.low1_fps, snapshot.window.max_ms);
	}

	const ImVec2 size(240.0f * scale, 44.0f * scale);
	ImGui::PlotLines("##frame_times", data.frame_ms.data(), static_cast<int>(data.frame_ms.size()),
	                 0, nullptr, 0.0f, GRAPH_MAX_MS, size);
	// Mark the 30 fps budget inside the plot's inner rectangle (the frame minus its padding).
	const ImVec2 padding = ImGui::GetStyle().FramePadding;
	const ImVec2 top_left {ImGui::GetItemRectMin().x + padding.x,
	                       ImGui::GetItemRectMin().y + padding.y};
	const ImVec2 bottom_right {ImGui::GetItemRectMax().x - padding.x,
	                           ImGui::GetItemRectMax().y - padding.y};
	const float  y = bottom_right.y - (bottom_right.y - top_left.y) * (GRAPH_TARGET_MS / GRAPH_MAX_MS);
	auto*        draw = ImGui::GetWindowDrawList();
	draw->AddLine({top_left.x, y}, {bottom_right.x, y}, GRAPH_TARGET_COLOR);
	draw->AddText({top_left.x + 2.0f, y - ImGui::GetTextLineHeight()}, GRAPH_TARGET_COLOR,
	              "33.3 ms");
}

void PerfOverlay::DrawCounters(const PerfOverlayData& data) {
	const auto& snapshot = data.snapshot;
	if (snapshot.sequence == 0) {
		return;
	}
	const auto& cpu = snapshot.cpu;
	ImGui::Text("CPU %.0f%% (%.0f%% of machine" KYTY_HUD_SEP "system %.0f%%)",
	            static_cast<double>(cpu.process_core_percent),
	            static_cast<double>(cpu.process_system_percent),
	            static_cast<double>(cpu.system_percent));
	const std::string cp = cpu.gpu_thread.has_value()
	                           ? fmt::format("CP {:.0f}%", cpu.gpu_thread->core_percent)
	                           : std::string("CP n/a");
	ImGui::Text("%s" KYTY_HUD_SEP "busiest %s/%llu %.0f%%", cp.c_str(),
	            cpu.busiest.name.empty() ? "?" : cpu.busiest.name.c_str(),
	            static_cast<unsigned long long>(cpu.busiest.id),
	            static_cast<double>(cpu.busiest.core_percent));

	if (snapshot.gpu.has_value()) {
		const auto& gpu  = *snapshot.gpu;
		std::string text = fmt::format("GPU {}%" KYTY_HUD_SEP "VRAM {:.1f}/{:.1f} GiB",
		                               gpu.utilization_percent, Gib(gpu.vram_used_bytes),
		                               Gib(gpu.vram_total_bytes));
		if (gpu.temperature_c.has_value()) {
			text += fmt::format(KYTY_HUD_SEP "{} C", *gpu.temperature_c);
		}
		if (gpu.power_w.has_value()) {
			text += fmt::format(KYTY_HUD_SEP "{} W", *gpu.power_w);
		}
		ImGui::TextUnformatted(text.c_str());
	} else {
		ImGui::TextDisabled("GPU n/a (NVML unavailable)");
	}

	const auto& memory = snapshot.memory;
	ImGui::Text("RAM %.1f/%.1f GiB" KYTY_HUD_SEP "emu %.1f GiB", Gib(memory.system_used_bytes),
	            Gib(memory.system_total_bytes), Gib(memory.process_bytes));
}

void PerfOverlay::DrawClip(const PerfOverlayData& data) {
	const auto& clip   = data.clip;
	const bool  recent = data.now_us - data.clip_state_since_us < CLIP_NOTICE_US;
	switch (clip.state) {
		case ClipStatus::State::Recording: {
			const uint64_t seconds =
			    data.now_us > clip.started_us ? (data.now_us - clip.started_us) / 1'000'000 : 0;
			RecDot(true);
			ImGui::Text("REC %02llu:%02llu", static_cast<unsigned long long>(seconds / 60),
			            static_cast<unsigned long long>(seconds % 60));
			break;
		}
		case ClipStatus::State::Finishing:
			RecDot(false);
			ImGui::TextUnformatted("saving...");
			break;
		case ClipStatus::State::Saved:
			if (recent) {
				ImGui::Text("saved %s", clip.file.filename().string().c_str());
			}
			break;
		case ClipStatus::State::Failed:
			if (recent) {
				ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "rec failed: %s",
				                   clip.message.c_str());
			}
			break;
		case ClipStatus::State::Idle: break;
	}
}

void PerfOverlay::Draw(const PerfOverlayData& data) {
	if (!m_specs_built) {
		BuildSpecLines(data.specs);
	}
	// Integer scaling keeps the bitmap font crisp; only very tall outputs need it.
	const float scale = data.output.height >= 1440 ? 2.0f : 1.0f;

	ImGui::SetNextWindowPos({6.0f * scale, 6.0f * scale}, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.6f);
	// Never focusable or interactive, so it cannot take keyboard/gamepad focus from the IME or
	// error dialogs drawn in the same ImGui frame.
	constexpr ImGuiWindowFlags flags =
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
	    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus |
	    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize;
	ImGui::PushFont(nullptr, BASE_FONT_SIZE * scale);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6.0f * scale, 4.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {4.0f * scale, 1.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f * scale);
	if (ImGui::Begin("##PerfOverlay", nullptr, flags)) {
		DrawSpecs(data);
		ImGui::Separator();
		DrawFrameTimes(data, scale);
		DrawCounters(data);
		DrawClip(data);
		ImGui::TextDisabled("hud %.2f ms" KYTY_HUD_SEP "sampler %.1f ms" KYTY_HUD_SEP
		                    "F10 hud" KYTY_HUD_SEP "F9 rec",
		                    data.hud_cost_ms, static_cast<double>(data.snapshot.sample_cost_ms));
	}
	ImGui::End();
	ImGui::PopStyleVar(3);
	ImGui::PopFont();
}

} // namespace Libs::Graphics

#undef KYTY_HUD_SEP
