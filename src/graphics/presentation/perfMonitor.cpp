#include "graphics/presentation/perfMonitor.h"

#include "common/emulatorConfig.h"
#include "common/perf/perfSampler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/perfOverlay.h"
#include "graphics/presentation/window/frameTimeLog.h"

#include <chrono>
#include <fmt/format.h>
#include <span>
#include <utility>

namespace Libs::Graphics {

namespace {

uint64_t NowUs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Fallback when the driver does not report VkPhysicalDeviceDriverProperties::driverInfo.
// driverVersion is vendor-encoded; NVIDIA and Intel-on-Windows use their own bit layouts.
std::string DecodeDriverVersion(uint32_t vendor_id, uint32_t version) {
	constexpr uint32_t VENDOR_NVIDIA = 0x10de;
	if (vendor_id == VENDOR_NVIDIA) {
		return fmt::format("{}.{}.{}", (version >> 22u) & 0x3ffu, (version >> 14u) & 0xffu,
		                   (version >> 6u) & 0xffu);
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr uint32_t VENDOR_INTEL = 0x8086;
	if (vendor_id == VENDOR_INTEL) {
		return fmt::format("{}.{}", version >> 14u, version & 0x3fffu);
	}
#endif
	return fmt::format("{}.{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
	                   VK_API_VERSION_PATCH(version));
}

Common::Perf::GpuIdentity QueryGpuIdentity(const GraphicContext& graphics) {
	const auto&               properties = graphics.GetPhysicalDeviceProperties();
	Common::Perf::GpuIdentity gpu;
	gpu.name      = properties.deviceName.data();
	gpu.vendor_id = properties.vendorID;

	// Core since Vulkan 1.2, which the emulator requires.
	vk::PhysicalDeviceDriverProperties driver {};
	driver.sType = vk::StructureType::ePhysicalDeviceDriverProperties;
	vk::PhysicalDeviceProperties2 properties2 {};
	properties2.sType = vk::StructureType::ePhysicalDeviceProperties2;
	properties2.pNext = &driver;
	graphics.physical_device.getProperties2(&properties2);
	gpu.driver_version = driver.driverInfo.data();
	if (gpu.driver_version.empty()) {
		gpu.driver_version = DecodeDriverVersion(properties.vendorID, properties.driverVersion);
	}

	const auto& memory = graphics.GetPhysicalDeviceMemoryProperties();
	for (uint32_t i = 0; i < memory.memoryHeapCount; i++) {
		if (memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
			gpu.vram_bytes += memory.memoryHeaps[i].size;
		}
	}
	return gpu;
}

ClipRecorderSettings ClipSettingsFromConfig(uint32_t gpu_vendor_id) {
	ClipRecorderSettings settings;
	settings.ffmpeg        = Config::GetFfmpegPath();
	settings.clips_dir     = Config::GetClipsFolder();
	settings.framerate     = Config::GetVblankFrequency();
	settings.gpu_vendor_id = gpu_vendor_id;
	return settings;
}

} // namespace

struct PerfMonitor::Probe {
	Common::Perf::HostSpecs                    specs;
	std::unique_ptr<Common::Perf::PerfSampler> sampler;
};

PerfMonitor::PerfMonitor(const GraphicContext& graphics, SDL_Window* window)
    : m_gpu(QueryGpuIdentity(graphics)),
      m_recorder(std::make_unique<ClipRecorder>(window, ClipSettingsFromConfig(m_gpu.vendor_id))),
      m_clip_include_overlay(Config::ClipIncludeOverlay()), m_record_pending(Config::RecordOnStart()),
      m_view(std::make_unique<PerfOverlay>()) {
	if (const auto path = Config::GetFrameTimeLogFile(); !path.empty()) {
		m_frame_time_log = FrameTimeLog::Open(path);
	}
	if (Config::PerfOverlayEnabled()) {
		ToggleOverlay();
	}
}

PerfMonitor::~PerfMonitor() = default;

void PerfMonitor::EnsureProbe() {
	if (m_probe_owner != nullptr) {
		return;
	}
	auto probe     = std::make_unique<Probe>();
	probe->specs   = Common::Perf::CollectHostSpecs(m_gpu);
	probe->sampler = std::make_unique<Common::Perf::PerfSampler>(
	    m_history, Common::Perf::NvmlGpuMonitor::Open(m_gpu.name));
	m_probe_owner = std::move(probe);
	m_probe.store(m_probe_owner.get(), std::memory_order_release);
}

void PerfMonitor::OnPresent() {
	const uint64_t now_us = NowUs();
	m_history.RecordPresent(now_us);
	if (m_frame_time_log != nullptr) {
		m_frame_time_log->Record(now_us);
	}
	m_recorder->OnPresent(now_us);
	if (!m_presented.load(std::memory_order_relaxed)) {
		m_presented.store(true, std::memory_order_release);
	}
}

bool PerfMonitor::OverlayVisible() const noexcept {
	return m_visible.load(std::memory_order_acquire);
}

bool PerfMonitor::ShouldDrawOverlay() const {
	// Clips capture the screen, so a HUD excluded from clips is hidden while ffmpeg runs.
	return OverlayVisible() && (m_clip_include_overlay || !m_recorder->IsRecording());
}

void PerfMonitor::ToggleOverlay() {
	const bool visible = !m_visible.load(std::memory_order_relaxed);
	if (visible) {
		EnsureProbe();
	}
	if (m_probe_owner != nullptr) {
		// An inactive sampler blocks on its condition variable, so a hidden HUD costs nothing.
		m_probe_owner->sampler->SetActive(visible);
	}
	m_visible.store(visible, std::memory_order_release);
}

void PerfMonitor::ToggleRecording() {
	m_record_pending = false;
	m_recorder->Toggle();
}

void PerfMonitor::PumpMainThread() {
	// The window only shows the game once something was presented; recording earlier would
	// capture a blank or half-created window.
	if (m_record_pending && m_presented.load(std::memory_order_acquire)) {
		m_record_pending = false;
		m_recorder->Start();
	}
}

void PerfMonitor::ReportOverlayCost(double ms) {
	constexpr double SMOOTHING = 0.1;
	m_overlay_cost_ms =
	    m_overlay_cost_ms == 0.0 ? ms : m_overlay_cost_ms + (ms - m_overlay_cost_ms) * SMOOTHING;
}

void PerfMonitor::DrawOverlay(vk::Extent2D output, vk::Extent2D guest_output) {
	const auto* probe = m_probe.load(std::memory_order_acquire);
	if (probe == nullptr) {
		return;
	}
	const uint64_t now_us   = NowUs();
	const auto     snapshot = probe->sampler->Latest();
	const auto     clip     = m_recorder->Status();
	if (clip.state != m_clip_state) {
		m_clip_state          = clip.state;
		m_clip_state_since_us = now_us;
	}
	const size_t frames = m_history.CopyLatest(std::span<float>(m_graph));

	const PerfOverlayData data {
	    .specs               = probe->specs,
	    .snapshot            = snapshot,
	    .frame_ms            = std::span<const float>(m_graph.data(), frames),
	    .clip                = clip,
	    .clip_state_since_us = m_clip_state_since_us,
	    .now_us              = now_us,
	    .hud_cost_ms         = m_overlay_cost_ms,
	    .output              = output,
	    .guest_output        = guest_output,
	};
	m_view->Draw(data);
}

std::string PerfMonitor::RecordingTitleSuffix() const {
	if (!m_recorder->IsRecording()) {
		return {};
	}
	const auto     status  = m_recorder->Status();
	const uint64_t now_us  = NowUs();
	const uint64_t seconds = now_us > status.started_us ? (now_us - status.started_us) / 1'000'000
	                                                    : 0;
	return fmt::format(" [REC {:02}:{:02}]", seconds / 60, seconds % 60);
}

} // namespace Libs::Graphics
