#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFMONITOR_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFMONITOR_H_

#include "common/common.h"
#include "common/perf/frameTimeStats.h"
#include "common/perf/hostStats.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/clipRecorder.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

struct SDL_Window;

namespace Common::Perf {
class PerfSampler;
} // namespace Common::Perf

namespace Libs::Graphics {

struct GraphicContext;
class FrameTimeLog;
class PerfOverlay;

// Performance instrumentation of the presented output: frame-time history, --frame-time-log,
// clip recording and the F10 HUD. Owned by WindowContext and created on the main thread once
// the Vulkan device exists; it must outlive the Presenter, which calls into it on every present.
//
// Threads: OnPresent/ShouldDrawOverlay/DrawOverlay/ReportOverlayCost/RecordingTitleSuffix run on
// the present (video-out flip) thread; ToggleOverlay/ToggleRecording/PumpMainThread on the main
// (SDL) thread.
class PerfMonitor final {
public:
	PerfMonitor(const GraphicContext& graphics, SDL_Window* window);
	~PerfMonitor();
	KYTY_CLASS_NO_COPY(PerfMonitor);

	// Right after a successful vkQueuePresentKHR. One timestamp fanned out to the history, the
	// frame-time log and the clip sidecar; a few uncontended locks, no allocation.
	void OnPresent();

	// Whether the user wants the HUD (F10 / --perf-overlay).
	[[nodiscard]] bool OverlayVisible() const noexcept;
	// Whether the HUD goes into this frame: visible, and not hidden for a clip recorded with
	// clip_include_overlay off.
	[[nodiscard]] bool ShouldDrawOverlay() const;
	// Draws the HUD into the current ImGui frame.
	void DrawOverlay(vk::Extent2D output, vk::Extent2D guest_output);
	// CPU time spent preparing and recording the HUD frame, averaged for display.
	void ReportOverlayCost(double ms);
	// " [REC mm:ss]" while recording, else empty.
	[[nodiscard]] std::string RecordingTitleSuffix() const;

	void ToggleOverlay();
	void ToggleRecording();
	// Called every main-loop iteration: starts the --record clip once frames are on screen.
	void PumpMainThread();

private:
	// Host specs and the sampler thread are only needed once the HUD is first shown (NVML and
	// the host probes cost startup time otherwise), so they are created lazily on the main
	// thread and published to the present thread through m_probe.
	struct Probe;
	void EnsureProbe();

	Common::Perf::GpuIdentity             m_gpu;
	Common::Perf::FrameTimeHistory        m_history; // outlives the sampler reading it
	std::unique_ptr<FrameTimeLog>         m_frame_time_log;
	std::unique_ptr<ClipRecorder>         m_recorder;
	std::unique_ptr<Probe>                m_probe_owner; // main thread
	std::atomic<const Probe*>             m_probe {nullptr};
	std::atomic_bool                      m_visible {false};
	std::atomic_bool                      m_presented {false};
	bool                                  m_clip_include_overlay = true;
	bool                                  m_record_pending       = false; // main thread

	// Present thread only.
	std::unique_ptr<PerfOverlay> m_view;
	std::array<float, 240>       m_graph {};
	double                       m_overlay_cost_ms = 0.0;
	ClipStatus::State            m_clip_state      = ClipStatus::State::Idle;
	uint64_t                     m_clip_state_since_us = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFMONITOR_H_
