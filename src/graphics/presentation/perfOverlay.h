#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <string>

namespace Common::Perf {
struct HostSpecs;
struct PerfSnapshot;
} // namespace Common::Perf

namespace Libs::Graphics {

struct ClipStatus;

// Everything the HUD shows for one frame. Plain data gathered by PerfMonitor, so the view
// never touches samplers, locks or the recorder itself.
struct PerfOverlayData {
	const Common::Perf::HostSpecs&    specs;
	const Common::Perf::PerfSnapshot& snapshot;
	std::span<const float>            frame_ms; // newest frame times, oldest first
	const ClipStatus&                 clip;
	uint64_t                          clip_state_since_us = 0; // when clip.state last changed
	uint64_t                          now_us              = 0;
	double                            hud_cost_ms         = 0.0;
	vk::Extent2D                      output {};      // swapchain extent
	vk::Extent2D                      guest_output {}; // guest video-out buffer size
};

// The performance HUD: a small, input-transparent ImGui window in the top-left corner.
// Must be called on the present thread between ImGui::NewFrame() and ImGui::Render().
class PerfOverlay final {
public:
	PerfOverlay()  = default;
	~PerfOverlay() = default;
	KYTY_CLASS_NO_COPY(PerfOverlay);

	void Draw(const PerfOverlayData& data);

private:
	void BuildSpecLines(const Common::Perf::HostSpecs& specs);
	void DrawSpecs(const PerfOverlayData& data) const;
	static void DrawFrameTimes(const PerfOverlayData& data, float scale);
	static void DrawCounters(const PerfOverlayData& data);
	static void DrawClip(const PerfOverlayData& data);

	// Host specs never change, so their text is formatted once.
	bool        m_specs_built = false;
	std::string m_build_line;
	std::string m_cpu_line;
	std::string m_gpu_line;
	std::string m_fidelity_line;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_
