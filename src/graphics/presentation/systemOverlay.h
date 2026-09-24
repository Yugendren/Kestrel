#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <memory>

union SDL_Event;
struct SDL_Window;

namespace Libs::Graphics {

struct GraphicContext;
class PerfMonitor;

struct SystemOverlayVisualState {
	bool     active;
	uint64_t revision;
};

void                     InitializeSystemOverlayInput(SDL_Window* window);
void                     ShutdownSystemOverlayInput();
bool                     ProcessSystemOverlayInput(const SDL_Event& event);
SystemOverlayVisualState GetSystemOverlayVisualState() noexcept;

class SystemOverlay final {
public:
	explicit SystemOverlay(GraphicContext& graphics);
	~SystemOverlay();
	KYTY_CLASS_NO_COPY(SystemOverlay);

	// Builds the ImGui frame for the active IME/error dialog and, when `hud` is non-null, the
	// performance HUD (both share the one process-wide ImGui context and Vulkan backend).
	// Returns true if anything was drawn and Record() must be called.
	[[nodiscard]] bool PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t image_count,
	                                PerfMonitor* hud, vk::Extent2D guest_output);
	void               Record(vk::CommandBuffer command, vk::ImageView target);
	void               ReleaseVulkan();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
