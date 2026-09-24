#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_CLIPRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_CLIPRECORDER_H_

#include "common/common.h"
#include "common/threads.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

struct SDL_Window;

namespace Common {
class ChildProcess;
} // namespace Common

namespace Libs::Graphics {

class FrameTimeLog;

struct ClipRecorderSettings {
	std::filesystem::path ffmpeg;        // explicit ffmpeg path; empty = search (exe dir, PATH)
	std::filesystem::path clips_dir;     // empty = "<exe dir>/clips"
	uint32_t              framerate = 60;
	uint32_t              gpu_vendor_id = 0; // picks the hardware encoder (0x10de -> h264_nvenc)
	bool                  frame_times_sidecar = true; // write "<clip>.frametimes.csv"
};

struct ClipStatus {
	enum class State : uint8_t {
		Idle,      // nothing recorded yet this session
		Recording, // ffmpeg running
		Finishing, // "q" sent, waiting for ffmpeg to finalise the file
		Saved,     // last clip finished; message holds its path
		Failed,    // message holds the reason ("ffmpeg not found", "ffmpeg exited: see <log>")
	};
	State                 state = State::Idle;
	uint64_t              started_us = 0; // steady_clock us when recording started
	std::filesystem::path file;
	std::string           message;
};

// Records the emulator window to "<clips>/kyty-YYYYmmdd-HHMMSS.mp4" with an external ffmpeg
// process capturing the window from the desktop (Linux: x11grab of the X11 window id;
// Windows: ddagrab of the window's client rectangle on its monitor), encoded on the GPU's
// hardware encoder. Out-of-process capture keeps encoding and colour conversion off the
// emulator's threads entirely; see the .cpp for the trade-off against swapchain readback.
// The clip therefore contains exactly what is on screen, including the overlay when visible.
class ClipRecorder final {
public:
	ClipRecorder(SDL_Window* window, ClipRecorderSettings settings);
	~ClipRecorder(); // finishes an active recording (bounded wait) before returning
	KYTY_CLASS_NO_COPY(ClipRecorder);

	// Main (SDL) thread only: reads the window geometry and spawns ffmpeg.
	void Toggle();
	void Start();
	// Any thread; returns immediately. ffmpeg is asked to stop ("q" on stdin) and a reaper
	// thread waits for it to finalise the MP4 (killing it after a timeout).
	void Stop();

	[[nodiscard]] ClipStatus Status() const; // thread-safe
	[[nodiscard]] bool       IsRecording() const;

	// Present thread: appends the present timestamp to the clip's frame-time sidecar while
	// recording. Costs one uncontended lock when not recording.
	void OnPresent(uint64_t time_us);

private:
	// (implementation-defined; ffmpeg ChildProcess, sidecar FrameTimeLog, reaper thread, all
	// guarded by a mutex)
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_CLIPRECORDER_H_
