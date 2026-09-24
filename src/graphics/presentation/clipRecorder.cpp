#include "graphics/presentation/clipRecorder.h"

#include "common/childProcess.h"
#include "common/logging/log.h"
#include "graphics/presentation/window/frameTimeLog.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// <windows.h> must stay lean and free of min/max/MemoryBarrier macros.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <dxgi.h>
#endif

#include <SDL3/SDL.h>

// Why an external ffmpeg capturing the window instead of reading back the swapchain:
//
// * Cost on the emulator. Swapchain readback needs a copy into a host-visible buffer per frame,
//   a fence wait (or a frame of latency), RGB->YUV conversion and an encoder feed on some
//   emulator thread; any hitch there is a hitch in the game, and it is exactly the frame pacing we
//   want to record. Out of process, ffmpeg does all of it on its own threads, and the capture
//   APIs keep the frames on the GPU: on Windows ddagrab hands D3D11 textures straight to NVENC;
//   on Linux x11grab copies via XShm in ffmpeg's address space, never ours.
// * Zero change to the present path: the only hook is OnPresent() appending a timestamp to the
//   frame-time sidecar, so recording cannot break presentation or change its timing.
// * Robustness: the MP4 is fragmented, so a clip survives the emulator crashing or EXIT()ing
//   mid-recording, and ffmpeg (own process group / kill-on-close job) is not taken down by a
//   Ctrl+C before it has flushed.
//
// Costs accepted: the clip is what is on screen (overlay included when visible, occluding windows
// on X11 without a compositor, and the window must stay on one monitor on Windows); the capture
// source is OS-specific (x11grab needs X11/XWayland, not native Wayland); and resizing the
// window mid-recording is not followed.

namespace Libs::Graphics {

namespace {

// Stop() asks ffmpeg to finish; if it has not finalised the file by then it is killed (the
// fragmented MP4 is still playable up to the last complete fragment).
constexpr uint32_t STOP_TIMEOUT_MS = 10000;
// How often the monitor thread checks whether ffmpeg died on its own while recording.
constexpr uint32_t MONITOR_POLL_US  = 50000;
constexpr uint32_t NVIDIA_VENDOR_ID = 0x10de;

uint64_t NowUs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// The ffmpeg input for the emulator window: arguments before the encoder, a filter the source
// always needs (empty = none), and whether its frames are GPU (D3D11) frames that a software
// encoder must first download.
struct CaptureSource {
	std::vector<std::string> input_args;
	std::string              filter;
	bool                     gpu_frames = false;
};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// The emulator is not per-monitor DPI aware, so window/monitor rectangles would come back in
// scaled logical units while Desktop Duplication works in physical pixels. Switch only this
// thread, only for the duration of the geometry queries. Resolved dynamically: the API is
// Windows 10 1607+.
class ScopedPhysicalDpi final {
public:
	ScopedPhysicalDpi() {
		HMODULE user32 = GetModuleHandleW(L"user32.dll");
		if (user32 != nullptr) {
			m_set = reinterpret_cast<SetContextFn>(
			    reinterpret_cast<void*>(GetProcAddress(user32, "SetThreadDpiAwarenessContext")));
		}
		if (m_set != nullptr) {
			// DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (DPI_AWARENESS_CONTEXT)-4
			m_previous = m_set(reinterpret_cast<HANDLE>(static_cast<intptr_t>(-4)));
		}
	}
	~ScopedPhysicalDpi() {
		if (m_set != nullptr && m_previous != nullptr) {
			m_set(m_previous);
		}
	}
	KYTY_CLASS_NO_COPY(ScopedPhysicalDpi);

private:
	using SetContextFn = HANDLE(WINAPI*)(HANDLE);
	SetContextFn m_set      = nullptr;
	HANDLE       m_previous = nullptr;
};

template <typename T>
struct ComRelease {
	void operator()(T* p) const { p->Release(); }
};
template <typename T>
using ComPtr = std::unique_ptr<T, ComRelease<T>>;

// ddagrab's output_idx counts the outputs of adapter 0 (the adapter of the D3D11 device it
// creates). Falls back to 0 when DXGI is unavailable or the monitor hangs off another adapter.
uint32_t DxgiOutputIndex(HMONITOR monitor) {
	HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
	if (dxgi == nullptr) {
		return 0;
	}
	using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
	auto create_factory =
	    reinterpret_cast<CreateFactoryFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
	uint32_t       index       = 0;
	IDXGIFactory1* raw_factory = nullptr;
	if (create_factory != nullptr &&
	    SUCCEEDED(create_factory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&raw_factory)))) {
		ComPtr<IDXGIFactory1> factory(raw_factory);
		IDXGIAdapter1*        raw_adapter = nullptr;
		if (SUCCEEDED(factory->EnumAdapters1(0, &raw_adapter))) {
			ComPtr<IDXGIAdapter1> adapter(raw_adapter);
			IDXGIOutput*          raw_output = nullptr;
			for (UINT i = 0; adapter->EnumOutputs(i, &raw_output) != DXGI_ERROR_NOT_FOUND; i++) {
				ComPtr<IDXGIOutput> output(raw_output);
				DXGI_OUTPUT_DESC    desc {};
				if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == monitor) {
					index = i;
					break;
				}
			}
		}
	}
	FreeLibrary(dxgi);
	return index;
}

std::optional<CaptureSource> DescribeCapture(SDL_Window* window, uint32_t fps, std::string* error) {
	HWND hwnd = window != nullptr ? static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(window),
	                                                                         SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr))
	                              : nullptr;
	if (hwnd == nullptr) {
		*error = "window capture not supported on this display";
		return std::nullopt;
	}

	ScopedPhysicalDpi physical_dpi;
	RECT              client {};
	POINT             origin {0, 0};
	if (GetClientRect(hwnd, &client) == FALSE || ClientToScreen(hwnd, &origin) == FALSE) {
		*error = "cannot read the window position";
		return std::nullopt;
	}
	HMONITOR    monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
	MONITORINFO monitor_info {};
	monitor_info.cbSize = sizeof(monitor_info);
	if (GetMonitorInfoW(monitor, &monitor_info) == FALSE) {
		*error = "cannot read the monitor geometry";
		return std::nullopt;
	}
	// ddagrab captures one monitor: clip the client area to it; encoders need even sizes.
	const RECT& screen = monitor_info.rcMonitor;
	const LONG  left   = origin.x > screen.left ? origin.x : screen.left;
	const LONG  top    = origin.y > screen.top ? origin.y : screen.top;
	const LONG  right  = origin.x + client.right < screen.right ? origin.x + client.right : screen.right;
	const LONG  bottom = origin.y + client.bottom < screen.bottom ? origin.y + client.bottom : screen.bottom;
	const LONG  width  = (right - left) & ~1L;
	const LONG  height = (bottom - top) & ~1L;
	if (width <= 0 || height <= 0) {
		*error = "window is not on a monitor";
		return std::nullopt;
	}

	CaptureSource source;
	source.gpu_frames = true;
	source.input_args = {"-f", "lavfi", "-i",
	                     "ddagrab=output_idx=" + std::to_string(DxgiOutputIndex(monitor)) +
	                         ":framerate=" + std::to_string(fps) + ":draw_mouse=0" +
	                         ":offset_x=" + std::to_string(left - screen.left) +
	                         ":offset_y=" + std::to_string(top - screen.top) +
	                         ":video_size=" + std::to_string(width) + "x" + std::to_string(height)};
	return source;
}

#else

std::optional<CaptureSource> DescribeCapture(SDL_Window* window, uint32_t fps, std::string* error) {
	const auto xid = window != nullptr ? SDL_GetNumberProperty(SDL_GetWindowProperties(window),
	                                                           SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)
	                                   : 0;
	if (xid == 0) {
		*error = "window capture not supported on this display";
		return std::nullopt;
	}

	const char*   display = std::getenv("DISPLAY");
	CaptureSource source;
	// x11grab reads the window's own contents (follows it when moved); frames are bgr0,
	// which both NVENC and libx264 (-pix_fmt yuv420p) accept.
	source.input_args = {"-f", "x11grab", "-framerate", std::to_string(fps), "-draw_mouse", "0", "-window_id",
	                     std::to_string(xid), "-i", display != nullptr && display[0] != '\0' ? display : ":0"};
	// The window may have odd dimensions, which libx264 rejects for 4:2:0; crop is zero-copy.
	source.filter = "crop=trunc(iw/2)*2:trunc(ih/2)*2";
	return source;
}

#endif

// Filter + encoder arguments.
std::vector<std::string> EncoderArgs(const CaptureSource& source, uint32_t gpu_vendor_id, uint32_t fps) {
	const bool  hardware = gpu_vendor_id == NVIDIA_VENDOR_ID;
	std::string filter   = source.filter;
	if (source.gpu_frames && !hardware) {
		filter += std::string(filter.empty() ? "" : ",") + "hwdownload,format=bgra";
	}
	std::vector<std::string> args;
	if (!filter.empty()) {
		args = {"-vf", filter};
	}
	if (hardware) {
		// NVENC takes both bgr0 system frames and ddagrab's D3D11 frames directly and converts
		// to 4:2:0 on the GPU (the stream is yuv420p).
		args.insert(args.end(), {"-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0",
		                         "-g", std::to_string(2 * fps)});
	} else {
		args.insert(args.end(), {"-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"});
	}
	return args;
}

std::tm LocalTime(std::time_t t) {
	std::tm tm {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	localtime_s(&tm, &t);
#else
	localtime_r(&t, &tm);
#endif
	return tm;
}

// "<dir>/kyty-YYYYmmdd-HHMMSS.mp4", with a numeric suffix if a clip from the same second exists.
std::filesystem::path NewClipPath(const std::filesystem::path& dir) {
	const std::tm tm = LocalTime(std::time(nullptr));
	char          stamp[32];
	std::strftime(stamp, sizeof(stamp), "kyty-%Y%m%d-%H%M%S", &tm);
	auto            path = dir / (std::string(stamp) + ".mp4");
	std::error_code ec;
	for (int n = 2; std::filesystem::exists(path, ec); n++) {
		path = dir / (std::string(stamp) + "-" + std::to_string(n) + ".mp4");
	}
	return path;
}

std::filesystem::path WithSuffix(std::filesystem::path clip, const char* suffix) {
	return clip.replace_extension(suffix);
}

} // namespace

struct ClipRecorder::Impl {
	SDL_Window*          window;
	ClipRecorderSettings settings;

	mutable Common::Mutex         mutex;
	Common::CondVar               wake;                   // signals stop_requested to the monitor
	ClipStatus                    status;                 // guarded by mutex
	bool                          stop_requested = false; // guarded by mutex
	std::unique_ptr<FrameTimeLog> sidecar;                // guarded by mutex; freed by the monitor
	// Owned by the monitor thread while it runs; Start() only touches it after joining.
	std::unique_ptr<Common::ChildProcess> child;
	std::filesystem::path                 log_path;
	std::unique_ptr<Common::Thread>       monitor;

	void SetFailed(std::string message) {
		Common::LockGuard lock(mutex);
		status.state   = ClipStatus::State::Failed;
		status.message = std::move(message);
	}

	static void MonitorMain(void* arg) { static_cast<Impl*>(arg)->Monitor(); }

	// Watches ffmpeg while recording (an early exit means bad arguments / no encoder / the window
	// went away), then performs the stop handshake so neither Stop() nor the present thread ever
	// blocks on ffmpeg.
	void Monitor() {
		std::optional<int> early_exit;
		for (;;) {
			{
				Common::LockGuard lock(mutex);
				if (!stop_requested) {
					wake.WaitFor(&mutex, MONITOR_POLL_US);
				}
				if (stop_requested) {
					break;
				}
			}
			early_exit = child->Wait(0);
			if (early_exit.has_value()) {
				break;
			}
		}

		bool killed = false;
		if (!early_exit.has_value()) {
			// "q" makes ffmpeg flush the encoder and write the last fragment.
			child->WriteStdin("q");
			early_exit = child->Wait(STOP_TIMEOUT_MS);
			if (!early_exit.has_value()) {
				child->Kill();
				killed = true;
			}
		}
		Finish(early_exit, killed);
	}

	void Finish(std::optional<int> exit_code, bool killed) {
		std::unique_ptr<FrameTimeLog> finished_sidecar;
		std::string                   log_message;
		{
			Common::LockGuard lock(mutex);
			finished_sidecar = std::move(sidecar);
			if (stop_requested && exit_code == 0) {
				status.state   = ClipStatus::State::Saved;
				status.message = status.file.string();
			} else {
				status.state = ClipStatus::State::Failed;
				if (killed) {
					status.message = "ffmpeg did not stop, clip may be truncated: " + status.file.string();
				} else {
					status.message = "ffmpeg exited, see " + log_path.string();
				}
			}
			log_message = status.message;
		}
		finished_sidecar.reset(); // joins its writer thread, outside our lock
		LOGF("clip: %s\n", log_message.c_str());
	}

	[[nodiscard]] bool Busy() const {
		Common::LockGuard lock(mutex);
		return status.state == ClipStatus::State::Recording || status.state == ClipStatus::State::Finishing;
	}

	void JoinMonitor() {
		if (monitor != nullptr) {
			monitor->Join();
			monitor.reset();
		}
		child.reset();
	}
};

ClipRecorder::ClipRecorder(SDL_Window* window, ClipRecorderSettings settings): m_impl(std::make_unique<Impl>()) {
	m_impl->window   = window;
	m_impl->settings = std::move(settings);
	if (m_impl->settings.framerate == 0) {
		m_impl->settings.framerate = 60;
	}
}

ClipRecorder::~ClipRecorder() {
	// Bounded: the monitor kills ffmpeg STOP_TIMEOUT_MS after the stop request.
	Stop();
	m_impl->JoinMonitor();
}

void ClipRecorder::Toggle() {
	const auto state = Status().state;
	if (state == ClipStatus::State::Recording) {
		Stop();
	} else if (state != ClipStatus::State::Finishing) {
		Start();
	}
}

void ClipRecorder::Start() {
	auto& impl = *m_impl;
	if (impl.Busy()) {
		return;
	}
	impl.JoinMonitor(); // previous clip's monitor has finished (state is Saved/Failed)
	const auto& settings = impl.settings;

	std::optional<std::filesystem::path> ffmpeg =
	    settings.ffmpeg.empty() ? Common::FindExecutable("ffmpeg") : std::optional(settings.ffmpeg);
	if (!ffmpeg.has_value()) {
		impl.SetFailed("ffmpeg not found");
		return;
	}

	std::string error;
	auto        source = DescribeCapture(impl.window, settings.framerate, &error);
	if (!source.has_value()) {
		impl.SetFailed(error);
		return;
	}

	const auto dir = settings.clips_dir.empty() ? Common::GetExecutableDirectory() / "clips" : settings.clips_dir;
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	if (ec) {
		impl.SetFailed("cannot create " + dir.string() + ": " + ec.message());
		return;
	}
	const auto clip = NewClipPath(dir);
	const auto log  = WithSuffix(clip, ".ffmpeg.log");

	std::vector<std::string> argv = {ffmpeg->string(), "-hide_banner", "-loglevel", "warning", "-nostats"};
	argv.insert(argv.end(), source->input_args.begin(), source->input_args.end());
	const auto encoder = EncoderArgs(*source, settings.gpu_vendor_id, settings.framerate);
	argv.insert(argv.end(), encoder.begin(), encoder.end());
	// Fragmented MP4: every keyframe closes a fragment, so the file is playable even if the
	// emulator dies before ffmpeg writes a trailer.
	argv.insert(argv.end(), {"-an", "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-y", clip.string()});

	auto child = Common::ChildProcess::Start(argv, log, &error);
	if (child == nullptr) {
		impl.SetFailed("cannot start ffmpeg: " + error);
		return;
	}
	std::unique_ptr<FrameTimeLog> sidecar;
	if (settings.frame_times_sidecar) {
		sidecar = FrameTimeLog::Open(WithSuffix(clip, ".frametimes.csv"), FrameTimeLog::Format::Csv);
	}

	{
		Common::LockGuard lock(impl.mutex);
		impl.status         = ClipStatus {ClipStatus::State::Recording, NowUs(), clip, {}};
		impl.stop_requested = false;
		impl.sidecar        = std::move(sidecar);
	}
	impl.child    = std::move(child);
	impl.log_path = log;
	impl.monitor  = std::make_unique<Common::Thread>(Impl::MonitorMain, &impl);
	LOGF("clip: recording %s\n", clip.string().c_str());
}

void ClipRecorder::Stop() {
	{
		Common::LockGuard lock(m_impl->mutex);
		if (m_impl->status.state != ClipStatus::State::Recording) {
			return;
		}
		m_impl->status.state   = ClipStatus::State::Finishing;
		m_impl->stop_requested = true;
	}
	m_impl->wake.Signal();
}

ClipStatus ClipRecorder::Status() const {
	Common::LockGuard lock(m_impl->mutex);
	return m_impl->status;
}

bool ClipRecorder::IsRecording() const {
	Common::LockGuard lock(m_impl->mutex);
	return m_impl->status.state == ClipStatus::State::Recording;
}

void ClipRecorder::OnPresent(uint64_t time_us) {
	Common::LockGuard lock(m_impl->mutex);
	if (m_impl->sidecar != nullptr && m_impl->status.state == ClipStatus::State::Recording) {
		m_impl->sidecar->Record(time_us);
	}
}

} // namespace Libs::Graphics
