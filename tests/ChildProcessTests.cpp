#include "common/childProcess.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ChildProcessTests: failed: %s\n", message);
		std::abort();
	}
}

std::string ReadFile(const std::filesystem::path& path) {
	std::ifstream in(path);
	return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::filesystem::path TempPath(const char* name) {
	return std::filesystem::temp_directory_path() / name;
}

std::filesystem::path Shell() {
	auto sh = Common::FindExecutable("sh");
	Check(sh.has_value(), "FindExecutable(\"sh\") found nothing on PATH");
	return *sh;
}

void TestStdinAndLog() {
	const auto log = TempPath("kyty_child_process_test.log");
	std::string error;
	auto child = Common::ChildProcess::Start({Shell().string(), "-c", "read x; echo got:$x"}, log, &error);
	Check(child != nullptr, "Start(sh) failed");
	Check(child->WriteStdin("q\n"), "WriteStdin to a running child failed");
	const auto code = child->Wait(5000);
	Check(code.has_value() && *code == 0, "sh did not exit 0");
	Check(!child->IsRunning(), "IsRunning after exit");
	Check(ReadFile(log).find("got:q") != std::string::npos, "log does not contain the child's output");
	std::filesystem::remove(log);
}

void TestKill() {
	auto child = Common::ChildProcess::Start({Shell().string(), "-c", "exec sleep 30"}, {}, nullptr);
	Check(child != nullptr, "Start(sleep) failed");
	Check(child->IsRunning(), "sleep is not running");
	Check(!child->Wait(50).has_value(), "Wait timed out yet reported an exit");
	const auto before = std::chrono::steady_clock::now();
	child->Kill();
	Check(std::chrono::steady_clock::now() - before < std::chrono::seconds(5), "Kill took too long");
	Check(!child->IsRunning(), "sleep still running after Kill");
	Check(child->Wait(0).has_value(), "no exit code after Kill");
}

void TestWriteAfterExit() {
	auto child = Common::ChildProcess::Start({Shell().string(), "-c", "exit 3"}, {}, nullptr);
	Check(child != nullptr, "Start(exit 3) failed");
	const auto code = child->Wait(5000);
	Check(code.has_value() && *code == 3, "exit code 3 not reported");
	// Several writes: the first may still land in a socket buffer on some kernels.
	bool failed = false;
	for (int i = 0; i < 16 && !failed; i++) {
		failed = !child->WriteStdin("q");
	}
	Check(failed, "writing to an exited child did not fail");
	child->CloseStdin();
	Check(!child->WriteStdin("q"), "write after CloseStdin succeeded");
}

void TestStartFailure() {
	std::string error;
	auto child = Common::ChildProcess::Start({"/nonexistent/kyty-no-such-binary"}, {}, &error);
	Check(child == nullptr, "Start of a missing binary succeeded");
	Check(!error.empty(), "Start failure left no error message");
}

std::string RunCapture(const std::string& command) {
	std::string out;
	FILE* pipe = popen(command.c_str(), "r");
	Check(pipe != nullptr, "popen failed");
	char buffer[256];
	while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
		out += buffer;
	}
	pclose(pipe);
	return out;
}

// Same encoder and movflags as ClipRecorder, with a synthetic source instead of the window.
void TestFfmpegEndToEnd() {
	auto ffmpeg  = Common::FindExecutable("ffmpeg");
	auto ffprobe = Common::FindExecutable("ffprobe");
	if (!ffmpeg || !ffprobe) {
		std::printf("ChildProcessTests: ffmpeg/ffprobe not found, skipping end-to-end test\n");
		return;
	}
	const auto clip = TempPath("kyty_child_process_test.mp4");
	const auto log  = TempPath("kyty_child_process_test.ffmpeg.log");
	std::string error;
	auto child = Common::ChildProcess::Start(
	    {ffmpeg->string(), "-hide_banner", "-loglevel", "warning", "-nostats", "-f", "lavfi", "-i",
	     "testsrc2=size=640x360:rate=30", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v",
	     "0", "-g", "60", "-an", "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-y", clip.string()},
	    log, &error);
	Check(child != nullptr, "Start(ffmpeg) failed");
	std::this_thread::sleep_for(std::chrono::seconds(2));
	Check(child->IsRunning(), "ffmpeg exited early (no NVENC?), see /tmp/kyty_child_process_test.ffmpeg.log");
	Check(child->WriteStdin("q"), "WriteStdin(q) to ffmpeg failed");
	const auto code = child->Wait(10000);
	Check(code.has_value() && *code == 0, "ffmpeg did not exit 0 after q");

	const auto probe = RunCapture(ffprobe->string() +
	                              " -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt"
	                              ":format=duration -of default=nw=1 " +
	                              clip.string());
	std::printf("ChildProcessTests: ffprobe:\n%s", probe.c_str());
	Check(probe.find("codec_name=h264") != std::string::npos, "clip is not h264");
	Check(probe.find("pix_fmt=yuv420p") != std::string::npos, "clip is not yuv420p");
	const auto pos = probe.find("duration=");
	Check(pos != std::string::npos, "no duration");
	Check(std::strtod(probe.c_str() + pos + 9, nullptr) >= 1.5, "clip shorter than 1.5 s");
	std::filesystem::remove(clip);
	std::filesystem::remove(log);
}

} // namespace

int main() {
	TestStdinAndLog();
	TestKill();
	TestWriteAfterExit();
	TestStartFailure();
	TestFfmpegEndToEnd();
	std::printf("ChildProcessTests: all passed\n");
	return 0;
}
