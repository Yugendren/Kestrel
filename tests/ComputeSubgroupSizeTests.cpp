#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The compute subgroup size a pipeline is pinned to (GraphicContext::ExactComputeSubgroupSize,
// CanRequireComputeSubgroupSize) is a pure function of the device's subgroup properties and the
// program's translation (ShaderWorkgroupInputInfo), so it is checked here for the four device
// profiles the emulator is known to meet, without a Vulkan device.

namespace {

using Libs::Graphics::GraphicContext;
using Libs::Graphics::ShaderComputeInputInfo;

int g_cases = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ComputeSubgroupSizeTests: failed: %s\n", text);
		std::abort();
	}
	g_cases++;
}

// Subgroup properties as VulkanCreateDevice records them (every profile here has compute in
// requiredSubgroupSizeStages and subgroupSizeControl, so size control is enabled).
struct DeviceProfile {
	uint32_t default_size;
	uint32_t min_size;
	uint32_t max_size;
	uint32_t max_workgroup_subgroups;
};

// NVIDIA (RTX 3060, as reported on the VM).
constexpr DeviceProfile Nvidia {32, 32, 32, 2097152};
// AMD RDNA (RADV reports UINT32_MAX subgroups per workgroup).
constexpr DeviceProfile AmdRdna {64, 32, 64, UINT32_MAX};
// Intel Xe (ANV).
constexpr DeviceProfile Intel {32, 8, 32, 64};
// lavapipe (Mesa 25.2, as reported on the VM).
constexpr DeviceProfile Lavapipe {8, 8, 8, 32};

void Configure(GraphicContext& graphics, const DeviceProfile& device) {
	graphics.subgroup_size                         = device.default_size;
	graphics.min_subgroup_size                     = device.min_size;
	graphics.max_subgroup_size                     = device.max_size;
	graphics.max_compute_workgroup_subgroups       = device.max_workgroup_subgroups;
	graphics.compute_subgroup_size_control_enabled = true;
}

struct Pinning {
	uint32_t exact;     // ExactComputeSubgroupSize
	uint32_t requested; // requiredSubgroupSize, 0 when CreatePipelineInternal requests none
};

// A compute program as GetComputeProgram prepares it, pinned as CreatePipelineInternal pins it.
Pinning Pin(const GraphicContext& graphics, uint32_t wave, uint32_t threads_x) {
	ShaderComputeInputInfo info;
	info.threads_num[0]     = threads_x;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.wave_size          = wave;
	info.host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto size         = info.TranslatedSubgroupSize();
	const auto invocations  = info.HostInvocations();
	return {graphics.ExactComputeSubgroupSize(size, invocations),
	        graphics.CanRequireComputeSubgroupSize(size, invocations) ? size : 0u};
}

void TranslationShape() {
	ShaderComputeInputInfo info;
	info.threads_num[0] = 100;
	info.threads_num[1] = 1;
	info.threads_num[2] = 1;
	info.wave_size          = 64;
	info.host_subgroup_size = 32;
	Check(info.SplitsWave64(), "wave64 on a 32-lane host is split");
	Check(info.TranslatedSubgroupSize() == 32, "a split wave64 is translated for 32 lanes");
	Check(info.HostInvocations() == 64, "100 wave64 invocations are two waves of 32 host lanes");
	info.host_subgroup_size = 64;
	Check(!info.SplitsWave64() && info.TranslatedSubgroupSize() == 64 &&
	          info.HostInvocations() == 100,
	      "native wave64 keeps the guest workgroup");
	info.wave_size          = 32;
	info.host_subgroup_size = 32;
	Check(!info.SplitsWave64() && info.TranslatedSubgroupSize() == 32 &&
	          info.HostInvocations() == 100,
	      "wave32 keeps the guest workgroup");
}

void NvidiaPinsEverythingTo32() {
	GraphicContext graphics;
	Configure(graphics, Nvidia);
	Check(!graphics.SupportsComputeWave64(), "NVIDIA splits wave64");
	Check(graphics.GuaranteesComputeSubgroupSize(32), "NVIDIA guarantees 32 lanes");
	const auto w32 = Pin(graphics, 32, 256);
	Check(w32.exact == 32 && w32.requested == 32, "NVIDIA wave32 runs at 32");
	const auto w64 = Pin(graphics, 64, 256);
	Check(w64.exact == 32 && w64.requested == 32, "NVIDIA split wave64 runs at 32");
}

void AmdRunsWavesNatively() {
	GraphicContext graphics;
	Configure(graphics, AmdRdna);
	Check(graphics.SupportsComputeWave64(), "RDNA runs wave64 natively");
	Check(graphics.GuaranteesComputeSubgroupSize(32), "RDNA guarantees 32 lanes");
	const auto w32 = Pin(graphics, 32, 1024);
	Check(w32.exact == 32 && w32.requested == 32, "RDNA wave32 runs at 32");
	const auto w64 = Pin(graphics, 64, 1024);
	Check(w64.exact == 64 && w64.requested == 64, "RDNA wave64 runs at 64");
}

void IntelPinsSplitWave64To32() {
	GraphicContext graphics;
	Configure(graphics, Intel);
	Check(!graphics.SupportsComputeWave64(), "Intel splits wave64");
	Check(graphics.GuaranteesComputeSubgroupSize(32), "Intel can pin 32 lanes");
	const auto w32 = Pin(graphics, 32, 256);
	Check(w32.exact == 32 && w32.requested == 32, "Intel wave32 runs at 32");
	// The split program assumes one guest wave64 per 32-lane host subgroup; without the request
	// the driver may pick 8 or 16.
	const auto w64 = Pin(graphics, 64, 256);
	Check(w64.exact == 32 && w64.requested == 32, "Intel split wave64 runs at 32");
	// 64 subgroups of 32: 2048 wave32 invocations fit, 4096 do not.
	const auto fits = Pin(graphics, 32, 2048);
	Check(fits.exact == 32 && fits.requested == 32, "Intel pins 64 subgroups");
	const auto over = Pin(graphics, 32, 4096);
	Check(over.exact == 0 && over.requested == 0,
	      "Intel leaves a workgroup over maxComputeWorkgroupSubgroups unpinned");
	// The split halves the host workgroup: 4096 wave64 invocations are 64 subgroups of 32.
	const auto split = Pin(graphics, 64, 4096);
	Check(split.exact == 32 && split.requested == 32,
	      "the subgroup limit counts host invocations of a split wave64");
}

void LavapipeCannotHold32Lanes() {
	GraphicContext graphics;
	Configure(graphics, Lavapipe);
	Check(!graphics.SupportsComputeWave64(), "lavapipe splits wave64");
	Check(!graphics.GuaranteesComputeSubgroupSize(32), "lavapipe cannot guarantee 32 lanes");
	const auto w32 = Pin(graphics, 32, 256);
	Check(w32.exact == 8 && w32.requested == 0, "lavapipe wave32 runs at its fixed 8");
	const auto w64 = Pin(graphics, 64, 256);
	Check(w64.exact == 8 && w64.requested == 0, "lavapipe split wave64 runs at its fixed 8");
}

void WithoutSizeControl() {
	GraphicContext graphics;
	Configure(graphics, Intel);
	graphics.compute_subgroup_size_control_enabled = false;
	Check(!graphics.GuaranteesComputeSubgroupSize(32), "a varying size is not guaranteed");
	const auto w32 = Pin(graphics, 32, 256);
	Check(w32.exact == 0 && w32.requested == 0, "the driver chooses among 8..32");
	Configure(graphics, Nvidia);
	graphics.compute_subgroup_size_control_enabled = false;
	Check(graphics.GuaranteesComputeSubgroupSize(32), "a fixed 32 needs no size control");
	const auto fixed = Pin(graphics, 64, 256);
	Check(fixed.exact == 32 && fixed.requested == 0, "a fixed 32 is exact without a request");
}

} // namespace

int main() {
	TranslationShape();
	NvidiaPinsEverythingTo32();
	AmdRunsWavesNatively();
	IntelPinsSplitWave64To32();
	LavapipeCannotHold32Lanes();
	WithoutSizeControl();
	std::printf("ComputeSubgroupSizeTests: %d checks passed\n", g_cases);
	return 0;
}
