#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPUTERESCALE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPUTERESCALE_H_

#include "common/abi.h"
#include "common/emulatorConfig.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

// Resolution rescaling of per-pixel tile compute, renderer side.
//
// The shader recompiler proves some compute programs compute every guest pixel independently
// (shader/recompiler/ir/passes/TileRescale.h) and makes them follow a runtime control word
// (IR::RescaleControl). With the word clear such a program runs exactly as translated, against
// the native twins of scaled images. With the remap bit set it runs one invocation per k x k
// block (k = 2^s, render scale 2^-s) and shifts its texel addresses into the images the mask bits
// name, which the renderer then binds scaled (TextureCache::TexelSpace::ScaledGuest).
//
// This file holds what decides the word. The gate (EvaluateGate) is a pure function of the
// bound images; ProgramState is the per-program state machine for the runtime proof the
// recompiler leaves to the renderer (cross-lane classes C1/C2, TileRescaleInfo::
// NeedsRuntimeCheck()); TileRescaleController owns both and the GPU side of that proof, the
// dual-run check: a native dispatch replayed remap-only into a private copy of its output and
// compared bit for bit at the representative pixels.
// Design: research/remap-detector-2026-09-26.md, sections 2.6 and 2.7.

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class CommandScheduler;
class Image;
class TextureCache;
struct GraphicContext;
using ImageId = Common::SlotId;

namespace TileRescale {

// Largest s a control word can carry (RescaleControl::ScaleBits) and a scaled image can have.
inline constexpr uint32_t MaxScaleLog2 = 3;
// Mask bits in the control word (RescaleControl::ImageMaskBits).
inline constexpr uint32_t MaskBits = 16;
// Dual-run checks a program that needs one must pass before Auto rescales it.
inline constexpr uint32_t ChecksToVerify = 8;
// Dispatches between two summary lines of one program.
inline constexpr uint64_t SummaryInterval = 4096;

[[nodiscard]] constexpr float ScaleOf(uint32_t scale_log2) {
	return 1.0F / static_cast<float>(1u << scale_log2);
}

// One image binding of a dispatch, as the gate sees it. For a binding served by a native scale
// twin, the image properties are those of the twin's scaled owner and `twin` is set.
struct BoundImage {
	uint64_t image         = 0; // identity of the host image the binding resolved to
	uint64_t guest_address = 0; // the guest range the descriptor names
	uint64_t guest_size    = 0;
	uint32_t width         = 0; // guest extent
	uint32_t height        = 0;
	float    scale         = 1.0F;
	uint32_t samples       = 1;
	uint32_t levels        = 1;
	bool     storage         = false;
	bool     texel_addressed = false;
	bool     written         = false;
	bool     twin            = false;
	bool     color_2d        = false;
	bool     depth           = false;
	bool     block           = false;

	[[nodiscard]] bool Overlaps(const BoundImage& other) const {
		return guest_size != 0 && other.guest_size != 0 &&
		       guest_address < other.guest_address + other.guest_size &&
		       other.guest_address < guest_address + guest_size;
	}
	// What a remapped store needs from the image it writes, whatever resolution it is bound at.
	[[nodiscard]] bool RemapWritable(uint32_t scale_log2) const {
		const uint32_t align = (1u << scale_log2) - 1u;
		return color_2d && !depth && samples == 1 && levels == 1 && !block &&
		       (width & align) == 0 && (height & align) == 0;
	}
};

// Why a dispatch of a rescalable program ran natively, or why its dual-run check was skipped.
enum class Reason : uint8_t {
	RenderScale,         // the render scale is not exactly 2^-s for the program's s
	Blacklisted,         // a dual-run check found a mismatch earlier in the session
	ChecksPending,       // Auto: all checks submitted, results not back yet
	NoWrittenImage,      // nothing to remap
	WrittenNotStorage,   // a written image is not bound as a storage image
	WrittenNotScaled,    // a written image is bound native, or scaled at another factor
	WrittenIneligible,   // a written image is not a single-sample, single-level colour 2D image
	ExtentNotAligned,    // a written image's guest extent is not a multiple of k
	TexelAlias,          // a texel-addressed binding of a written range resolved elsewhere
	SampledAlias,        // a written range is also sampled: it would read pixels being written
	MaskOverflow,        // a binding past the control word's mask bits would need its bit
	Rebound,             // the images changed while they were being bound
	// From here on: reasons a dual-run check was skipped (the dispatch itself ran natively).
	CheckMultipleWrites, // the check handles programs writing one image
	CheckFormat,         // no raw view to compare the written image's bits through
	CheckNoSlot,         // every result slot is waiting for the GPU
	Count,
};
inline constexpr size_t ReasonCount = static_cast<size_t>(Reason::Count);

[[nodiscard]] constexpr const char* ReasonName(Reason reason) {
	constexpr std::array<const char*, ReasonCount> names {
	    "render-scale",
	    "blacklisted",
	    "checks-pending",
	    "no-written-image",
	    "written-not-storage",
	    "written-not-scaled",
	    "written-ineligible",
	    "extent-not-aligned",
	    "texel-alias",
	    "sampled-alias",
	    "mask-overflow",
	    "rebound",
	    "check-multiple-writes",
	    "check-format",
	    "check-no-slot",
	};
	return names[static_cast<size_t>(reason)];
}

struct GateDecision {
	bool     on         = false;
	Reason   reason     = Reason::Count; // why not, when !on
	uint32_t image_mask = 0;             // RescaleControl mask, when on
};

// Whether a dispatch of a program proven for downscale 2^-scale_log2 may run remapped against
// the images it was bound to (TexelSpace::ScaledGuest), and with which mask:
//  - the render scale is exactly 2^-s;
//  - every written image is bound as storage to a scaled image itself (not a twin) at exactly
//    2^-s that RemapWritable() accepts;
//  - every other texel-addressed binding of a written image's guest range resolved to that same
//    image, and no sampled binding reads that range (it would see the pixels being written);
//  - mask bit i is set exactly for a storage or texel-addressed binding i bound scaled at 2^-s.
[[nodiscard]] inline GateDecision EvaluateGate(float render_scale, uint32_t scale_log2,
                                               std::span<const BoundImage> images) {
	const auto off = [](Reason reason) { return GateDecision {false, reason, 0}; };
	if (scale_log2 == 0 || scale_log2 > MaxScaleLog2 || render_scale != ScaleOf(scale_log2)) {
		return off(Reason::RenderScale);
	}
	const float scale   = ScaleOf(scale_log2);
	bool        written = false;
	for (const auto& image: images) {
		if (!image.written) {
			continue;
		}
		written = true;
		if (!image.storage) {
			return off(Reason::WrittenNotStorage);
		}
		if (image.twin || image.scale != scale) {
			return off(Reason::WrittenNotScaled);
		}
		if (!image.RemapWritable(0)) {
			return off(Reason::WrittenIneligible);
		}
		if (!image.RemapWritable(scale_log2)) {
			return off(Reason::ExtentNotAligned);
		}
		for (const auto& other: images) {
			if (&other == &image || !other.Overlaps(image)) {
				continue;
			}
			if (!other.storage && !other.texel_addressed) {
				return off(Reason::SampledAlias);
			}
			if (other.image != image.image) {
				return off(Reason::TexelAlias);
			}
		}
	}
	if (!written) {
		return off(Reason::NoWrittenImage);
	}
	uint32_t mask = 0;
	for (uint32_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if ((image.storage || image.texel_addressed) && !image.twin && image.scale == scale) {
			if (i >= MaskBits) {
				return off(Reason::MaskOverflow);
			}
			mask |= 1u << i;
		}
	}
	return {true, Reason::Count, mask};
}

struct CheckTarget {
	bool     eligible = false;
	Reason   reason   = Reason::Count;
	uint32_t written  = 0; // binding index of the written image
};

// Whether a natively bound dispatch (TexelSpace::Guest) can be dual-run checked: it writes a
// single image, through a native twin whose scaled owner the gate would accept at 2^-s, and
// every texel-addressed binding of that range is the twin too (the replay swaps them all for
// the scratch copy, so a read-modify-write reads the pre-dispatch pixels in both runs).
[[nodiscard]] inline CheckTarget EvaluateCheck(float render_scale, uint32_t scale_log2,
                                               std::span<const BoundImage> images) {
	const auto off = [](Reason reason) { return CheckTarget {false, reason, 0}; };
	if (scale_log2 == 0 || scale_log2 > MaxScaleLog2 || render_scale != ScaleOf(scale_log2)) {
		return off(Reason::RenderScale);
	}
	CheckTarget target;
	bool        found = false;
	for (uint32_t i = 0; i < images.size(); i++) {
		if (!images[i].written) {
			continue;
		}
		if (found) {
			return off(Reason::CheckMultipleWrites);
		}
		found          = true;
		target.written = i;
	}
	if (!found) {
		return off(Reason::NoWrittenImage);
	}
	const auto& image = images[target.written];
	if (!image.storage) {
		return off(Reason::WrittenNotStorage);
	}
	if (!image.twin || image.scale != ScaleOf(scale_log2)) {
		return off(Reason::WrittenNotScaled);
	}
	if (!image.RemapWritable(0)) {
		return off(Reason::WrittenIneligible);
	}
	if (!image.RemapWritable(scale_log2)) {
		return off(Reason::ExtentNotAligned);
	}
	for (const auto& other: images) {
		if (&other == &image || !other.Overlaps(image)) {
			continue;
		}
		if (!other.storage && !other.texel_addressed) {
			return off(Reason::SampledAlias);
		}
		if (other.image != image.image) {
			return off(Reason::TexelAlias);
		}
	}
	target.eligible = true;
	return target;
}

// How one dispatch of a rescalable program runs.
enum class Plan : uint8_t {
	Native,          // as translated (control word 0)
	Rescale,         // remapped at the render resolution, if the gate agrees
	NativeWithCheck, // as translated, plus a dual-run check
};

enum class CheckOutcome : uint8_t {
	Passed,
	Verified,     // Auto: this pass completed the checks the program needed
	Blacklisted,  // the first mismatch of the program
	Inconclusive, // the compare saw no pixel; counts neither way
	Ignored,      // a result for a program already blacklisted
};

// Per-program state machine of the rescale decision and its runtime proof.
struct ProgramState {
	uint64_t                          dispatches       = 0;
	uint64_t                          rescaled         = 0;
	uint32_t                          checks_submitted = 0;
	uint32_t                          checks_passed    = 0;
	bool                              verified         = false;
	bool                              blacklisted      = false;
	std::array<uint64_t, ReasonCount> off {};
	std::bitset<ReasonCount>          logged;

	// Counts one dispatch and picks its plan. Verify checks every dispatch and never rescales.
	// Auto rescales a program whose proof is complete (static, or verified by checks) and not
	// refuted, and until then checks its first ChecksToVerify dispatches natively.
	[[nodiscard]] Plan Decide(Config::ComputeRescale mode, bool needs_runtime_check) {
		dispatches++;
		switch (mode) {
			case Config::ComputeRescale::Off: return Plan::Native;
			case Config::ComputeRescale::Verify: return Plan::NativeWithCheck;
			case Config::ComputeRescale::Auto: break;
		}
		if (blacklisted) {
			return Plan::Native;
		}
		if (!needs_runtime_check || verified) {
			return Plan::Rescale;
		}
		return checks_submitted < ChecksToVerify ? Plan::NativeWithCheck : Plan::Native;
	}

	// Counts a native dispatch or a skipped check; true on the first occurrence of `reason`.
	bool CountOff(Reason reason) {
		const auto index = static_cast<size_t>(reason);
		off[index]++;
		if (logged.test(index)) {
			return false;
		}
		logged.set(index);
		return true;
	}

	[[nodiscard]] CheckOutcome NoteCheckResult(Config::ComputeRescale mode, uint32_t mismatches,
	                                           uint32_t compared) {
		if (blacklisted) {
			return CheckOutcome::Ignored;
		}
		if (mismatches != 0) {
			blacklisted = true;
			return CheckOutcome::Blacklisted;
		}
		if (compared == 0) {
			return CheckOutcome::Inconclusive;
		}
		checks_passed++;
		if (mode == Config::ComputeRescale::Auto && !verified && checks_passed >= ChecksToVerify) {
			verified = true;
			return CheckOutcome::Verified;
		}
		return CheckOutcome::Passed;
	}

	[[nodiscard]] bool SummaryDue() const { return dispatches % SummaryInterval == 0; }
};

} // namespace TileRescale

// Owned by RenderExecutor: the per-program rescale state of the session and the GPU resources
// of the dual-run check. Every GPU object is created on the first check, so a session that
// never checks anything allocates nothing.
class TileRescaleController {
public:
	TileRescaleController();
	~TileRescaleController();
	KYTY_CLASS_NO_COPY(TileRescaleController);

	// --compute-rescale, read on first use: nothing reaches the controller before a program the
	// recompiler rescaled, which it only does once the configuration is loaded.
	[[nodiscard]] Config::ComputeRescale Mode();

	// The state of one compiled program; `identity` tells apart programs sharing a hash that
	// were compiled with different options.
	[[nodiscard]] TileRescale::ProgramState& Track(uint64_t hash, const void* identity);
	// Counts a native dispatch or skipped check and logs the first one per program and reason.
	void Off(TileRescale::ProgramState& state, uint64_t hash, TileRescale::Reason reason);
	// Called once per dispatch of a tracked program after it is recorded: logs the summary.
	void NoteDispatched(const TileRescale::ProgramState& state, uint64_t hash) const;

	// Reads back every check whose command buffer has completed and applies its result; frees
	// idle scratch images. Cheap when nothing is outstanding.
	void Poll(CommandScheduler& scheduler, TextureCache& cache);

	// Whether the compare pass can read `image`'s stored bits (RawViewFormat()).
	[[nodiscard]] static bool CanCompare(const Image& image);
	[[nodiscard]] bool        HasFreeSlot() const;
	// A scratch image shaped like `native`, reused across checks of the same shape.
	[[nodiscard]] ImageId AcquireScratch(TextureCache& cache, const Image& native);
	// Records the compare pass of a check whose native dispatch wrote `native` and whose
	// remap-only replay wrote `scratch`, and remembers it for Poll(). Requires HasFreeSlot().
	void RecordCompare(CommandScheduler& scheduler, Image& native, Image& scratch,
	                   uint32_t scale_log2, TileRescale::ProgramState& state, uint64_t hash);

private:
	struct PendingCheck {
		TileRescale::ProgramState* state    = nullptr;
		uint64_t                   hash     = 0;
		uint64_t                   dispatch = 0;
		uint64_t                   tick     = 0;
		uint32_t                   slot     = 0;
	};
	struct ProgramKey {
		uint64_t    hash     = 0;
		const void* identity = nullptr;
		bool        operator==(const ProgramKey&) const = default;
	};
	struct ProgramKeyHash {
		size_t operator()(const ProgramKey& key) const noexcept {
			return std::hash<uint64_t> {}(key.hash) ^
			       (std::hash<const void*> {}(key.identity) << 1u);
		}
	};
	struct Scratch {
		vk::Format format = vk::Format::eUndefined;
		uint32_t   width  = 0;
		uint32_t   height = 0;
		uint32_t   layers = 0;
		ImageId    image {};
	};

	// The unsigned-integer format of `format`'s texel size, through which the compare pass reads
	// stored bits whatever they encode; eUndefined when there is none.
	[[nodiscard]] static vk::Format RawViewFormat(vk::Format format);
	void                            InitializeGpu(CommandScheduler& scheduler);

	std::optional<Config::ComputeRescale> m_mode;
	// Node-based, so the PendingCheck::state pointers stay valid.
	std::unordered_map<ProgramKey, TileRescale::ProgramState, ProgramKeyHash> m_programs;
	std::vector<PendingCheck> m_pending;
	std::vector<bool>         m_slot_busy;
	std::vector<Scratch>      m_scratch;
	// Poll() calls since the last check; scratch images are freed once this grows large.
	uint64_t                  m_idle_polls = 0;

	GraphicContext*         m_graphics        = nullptr;
	std::unique_ptr<Buffer> m_results;
	vk::Sampler             m_sampler         = nullptr;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMPUTERESCALE_H_
