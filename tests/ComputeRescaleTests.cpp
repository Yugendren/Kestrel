#include "graphics/host_gpu/renderer/computeRescale.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace TR = Libs::Graphics::TileRescale;
using Config::ComputeRescale;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ComputeRescaleTests: failed: %s\n", text);
		std::abort();
	}
}

constexpr float    Half  = 0.5F;
constexpr uint32_t Shift = 1;

// A 3840x2160 colour target at `scale`, bound as storage (written or not) or texel-read.
TR::BoundImage Target(uint64_t image, uint64_t address, float scale, bool storage, bool written) {
	TR::BoundImage bound;
	bound.image           = image;
	bound.guest_address   = address;
	bound.guest_size      = 0x2000000;
	bound.width           = 3840;
	bound.height          = 2160;
	bound.scale           = scale;
	bound.storage         = storage;
	bound.texel_addressed = !storage;
	bound.written         = written;
	bound.color_2d        = true;
	return bound;
}

TR::BoundImage Output(float scale = Half) { return Target(1, 0x10000000, scale, true, true); }
TR::BoundImage GBuffer(float scale = Half) { return Target(2, 0x20000000, scale, false, false); }

TR::GateDecision Gate(std::vector<TR::BoundImage> images, float render_scale = Half,
                      uint32_t shift = Shift) {
	return TR::EvaluateGate(render_scale, shift, images);
}

void GateAcceptsScaledOutputAndMasksScaledReads() {
	const auto native_read = Target(3, 0x30000000, 1.0F, false, false);
	const auto d           = Gate({GBuffer(), Output(), native_read});
	Check(d.on, "scaled output and scaled G-buffer read pass the gate");
	Check(d.image_mask == 0b011u, "scaled bindings get mask bits, the native read does not");
}

void GateRefusesRenderScaleMismatch() {
	Check(Gate({Output()}, 1.0F).reason == TR::Reason::RenderScale, "render scale 1.0");
	Check(Gate({Output(0.25F)}, 0.25F).reason == TR::Reason::RenderScale,
	      "a program proven for k=2 is not remapped at 0.25");
	Check(Gate({Output()}, Half, 0).reason == TR::Reason::RenderScale, "s = 0 is not rescaled");
}

void GateRefusesOutputNotBoundScaled() {
	auto twin = Output();
	twin.twin = true;
	Check(Gate({twin}).reason == TR::Reason::WrittenNotScaled, "a twin-bound output");
	Check(Gate({Output(1.0F)}).reason == TR::Reason::WrittenNotScaled, "a native output");
	Check(Gate({Output(0.25F)}).reason == TR::Reason::WrittenNotScaled,
	      "an output scaled at another factor");
}

void GateRefusesIneligibleOutputs() {
	auto sampled_written = Output();
	sampled_written.storage = false;
	Check(Gate({sampled_written}).reason == TR::Reason::WrittenNotStorage, "non-storage write");
	auto depth  = Output();
	depth.depth = true;
	Check(Gate({depth}).reason == TR::Reason::WrittenIneligible, "depth output");
	auto msaa    = Output();
	msaa.samples = 4;
	Check(Gate({msaa}).reason == TR::Reason::WrittenIneligible, "multisample output");
	auto mips   = Output();
	mips.levels = 2;
	Check(Gate({mips}).reason == TR::Reason::WrittenIneligible, "mipped output");
	auto odd  = Output();
	odd.width = 3841;
	Check(Gate({odd}).reason == TR::Reason::ExtentNotAligned, "odd guest width");
	Check(Gate({GBuffer()}).reason == TR::Reason::NoWrittenImage, "nothing written");
}

void GateRefusesAliases() {
	// Same guest range, texel-read through another image (a format alias, say).
	auto alias = Target(9, 0x10000000, Half, false, false);
	Check(Gate({Output(), alias}).reason == TR::Reason::TexelAlias,
	      "a texel read of the output range bound to another image");
	// Same image read through a texel-addressed binding: allowed and masked.
	auto own = Target(1, 0x10000000, Half, false, false);
	const auto d = Gate({Output(), own});
	Check(d.on && d.image_mask == 0b11u, "a texel read of the output itself is fine");
	auto sampled            = own;
	sampled.texel_addressed = false;
	Check(Gate({Output(), sampled}).reason == TR::Reason::SampledAlias,
	      "sampling the range being written");
}

void GateRefusesMaskOverflow() {
	std::vector<TR::BoundImage> images(16, Target(7, 0x70000000, 1.0F, false, false));
	images.push_back(GBuffer());
	images.push_back(Output());
	Check(Gate(images).reason == TR::Reason::MaskOverflow,
	      "a scaled binding past bit 15 cannot be described");
	images[16] = Target(8, 0x80000000, 1.0F, false, false);
	images[17].storage = true;
	auto d = Gate(images);
	Check(!d.on && d.reason == TR::Reason::MaskOverflow, "the output past bit 15 too");
}

void CheckTargetsTheTwinOfAScaledOwner() {
	auto twin = Output();
	twin.twin = true;
	const std::vector<TR::BoundImage> images {GBuffer(), twin};
	const auto t = TR::EvaluateCheck(Half, Shift, images);
	Check(t.eligible && t.written == 1, "a single written twin of a 0.5 owner is checkable");
	Check(TR::EvaluateCheck(Half, Shift, std::vector {Output(1.0F)}).reason ==
	          TR::Reason::WrittenNotScaled,
	      "a plain native output has no scaled owner to verify against");
	auto second = twin;
	second.image = 5;
	second.guest_address = 0x50000000;
	Check(TR::EvaluateCheck(Half, Shift, std::vector {twin, second}).reason ==
	          TR::Reason::CheckMultipleWrites,
	      "two written images");
}

void AutoChecksThenRescales() {
	TR::ProgramState state;
	for (uint32_t i = 0; i < TR::ChecksToVerify; i++) {
		Check(state.Decide(ComputeRescale::Auto, true) == TR::Plan::NativeWithCheck,
		      "the first dispatches of a C1/C2 program are checked");
		state.checks_submitted++;
	}
	Check(state.Decide(ComputeRescale::Auto, true) == TR::Plan::Native,
	      "with all checks submitted and none back it runs natively");
	for (uint32_t i = 0; i + 1 < TR::ChecksToVerify; i++) {
		Check(state.NoteCheckResult(ComputeRescale::Auto, 0, 100) == TR::CheckOutcome::Passed,
		      "a clean check passes");
	}
	Check(state.NoteCheckResult(ComputeRescale::Auto, 0, 100) == TR::CheckOutcome::Verified,
	      "the eighth clean check verifies");
	Check(state.Decide(ComputeRescale::Auto, true) == TR::Plan::Rescale, "verified rescales");
}

void AutoBlacklistsOnMismatch() {
	TR::ProgramState state;
	(void)state.Decide(ComputeRescale::Auto, true);
	state.checks_submitted++;
	Check(state.NoteCheckResult(ComputeRescale::Auto, 3, 100) == TR::CheckOutcome::Blacklisted,
	      "a mismatch blacklists");
	Check(state.NoteCheckResult(ComputeRescale::Auto, 0, 100) == TR::CheckOutcome::Ignored,
	      "later results are ignored");
	Check(state.Decide(ComputeRescale::Auto, true) == TR::Plan::Native, "blacklisted is native");
	Check(state.Decide(ComputeRescale::Auto, false) == TR::Plan::Native,
	      "blacklisting wins over a static proof");
}

void StaticProofRescalesAtOnce() {
	TR::ProgramState state;
	Check(state.Decide(ComputeRescale::Auto, false) == TR::Plan::Rescale,
	      "a C0 program needs no runtime check");
	Check(state.Decide(ComputeRescale::Off, false) == TR::Plan::Native, "off is native");
}

void VerifyChecksForeverAndNeverVerifies() {
	TR::ProgramState state;
	for (uint32_t i = 0; i < 3 * TR::ChecksToVerify; i++) {
		Check(state.Decide(ComputeRescale::Verify, false) == TR::Plan::NativeWithCheck,
		      "verify checks every dispatch");
		state.checks_submitted++;
		Check(state.NoteCheckResult(ComputeRescale::Verify, 0, 100) == TR::CheckOutcome::Passed,
		      "verify never marks a program verified");
	}
	Check(!state.verified, "verify leaves the program unverified");
}

void InconclusiveChecksCountNeitherWay() {
	TR::ProgramState state;
	Check(state.NoteCheckResult(ComputeRescale::Auto, 0, 0) == TR::CheckOutcome::Inconclusive,
	      "a compare of no pixel");
	Check(state.checks_passed == 0 && !state.blacklisted, "changes nothing");
}

void OffReasonsLogOnceAndSummariesAreSpaced() {
	TR::ProgramState state;
	Check(state.CountOff(TR::Reason::SampledAlias), "first occurrence logs");
	Check(!state.CountOff(TR::Reason::SampledAlias), "second does not");
	Check(state.off[static_cast<size_t>(TR::Reason::SampledAlias)] == 2, "both are counted");
	for (uint64_t i = 1; i < TR::SummaryInterval; i++) {
		(void)state.Decide(ComputeRescale::Auto, false);
		Check(!state.SummaryDue(), "no summary before the interval");
	}
	(void)state.Decide(ComputeRescale::Auto, false);
	Check(state.SummaryDue(), "a summary every interval");
}

} // namespace

int main() {
	GateAcceptsScaledOutputAndMasksScaledReads();
	GateRefusesRenderScaleMismatch();
	GateRefusesOutputNotBoundScaled();
	GateRefusesIneligibleOutputs();
	GateRefusesAliases();
	GateRefusesMaskOverflow();
	CheckTargetsTheTwinOfAScaledOwner();
	AutoChecksThenRescales();
	AutoBlacklistsOnMismatch();
	StaticProofRescalesAtOnce();
	VerifyChecksForeverAndNeverVerifies();
	InconclusiveChecksCountNeitherWay();
	OffReasonsLogOnceAndSummariesAreSpaced();
	std::printf("ComputeRescaleTests: all cases passed\n");
	return 0;
}
