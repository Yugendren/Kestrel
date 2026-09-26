#include "graphics/host_gpu/renderer/passScale.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <vector>

namespace {

using Libs::Graphics::PassScale::ColorAttachment;
using Libs::Graphics::PassScale::Decide;
using Libs::Graphics::PassScale::Decision;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "PassScaleTests: failed: %s\n", text);
		std::abort();
	}
}

Decision Run(std::initializer_list<ColorAttachment> colors, std::optional<float> depth) {
	const std::vector<ColorAttachment> list(colors);
	return Decide(list, depth);
}

constexpr ColorAttachment Written(float scale) { return {scale, true}; }
constexpr ColorAttachment Unwritten(float scale) { return {scale, false}; }

void UniformPassKeepsEverything() {
	const auto d = Run({Written(0.5F), Written(0.5F), Unwritten(0.5F)}, 0.5F);
	Check(d.uniform && d.color_count == 3, "a uniform pass keeps all attachments");
}

void StaleTrailingTargetsAreDropped() {
	// The Astro reflection-cube pass: a native 256x256 cube face exported by the shader, with the
	// four scaled G-buffer targets still enabled but not exported.
	const auto d = Run({Written(1.0F), Unwritten(0.5F), Unwritten(0.5F), Unwritten(0.5F),
	                    Unwritten(0.5F)},
	                   std::nullopt);
	Check(d.uniform, "stale trailing targets do not make the pass mixed");
	Check(d.color_count == 1, "stale trailing targets leave the pass");
}

void UnwrittenAtPassScaleStays() {
	const auto d = Run({Written(0.5F), Unwritten(0.5F), Unwritten(1.0F)}, std::nullopt);
	Check(d.uniform && d.color_count == 2,
	      "only the trailing run at another scale is dropped, not a matching attachment");
}

void DepthDecidesWhenNoColourIsWritten() {
	// Depth-only pass (shadow, pre-pass) with a stale native slot 0 over a scaled depth buffer.
	const auto d = Run({Unwritten(1.0F)}, 0.5F);
	Check(d.uniform && d.color_count == 0, "a stale slot 0 leaves a depth-only pass");
}

void DepthConflictWithWrittenColourIsMixed() {
	const auto d = Run({Written(1.0F), Unwritten(0.5F)}, 0.5F);
	Check(!d.uniform, "written colour and depth at different scales stay mixed");
	Check(d.color_count == 2, "a mixed decision keeps every attachment");
}

void WrittenColoursDisagreeIsMixed() {
	const auto d = Run({Written(0.5F), Written(1.0F)}, std::nullopt);
	Check(!d.uniform, "two written attachments at different scales stay mixed");
}

void UnwrittenInFrontOfKeptIsMixed() {
	// Dropping slot 0 would move the written attachment onto the wrong output location.
	const auto d = Run({Unwritten(1.0F), Written(0.5F)}, std::nullopt);
	Check(!d.uniform, "an unwritten attachment in front of a kept one cannot be dropped");
	Check(d.color_count == 2, "a mixed decision keeps every attachment");
}

void NothingWrittenAnchorsOnFirst() {
	const auto d = Run({Unwritten(0.5F), Unwritten(1.0F), Unwritten(1.0F)}, std::nullopt);
	Check(d.uniform && d.color_count == 1, "without writes or depth the first attachment anchors");
}

void EmptyPass() {
	const auto a = Run({}, std::nullopt);
	Check(a.uniform && a.color_count == 0, "an empty pass is uniform");
	const auto b = Run({}, 0.5F);
	Check(b.uniform && b.color_count == 0, "a depth-only pass is uniform");
}

} // namespace

int main() {
	UniformPassKeepsEverything();
	StaleTrailingTargetsAreDropped();
	UnwrittenAtPassScaleStays();
	DepthDecidesWhenNoColourIsWritten();
	DepthConflictWithWrittenColourIsMixed();
	WrittenColoursDisagreeIsMixed();
	UnwrittenInFrontOfKeptIsMixed();
	NothingWrittenAnchorsOnFirst();
	EmptyPass();
	std::printf("PassScaleTests: all passed\n");
	return 0;
}
