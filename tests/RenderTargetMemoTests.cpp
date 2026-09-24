#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::GenerationMemo;
namespace HW       = Libs::Graphics::HW;
namespace Prospero = Libs::Graphics::Prospero;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "RenderTargetMemoTests: failed: %s\n", text);
		std::abort();
	}
}

// A minimal equality-comparable key/value pair, independent of the render-target types, so the
// GenerationMemo cases below exercise only the template's own hit/miss/overwrite behaviour.
struct TestKey {
	int a = 0;
	int b = 0;

	bool operator==(const TestKey&) const = default;
};

struct TestValue {
	int payload = 0;
};

void TestHitOnSameKeyAndGeneration() {
	GenerationMemo<TestKey, TestValue> memo;
	memo.Store({1, 2}, 5, {42});
	const auto* hit = memo.Find({1, 2}, 5);
	Check(hit != nullptr && hit->payload == 42, "identical key and generation hits");
}

void TestMissOnGenerationChange() {
	GenerationMemo<TestKey, TestValue> memo;
	memo.Store({1, 2}, 5, {42});
	Check(memo.Find({1, 2}, 6) == nullptr, "a moved generation misses even with the same key");
}

void TestMissOnKeyChange() {
	GenerationMemo<TestKey, TestValue> memo;
	memo.Store({1, 2}, 5, {42});
	Check(memo.Find({1, 3}, 5) == nullptr, "a different key misses at the same generation");
	Check(memo.Find({0, 2}, 5) == nullptr, "any differing key field misses at the same generation");
}

void TestInvalidate() {
	GenerationMemo<TestKey, TestValue> memo;
	memo.Store({1, 2}, 5, {42});
	memo.Invalidate();
	Check(memo.Find({1, 2}, 5) == nullptr, "Invalidate() forces the next Find() to miss");
}

void TestStoreOverwrites() {
	GenerationMemo<TestKey, TestValue> memo;
	memo.Store({1, 2}, 5, {42});
	memo.Store({3, 4}, 7, {99});
	Check(memo.Find({1, 2}, 5) == nullptr, "Store() replaces the previous entry's key wholesale");
	const auto* hit = memo.Find({3, 4}, 7);
	Check(hit != nullptr && hit->payload == 99, "the newly stored entry is retrievable");
}

void TestDefaultRenderTargetsCompareEqual() {
	Check(HW::RenderTarget {} == HW::RenderTarget {},
	      "two default-constructed HW::RenderTarget compare equal");
}

// ColorTargetKey (render.h) embeds HW::RenderTarget wholesale, so every field the discovery memo
// relies on to detect a changed slot has to participate in equality. Exercise one field per
// nested register struct, matching the set colorRenderTarget.cpp reads before calling
// TextureCache::FindImage().
void TestRenderTargetFieldsAffectEquality() {
	const HW::RenderTarget base {};

	auto changed_base = base;
	changed_base.base.addr = 0x1000;
	Check(!(changed_base == base), "base.addr participates in HW::RenderTarget equality");

	auto changed_view = base;
	changed_view.view.current_mip_level = 3;
	Check(!(changed_view == base), "view.current_mip_level participates in HW::RenderTarget equality");

	auto changed_info = base;
	changed_info.info.format = Prospero::ChannelLayout::k8_8_8_8;
	Check(!(changed_info == base), "info.format participates in HW::RenderTarget equality");

	auto changed_attrib3 = base;
	changed_attrib3.attrib3.tile_mode = Prospero::TileMode::kDepth;
	Check(!(changed_attrib3 == base), "attrib3.tile_mode participates in HW::RenderTarget equality");

	auto changed_dcc_addr = base;
	changed_dcc_addr.dcc_addr.addr = 0x2000;
	Check(!(changed_dcc_addr == base), "dcc_addr.addr participates in HW::RenderTarget equality");
}

void TestDefaultDepthRenderTargetsCompareEqual() {
	Check(HW::DepthRenderTarget {} == HW::DepthRenderTarget {},
	      "two default-constructed HW::DepthRenderTarget compare equal");
}

// RenderExecutor::m_depth_target_memo keys directly on HW::DepthRenderTarget (render.h), so the
// same requirement applies here: every nested register struct that feeds MakeDepthTargetDesc()
// has to affect equality.
void TestDepthRenderTargetFieldsAffectEquality() {
	const HW::DepthRenderTarget base {};

	auto changed_addr = base;
	changed_addr.z_read_base_addr = 0x1000;
	Check(!(changed_addr == base),
	      "z_read_base_addr participates in HW::DepthRenderTarget equality");

	auto changed_view = base;
	changed_view.depth_view.slice_max = 4;
	Check(!(changed_view == base),
	      "depth_view.slice_max participates in HW::DepthRenderTarget equality");

	auto changed_size = base;
	changed_size.size.x_max = 128;
	Check(!(changed_size == base), "size.x_max participates in HW::DepthRenderTarget equality");

	auto changed_z_info = base;
	changed_z_info.z_info.htile_acceleration = true;
	Check(!(changed_z_info == base),
	      "z_info.htile_acceleration participates in HW::DepthRenderTarget equality");
}

} // namespace

int main() {
	TestHitOnSameKeyAndGeneration();
	TestMissOnGenerationChange();
	TestMissOnKeyChange();
	TestInvalidate();
	TestStoreOverwrites();
	TestDefaultRenderTargetsCompareEqual();
	TestRenderTargetFieldsAffectEquality();
	TestDefaultDepthRenderTargetsCompareEqual();
	TestDepthRenderTargetFieldsAffectEquality();
	std::printf("RenderTargetMemoTests: all cases passed\n");
	return 0;
}
