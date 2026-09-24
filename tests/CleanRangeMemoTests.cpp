#include "graphics/host_gpu/renderer/cache/cleanRangeMemo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Common::SlotId;
using Libs::Graphics::CleanRangeMemo;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "CleanRangeMemoTests: failed: %s\n", message);
		std::abort();
	}
}

void TestHitOnlyAtSameEpoch() {
	CleanRangeMemo memo;
	memo.Remember(0x10000, 0x400, 7, SlotId {3, 2});
	const auto hit = memo.Find(0x10000, 0x400, 7);
	Check(hit.has_value() && hit->index == 3 && hit->generation == 2, "exact entry missed");
	Check(!memo.Find(0x10000, 0x400, 8).has_value(), "entry trusted after the epoch advanced");
}

void TestExactRangeOnly() {
	CleanRangeMemo memo;
	memo.Remember(0x20000, 0x400, 5, SlotId {1});
	Check(!memo.Find(0x20000, 0x200, 5).has_value(), "a smaller range reused the entry");
	Check(!memo.Find(0x20000, 0x800, 5).has_value(), "a larger range reused the entry");
	Check(!memo.Find(0x20100, 0x400, 5).has_value(), "a shifted range reused the entry");
}

void TestUntrustedEpochNeverStoredOrFound() {
	CleanRangeMemo memo;
	memo.Remember(0x30000, 0x40, 0, SlotId {1});
	Check(!memo.Find(0x30000, 0x40, 0).has_value(), "epoch 0 was trusted");
}

void TestCollisionReplaces() {
	CleanRangeMemo memo;
	// Many distinct ranges: every lookup either finds its own entry or misses, never another's.
	for (uint64_t i = 0; i < 4096; i++) {
		memo.Remember(0x100000 + i * 0x100, 0x100, 9, SlotId {static_cast<uint32_t>(i)});
	}
	uint32_t hits = 0;
	for (uint64_t i = 0; i < 4096; i++) {
		if (const auto id = memo.Find(0x100000 + i * 0x100, 0x100, 9)) {
			Check(id->index == i, "a colliding range returned another range's buffer");
			hits++;
		}
	}
	Check(hits > 0, "no entry survived");
}

} // namespace

int main() {
	TestHitOnlyAtSameEpoch();
	TestExactRangeOnly();
	TestUntrustedEpochNeverStoredOrFound();
	TestCollisionReplaces();
	std::printf("CleanRangeMemoTests: all passed\n");
	return 0;
}
