#include "common/slotVector.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// DbgExitIfHandler/DbgExit come from the common library (linked like image_page_table_tests).

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SlotVectorTests: failed: %s\n", text);
		std::abort();
	}
}

int g_live = 0;

// Large and non-movable, like Image/Buffer: exercises the chunked storage path.
struct Resource {
	explicit Resource(uint32_t v): value(v) { ++g_live; }
	~Resource() { --g_live; }
	Resource(const Resource&)            = delete;
	Resource& operator=(const Resource&) = delete;
	Resource(Resource&&)                 = delete;
	Resource& operator=(Resource&&)      = delete;

	uint32_t value;
	char     pad[900] {};
};

void TestInsertLookupErase() {
	Common::SlotVector<Resource> slots;
	const auto a = slots.insert(1u);
	const auto b = slots.insert(2u);
	Check(slots.size() == 2 && slots.capacity() == 2, "two live slots");
	Check(slots[a].value == 1 && slots[b].value == 2, "lookup returns the inserted values");
	Check(slots.is_allocated(a) && slots.try_get(b) != nullptr, "ids are allocated");
	Check(!slots.is_allocated(Common::SlotId {}), "invalid id is not allocated");
	Check(!slots.is_allocated(Common::SlotId {7}), "out-of-range id is not allocated");

	slots.erase(a);
	Check(slots.size() == 1 && g_live == 1, "erase destroys the value");
	Check(!slots.is_allocated(a) && slots.try_get(a) == nullptr, "stale id is rejected");

	// The freed index is reused with a new generation, so the old id stays stale.
	const auto c = slots.insert(3u);
	Check(c.index == a.index && c.generation != a.generation, "freed index reused, new generation");
	Check(!slots.is_allocated(a) && slots[c].value == 3, "reused slot holds the new value");
	Check(!slots.is_allocated(Common::SlotId {c.index, c.generation + 1}), "wrong generation rejected");
}

void TestReferenceStabilityAcrossGrowth() {
	Common::SlotVector<Resource> slots;
	const auto first  = slots.insert(100u);
	auto&      held   = slots[first];
	const auto* where = &held;
	std::vector<Common::SlotId> ids;
	// Enough inserts to allocate many chunks and grow every internal array several times.
	for (uint32_t i = 0; i < 1000; ++i) {
		ids.push_back(slots.insert(i));
	}
	Check(&slots[first] == where && held.value == 100, "reference survives growth");
	for (uint32_t i = 0; i < 1000; ++i) {
		Check(slots[ids[i]].value == i, "every value survives growth");
	}
}

void TestForEachAndDestructor() {
	g_live = 0;
	{
		Common::SlotVector<Resource> slots;
		std::vector<Common::SlotId>  ids;
		for (uint32_t i = 0; i < 200; ++i) {
			ids.push_back(slots.insert(i));
		}
		for (uint32_t i = 0; i < 200; i += 2) {
			slots.erase(ids[i]);
		}
		uint32_t count = 0;
		uint64_t sum   = 0;
		slots.ForEach([&](Common::SlotId id, const Resource& r) {
			Check(slots.is_allocated(id) && &slots[id] == &r, "ForEach yields live ids");
			++count;
			sum += r.value;
		});
		Check(count == 100 && sum == 100 * 100, "ForEach visits exactly the live odd values");
		Check(g_live == 100, "erased values destroyed");
	}
	Check(g_live == 0, "destructor destroys remaining values");
}

void TestSmallType() {
	Common::SlotVector<uint32_t> slots;
	std::vector<Common::SlotId>  ids;
	for (uint32_t i = 0; i < 130; ++i) {
		ids.push_back(slots.insert(i * 3));
	}
	for (uint32_t i = 0; i < 130; ++i) {
		Check(slots[ids[i]] == i * 3, "small values round-trip across chunk boundaries");
	}
}

} // namespace

int main() {
	TestInsertLookupErase();
	TestReferenceStabilityAcrossGrowth();
	TestForEachAndDestructor();
	TestSmallType();
	std::printf("SlotVectorTests: ok\n");
	return 0;
}
