#include "graphics/host_gpu/renderer/cache/cleanPageMemo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace {

using Libs::Graphics::CleanPageMemo;

constexpr uint64_t Page = CleanPageMemo::PageSize;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "CleanPageMemoTests: failed: %s\n", message);
		std::abort();
	}
}

// Records every proof it is asked for and answers from a fixed verdict.
struct Prover {
	bool                  clean = true;
	std::vector<uint64_t> asked;

	bool operator()(uint64_t page, uint64_t size) {
		Check(page % Page == 0 && size == Page, "a proof was asked for something other than a page");
		asked.push_back(page);
		return clean;
	}
};

bool Contains(CleanPageMemo& memo, uint64_t vaddr, uint64_t size, Prover& prove) {
	return memo.Contains(vaddr, size, std::ref(prove));
}

void TestProvesEachPageOnce() {
	CleanPageMemo memo;
	Prover        prove;
	memo.Begin(1);
	for (uint64_t offset = 0; offset < Page; offset += 4) {
		Check(Contains(memo, 0x10000 + offset, 4, prove), "a dword on a clean page");
	}
	Check(Contains(memo, 0x10000, Page, prove), "the whole clean page");
	Check(prove.asked.size() == 1 && prove.asked[0] == 0x10000, "the page was proven again");
	Check(Contains(memo, 0x11000, 4, prove), "the next page");
	Check(prove.asked.size() == 2 && prove.asked[1] == 0x11000, "the next page was not proven");
	Check(Contains(memo, 0x10ff0, 16, prove) && prove.asked.size() == 2,
	      "a remembered page was forgotten when the next was proven");
}

void TestEpochBoundsTheProof() {
	CleanPageMemo memo;
	Prover        prove;
	memo.Begin(5);
	Check(Contains(memo, 0x20000, 4, prove), "the page at the first epoch");
	memo.Begin(5);
	Check(Contains(memo, 0x20004, 4, prove) && prove.asked.size() == 1,
	      "an unchanged epoch forgot the page");
	// Something may have become GPU-dirty: the page must be proven again, and can now fail.
	memo.Begin(6);
	prove.clean = false;
	Check(!Contains(memo, 0x20008, 4, prove), "a page stayed trusted after the epoch moved");
	Check(prove.asked.size() == 2, "the page was not proven again after the epoch moved");
}

void TestUnknownEpochAnswersNothing() {
	CleanPageMemo memo;
	Prover        prove;
	Check(!Contains(memo, 0x30000, 4, prove), "a memo that never began answered");
	memo.Begin(3);
	Check(Contains(memo, 0x30000, 4, prove), "the page at a known epoch");
	memo.Begin(0);
	Check(!Contains(memo, 0x30000, 4, prove), "epoch 0 answered from a remembered page");
	Check(prove.asked.size() == 1, "epoch 0 asked for a proof");
}

void TestFailedProofIsNotRemembered() {
	CleanPageMemo memo;
	Prover        prove;
	memo.Begin(1);
	prove.clean = false;
	Check(!Contains(memo, 0x40000, 4, prove), "a dirty page answered clean");
	Check(!Contains(memo, 0x40004, 4, prove), "a dirty page answered clean");
	Check(prove.asked.size() == 2, "a dirty page was remembered instead of asked again");
	// A download publishing the bytes makes the page clean within the same epoch.
	prove.clean = true;
	Check(Contains(memo, 0x40008, 4, prove), "a page that became clean stayed dirty");
}

void TestStraddlingAndDegenerateReadsAreRefused() {
	CleanPageMemo memo;
	Prover        prove;
	memo.Begin(1);
	Check(Contains(memo, 0x50000, 4, prove), "the first page");
	Check(Contains(memo, 0x51000, 4, prove), "the second page");
	const auto asked = prove.asked.size();
	// Both pages are remembered, but one proof never covers a read that crosses them.
	Check(!Contains(memo, 0x50ffc, 8, prove), "a read across two pages answered");
	Check(!Contains(memo, 0x50000, Page + 4, prove), "a read larger than a page answered");
	Check(!Contains(memo, 0x50000, 0, prove), "an empty read answered");
	Check(!Contains(memo, 0x10, 4, prove), "a read on page 0 answered");
	Check(prove.asked.size() == asked, "a refused read asked for a proof");
}

void TestCollisionsOnlyCostAProof() {
	CleanPageMemo memo;
	Prover        prove;
	memo.Begin(1);
	constexpr uint64_t pages = 4096;
	for (uint64_t round = 0; round < 2; round++) {
		for (uint64_t i = 0; i < pages; i++) {
			Check(Contains(memo, 0x1000000 + i * Page + 8, 4, prove),
			      "a clean page answered dirty after it was replaced");
		}
	}
	Check(prove.asked.size() > pages && prove.asked.size() <= 2 * pages,
	      "replacement changed how often pages are proven");
	// Pages that share a slot never answer for each other.
	prove.clean = false;
	const auto asked = prove.asked.size();
	Check(!Contains(memo, 0x9000000, 4, prove) && prove.asked.size() == asked + 1,
	      "an unproven page was answered from another page's slot");
}

} // namespace

int main() {
	TestProvesEachPageOnce();
	TestEpochBoundsTheProof();
	TestUnknownEpochAnswersNothing();
	TestFailedProofIsNotRemembered();
	TestStraddlingAndDegenerateReadsAreRefused();
	TestCollisionsOnlyCostAProof();
	std::puts("CleanPageMemoTests: all passed");
	return 0;
}
