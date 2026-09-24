#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

struct StreamBufferTestAccess {
	static bool MayWaitFor(RingWait wait, uint64_t tick, uint64_t current_tick) {
		return StreamBuffer::MayWaitFor(wait, tick, current_tick);
	}
};

} // namespace Libs::Graphics

namespace {

using Libs::Graphics::RingWait;
using Libs::Graphics::StreamBufferTestAccess;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "StreamBufferWaitTests: failed: %s\n", message);
		std::abort();
	}
}

void TestNeverWaits() {
	Check(!StreamBufferTestAccess::MayWaitFor(RingWait::Never, 3, 10), "Never waited for old work");
	Check(!StreamBufferTestAccess::MayWaitFor(RingWait::Never, 10, 10),
	      "Never waited for the recording tick");
}

void TestSubmittedNeverSubmits() {
	// Waiting for the recording tick is what submits it; an upload callback must never do that.
	Check(!StreamBufferTestAccess::MayWaitFor(RingWait::Submitted, 10, 10),
	      "Submitted would submit the recording command buffer");
	Check(StreamBufferTestAccess::MayWaitFor(RingWait::Submitted, 9, 10),
	      "Submitted refused to wait for the previous submission");
	Check(StreamBufferTestAccess::MayWaitFor(RingWait::Submitted, 1, 10),
	      "Submitted refused to wait for old work");
}

void TestAnyWaitsForEverything() {
	Check(StreamBufferTestAccess::MayWaitFor(RingWait::Any, 10, 10),
	      "Any refused the recording tick");
	Check(StreamBufferTestAccess::MayWaitFor(RingWait::Any, 1, 10), "Any refused old work");
}

} // namespace

int main() {
	TestNeverWaits();
	TestSubmittedNeverSubmits();
	TestAnyWaitsForEverything();
	std::printf("StreamBufferWaitTests: all passed\n");
	return 0;
}
