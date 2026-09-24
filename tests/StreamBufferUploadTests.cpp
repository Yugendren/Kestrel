#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

struct StreamBufferTestAccess {
	static void AppendUpload(std::vector<vk::BufferCopy>& uploads, uint64_t offset,
	                         uint64_t size) {
		StreamBuffer::AppendUpload(uploads, offset, size);
	}
};

} // namespace Libs::Graphics

namespace {

using Libs::Graphics::StreamBufferTestAccess;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "StreamBufferUploadTests: failed: %s\n", message);
		std::abort();
	}
}

bool Is(const vk::BufferCopy& copy, uint64_t offset, uint64_t size) {
	return copy.srcOffset == offset && copy.dstOffset == offset && copy.size == size;
}

void TestPackedRangesMerge() {
	std::vector<vk::BufferCopy> uploads;
	StreamBufferTestAccess::AppendUpload(uploads, 0, 100);
	StreamBufferTestAccess::AppendUpload(uploads, 100, 28);
	// Alignment padding between two allocations is copied along rather than splitting the copy.
	StreamBufferTestAccess::AppendUpload(uploads, 256, 64);
	Check(uploads.size() == 1 && Is(uploads[0], 0, 320), "packed ranges did not merge");
}

void TestDistantRangeStartsNewCopy() {
	std::vector<vk::BufferCopy> uploads;
	StreamBufferTestAccess::AppendUpload(uploads, 0, 64);
	StreamBufferTestAccess::AppendUpload(uploads, 64 + 4096 + 1, 64);
	Check(uploads.size() == 2 && Is(uploads[0], 0, 64) && Is(uploads[1], 4161, 64),
	      "a distant range was merged across unrelated bytes");
}

void TestWrapStartsNewCopy() {
	std::vector<vk::BufferCopy> uploads;
	StreamBufferTestAccess::AppendUpload(uploads, 1000, 24);
	// The ring wrapped: the next range lies before the previous one and must not extend it.
	StreamBufferTestAccess::AppendUpload(uploads, 0, 16);
	Check(uploads.size() == 2 && Is(uploads[0], 1000, 24) && Is(uploads[1], 0, 16),
	      "a wrapped range was merged backwards");
}

} // namespace

int main() {
	TestPackedRangesMerge();
	TestDistantRangeStartsNewCopy();
	TestWrapStartsNewCopy();
	std::printf("StreamBufferUploadTests: all passed\n");
	return 0;
}
