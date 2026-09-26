#include "graphics/host_gpu/renderer/pipeline/driverCacheBlob.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace Blob = Libs::Graphics::DriverCacheBlob;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "DriverCacheBlobTests: failed: %s\n", text);
		std::abort();
	}
}

Blob::DeviceIdentity Device() {
	Blob::DeviceIdentity device;
	for (size_t i = 0; i < Blob::kUuidSize; i++) {
		device.pipeline_cache_uuid[i] = static_cast<uint8_t>(0xa0 + i);
	}
	device.vendor_id      = 0x10de;
	device.device_id      = 0x2504;
	device.driver_version = 0x8e3c4000;
	return device;
}

std::vector<uint8_t> Payload() {
	std::vector<uint8_t> payload(4096);
	for (size_t i = 0; i < payload.size(); i++) {
		payload[i] = static_cast<uint8_t>(i * 131 + 7);
	}
	return payload;
}

// What SerializeAndWrite() puts on disk.
std::vector<uint8_t> File(const std::string& signature, const std::vector<uint8_t>& payload) {
	const auto           prefix = Blob::Prefix(signature, payload);
	std::vector<uint8_t> file(prefix.begin(), prefix.end());
	file.insert(file.end(), payload.begin(), payload.end());
	return file;
}

void TestSignature() {
	const auto device = Device();
	Check(Blob::Signature(device) ==
	          "KytyPC:2:000010de:00002504:8e3c4000:a0a1a2a3a4a5a6a7a8a9aaabacadaeaf\n",
	      "signature text");
	Check(Blob::Signature(device) == Blob::Signature(device, Blob::kFormatVersion),
	      "default format version");

	// Every field of the device identity, and the format version, must change the key.
	auto other = device;
	other.pipeline_cache_uuid[15] ^= 1u;
	Check(Blob::Signature(other) != Blob::Signature(device), "uuid in key");
	other = device;
	other.vendor_id++;
	Check(Blob::Signature(other) != Blob::Signature(device), "vendor in key");
	other = device;
	other.device_id++;
	Check(Blob::Signature(other) != Blob::Signature(device), "device in key");
	other = device;
	other.driver_version++;
	Check(Blob::Signature(other) != Blob::Signature(device), "driver version in key");
	Check(Blob::Signature(device, Blob::kFormatVersion + 1) != Blob::Signature(device),
	      "format version in key");
}

void TestAccept() {
	const auto signature = Blob::Signature(Device());
	const auto payload   = Payload();
	const auto file      = File(signature, payload);
	const auto result    = Blob::Check(file, signature);
	Check(result.verdict == Blob::Verdict::Accepted, "matching file accepted");
	Check(result.payload.size() == payload.size() &&
	          std::equal(result.payload.begin(), result.payload.end(), payload.begin()),
	      "accepted payload is the driver blob");
}

void TestRejectDifferentUuid() {
	auto other = Device();
	other.pipeline_cache_uuid[0] ^= 0xffu;
	const auto file   = File(Blob::Signature(other), Payload());
	const auto result = Blob::Check(file, Blob::Signature(Device()));
	Check(result.verdict == Blob::Verdict::SignatureMismatch, "different uuid rejected");
	Check(result.payload.empty(), "rejected payload empty");
}

void TestRejectDifferentFormatVersion() {
	const auto file = File(Blob::Signature(Device(), Blob::kFormatVersion + 1), Payload());
	Check(Blob::Check(file, Blob::Signature(Device())).verdict == Blob::Verdict::SignatureMismatch,
	      "newer format version rejected");
	const auto old_file = File(Blob::Signature(Device(), Blob::kFormatVersion - 1), Payload());
	Check(Blob::Check(old_file, Blob::Signature(Device())).verdict ==
	          Blob::Verdict::SignatureMismatch,
	      "older format version rejected");
}

void TestRejectTruncated() {
	const auto signature = Blob::Signature(Device());
	const auto file      = File(signature, Payload());
	// Cut inside the payload, exactly at the end of the prefix, inside the hash, inside the
	// signature, and to nothing.
	for (size_t size: {file.size() - 1, signature.size() + sizeof(uint64_t) + 1,
	                   signature.size() + sizeof(uint64_t), signature.size() + 3,
	                   signature.size() / 2, size_t {0}}) {
		const std::vector<uint8_t> cut(file.begin(), file.begin() + static_cast<ptrdiff_t>(size));
		const auto                 result = Blob::Check(cut, signature);
		Check(result.verdict != Blob::Verdict::Accepted, "truncated file rejected");
		Check(result.payload.empty(), "truncated payload empty");
	}
	const std::vector<uint8_t> cut(file.begin(), file.end() - 1);
	Check(Blob::Check(cut, signature).verdict == Blob::Verdict::PayloadMismatch,
	      "payload truncation is a payload mismatch");
}

void TestRejectCorruptPayload() {
	const auto signature = Blob::Signature(Device());
	auto       file      = File(signature, Payload());
	file[file.size() / 2 + signature.size()] ^= 0x40u;
	Check(Blob::Check(file, signature).verdict == Blob::Verdict::PayloadMismatch,
	      "corrupt payload rejected");
}

void TestRejectGarbage() {
	const auto           signature = Blob::Signature(Device());
	std::vector<uint8_t> garbage(64 * 1024);
	uint32_t             state = 12345;
	for (auto& byte: garbage) {
		state = state * 1664525u + 1013904223u;
		byte  = static_cast<uint8_t>(state >> 24u);
	}
	Check(Blob::Check(garbage, signature).verdict == Blob::Verdict::SignatureMismatch,
	      "garbage rejected");
	// A file from the previous build-keyed layout ("KytyPC1:<git revision>:...").
	const std::string old_layout = "KytyPC1:1234:000010de:00002504:8e3c4000:a0a1\n";
	auto              old_file   = File(old_layout, Payload());
	Check(Blob::Check(old_file, signature).verdict != Blob::Verdict::Accepted,
	      "old build-keyed layout rejected");
}

} // namespace

int main() {
	TestSignature();
	TestAccept();
	TestRejectDifferentUuid();
	TestRejectDifferentFormatVersion();
	TestRejectTruncated();
	TestRejectCorruptPayload();
	TestRejectGarbage();
	std::printf("DriverCacheBlobTests: all passed\n");
	return 0;
}
