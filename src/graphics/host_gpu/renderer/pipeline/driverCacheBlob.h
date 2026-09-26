#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEBLOB_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEBLOB_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <span>
#include <string>
#include <string_view>
#include <xxhash.h>

// On-disk layout of the persisted Vulkan driver pipeline cache (_PipelineCache/<title>.bin):
//
//   <signature line, '\n'-terminated> <XXH3-64 of payload, 8 bytes> <vkGetPipelineCacheData blob>
//
// Kept free of Vulkan and file I/O so the accept/reject decision can be unit-tested without a GPU
// (tests/DriverCacheBlobTests.cpp).
//
// Why the signature names the driver and not the emulator build: the blob used to be keyed on
// KYTY_GIT_REVISION, so every new emulator build threw the whole blob away and recompiled every
// pipeline cold (up to 8 s per compute pipeline, ~74 s over a cold load), with only the driver's
// own shader cache (NVIDIA's GLCache, size-capped and evicting) hiding it. That key protected
// nothing: the driver hashes each entry by its full pipeline create info (SPIR-V included) and
// validates the blob's VkPipelineCacheHeaderVersionOne itself, so an entry an older build
// produced is at worst never hit. What the driver cannot survive is a blob from another
// device/driver, hence pipelineCacheUUID + vendor/device ID + driverVersion, the same key yuzu
// and RPCS3 use. kFormatVersion covers the only thing the driver cannot see: this file's own
// layout (and, conservatively, an incompatible change to how the emulator builds pipelines).
namespace Libs::Graphics::DriverCacheBlob {

// Bump only when the layout above, or the emulator's pipeline-creation inputs, change in a way
// that makes an existing blob unusable. An ordinary emulator rebuild must not bump it.
constexpr uint32_t kFormatVersion = 2;

constexpr size_t kUuidSize = 16; // VK_UUID_SIZE; checked against Vulkan in pipelineCache.cpp.

struct DeviceIdentity {
	std::array<uint8_t, kUuidSize> pipeline_cache_uuid {};
	uint32_t                       vendor_id      = 0;
	uint32_t                       device_id      = 0;
	uint32_t                       driver_version = 0;
};

inline std::string Signature(const DeviceIdentity& device, uint32_t format_version = kFormatVersion) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(kUuidSize * 2, '0');
	for (size_t i = 0; i < kUuidSize; i++) {
		uuid[i * 2]     = hex[device.pipeline_cache_uuid[i] >> 4u];
		uuid[i * 2 + 1] = hex[device.pipeline_cache_uuid[i] & 0xfu];
	}
	return fmt::format("KytyPC:{}:{:08x}:{:08x}:{:08x}:{}\n", format_version, device.vendor_id,
	                   device.device_id, device.driver_version, uuid);
}

// Signature line plus payload hash; the payload is appended after it.
inline std::string Prefix(std::string_view signature, std::span<const uint8_t> payload) {
	const uint64_t payload_hash = XXH3_64bits(payload.data(), payload.size());
	std::string    prefix(signature);
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	return prefix;
}

enum class Verdict {
	Accepted,
	TooShort,          // truncated inside the prefix, or empty payload
	SignatureMismatch, // other device, driver, or format version (or not a cache file at all)
	PayloadMismatch,   // payload truncated or corrupted after the prefix
};

inline const char* VerdictText(Verdict verdict) {
	switch (verdict) {
		case Verdict::Accepted: return "accepted";
		case Verdict::TooShort: return "file too short";
		case Verdict::SignatureMismatch: return "different device, driver, or cache format";
		case Verdict::PayloadMismatch: return "truncated or corrupt payload";
	}
	return "unknown";
}

struct CheckResult {
	Verdict                  verdict = Verdict::TooShort;
	std::span<const uint8_t> payload; // the driver blob inside `file`; empty unless Accepted
};

// Decides whether a whole cache file may be handed to vkCreatePipelineCache.
inline CheckResult Check(std::span<const uint8_t> file, std::string_view expected_signature) {
	const size_t prefix_size = expected_signature.size() + sizeof(uint64_t);
	if (file.size() <= prefix_size) {
		return {Verdict::TooShort, {}};
	}
	if (std::memcmp(file.data(), expected_signature.data(), expected_signature.size()) != 0) {
		return {Verdict::SignatureMismatch, {}};
	}
	uint64_t payload_hash = 0;
	std::memcpy(&payload_hash, file.data() + expected_signature.size(), sizeof(payload_hash));
	const auto payload = file.subspan(prefix_size);
	if (XXH3_64bits(payload.data(), payload.size()) != payload_hash) {
		return {Verdict::PayloadMismatch, {}};
	}
	return {Verdict::Accepted, payload};
}

} // namespace Libs::Graphics::DriverCacheBlob

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEBLOB_H_
