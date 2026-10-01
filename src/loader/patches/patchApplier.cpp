#include "loader/patches/patchApplier.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <numeric>

namespace Loader::Patches {

namespace {

// Long code patches would flood the log; the first bytes are enough to see what differs.
std::string FormatBytes(std::span<const uint8_t> bytes) {
	constexpr size_t MAX_SHOWN = 16;
	std::string      text;
	for (size_t i = 0; i < bytes.size() && i < MAX_SHOWN; i++) {
		text += fmt::format("{:02x}", bytes[i]);
	}
	if (bytes.size() > MAX_SHOWN) {
		text += "...";
	}
	return text;
}

ApplyResult Rejected(std::string reason) {
	ApplyResult result;
	result.status = ApplyStatus::Rejected;
	result.reason = std::move(reason);
	return result;
}

// The bytes of `image` holding [vaddr, vaddr + size), or an empty span when no single segment's
// file bytes contain the whole range.
std::span<uint8_t> FindTarget(std::span<const ImageSegment> image, uint64_t vaddr, uint64_t size) {
	for (const auto& segment: image) {
		if (vaddr < segment.vaddr) {
			continue;
		}
		const uint64_t offset = vaddr - segment.vaddr;
		if (offset <= segment.bytes.size() && size <= segment.bytes.size() - offset) {
			return segment.bytes.subspan(offset, size);
		}
	}
	return {};
}

} // namespace

bool IsForGame(const PatchEntry& entry, std::string_view title_id, std::string_view app_version) {
	return entry.app_version == app_version &&
	       std::any_of(entry.title_ids.begin(), entry.title_ids.end(),
	                   [&](const auto& id) { return Common::EqualNoCase(id, title_id); });
}

MatchResult MatchModule(const PatchEntry& entry, const ModuleIdentity& module) {
	if (!IsForGame(entry, module.title_id, module.app_version) ||
	    !Common::EqualNoCase(entry.app_elf, module.module_name)) {
		return MatchResult::OtherModule;
	}
	if (entry.elf_xxh3.has_value() && module.file_xxh3() != entry.elf_xxh3) {
		return MatchResult::HashMismatch;
	}
	return MatchResult::Match;
}

bool PatchSelection::IsNamed(std::string_view name) const {
	return std::any_of(m_names.begin(), m_names.end(),
	                   [&](const auto& enabled) { return Common::EqualNoCase(enabled, name); });
}

bool PatchSelection::IsEnabled(const PatchEntry& entry) const {
	return entry.enabled_by_default || IsNamed(entry.name);
}

std::vector<std::string> PatchSelection::UnknownNames(std::span<const PatchEntry> entries,
                                                      std::string_view            title_id,
                                                      std::string_view app_version) const {
	std::vector<std::string> unknown;
	for (const auto& name: m_names) {
		const bool known = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
			return Common::EqualNoCase(entry.name, name) &&
			       IsForGame(entry, title_id, app_version);
		});
		if (!known) {
			unknown.push_back(name);
		}
	}
	return unknown;
}

ApplyResult ApplyPatch(const PatchEntry& entry, std::span<const ImageSegment> image,
                       const ApplyOptions& options) {
	if (!entry.error.empty()) {
		return Rejected("invalid patch: " + entry.error);
	}
	if (entry.required_frame_cap != 0 && entry.required_frame_cap != options.frame_cap) {
		ApplyResult result;
		result.status = ApplyStatus::NeedsFrameCap;
		result.reason = fmt::format(
		    "requires --frame-cap {} (current: {})", entry.required_frame_cap,
		    options.frame_cap == 0 ? std::string("uncapped") : std::to_string(options.frame_cap));
		return result;
	}

	const auto& lines = entry.lines;

	std::vector<size_t> by_address(lines.size());
	std::iota(by_address.begin(), by_address.end(), size_t {0});
	std::sort(by_address.begin(), by_address.end(),
	          [&](size_t a, size_t b) { return lines[a].address < lines[b].address; });
	for (size_t i = 1; i < by_address.size(); i++) {
		const auto& previous = lines[by_address[i - 1]];
		const auto& next     = lines[by_address[i]];
		if (next.address - previous.address < previous.value.size()) {
			return Rejected(fmt::format("lines at +0x{:x} and +0x{:x} overlap", previous.address,
			                            next.address));
		}
	}

	// Verify every line before writing any: a half-applied patch is worse than none.
	std::vector<std::span<uint8_t>> targets;
	targets.reserve(lines.size());
	for (const auto& line: lines) {
		const auto target = FindTarget(image, line.address, line.value.size());
		if (target.empty()) {
			return Rejected(fmt::format("+0x{:x} ({} bytes) is outside the module's file data",
			                            line.address, line.value.size()));
		}
		if (line.original.empty()) {
			if (!options.file_hash_verified) {
				return Rejected(fmt::format(
				    "line at +0x{:x} has no Original bytes and the patch has no ElfXXH3 pin",
				    line.address));
			}
		} else if (!std::equal(line.original.begin(), line.original.end(), target.begin())) {
			return Rejected(fmt::format("original bytes differ at +0x{:x} (expected {}, found {})",
			                            line.address, FormatBytes(line.original),
			                            FormatBytes(target)));
		}
		targets.push_back(target);
	}

	ApplyResult result;
	result.status = ApplyStatus::Applied;
	for (size_t i = 0; i < lines.size(); i++) {
		std::memcpy(targets[i].data(), lines[i].value.data(), lines[i].value.size());
		result.bytes_written += lines[i].value.size();
		result.writes.push_back({lines[i].address, lines[i].value.size()});
	}
	return result;
}

} // namespace Loader::Patches
