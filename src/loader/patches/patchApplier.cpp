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

// The tail code cave holding [vaddr, vaddr + size), or an empty span.
std::span<uint8_t> FindTail(std::span<const ImageSegment> image, uint64_t vaddr, uint64_t size) {
	for (const auto& segment: image) {
		const uint64_t start = segment.vaddr + segment.bytes.size();
		if (vaddr < start) {
			continue;
		}
		const uint64_t offset = vaddr - start;
		if (offset <= segment.tail.size() && size <= segment.tail.size() - offset) {
			return segment.tail.subspan(offset, size);
		}
	}
	return {};
}

bool IsZero(std::span<const uint8_t> bytes) {
	return std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0; });
}

// Lines whose Original locates them in the module: present and not all zero.
bool IsLocatable(const PatchLine& line) {
	return !line.original.empty() && !IsZero(line.original);
}

const PatchLine* FindAnchor(const PatchEntry& entry) {
	const auto anchor = std::find_if(entry.lines.begin(), entry.lines.end(), IsLocatable);
	return anchor == entry.lines.end() ? nullptr : &*anchor;
}

// Whether every locatable line of `entry` finds its Original at `line.address - image_base`.
bool VerifiesAt(const PatchEntry& entry, std::span<const ImageSegment> image, uint64_t image_base) {
	return std::all_of(entry.lines.begin(), entry.lines.end(), [&](const PatchLine& line) {
		if (!IsLocatable(line)) {
			return true;
		}
		const auto target = FindTarget(image, line.address - image_base, line.original.size());
		return !target.empty() &&
		       std::equal(line.original.begin(), line.original.end(), target.begin());
	});
}

bool IsCaveLine(const PatchLine& line, uint64_t module_size) {
	// value is never empty, so `size - 1` cannot wrap; the second test keeps the line from
	// wrapping around 2^64 into the module.
	return !line.original.empty() && IsZero(line.original) && line.address >= module_size &&
	       line.value.size() - 1 <= UINT64_MAX - line.address;
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

std::optional<uint64_t> ResolveImageBase(std::span<const PatchEntry* const> entries,
                                         std::span<const ImageSegment>     image,
                                         std::string*                      reason) {
	std::vector<const PatchEntry*> located;
	for (const auto* entry: entries) {
		if (entry->error.empty() && FindAnchor(*entry) != nullptr) {
			located.push_back(entry);
		}
	}
	if (located.empty()) {
		*reason = "no line has non-zero original bytes to locate the image base with";
		return std::nullopt;
	}

	std::optional<uint64_t> best;
	size_t                  best_count = 0;
	for (const auto* anchor_entry: located) {
		const auto& anchor = *FindAnchor(*anchor_entry);
		const auto& needle = anchor.original;
		for (const auto& segment: image) {
			const auto begin = segment.bytes.begin();
			const auto end   = segment.bytes.end();
			for (auto found = std::search(begin, end, needle.begin(), needle.end()); found != end;
			     found      = std::search(found + 1, end, needle.begin(), needle.end())) {
				const uint64_t vaddr = segment.vaddr + static_cast<uint64_t>(found - begin);
				// The anchor's file address is at least its module-relative one: a base is never
				// negative.
				if (anchor.address < vaddr) {
					continue;
				}
				const uint64_t image_base = anchor.address - vaddr;
				const auto     verifies   = [&](const PatchEntry* entry) {
					return VerifiesAt(*entry, image, image_base);
				};
				const auto count =
				    static_cast<size_t>(std::count_if(located.begin(), located.end(), verifies));
				if (count == located.size()) {
					return image_base;
				}
				if (count > best_count) {
					best       = image_base;
					best_count = count;
				}
			}
		}
	}
	if (!best.has_value()) {
		*reason = "the original bytes are not in the module at any image base";
	}
	return best;
}

uint64_t SegmentTailSize(uint64_t vaddr, uint64_t filesz, uint64_t memsz, uint64_t limit) {
	if (memsz > UINT64_MAX - vaddr - (CAVE_PAGE_SIZE - 1)) {
		return 0;
	}
	const uint64_t start = vaddr + filesz;
	const uint64_t end   = std::min((vaddr + memsz + (CAVE_PAGE_SIZE - 1)) & ~(CAVE_PAGE_SIZE - 1),
	                                limit);
	return end > start ? end - start : 0;
}

PatchEntry Rebase(const PatchEntry& entry, uint64_t image_base) {
	PatchEntry rebased      = entry;
	rebased.auto_image_base = false;
	for (auto& line: rebased.lines) {
		line.address -= image_base;
	}
	return rebased;
}

std::vector<AddressRange> CaveRanges(std::span<const PatchEntry* const> entries,
                                     uint64_t module_base, uint64_t module_size) {
	// Pages are aligned in absolute addresses, then expressed module-relative again; both
	// conversions are modulo 2^64.
	std::vector<uint64_t> pages;
	for (const auto* entry: entries) {
		if (!entry->error.empty() || entry->auto_image_base) {
			continue;
		}
		for (const auto& line: entry->lines) {
			if (!IsCaveLine(line, module_size)) {
				continue;
			}
			const uint64_t first = (module_base + line.address) & ~(CAVE_PAGE_SIZE - 1);
			const uint64_t last =
			    (module_base + line.address + (line.value.size() - 1)) & ~(CAVE_PAGE_SIZE - 1);
			for (uint64_t page = first;; page += CAVE_PAGE_SIZE) {
				pages.push_back(page - module_base);
				if (page == last) {
					break;
				}
			}
		}
	}
	std::sort(pages.begin(), pages.end());
	pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

	std::vector<AddressRange> ranges;
	for (const auto page: pages) {
		if (!ranges.empty() && ranges.back().vaddr + ranges.back().size == page) {
			ranges.back().size += CAVE_PAGE_SIZE;
		} else {
			ranges.push_back({page, CAVE_PAGE_SIZE});
		}
	}
	return ranges;
}

ApplyResult ApplyPatch(const PatchEntry& entry, std::span<const ImageSegment> image,
                       const ApplyOptions& options) {
	if (!entry.error.empty()) {
		return Rejected("invalid patch: " + entry.error);
	}
	if (entry.auto_image_base) {
		return Rejected("the image base is not resolved");
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
		auto target = FindTarget(image, line.address, line.value.size());
		if (target.empty()) {
			target = FindTail(image, line.address, line.value.size());
			if (!target.empty() && (line.original.empty() || !IsZero(line.original))) {
				return Rejected(fmt::format("line at +0x{:x} lies in a segment tail code cave and "
				                            "its Original is not all zero",
				                            line.address));
			}
		}
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
