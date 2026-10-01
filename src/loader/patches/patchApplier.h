#ifndef KYTY_LOADER_PATCHES_PATCH_APPLIER_H_
#define KYTY_LOADER_PATCHES_PATCH_APPLIER_H_

#include "common/common.h"
#include "loader/patches/patchFile.h"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Deciding which patches apply to a loaded module and applying one of them to the module's image.
// Pure (no emulator state): the caller describes the module and its memory.
namespace Loader::Patches {

// The module a patch is matched against.
struct ModuleIdentity {
	std::string title_id;
	std::string app_version;
	// File name of the module, e.g. eboot.bin.
	std::string module_name;
	// XXH3-64 of the module file, nullopt when the file cannot be read. Hashing a whole module is
	// expensive, so it is only called for entries pinned with ElfXXH3.
	std::function<std::optional<uint64_t>()> file_xxh3;
};

enum class MatchResult {
	OtherModule,  // not for this game, game version or module
	HashMismatch, // for this module by name, but made against a different file (ElfXXH3)
	Match,
};

// Whether the entry targets this game (title id) and game version (exact AppVer).
bool IsForGame(const PatchEntry& entry, std::string_view title_id, std::string_view app_version);

MatchResult MatchModule(const PatchEntry& entry, const ModuleIdentity& module);

// The patches the user turned on: names given on the command line or in patches.json (compared
// case-insensitively), plus entries marked isEnabled="true".
class PatchSelection {
public:
	PatchSelection() = default;
	explicit PatchSelection(std::vector<std::string> names): m_names(std::move(names)) {}

	[[nodiscard]] bool IsEnabled(const PatchEntry& entry) const;

	// Requested names that no entry for this game version carries (typos, wrong game version).
	[[nodiscard]] std::vector<std::string> UnknownNames(std::span<const PatchEntry> entries,
	                                                    std::string_view            title_id,
	                                                    std::string_view app_version) const;

private:
	[[nodiscard]] bool IsNamed(std::string_view name) const;

	std::vector<std::string> m_names;
};

// The file-backed part of one loaded segment: module-relative p_vaddr and the p_filesz bytes
// in memory. Patches only touch file bytes; the zero-filled tail of a segment is not part of the
// file a patch was made against, except for the segment tail code cave below. A code cave (see
// CaveRanges()) is passed as one more segment.
struct ImageSegment {
	uint64_t           vaddr = 0;
	std::span<uint8_t> bytes;
	// Segment tail code cave: the zero-filled memory that directly follows `bytes` (p_filesz up
	// to the end of the segment's mapped guest pages, see SegmentTailSize()). Empty unless the
	// segment is executable. Only a line whose Original is all zero may lie in it.
	std::span<uint8_t> tail;
};

// A module-relative address range.
struct AddressRange {
	uint64_t vaddr = 0;
	uint64_t size  = 0;
};

// Code caves are allocated in whole guest pages of this size.
constexpr uint64_t CAVE_PAGE_SIZE = 0x4000;

// Size of the tail code cave of an executable PT_LOAD segment: from vaddr + filesz up to
// AlignUp(vaddr + memsz, CAVE_PAGE_SIZE), but never beyond `limit`, the module-relative address
// where the next segment starts or where the loader's protection of this segment ends, whichever
// is lower. 0 when the segment has no room left.
uint64_t SegmentTailSize(uint64_t vaddr, uint64_t filesz, uint64_t memsz, uint64_t limit);

// Finds the image base the auto_image_base entries of one source were written against, so that
// the module-relative address of a line is its address minus the base (see Rebase()).
//
// An entry is located by its lines with a non-zero Original (lines whose Original is all zero
// may be code caves, which are not part of the module). The anchor is the first such line of an
// entry; every place its Original bytes occur in the image's file bytes gives a candidate base.
// The candidates of the first entry's anchor are tried first, in segment and address order, then
// those of the next entry's anchor, and so on. A candidate verifies an entry when every one of the
// entry's non-zero-Original lines lies inside one segment's file bytes and finds its Original
// there. The first candidate that verifies every entry is the base; if none does, the candidate
// that verifies the most entries (the earliest on a tie), so one outdated entry does not hide the
// others -- ApplyPatch() then rejects the entries that do not verify. Returns nullopt with the
// reason when no entry has a non-zero Original or no candidate verifies any entry. Entries with an
// `error` are ignored.
std::optional<uint64_t> ResolveImageBase(std::span<const PatchEntry* const> entries,
                                         std::span<const ImageSegment>     image,
                                         std::string*                      reason);

// A copy of `entry` with module-relative line addresses: each address minus `image_base`, in
// uint64 modular arithmetic. A code cave below the module therefore gets an address that wraps
// around 2^64; adding the module's load address (again modulo 2^64) gives the cave's absolute
// address. The copy has auto_image_base cleared.
PatchEntry Rebase(const PatchEntry& entry, uint64_t image_base);

// The code caves the entries need. A cave line is a line whose Original is present and all zero
// and which lies entirely outside the module, i.e. does not overlap [0, module_size) and does not
// wrap around 2^64. The result is the module-relative ranges of the CAVE_PAGE_SIZE pages (aligned
// in absolute addresses, given the module's load address `module_base`) that hold those lines,
// sorted, with adjacent pages merged into one range. The caller maps the ranges, zero-filled, and
// passes them to ApplyPatch() as extra segments, so cave lines are verified (against their zero
// Original) like every other line. Entries with an `error` or auto_image_base are ignored.
std::vector<AddressRange> CaveRanges(std::span<const PatchEntry* const> entries,
                                     uint64_t module_base, uint64_t module_size);

struct ApplyOptions {
	// The module file's hash matched the entry's ElfXXH3 pin.
	bool file_hash_verified = false;
	// The configured presentation cap (Config::GetFrameCap()), 0 when uncapped.
	uint32_t frame_cap = 0;
};

enum class ApplyStatus {
	Applied,
	NeedsFrameCap, // RequiresFrameCap is not the configured cap
	Rejected,      // invalid entry, or the image does not verify
};

struct ApplyResult {
	ApplyStatus               status = ApplyStatus::Rejected;
	std::string               reason;
	uint64_t                  bytes_written = 0;
	std::vector<AddressRange> writes;
};

// Applies one patch atomically. Invariant: no byte is written unless every line of the patch has
// been verified first; each line must lie inside one segment's file bytes (or a code cave passed
// as a segment), must not overlap another line, and must find its Original bytes in memory -- or,
// for lines without Original, the module file must have matched the entry's ElfXXH3 pin. On any
// failure nothing is written and the result says why. An auto_image_base entry is rejected: it
// has to be rebased first.
ApplyResult ApplyPatch(const PatchEntry& entry, std::span<const ImageSegment> image,
                       const ApplyOptions& options);

} // namespace Loader::Patches

#endif /* KYTY_LOADER_PATCHES_PATCH_APPLIER_H_ */
