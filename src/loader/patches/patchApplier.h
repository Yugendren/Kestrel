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
// file a patch was made against.
struct ImageSegment {
	uint64_t           vaddr = 0;
	std::span<uint8_t> bytes;
};

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

struct WrittenRange {
	uint64_t vaddr = 0;
	uint64_t size  = 0;
};

struct ApplyResult {
	ApplyStatus               status = ApplyStatus::Rejected;
	std::string               reason;
	uint64_t                  bytes_written = 0;
	std::vector<WrittenRange> writes;
};

// Applies one patch atomically. Invariant: no byte is written unless every line of the patch has
// been verified first; each line must lie inside one segment's file bytes, must not overlap
// another line, and must find its Original bytes in memory -- or, for lines without Original,
// the module file must have matched the entry's ElfXXH3 pin. On any failure nothing is written
// and the result says why.
ApplyResult ApplyPatch(const PatchEntry& entry, std::span<const ImageSegment> image,
                       const ApplyOptions& options);

} // namespace Loader::Patches

#endif /* KYTY_LOADER_PATCHES_PATCH_APPLIER_H_ */
