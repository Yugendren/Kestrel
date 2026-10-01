#ifndef KYTY_LOADER_PATCHES_PATCH_MANAGER_H_
#define KYTY_LOADER_PATCHES_PATCH_MANAGER_H_

#include "common/common.h"
#include "loader/patches/patchApplier.h"
#include "loader/patches/patchFile.h"

#include <string>
#include <vector>

namespace Loader {

struct Program;

namespace Patches {

// Applies the enabled game patches to modules as the runtime linker loads them. The patch
// directory (Config::GetPatchDirectory(): every *.xml plus patches.json) is read on the first
// module load, when the configuration and the game's param.json are known. Every decision is
// reported on one "[patch]" line. When no patch is enabled for a module nothing is written.
//
// Not thread-safe: the runtime linker calls it under its own lock.
class PatchManager {
public:
	PatchManager() = default;
	~PatchManager() = default;

	KYTY_CLASS_NO_COPY(PatchManager);

	// Applies the enabled patches matching `program` to its file bytes. Must run after every
	// file-backed segment is copied and before the loader rewrites guest code or write-protects
	// the mapping, so patches verify against the pristine file and the loader's own code
	// rewriting then sees patched code like any other guest code.
	void ApplyToModule(const Program& program);

	// The bytes the patches wrote into `program`, read now (empty when nothing was patched).
	// Taken right before relocation and handed to CheckAfterRelocation().
	[[nodiscard]] std::vector<uint8_t> CaptureWrites(const Program& program) const;

	// Warns when relocation changed a byte a patch wrote: the patch targets data the loader
	// fills in, which is an error in the patch.
	void CheckAfterRelocation(const Program& program, const std::vector<uint8_t>& before) const;

	// Drops the records of an unloaded module.
	void ForgetModule(const Program& program);

	void Clear();

private:
	struct AppliedWrite {
		const Program* program = nullptr;
		std::string    patch_name;
		uint64_t       vaddr = 0;
		uint64_t       size  = 0;
	};

	void LoadPatchDirectory();

	bool                      m_directory_loaded = false;
	std::string               m_title_id;
	std::string               m_app_version;
	std::vector<PatchEntry>   m_entries;
	PatchSelection            m_selection;
	std::vector<AppliedWrite> m_writes;
};

} // namespace Patches

} // namespace Loader

#endif /* KYTY_LOADER_PATCHES_PATCH_MANAGER_H_ */
