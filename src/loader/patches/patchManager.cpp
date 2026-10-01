#include "loader/patches/patchManager.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <xxhash.h>

namespace Loader::Patches {

namespace {

// Patch decisions always go to stdout, so a refused patch is visible without a log file; when the
// log goes to a file they are mirrored there too.
template <typename... Args>
void Report(fmt::format_string<Args...> format, Args&&... args) {
	const auto line = "[patch] " + fmt::format(format, std::forward<Args>(args)...) + "\n";
	std::fputs(line.c_str(), stdout);
	std::fflush(stdout);
	if (Log::GetDirection() == Log::Direction::File) {
		Log::Write(line);
	}
}

std::optional<std::string> ReadTextFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return std::nullopt;
	}
	return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::optional<uint64_t> HashFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return std::nullopt;
	}
	const std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)> state(XXH3_createState(),
	                                                                     &XXH3_freeState);
	if (state == nullptr || XXH3_64bits_reset(state.get()) == XXH_ERROR) {
		return std::nullopt;
	}
	std::vector<char> chunk(static_cast<size_t>(1) << 20u);
	while (file) {
		file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
		if (file.gcount() > 0) {
			XXH3_64bits_update(state.get(), chunk.data(), static_cast<size_t>(file.gcount()));
		}
	}
	if (!file.eof()) {
		return std::nullopt;
	}
	return XXH3_64bits_digest(state.get());
}

// The file-backed bytes of every segment the loader copied from the module file.
std::vector<ImageSegment> FileImage(const Program& program) {
	std::vector<ImageSegment> image;
	const auto*               ehdr = program.elf->GetEhdr();
	const auto*               phdr = program.elf->GetPhdr();
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (IsMappedSegment(phdr[i]) && phdr[i].p_filesz != 0) {
			auto* bytes = reinterpret_cast<uint8_t*>(program.base_vaddr + phdr[i].p_vaddr);
			image.push_back({phdr[i].p_vaddr, {bytes, phdr[i].p_filesz}});
		}
	}
	return image;
}

std::vector<std::filesystem::path> ListPatchFiles(const std::filesystem::path& directory) {
	std::vector<std::filesystem::path> files;
	std::error_code                    error;
	for (const auto& item: std::filesystem::directory_iterator(directory, error)) {
		if (item.is_regular_file(error) &&
		    Common::EqualNoCase(Common::PathToString(item.path().extension()), ".xml")) {
			files.push_back(item.path());
		}
	}
	// Directory order is filesystem-specific; patches must apply in the same order everywhere.
	std::sort(files.begin(), files.end());
	return files;
}

using PatchParser = bool (*)(std::string_view text, PatchFile* out, std::string* error);

// Reads one patch file and appends its entries, tagged with their source, to `entries`. Returns
// the number of entries read.
size_t ReadPatchSource(const std::filesystem::path& path, PatchParser parse,
                       std::vector<PatchEntry>* entries) {
	const auto  text = ReadTextFile(path);
	PatchFile   file;
	std::string error;
	if (!text.has_value()) {
		Report("warning: cannot read {}", Common::PathToString(path));
		return 0;
	}
	if (!parse(*text, &file, &error)) {
		Report("warning: {} is not a valid patch file: {}", Common::PathToString(path), error);
		return 0;
	}
	for (auto& entry: file.entries) {
		entry.source = Common::PathToString(path);
		entries->push_back(std::move(entry));
	}
	return file.entries.size();
}

// Maps zero-filled guest pages at exactly `address`; null when the range is not free.
CodeCave MapCodeCave(uint64_t address, uint64_t size) {
	const auto mapped = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    address, size, Common::VirtualMemory::Mode::ExecuteReadWrite, "game_patch_code_cave", true);
	if (mapped == 0) {
		return CodeCave(nullptr, CodeCaveDeleter {size});
	}
	EXIT_IF(mapped != address);
	auto* pages = reinterpret_cast<uint8_t*>(mapped);
	// The cave lines verify against an all-zero Original, so the pages must start out zeroed
	// whatever the allocator guarantees.
	std::memset(pages, 0, size);
	return CodeCave(pages, CodeCaveDeleter {size});
}

} // namespace

void CodeCaveDeleter::operator()(uint8_t* pages) const {
	EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(reinterpret_cast<uint64_t>(pages), size));
}

void PatchManager::LoadPatchSources() {
	m_sources_loaded = true;
	SystemContentParamSfoGetString("TITLE_ID", &m_title_id);
	SystemContentParamSfoGetString("APP_VER", &m_app_version);

	const auto directory = Config::GetPatchDirectory();
	auto       names     = Config::GetEnabledPatches();

	if (!std::filesystem::is_directory(directory)) {
		Report("no patch directory at {}", Common::PathToString(directory));
	} else {
		for (const auto& path: ListPatchFiles(directory)) {
			ReadPatchSource(path, ParsePatchFile, &m_entries);
		}

		const auto enabled_list = directory / "patches.json";
		if (std::filesystem::is_regular_file(enabled_list)) {
			const auto               text = ReadTextFile(enabled_list);
			std::vector<std::string> listed;
			std::string              error;
			if (!text.has_value() || !ParseEnabledNames(*text, &listed, &error)) {
				Report("warning: ignoring {}: {}", Common::PathToString(enabled_list),
				       text.has_value() ? error : std::string("cannot read it"));
			} else {
				std::move(listed.begin(), listed.end(), std::back_inserter(names));
			}
		}
	}

	for (const auto& path: Config::GetPatchFiles()) {
		const auto first = m_entries.size();
		if (ReadPatchSource(path, ParseCheatJson, &m_entries) != 0 &&
		    std::none_of(m_entries.begin() + static_cast<ptrdiff_t>(first), m_entries.end(),
		                 [&](const auto& entry) {
			                 return IsForGame(entry, m_title_id, m_app_version);
		                 })) {
			Report("warning: {} has no patch for {} version {}", Common::PathToString(path),
			       m_title_id, m_app_version);
		}
	}

	m_selection = PatchSelection(std::move(names));
	for (const auto& name: m_selection.UnknownNames(m_entries, m_title_id, m_app_version)) {
		Report("warning: no patch named \"{}\" for {} version {}", name, m_title_id,
		       m_app_version);
	}
}

void PatchManager::ApplyToModule(const Program& program) {
	EXIT_IF(program.elf == nullptr || program.base_vaddr == 0);

	if (!m_sources_loaded) {
		LoadPatchSources();
	}
	if (m_entries.empty()) {
		return;
	}

	const auto module_name = Common::PathToString(program.file_name.filename());

	// Hash the module at most once, and only if an entry for it is pinned with ElfXXH3.
	std::optional<std::optional<uint64_t>> file_hash;
	ModuleIdentity module {m_title_id, m_app_version, module_name, [&]() {
		                       if (!file_hash.has_value()) {
			                       file_hash = HashFile(program.file_name);
		                       }
		                       return *file_hash;
	                       }};

	std::vector<const PatchEntry*> selected;
	for (const auto& entry: m_entries) {
		const auto match = MatchModule(entry, module);
		if (match == MatchResult::OtherModule) {
			continue;
		}
		if (!m_selection.IsEnabled(entry)) {
			if (match == MatchResult::Match) {
				Report("available (disabled): {}", entry.name);
			}
			continue;
		}
		if (match == MatchResult::HashMismatch) {
			const auto hash = module.file_xxh3();
			Report("skipped \"{}\" for {}: made for a different file (ElfXXH3 {:016x}, file {})",
			       entry.name, module_name, *entry.elf_xxh3,
			       hash.has_value() ? fmt::format("{:016x}", *hash) : std::string("unreadable"));
			continue;
		}
		selected.push_back(&entry);
	}
	if (selected.empty()) {
		return;
	}

	const auto file_image = FileImage(program);
	// m_entries keeps the entries of one source together, so the selected ones are too.
	for (auto begin = selected.begin(); begin != selected.end();) {
		const auto end = std::find_if(begin, selected.end(), [&](const PatchEntry* entry) {
			return entry->source != (*begin)->source;
		});
		ApplySource(program, module_name, {begin, end}, file_image);
		begin = end;
	}
}

void PatchManager::ApplySource(const Program& program, std::string_view module_name,
                               std::span<const PatchEntry* const> entries,
                               std::span<const ImageSegment>      file_image) {
	std::vector<const PatchEntry*> unresolved;
	std::copy_if(entries.begin(), entries.end(), std::back_inserter(unresolved),
	             [](const PatchEntry* entry) {
		             return entry->auto_image_base && entry->error.empty();
	             });
	std::optional<uint64_t> image_base;
	if (!unresolved.empty()) {
		std::string reason;
		image_base = ResolveImageBase(unresolved, file_image, &reason);
		if (image_base.has_value()) {
			Report("image base of {} in {} is 0x{:x}",
			       Common::PathToString(std::filesystem::path(entries.front()->source).filename()),
			       module_name, *image_base);
		} else {
			for (const auto* entry: unresolved) {
				Report("skipped \"{}\" for {}: {}", entry->name, module_name, reason);
			}
		}
	}

	// The entries to apply, rebased where needed. `rebased` owns the rebased copies; it is
	// reserved up front so the pointers into it stay valid.
	std::vector<PatchEntry> rebased;
	rebased.reserve(unresolved.size());
	std::vector<const PatchEntry*> resolved;
	for (const auto* entry: entries) {
		if (!entry->auto_image_base || !entry->error.empty()) {
			// Invalid entries go through so ApplyPatch() reports their error.
			resolved.push_back(entry);
		} else if (image_base.has_value()) {
			rebased.push_back(Rebase(*entry, *image_base));
			resolved.push_back(&rebased.back());
		}
	}

	struct MappedCave {
		AddressRange range;
		CodeCave     pages;
	};
	std::vector<MappedCave>   caves;
	std::vector<ImageSegment> image(file_image.begin(), file_image.end());
	for (const auto& range: CaveRanges(resolved, program.base_vaddr, program.mapped_size)) {
		// Module-relative to absolute, modulo 2^64 like the cave's relative address.
		const uint64_t address = program.base_vaddr + range.vaddr;
		auto           pages   = MapCodeCave(address, range.size);
		if (pages == nullptr) {
			Report("warning: cannot map a code cave at 0x{:x} ({} bytes) for {}", address,
			       range.size, module_name);
			continue;
		}
		image.push_back({range.vaddr, {pages.get(), range.size}});
		caves.push_back({range, std::move(pages)});
	}

	const auto first_write = m_writes.size();
	for (const auto* entry: resolved) {
		ApplyOptions options;
		options.file_hash_verified = entry->elf_xxh3.has_value();
		options.frame_cap          = Config::GetFrameCap();

		const auto result = ApplyPatch(*entry, image, options);
		switch (result.status) {
			case ApplyStatus::Applied:
				for (const auto& write: result.writes) {
					m_writes.push_back({&program, entry->name, write.vaddr, write.size});
				}
				Report("applied \"{}\" to {} ({} lines, {} bytes)", entry->name, module_name,
				       entry->lines.size(), result.bytes_written);
				break;
			case ApplyStatus::NeedsFrameCap:
				Report("patch \"{}\" {}; not applied", entry->name, result.reason);
				break;
			case ApplyStatus::Rejected:
				Report("skipped \"{}\" for {}: {}", entry->name, module_name, result.reason);
				break;
		}
	}

	// Keep the caves an applied patch wrote to; the others are unmapped as `caves` goes away. A
	// write lies within one segment, so its start decides which cave holds it.
	const std::span<const AppliedWrite> writes(
	    m_writes.begin() + static_cast<ptrdiff_t>(first_write), m_writes.end());
	for (auto& cave: caves) {
		if (std::any_of(writes.begin(), writes.end(), [&](const AppliedWrite& write) {
			    return write.vaddr - cave.range.vaddr < cave.range.size;
		    })) {
			m_caves.push_back({&program, std::move(cave.pages)});
		}
	}
}

std::vector<uint8_t> PatchManager::CaptureWrites(const Program& program) const {
	std::vector<uint8_t> bytes;
	for (const auto& write: m_writes) {
		if (write.program == &program) {
			const auto* begin = reinterpret_cast<const uint8_t*>(program.base_vaddr + write.vaddr);
			bytes.insert(bytes.end(), begin, begin + write.size);
		}
	}
	return bytes;
}

void PatchManager::CheckAfterRelocation(const Program&              program,
                                        const std::vector<uint8_t>& before) const {
	size_t offset = 0;
	for (const auto& write: m_writes) {
		if (write.program != &program) {
			continue;
		}
		const auto* now = reinterpret_cast<const uint8_t*>(program.base_vaddr + write.vaddr);
		EXIT_IF(offset + write.size > before.size());
		if (!std::equal(now, now + write.size, before.begin() + static_cast<ptrdiff_t>(offset))) {
			Report("warning: relocation of {} overwrote bytes written by \"{}\" at +0x{:x}",
			       Common::PathToString(program.file_name.filename()), write.patch_name,
			       write.vaddr);
		}
		offset += write.size;
	}
}

void PatchManager::ForgetModule(const Program& program) {
	std::erase_if(m_writes, [&](const auto& write) { return write.program == &program; });
	std::erase_if(m_caves, [&](const auto& cave) { return cave.program == &program; });
}

void PatchManager::Clear() {
	m_sources_loaded = false;
	m_title_id.clear();
	m_app_version.clear();
	m_entries.clear();
	m_selection = {};
	m_writes.clear();
	m_caves.clear();
}

} // namespace Loader::Patches
