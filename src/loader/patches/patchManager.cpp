#include "loader/patches/patchManager.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <cstdio>
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

} // namespace

void PatchManager::LoadPatchDirectory() {
	m_directory_loaded = true;
	SystemContentParamSfoGetString("TITLE_ID", &m_title_id);
	SystemContentParamSfoGetString("APP_VER", &m_app_version);

	const auto directory = Config::GetPatchDirectory();
	auto       names     = Config::GetEnabledPatches();

	if (!std::filesystem::is_directory(directory)) {
		Report("no patch directory at {}", Common::PathToString(directory));
	} else {
		for (const auto& path: ListPatchFiles(directory)) {
			const auto text  = ReadTextFile(path);
			PatchFile  file;
			std::string error;
			if (!text.has_value()) {
				Report("warning: cannot read {}", Common::PathToString(path));
			} else if (!ParsePatchFile(*text, &file, &error)) {
				Report("warning: {} is not a valid patch file: {}", Common::PathToString(path),
				       error);
			} else {
				std::move(file.entries.begin(), file.entries.end(), std::back_inserter(m_entries));
			}
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

	m_selection = PatchSelection(std::move(names));
	for (const auto& name: m_selection.UnknownNames(m_entries, m_title_id, m_app_version)) {
		Report("warning: no patch named \"{}\" for {} version {}", name, m_title_id,
		       m_app_version);
	}
}

void PatchManager::ApplyToModule(const Program& program) {
	EXIT_IF(program.elf == nullptr || program.base_vaddr == 0);

	if (!m_directory_loaded) {
		LoadPatchDirectory();
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

	std::vector<ImageSegment> image;
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

		if (image.empty()) {
			image = FileImage(program);
		}
		ApplyOptions options;
		options.file_hash_verified = entry.elf_xxh3.has_value();
		options.frame_cap          = Config::GetFrameCap();

		const auto result = ApplyPatch(entry, image, options);
		switch (result.status) {
			case ApplyStatus::Applied:
				for (const auto& write: result.writes) {
					m_writes.push_back({&program, entry.name, write.vaddr, write.size});
				}
				Report("applied \"{}\" to {} ({} lines, {} bytes)", entry.name, module_name,
				       entry.lines.size(), result.bytes_written);
				break;
			case ApplyStatus::NeedsFrameCap:
				Report("patch \"{}\" {}; not applied", entry.name, result.reason);
				break;
			case ApplyStatus::Rejected:
				Report("skipped \"{}\" for {}: {}", entry.name, module_name, result.reason);
				break;
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
}

void PatchManager::Clear() {
	m_directory_loaded = false;
	m_title_id.clear();
	m_app_version.clear();
	m_entries.clear();
	m_selection = {};
	m_writes.clear();
}

} // namespace Loader::Patches
