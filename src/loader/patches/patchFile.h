#ifndef KYTY_LOADER_PATCHES_PATCH_FILE_H_
#define KYTY_LOADER_PATCHES_PATCH_FILE_H_

#include "common/common.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Game patch files in the GoldHEN/etaHEN "xml_prospero" format that shadPS4 also reads:
//
//   <Patch>
//     <TitleID><ID>PPSA01325</ID>...</TitleID>
//     <Metadata Name="..." AppVer="01.905.000" AppElf="eboot.bin" ImageBase="0x0" ...>
//       <PatchList><Line Type="bytes" Address="0x1b4b8c0" Value="c5fb..."/>...</PatchList>
//     </Metadata>
//     ...
//   </Patch>
//
// plus three optional extensions that let the loader prove a patch matches the image before it
// writes anything: Line Original="<hex>" (the bytes the line replaces), Metadata
// ElfXXH3="<hex>" (XXH3-64 of the module file on disk) and Metadata RequiresFrameCap="<fps>".
//
// This part is pure (no emulator state), so it is unit-tested on its own.
namespace Loader::Patches {

// One write. `address` is module-relative, i.e. an ELF p_vaddr: the file's Address minus the
// entry's ImageBase. `original` is empty when the file gives no Original attribute; otherwise it
// has exactly the length of `value`.
struct PatchLine {
	uint64_t             address = 0;
	std::vector<uint8_t> value;
	std::vector<uint8_t> original;
};

// One <Metadata> block: a named patch for one version of one module of a game.
struct PatchEntry {
	std::vector<std::string> title_ids;
	std::string              title;
	std::string              name;
	std::string              author;
	std::string              note;
	std::string              patch_version;
	std::string              app_version;
	std::string              app_elf;
	// shadPS4's isEnabled="true": on without being named by the user.
	bool enabled_by_default = false;
	// The presented frame rate the patch is written for; 0 when it does not depend on it.
	uint32_t required_frame_cap = 0;
	// XXH3-64 of the module file the patch was made against.
	std::optional<uint64_t> elf_xxh3;
	std::vector<PatchLine>  lines;
	// Non-empty when the block is well-formed XML but cannot be applied (an unsupported line
	// type, a value that does not fit its type, ...). The entry is kept rather than dropped so
	// the reason is reported when the user asks for the patch.
	std::string error;
};

struct PatchFile {
	std::vector<PatchEntry> entries;
};

// Parses a patch document. Returns false only when the text is not a patch document at all
// (malformed XML, no <Patch> root); problems inside one <Metadata> block land in that entry's
// `error` and leave the other blocks usable.
bool ParsePatchFile(std::string_view xml, PatchFile* out, std::string* error);

// Encodes a <Line> Value of the given Type into the bytes written to guest memory:
//   bytes                      hex string, spaces allowed
//   byte, bytes16/32/64        unsigned integer of 1/2/4/8 bytes, little-endian; a 0x, # or $
//                              prefix means hexadecimal, otherwise decimal
//   float32, float64           decimal floating point, little-endian IEEE 754
//   utf8                       the string plus a NUL byte
//   utf16                      the string as UTF-16LE plus a 2-byte NUL
// Any other type (mask, mask_jump32, ...) is reported as not supported.
bool EncodeLineValue(std::string_view type, std::string_view text, std::vector<uint8_t>* out,
                     std::string* error);

// Hex string to bytes ("c5 fb 10" or "c5fb10"). False for odd digit counts and non-hex text.
bool ParseHexBytes(std::string_view text, std::vector<uint8_t>* out);

// The patch directory's patches.json: {"enabled": ["<patch name>", ...]}. A missing "enabled"
// key is an empty list.
bool ParseEnabledNames(std::string_view json, std::vector<std::string>* out, std::string* error);

} // namespace Loader::Patches

#endif /* KYTY_LOADER_PATCHES_PATCH_FILE_H_ */
