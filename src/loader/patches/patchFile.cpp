#include "loader/patches/patchFile.h"

#include "common/stringUtils.h"

#include <bit>
#include <charconv>
#include <cstdio>
#include <fmt/format.h>
#include <locale>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include <sstream>
#include <utility>

namespace Loader::Patches {

namespace {

std::string_view Trim(std::string_view text) {
	while (!text.empty() && Common::IsSpace(text.front())) {
		text.remove_prefix(1);
	}
	while (!text.empty() && Common::IsSpace(text.back())) {
		text.remove_suffix(1);
	}
	return text;
}

int HexDigit(char c) {
	if (c >= '0' && c <= '9') {
		return c - '0';
	}
	if (c >= 'a' && c <= 'f') {
		return c - 'a' + 10;
	}
	if (c >= 'A' && c <= 'F') {
		return c - 'A' + 10;
	}
	return -1;
}

// GoldHEN integer syntax: a 0x, # or $ prefix selects hexadecimal, anything else is decimal.
bool ParseUnsigned(std::string_view text, uint64_t* out) {
	text     = Trim(text);
	int base = 10;
	if (text.starts_with("0x") || text.starts_with("0X")) {
		text.remove_prefix(2);
		base = 16;
	} else if (text.starts_with('#') || text.starts_with('$')) {
		text.remove_prefix(1);
		base = 16;
	}
	if (text.empty()) {
		return false;
	}
	const auto [end, result] = std::from_chars(text.data(), text.data() + text.size(), *out, base);
	return result == std::errc {} && end == text.data() + text.size();
}

// Locale-independent: a patch file must mean the same on every host.
template <typename T>
bool ParseFloat(std::string_view text, T* out) {
	std::istringstream stream {std::string(Trim(text))};
	stream.imbue(std::locale::classic());
	stream >> *out;
	return !stream.fail() && stream.peek() == std::char_traits<char>::eof();
}

template <typename T>
void AppendLittleEndian(T value, std::vector<uint8_t>* out) {
	for (size_t i = 0; i < sizeof(T); i++) {
		out->push_back(static_cast<uint8_t>(value >> (8u * i)));
	}
}

// Decodes one UTF-8 sequence at `text[*pos]`, advancing *pos. False on malformed input
// (truncated, overlong, surrogate or out-of-range code points).
bool DecodeUtf8(std::string_view text, size_t* pos, char32_t* code_point) {
	const auto lead = static_cast<uint8_t>(text[*pos]);
	size_t     length = 0;
	char32_t   value  = 0;
	char32_t   min    = 0;
	if (lead < 0x80u) {
		*code_point = lead;
		(*pos)++;
		return true;
	}
	if ((lead & 0xE0u) == 0xC0u) {
		length = 2;
		value  = lead & 0x1Fu;
		min    = 0x80;
	} else if ((lead & 0xF0u) == 0xE0u) {
		length = 3;
		value  = lead & 0x0Fu;
		min    = 0x800;
	} else if ((lead & 0xF8u) == 0xF0u) {
		length = 4;
		value  = lead & 0x07u;
		min    = 0x10000;
	} else {
		return false;
	}
	if (*pos + length > text.size()) {
		return false;
	}
	for (size_t i = 1; i < length; i++) {
		const auto next = static_cast<uint8_t>(text[*pos + i]);
		if ((next & 0xC0u) != 0x80u) {
			return false;
		}
		value = (value << 6u) | (next & 0x3Fu);
	}
	if (value < min || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
		return false;
	}
	*code_point = value;
	*pos += length;
	return true;
}

bool EncodeUtf16(std::string_view text, std::vector<uint8_t>* out) {
	for (size_t pos = 0; pos < text.size();) {
		char32_t code_point = 0;
		if (!DecodeUtf8(text, &pos, &code_point)) {
			return false;
		}
		if (code_point < 0x10000) {
			AppendLittleEndian(static_cast<uint16_t>(code_point), out);
		} else {
			const char32_t offset = code_point - 0x10000;
			AppendLittleEndian(static_cast<uint16_t>(0xD800u + (offset >> 10u)), out);
			AppendLittleEndian(static_cast<uint16_t>(0xDC00u + (offset & 0x3FFu)), out);
		}
	}
	AppendLittleEndian(uint16_t {0}, out);
	return true;
}

struct IntegerType {
	std::string_view name;
	size_t           size;
};

constexpr IntegerType INTEGER_TYPES[] = {
    {"byte", 1},
    {"bytes16", 2},
    {"bytes32", 4},
    {"bytes64", 8},
};

bool EncodeInteger(const IntegerType& type, std::string_view text, std::vector<uint8_t>* out,
                   std::string* error) {
	uint64_t value = 0;
	if (!ParseUnsigned(text, &value)) {
		*error = fmt::format("{} value \"{}\" is not an unsigned integer", type.name, text);
		return false;
	}
	if (type.size < sizeof(uint64_t) && (value >> (8u * type.size)) != 0) {
		*error = fmt::format("{} value \"{}\" does not fit in {} byte(s)", type.name, text,
		                     type.size);
		return false;
	}
	for (size_t i = 0; i < type.size; i++) {
		out->push_back(static_cast<uint8_t>(value >> (8u * i)));
	}
	return true;
}

std::string Attribute(const pugi::xml_node& node, const char* name) {
	return std::string(Trim(node.attribute(name).as_string()));
}

// Fills the parts of `entry` that can fail; returns the reason the entry cannot be applied, or an
// empty string.
std::string ParseEntryBody(const pugi::xml_node& metadata, PatchEntry* entry) {
	if (entry->name.empty()) {
		return "missing Name";
	}
	if (entry->app_version.empty() || entry->app_elf.empty()) {
		return "missing AppVer or AppElf";
	}

	uint64_t image_base = 0;
	if (const auto text = Attribute(metadata, "ImageBase");
	    !text.empty() && !ParseUnsigned(text, &image_base)) {
		return fmt::format("invalid ImageBase \"{}\"", text);
	}
	if (const auto text = Attribute(metadata, "RequiresFrameCap"); !text.empty()) {
		uint64_t cap = 0;
		if (!ParseUnsigned(text, &cap) || cap == 0 || cap > UINT32_MAX) {
			return fmt::format("invalid RequiresFrameCap \"{}\"", text);
		}
		entry->required_frame_cap = static_cast<uint32_t>(cap);
	}
	if (auto text = Attribute(metadata, "ElfXXH3"); !text.empty()) {
		if (!text.starts_with("0x") && !text.starts_with("0X")) {
			text = "0x" + text;
		}
		uint64_t hash = 0;
		if (!ParseUnsigned(text, &hash)) {
			return fmt::format("invalid ElfXXH3 \"{}\"", text);
		}
		entry->elf_xxh3 = hash;
	}

	for (const auto& line_node: metadata.child("PatchList").children("Line")) {
		const auto type    = Attribute(line_node, "Type");
		const auto address = Attribute(line_node, "Address");
		// Values are taken verbatim: leading/trailing spaces are part of a utf8/utf16 string.
		const std::string value = line_node.attribute("Value").as_string();

		PatchLine line;
		uint64_t  absolute = 0;
		if (!ParseUnsigned(address, &absolute) || absolute < image_base) {
			return fmt::format("invalid Address \"{}\"", address);
		}
		line.address = absolute - image_base;

		std::string value_error;
		if (!EncodeLineValue(type, value, &line.value, &value_error)) {
			return fmt::format("line at {}: {}", address, value_error);
		}
		if (const auto original = line_node.attribute("Original"); original) {
			if (!ParseHexBytes(original.as_string(), &line.original)) {
				return fmt::format("line at {}: invalid Original \"{}\"", address,
				                   original.as_string());
			}
			if (line.original.size() != line.value.size()) {
				return fmt::format("line at {}: Original has {} byte(s), Value has {}", address,
				                   line.original.size(), line.value.size());
			}
		}
		entry->lines.push_back(std::move(line));
	}
	if (entry->lines.empty()) {
		return "no Line in PatchList";
	}
	return {};
}

PatchEntry ParseEntry(const pugi::xml_node& metadata, const std::vector<std::string>& title_ids) {
	PatchEntry entry;
	entry.title_ids          = title_ids;
	entry.title              = Attribute(metadata, "Title");
	entry.name               = Attribute(metadata, "Name");
	entry.author             = Attribute(metadata, "Author");
	entry.note               = Attribute(metadata, "Note");
	entry.patch_version      = Attribute(metadata, "PatchVer");
	entry.app_version        = Attribute(metadata, "AppVer");
	entry.app_elf            = Attribute(metadata, "AppElf");
	entry.enabled_by_default = Common::EqualNoCase(Attribute(metadata, "isEnabled"), "true");
	entry.error              = ParseEntryBody(metadata, &entry);
	return entry;
}

} // namespace

bool ParseHexBytes(std::string_view text, std::vector<uint8_t>* out) {
	out->clear();
	int high = -1;
	for (const char c: text) {
		if (Common::IsSpace(c)) {
			continue;
		}
		const int digit = HexDigit(c);
		if (digit < 0) {
			return false;
		}
		if (high < 0) {
			high = digit;
		} else {
			out->push_back(static_cast<uint8_t>((high << 4) | digit));
			high = -1;
		}
	}
	return high < 0 && !out->empty();
}

bool EncodeLineValue(std::string_view type, std::string_view text, std::vector<uint8_t>* out,
                     std::string* error) {
	out->clear();
	if (Common::EqualNoCase(type, "bytes")) {
		if (!ParseHexBytes(text, out)) {
			*error = fmt::format("bytes value \"{}\" is not a hex byte string", text);
			return false;
		}
		return true;
	}
	for (const auto& integer: INTEGER_TYPES) {
		if (Common::EqualNoCase(type, integer.name)) {
			return EncodeInteger(integer, text, out, error);
		}
	}
	if (Common::EqualNoCase(type, "float32")) {
		float value = 0.0F;
		if (!ParseFloat(text, &value)) {
			*error = fmt::format("float32 value \"{}\" is not a number", text);
			return false;
		}
		AppendLittleEndian(std::bit_cast<uint32_t>(value), out);
		return true;
	}
	if (Common::EqualNoCase(type, "float64")) {
		double value = 0.0;
		if (!ParseFloat(text, &value)) {
			*error = fmt::format("float64 value \"{}\" is not a number", text);
			return false;
		}
		AppendLittleEndian(std::bit_cast<uint64_t>(value), out);
		return true;
	}
	if (Common::EqualNoCase(type, "utf8")) {
		out->assign(text.begin(), text.end());
		out->push_back(0);
		return true;
	}
	if (Common::EqualNoCase(type, "utf16")) {
		if (!EncodeUtf16(text, out)) {
			*error = "utf16 value is not valid UTF-8 text";
			return false;
		}
		return true;
	}
	*error = fmt::format("line type \"{}\" is not supported yet", type);
	return false;
}

bool ParsePatchFile(std::string_view xml, PatchFile* out, std::string* error) {
	pugi::xml_document doc;
	if (const auto result = doc.load_buffer(xml.data(), xml.size()); !result) {
		*error = fmt::format("{} at offset {}", result.description(), result.offset);
		return false;
	}
	const auto root = doc.child("Patch");
	if (!root) {
		*error = "no <Patch> root element";
		return false;
	}

	std::vector<std::string> title_ids;
	for (const auto& id: root.child("TitleID").children("ID")) {
		title_ids.emplace_back(Trim(id.text().as_string()));
	}
	out->entries.clear();
	for (const auto& metadata: root.children("Metadata")) {
		out->entries.push_back(ParseEntry(metadata, title_ids));
	}
	return true;
}

bool ParseEnabledNames(std::string_view json, std::vector<std::string>* out, std::string* error) {
	out->clear();
	const auto root = nlohmann::json::parse(json.begin(), json.end(), nullptr, false);
	if (root.is_discarded() || !root.is_object()) {
		*error = "expected a JSON object such as {\"enabled\": [\"<patch name>\"]}";
		return false;
	}
	const auto enabled = root.find("enabled");
	if (enabled == root.end()) {
		return true;
	}
	if (!enabled->is_array()) {
		*error = "\"enabled\" must be an array of patch names";
		return false;
	}
	for (const auto& name: *enabled) {
		if (!name.is_string()) {
			*error = "\"enabled\" must be an array of patch names";
			return false;
		}
		out->push_back(name.get<std::string>());
	}
	return true;
}

} // namespace Loader::Patches
