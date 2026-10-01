#include "loader/patches/patchApplier.h"
#include "loader/patches/patchFile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace Loader::Patches;

using Bytes = std::vector<uint8_t>;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PatchLoaderTests: failed: %s\n", message);
		std::abort();
	}
}

Bytes Encode(const char* type, const char* text) {
	Bytes       out;
	std::string error;
	if (!EncodeLineValue(type, text, &out, &error)) {
		std::fprintf(stderr, "PatchLoaderTests: %s \"%s\" did not encode: %s\n", type, text,
		             error.c_str());
		std::abort();
	}
	return out;
}

bool EncodeFails(const char* type, const char* text) {
	Bytes       out;
	std::string error;
	return !EncodeLineValue(type, text, &out, &error) && !error.empty();
}

PatchFile Parse(const std::string& xml) {
	PatchFile   file;
	std::string error;
	if (!ParsePatchFile(xml, &file, &error)) {
		std::fprintf(stderr, "PatchLoaderTests: parse failed: %s\n", error.c_str());
		std::abort();
	}
	return file;
}

PatchLine Line(uint64_t address, Bytes value, Bytes original = {}) {
	return PatchLine {address, std::move(value), std::move(original)};
}

PatchEntry Entry(std::vector<PatchLine> lines) {
	PatchEntry entry;
	entry.title_ids   = {"PPSA01325"};
	entry.name        = "test";
	entry.app_version = "01.905.000";
	entry.app_elf     = "eboot.bin";
	entry.lines       = std::move(lines);
	return entry;
}

// A two-segment module: text at 0x1000 (16 bytes 0x10..0x1f), data at 0x4000 (8 bytes 0xa0..).
struct Image {
	Image() {
		for (uint8_t i = 0; i < 16; i++) {
			text.push_back(0x10 + i);
		}
		for (uint8_t i = 0; i < 8; i++) {
			data.push_back(0xa0 + i);
		}
		segments = {{0x1000, text}, {0x4000, data}};
	}
	Image(const Image&)            = delete;
	Image& operator=(const Image&) = delete;

	Bytes                     text;
	Bytes                     data;
	std::vector<ImageSegment> segments;
};

ModuleIdentity Identity(std::optional<uint64_t> hash = std::nullopt, int* hash_calls = nullptr) {
	return {"PPSA01325", "01.905.000", "EBOOT.BIN", [hash, hash_calls]() {
		        if (hash_calls != nullptr) {
			        (*hash_calls)++;
		        }
		        return hash;
	        }};
}

const char* const ASTRO_XML = R"(<?xml version="1.0" encoding="utf-8"?>
<!-- comment -->
<Patch>
  <TitleID>
    <ID>PPSA01325</ID>
    <ID> PPSA01323 </ID>
  </TitleID>
  <Metadata Title="Astro&apos;s Playroom" Name="Astro&apos;s Playroom - correct speed at 30 fps"
            Note="n" Author="a" PatchVer="1.0" AppVer="01.905.000" AppElf="eboot.bin"
            ImageBase="0x400000" RequiresFrameCap="30" ElfXXH3="0123456789abcdef">
    <PatchList>
      <Line Type="bytes" Address="0x401000" Value="c5 fb 59" Original="c5fb10"/>
      <Line Type="float32" Address="0x401010" Value="0.033333335" Original="8988883c"/>
    </PatchList>
  </Metadata>
  <Metadata Name="Second" AppVer="01.000.000" AppElf="eboot.bin" isEnabled="TRUE">
    <PatchList><Line Type="bytes64" Address="#10" Value="33333"/></PatchList>
  </Metadata>
  <Metadata Name="Masked" AppVer="01.905.000" AppElf="eboot.bin">
    <PatchList><Line Type="mask" Address="0x0" Value="90"/></PatchList>
  </Metadata>
</Patch>)";

void TestParse() {
	const auto file = Parse(ASTRO_XML);
	Check(file.entries.size() == 3, "three Metadata blocks");

	const auto& astro = file.entries[0];
	Check(astro.error.empty(), "astro entry is valid");
	Check(astro.title_ids == std::vector<std::string> {"PPSA01325", "PPSA01323"},
	      "every TitleID is kept, trimmed");
	Check(astro.title == "Astro's Playroom", "&apos; in Title");
	Check(astro.name == "Astro's Playroom - correct speed at 30 fps", "&apos; in Name");
	Check(astro.app_version == "01.905.000" && astro.app_elf == "eboot.bin", "AppVer and AppElf");
	Check(astro.required_frame_cap == 30, "RequiresFrameCap");
	Check(astro.elf_xxh3 == 0x0123456789abcdefULL, "ElfXXH3 without 0x is hex");
	Check(!astro.enabled_by_default, "isEnabled defaults to false");
	Check(astro.lines.size() == 2, "two lines");
	Check(astro.lines[0].address == 0x1000, "Address is relative to ImageBase");
	Check(astro.lines[0].value == Bytes {0xc5, 0xfb, 0x59}, "bytes with spaces");
	Check(astro.lines[0].original == Bytes {0xc5, 0xfb, 0x10}, "Original");
	Check(astro.lines[1].value == Bytes {0x89, 0x88, 0x08, 0x3d}, "float32 line");

	const auto& second = file.entries[1];
	Check(second.error.empty(), "second entry is valid");
	Check(second.title_ids.size() == 2, "TitleID list is shared by every Metadata block");
	Check(second.enabled_by_default, "isEnabled=\"TRUE\"");
	Check(second.lines.size() == 1 && second.lines[0].address == 0x10, "# address is hex");
	Check(second.lines[0].original.empty(), "no Original");
	Check(!second.elf_xxh3.has_value(), "no ElfXXH3");

	const auto& masked = file.entries[2];
	Check(masked.error.find("\"mask\" is not supported yet") != std::string::npos,
	      "unsupported line type is recorded on the entry");

	PatchFile   broken;
	std::string error;
	Check(!ParsePatchFile("<Patch><Metadata></Patch>", &broken, &error) && !error.empty(),
	      "malformed XML is an error");
	Check(!ParsePatchFile("<NotAPatch/>", &broken, &error), "missing <Patch> root is an error");
}

void TestParseEntryErrors() {
	const auto file = Parse(R"(<Patch><TitleID><ID>X</ID></TitleID>
  <Metadata Name="odd" AppVer="1" AppElf="eboot.bin"><PatchList>
    <Line Type="bytes" Address="0x0" Value="abc"/></PatchList></Metadata>
  <Metadata Name="length" AppVer="1" AppElf="eboot.bin"><PatchList>
    <Line Type="bytes32" Address="0x0" Value="1" Original="0000"/></PatchList></Metadata>
  <Metadata Name="overflow" AppVer="1" AppElf="eboot.bin"><PatchList>
    <Line Type="byte" Address="0x0" Value="256"/></PatchList></Metadata>
  <Metadata Name="below base" AppVer="1" AppElf="eboot.bin" ImageBase="0x1000"><PatchList>
    <Line Type="byte" Address="0x10" Value="1"/></PatchList></Metadata>
  <Metadata Name="empty" AppVer="1" AppElf="eboot.bin"><PatchList/></Metadata>
  <Metadata AppVer="1" AppElf="eboot.bin"><PatchList>
    <Line Type="byte" Address="0x0" Value="1"/></PatchList></Metadata>
</Patch>)");
	Check(file.entries.size() == 6, "every block is kept");
	for (const auto& entry: file.entries) {
		Check(!entry.error.empty(), "each broken block carries an error");
	}
	Check(file.entries[0].error.find("hex") != std::string::npos, "odd hex digit count");
	Check(file.entries[1].error.find("Original has 2 byte(s), Value has 4") != std::string::npos,
	      "Original length mismatch");
	Check(file.entries[2].error.find("does not fit") != std::string::npos, "byte overflow");
	Check(file.entries[3].error.find("Address") != std::string::npos, "Address below ImageBase");
	Check(file.entries[5].error == "missing Name", "missing Name");
}

void TestEncoding() {
	Check(Encode("bytes", "c5fb 59") == Bytes {0xc5, 0xfb, 0x59}, "bytes");
	Check(Encode("BYTES", "AB") == Bytes {0xab}, "type names are case-insensitive");
	Check(Encode("byte", "0xff") == Bytes {0xff}, "byte hex");
	Check(Encode("byte", "200") == Bytes {200}, "byte decimal");
	Check(Encode("bytes16", "$1234") == Bytes {0x34, 0x12}, "bytes16 $ hex, little-endian");
	Check(Encode("bytes32", "#deadbeef") == Bytes {0xef, 0xbe, 0xad, 0xde}, "bytes32 # hex");
	Check(Encode("bytes32", "4294967295") == Bytes {0xff, 0xff, 0xff, 0xff}, "bytes32 max");
	Check(Encode("bytes64", "33333") == Bytes {0x35, 0x82, 0, 0, 0, 0, 0, 0}, "bytes64 33333");
	Check(Encode("bytes64", "0xffffffffffffffff") == Bytes(8, 0xff), "bytes64 max");
	Check(Encode("float32", "0.033333335") == Bytes {0x89, 0x88, 0x08, 0x3d}, "float32 1/30");
	Check(Encode("float32", "1") == Bytes {0x00, 0x00, 0x80, 0x3f}, "float32 1");
	Check(Encode("float64", "30.0") == Bytes {0, 0, 0, 0, 0, 0, 0x3e, 0x40}, "float64 30.0");
	Check(Encode("float64", "60") == Bytes {0, 0, 0, 0, 0, 0, 0x4e, 0x40}, "float64 60");
	Check(Encode("float64", "-0.5") == Bytes {0, 0, 0, 0, 0, 0, 0xe0, 0xbf}, "float64 -0.5");
	Check(Encode("utf8", "Hi") == Bytes {'H', 'i', 0}, "utf8 adds NUL");
	Check(Encode("utf16", "A\xc3\xa9") == Bytes {'A', 0, 0xe9, 0, 0, 0}, "utf16 adds 2-byte NUL");
	Check(Encode("utf16", "\xf0\x9f\x98\x80") == Bytes {0x3d, 0xd8, 0x00, 0xde, 0, 0},
	      "utf16 surrogate pair");

	Check(EncodeFails("bytes", "abc"), "odd hex digit count");
	Check(EncodeFails("bytes", "zz"), "non-hex bytes");
	Check(EncodeFails("bytes", ""), "empty bytes");
	Check(EncodeFails("byte", "256"), "byte overflow");
	Check(EncodeFails("bytes16", "0x10000"), "bytes16 overflow");
	Check(EncodeFails("bytes32", "4294967296"), "bytes32 overflow");
	Check(EncodeFails("bytes64", "18446744073709551616"), "bytes64 overflow");
	Check(EncodeFails("bytes32", "-1"), "negative integer");
	Check(EncodeFails("bytes32", "12abc"), "trailing garbage");
	Check(EncodeFails("float32", "fast"), "float32 not a number");
	Check(EncodeFails("float32", "1e60"), "float32 out of range");
	Check(EncodeFails("float64", "1.0x"), "float64 trailing garbage");
	Check(EncodeFails("utf16", "\xc3"), "utf16 of truncated UTF-8");
	Check(EncodeFails("mask_jump32", "90"), "mask_jump32 not supported");
}

void TestEnabledNamesJson() {
	std::vector<std::string> names;
	std::string              error;
	Check(ParseEnabledNames(R"({"enabled": ["A", "b c"]})", &names, &error) &&
	          names == std::vector<std::string> {"A", "b c"},
	      "patches.json names");
	Check(ParseEnabledNames("{}", &names, &error) && names.empty(), "no enabled key");
	Check(!ParseEnabledNames(R"({"enabled": "A"})", &names, &error), "enabled not an array");
	Check(!ParseEnabledNames(R"({"enabled": [1]})", &names, &error), "non-string name");
	Check(!ParseEnabledNames("{", &names, &error), "malformed json");
}

void TestMatching() {
	auto entry = Entry({Line(0x1000, {0x90})});
	int  calls = 0;
	Check(MatchModule(entry, Identity(std::nullopt, &calls)) == MatchResult::Match,
	      "title, AppVer and AppElf (case-insensitive) match");
	Check(calls == 0, "no hash computed for an unpinned entry");

	auto other_title = Identity();
	other_title.title_id = "PPSA00000";
	Check(MatchModule(entry, other_title) == MatchResult::OtherModule, "other title");
	auto other_version = Identity();
	other_version.app_version = "01.905.001";
	Check(MatchModule(entry, other_version) == MatchResult::OtherModule, "AppVer is exact");
	auto other_module = Identity();
	other_module.module_name = "libgame.prx";
	Check(MatchModule(entry, other_module) == MatchResult::OtherModule, "other module");

	entry.elf_xxh3 = 0x1234;
	Check(MatchModule(entry, Identity(0x1234, &calls)) == MatchResult::Match, "hash match");
	Check(calls == 1, "hash computed for a pinned entry");
	Check(MatchModule(entry, Identity(0x9999)) == MatchResult::HashMismatch, "hash mismatch");
	Check(MatchModule(entry, Identity(std::nullopt)) == MatchResult::HashMismatch,
	      "unreadable file does not match a pin");
	calls = 0;
	Check(MatchModule(entry, other_module) == MatchResult::OtherModule && calls == 0,
	      "no hash for another module");
}

void TestSelection() {
	const auto file = Parse(ASTRO_XML);

	const PatchSelection none;
	Check(!none.IsEnabled(file.entries[0]), "not enabled unless asked for");
	Check(none.IsEnabled(file.entries[1]), "isEnabled=\"true\" enables without a name");

	const PatchSelection cli({"astro's playroom - CORRECT SPEED AT 30 FPS", "Typo", "Second"});
	Check(cli.IsEnabled(file.entries[0]), "names compare case-insensitively");
	Check(!cli.IsEnabled(file.entries[2]), "other names stay disabled");

	// "Second" exists, but for another game version: it is unknown for 01.905.000.
	const auto unknown = cli.UnknownNames(file.entries, "PPSA01325", "01.905.000");
	Check(unknown == std::vector<std::string> {"Typo", "Second"}, "unknown names for this game");
	Check(cli.UnknownNames(file.entries, "PPSA01325", "01.000.000").size() == 2,
	      "the astro name is unknown for another game version");
}

void TestApplySuccess() {
	Image image;
	auto  entry = Entry({Line(0x1002, {0xaa, 0xbb}, {0x12, 0x13}), Line(0x4007, {0xcc}, {0xa7})});
	auto  expected_text = image.text;
	auto  expected_data = image.data;
	expected_text[2]    = 0xaa;
	expected_text[3]    = 0xbb;
	expected_data[7]    = 0xcc;

	const auto result = ApplyPatch(entry, image.segments, {});
	Check(result.status == ApplyStatus::Applied, "applied");
	Check(result.bytes_written == 3 && result.writes.size() == 2, "written byte count");
	Check(result.writes[0].vaddr == 0x1002 && result.writes[0].size == 2, "first write range");
	Check(image.text == expected_text && image.data == expected_data,
	      "exactly the patched bytes changed");

	const auto again = ApplyPatch(entry, image.segments, {});
	Check(again.status == ApplyStatus::Rejected, "re-applying fails verification");
	Check(again.reason == "original bytes differ at +0x1002 (expected 1213, found aabb)",
	      "mismatch reason names the address and both byte strings");
	Check(image.text == expected_text && image.data == expected_data, "nothing written twice");
}

void TestApplyIsAtomic() {
	Image       image;
	const auto  pristine_text = image.text;
	const auto  pristine_data = image.data;
	const auto& segments      = image.segments;

	auto unchanged = [&]() { return image.text == pristine_text && image.data == pristine_data; };

	// The first line verifies; the second, in the other segment, does not.
	auto mismatch = Entry({Line(0x1000, {0x90}, {0x10}), Line(0x4000, {0x90}, {0x00})});
	Check(ApplyPatch(mismatch, segments, {}).status == ApplyStatus::Rejected, "mismatch rejected");
	Check(unchanged(), "a mismatch in any line writes nothing");

	auto outside = Entry({Line(0x1000, {0x90}, {0x10}), Line(0x100f, {0x90, 0x90}, {0x1f, 0})});
	const auto outside_result = ApplyPatch(outside, segments, {});
	Check(outside_result.status == ApplyStatus::Rejected &&
	          outside_result.reason.find("outside") != std::string::npos,
	      "a line crossing the end of a segment's file bytes is rejected");
	Check(unchanged(), "out-of-segment writes nothing");

	auto gap = Entry({Line(0x2000, {0x90}, {0x00})});
	Check(ApplyPatch(gap, segments, {}).status == ApplyStatus::Rejected, "unmapped address");

	auto overlap = Entry({Line(0x1004, {0x90}, {0x14}), Line(0x1003, {0x90, 0x90}, {0x13, 0x14})});
	const auto overlap_result = ApplyPatch(overlap, segments, {});
	Check(overlap_result.status == ApplyStatus::Rejected &&
	          overlap_result.reason.find("overlap") != std::string::npos,
	      "overlapping lines are rejected");
	Check(unchanged(), "overlap writes nothing");

	auto unverified = Entry({Line(0x1000, {0x90}, {0x10}), Line(0x1001, {0x90})});
	const auto unverified_result = ApplyPatch(unverified, segments, {});
	Check(unverified_result.status == ApplyStatus::Rejected &&
	          unverified_result.reason.find("no Original") != std::string::npos,
	      "a line without Original needs a matched ElfXXH3 pin");
	Check(unchanged(), "unverifiable writes nothing");

	auto invalid  = Entry({Line(0x1000, {0x90}, {0x10})});
	invalid.error = "line type \"mask\" is not supported yet";
	Check(ApplyPatch(invalid, segments, {}).status == ApplyStatus::Rejected && unchanged(),
	      "an invalid entry writes nothing");

	ApplyOptions hashed;
	hashed.file_hash_verified = true;
	Check(ApplyPatch(unverified, segments, hashed).status == ApplyStatus::Applied,
	      "with a matched hash pin, lines without Original are allowed");
	Check(image.text[0] == 0x90 && image.text[1] == 0x90, "pinned patch written");
}

void TestFrameCap() {
	Image image;
	auto  entry              = Entry({Line(0x1000, {0x90}, {0x10})});
	entry.required_frame_cap = 30;

	const auto uncapped = ApplyPatch(entry, image.segments, {});
	Check(uncapped.status == ApplyStatus::NeedsFrameCap, "refused when uncapped");
	Check(uncapped.reason == "requires --frame-cap 30 (current: uncapped)", "uncapped reason");
	Check(image.text[0] == 0x10, "refused patch writes nothing");

	ApplyOptions sixty;
	sixty.frame_cap   = 60;
	const auto at_60  = ApplyPatch(entry, image.segments, sixty);
	Check(at_60.status == ApplyStatus::NeedsFrameCap &&
	          at_60.reason == "requires --frame-cap 30 (current: 60)",
	      "refused at another cap");

	ApplyOptions thirty;
	thirty.frame_cap = 30;
	Check(ApplyPatch(entry, image.segments, thirty).status == ApplyStatus::Applied,
	      "applied at the required cap");
	Check(image.text[0] == 0x90, "written at the required cap");
}

const char* const CHEAT_JSON = R"({
  "name": "Astro's Playroom", "id": "PPSA01325", "version": "01.905.000", "process": "eboot.bin",
  "mods": [
    {"name": "Default on", "memory": [{"offset": "0x401002", "off": "1213", "on": "aabb"}]},
    {"name": "Off", "enabled": false, "memory": [{"offset": "401000", "off": "10", "on": "90"}]},
    {"name": "Bad bytes", "memory": [{"offset": "401000", "off": "10", "on": "9090"}]},
    {"name": "Bad offset", "memory": [{"offset": "zz", "off": "10", "on": "90"}]},
    {"memory": [{"offset": "401000", "off": "10", "on": "90"}]},
    {"name": "Empty", "memory": []}
  ]
})";

PatchFile ParseCheat(const std::string& json) {
	PatchFile   file;
	std::string error;
	if (!ParseCheatJson(json, &file, &error)) {
		std::fprintf(stderr, "PatchLoaderTests: cheat parse failed: %s\n", error.c_str());
		std::abort();
	}
	return file;
}

void TestParseCheatJson() {
	const auto file = ParseCheat(CHEAT_JSON);
	Check(file.entries.size() == 6, "one entry per mod");

	const auto& on = file.entries[0];
	Check(on.error.empty(), "valid mod");
	Check(on.title_ids == std::vector<std::string> {"PPSA01325"}, "id is the title id");
	Check(on.title == "Astro's Playroom", "name is the title");
	Check(on.app_version == "01.905.000" && on.app_elf == "eboot.bin", "version and process");
	Check(on.name == "Default on", "mod name");
	Check(on.enabled_by_default, "a mod without \"enabled\" is enabled");
	Check(on.auto_image_base, "cheat entries have no stated image base");
	Check(on.lines.size() == 1 && on.lines[0].address == 0x401002, "hex offset with 0x");
	Check(on.lines[0].value == Bytes {0xaa, 0xbb}, "on is the value");
	Check(on.lines[0].original == Bytes {0x12, 0x13}, "off is the original");

	const auto& off = file.entries[1];
	Check(off.error.empty() && !off.enabled_by_default, "\"enabled\": false");
	Check(off.lines[0].address == 0x401000, "hex offset without 0x");

	Check(file.entries[2].error == "write at 401000: off has 1 byte(s), on has 2",
	      "off and on lengths differ");
	Check(file.entries[3].error == "invalid offset \"zz\"", "invalid offset");
	Check(file.entries[4].error == "missing name", "missing name");
	Check(file.entries[5].error == "no \"memory\" writes", "empty memory list");

	PatchFile   broken;
	std::string error;
	Check(!ParseCheatJson("{", &broken, &error) && !error.empty(), "malformed JSON");
	Check(!ParseCheatJson(R"({"id": "X", "version": "1", "mods": []})", &broken, &error),
	      "missing process");
	Check(!ParseCheatJson(R"({"id": "X", "version": "1", "process": "e", "mods": {}})", &broken,
	                      &error),
	      "mods is not an array");
}

PatchEntry AutoEntry(std::vector<PatchLine> lines) {
	auto entry            = Entry(std::move(lines));
	entry.auto_image_base = true;
	return entry;
}

void TestResolveImageBase() {
	Image       image;
	std::string reason;

	// Written against an image loaded at 0x400000: text+2 is 0x401002.
	const auto shifted = AutoEntry({Line(0x401002, {0xaa, 0xbb}, {0x12, 0x13}),
	                                Line(0x404007, {0xcc}, {0xa7})});
	const std::vector<const PatchEntry*> one {&shifted};
	Check(ResolveImageBase(one, image.segments, &reason) == 0x400000, "anchor at a shifted base");

	const auto rebased = Rebase(shifted, 0x400000);
	Check(!rebased.auto_image_base, "rebased entry is module-relative");
	Check(rebased.lines[0].address == 0x1002 && rebased.lines[1].address == 0x4007,
	      "rebased addresses");
	Image applied;
	Check(ApplyPatch(shifted, applied.segments, {}).status == ApplyStatus::Rejected,
	      "an unresolved entry is not applied");
	Check(ApplyPatch(rebased, applied.segments, {}).status == ApplyStatus::Applied &&
	          applied.text[2] == 0xaa && applied.data[7] == 0xcc,
	      "the rebased entry applies");

	// The anchor byte 0x11 occurs at text+1 and data+0 (after the writes below). The first
	// candidate (text+1, base 0x7fff) puts the second line on 0x12 instead of 0x55; the second
	// (data+0, base 0x5000) verifies both.
	Image twice;
	twice.data[0]   = 0x11;
	twice.data[1]   = 0x55;
	const auto pair = AutoEntry({Line(0x9000, {0x90}, {0x11}), Line(0x9001, {0x90}, {0x55})});
	const std::vector<const PatchEntry*> pair_list {&pair};
	Check(ResolveImageBase(pair_list, twice.segments, &reason) == 0x5000,
	      "first candidate fails validation, the second is accepted");

	// Zero-Original lines do not anchor; nothing else does either.
	const auto zeros = AutoEntry({Line(0x9000, {0x90}, {0x00})});
	const std::vector<const PatchEntry*> zero_list {&zeros};
	Check(!ResolveImageBase(zero_list, image.segments, &reason).has_value() &&
	          reason.find("no line has non-zero original") != std::string::npos,
	      "no anchor: rejected");

	const auto missing = AutoEntry({Line(0x9000, {0x90}, {0xee, 0xef})});
	const std::vector<const PatchEntry*> missing_list {&missing};
	Check(!ResolveImageBase(missing_list, image.segments, &reason).has_value() &&
	          reason.find("not in the module") != std::string::npos,
	      "anchor not found: rejected");

	// One source: the base must suit every entry; an entry that verifies nowhere does not hide
	// the others, and is rejected when applied.
	const std::vector<const PatchEntry*> with_outdated {&missing, &shifted};
	Check(ResolveImageBase(with_outdated, image.segments, &reason) == 0x400000,
	      "the base verifying the most entries");
	const auto outdated = Rebase(missing, 0x400000);
	Check(ApplyPatch(outdated, image.segments, {}).status == ApplyStatus::Rejected,
	      "the outdated entry is rejected at that base");

	// The anchor must not imply a negative base.
	const auto low = AutoEntry({Line(0x1001, {0x90}, {0x14})});
	const std::vector<const PatchEntry*> low_list {&low};
	Check(!ResolveImageBase(low_list, image.segments, &reason).has_value(), "negative base");
}

void TestCodeCaves() {
	Image image;
	// The module is loaded at 0x80000000 and maps 0x8000 bytes. A cave line 0x10 bytes below the
	// module wraps around 2^64 in module-relative terms; one more crosses two pages above it.
	constexpr uint64_t BASE  = 0x80000000;
	constexpr uint64_t SIZE  = 0x8000;
	const uint64_t     below = 0 - uint64_t {0x10};
	auto entry = Entry({Line(below, {0xe9, 0x01}, {0x00, 0x00}),
	                    Line(0xbffe, {0xc3, 0xc3, 0xc3, 0xc3}, Bytes(4, 0)),
	                    Line(0x1000, {0xe9}, {0x10})});
	const std::vector<const PatchEntry*> list {&entry};
	const auto ranges = CaveRanges(list, BASE, SIZE);
	Check(ranges.size() == 2, "two cave ranges");
	Check(ranges[0].vaddr == 0x8000 && ranges[0].size == 2 * CAVE_PAGE_SIZE,
	      "adjacent pages above the module are merged");
	Check(ranges[1].vaddr == 0 - CAVE_PAGE_SIZE && ranges[1].size == CAVE_PAGE_SIZE,
	      "the page below the module wraps around 2^64");

	// Unaligned load address: pages are aligned in absolute terms.
	// Loaded 0x1000 bytes higher: the upper line moves to BASE + 0xcffe..0xd001, one page at
	// BASE + 0xc000; the lower one to BASE + 0xff0, in the page at BASE.
	const auto shifted = CaveRanges(list, BASE + 0x1000, SIZE);
	Check(shifted.size() == 2 && shifted[0].vaddr == 0xb000 &&
	          shifted[0].size == CAVE_PAGE_SIZE && shifted[1].vaddr == 0 - uint64_t {0x1000},
	      "pages aligned in absolute addresses");

	// Lines inside the module are never caves, whatever their Original.
	const auto inside = Entry({Line(0x4000, {0x90}, {0x00})});
	const std::vector<const PatchEntry*> inside_list {&inside};
	Check(CaveRanges(inside_list, BASE, SIZE).empty(), "zero Original inside the module");
	const auto nonzero = Entry({Line(0x9000, {0x90}, {0x01})});
	const std::vector<const PatchEntry*> nonzero_list {&nonzero};
	Check(CaveRanges(nonzero_list, BASE, SIZE).empty(), "non-zero Original is not a cave");
	const auto across = Entry({Line(0 - uint64_t {1}, {0x90, 0x90}, {0x00, 0x00})});
	const std::vector<const PatchEntry*> across_list {&across};
	Check(CaveRanges(across_list, BASE, SIZE).empty(), "a line wrapping into the module");

	// The caller maps the ranges zero-filled and passes them as segments.
	Bytes above(ranges[0].size, 0);
	Bytes under(ranges[1].size, 0);
	auto  segments = image.segments;
	segments.push_back({ranges[0].vaddr, above});
	segments.push_back({ranges[1].vaddr, under});
	const auto result = ApplyPatch(entry, segments, {});
	Check(result.status == ApplyStatus::Applied && result.bytes_written == 7, "cave lines applied");
	Check(under[CAVE_PAGE_SIZE - 0x10] == 0xe9 && under[CAVE_PAGE_SIZE - 0xf] == 0x01,
	      "cave below the module written");
	Check(above[0x3ffe] == 0xc3 && above[0x4001] == 0xc3, "write across two cave pages");
	Check(image.text[0] == 0xe9, "module line written");

	// Cave bytes verify against their zero Original: applying again fails and writes nothing.
	const auto pristine_above = above;
	auto       again          = Entry({Line(0xbffe, {0x90}, {0x00}), Line(0x1001, {0x90}, {0x11})});
	Check(ApplyPatch(again, segments, {}).status == ApplyStatus::Rejected &&
	          above == pristine_above && image.text[1] == 0x11,
	      "a used cave byte does not verify; nothing written");

	// Without the cave segment the line is outside the module's file data.
	Check(ApplyPatch(Entry({Line(0x9000, {0x90}, {0x00})}), image.segments, {}).status ==
	          ApplyStatus::Rejected,
	      "unmapped cave line rejected");
}

void TestZeroOriginalInModuleIsVerified() {
	Image image;
	image.text[4] = 0x00;
	// The old cheat loader skipped zero-off lines; now they are verified like any other.
	auto mismatch = Entry({Line(0x1000, {0x90}, {0x10}), Line(0x1005, {0x90}, {0x00})});
	const auto result = ApplyPatch(mismatch, image.segments, {});
	Check(result.status == ApplyStatus::Rejected &&
	          result.reason.find("original bytes differ at +0x1005") != std::string::npos,
	      "zero Original inside the module must match");
	Check(image.text[0] == 0x10, "nothing written");
	auto match = Entry({Line(0x1004, {0x90}, {0x00})});
	Check(ApplyPatch(match, image.segments, {}).status == ApplyStatus::Applied &&
	          image.text[4] == 0x90,
	      "a matching zero Original applies");
}

} // namespace

int main() {
	TestParse();
	TestParseEntryErrors();
	TestEncoding();
	TestEnabledNamesJson();
	TestMatching();
	TestSelection();
	TestApplySuccess();
	TestApplyIsAtomic();
	TestFrameCap();
	TestParseCheatJson();
	TestResolveImageBase();
	TestCodeCaves();
	TestZeroOriginalInModuleIsVerified();
	std::printf("PatchLoaderTests: all passed\n");
	return 0;
}
