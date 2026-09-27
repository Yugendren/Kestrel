#include "common/cpuCheck.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define KYTY_CPU_CHECK_X86_MSVC 1
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define KYTY_CPU_CHECK_X86_GNU 1
#endif

// This file is compiled at the baseline ISA whatever KYTY_MARCH says (kyty_add_cpu_check in
// cmake/KytyOptimization.cmake) and only calls the C library before the check has passed, so
// nothing here can itself need an instruction the check is about to report missing.
// KYTY_CPU_CHECK_LEVEL (0 = no check) is the psABI level KYTY_MARCH selected.

namespace Common::CpuCheck {

namespace {

enum class Word : uint8_t { Leaf1Ecx, Leaf7Ebx, Ext1Ecx, Xcr0 };

struct Requirement {
	X86Feature feature;
	uint32_t   level; // lowest psABI level that needs it
	Word       word;
	uint64_t   mask;  // every bit must be set
};

// x86-64 psABI micro-architecture levels. v2: CMPXCHG16B, LAHF/SAHF, POPCNT, SSE3, SSE4.1,
// SSE4.2, SSSE3. v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE (and the OS YMM
// state). v4: AVX512F/BW/CD/DQ/VL (and the OS opmask/ZMM state).
constexpr std::array<Requirement, static_cast<size_t>(X86Feature::Count)> Requirements {{
    {X86Feature::Cx16, 2, Word::Leaf1Ecx, 1u << 13u},
    {X86Feature::LahfSahf, 2, Word::Ext1Ecx, 1u << 0u},
    {X86Feature::Popcnt, 2, Word::Leaf1Ecx, 1u << 23u},
    {X86Feature::Sse3, 2, Word::Leaf1Ecx, 1u << 0u},
    {X86Feature::Sse41, 2, Word::Leaf1Ecx, 1u << 19u},
    {X86Feature::Sse42, 2, Word::Leaf1Ecx, 1u << 20u},
    {X86Feature::Ssse3, 2, Word::Leaf1Ecx, 1u << 9u},
    {X86Feature::Avx, 3, Word::Leaf1Ecx, 1u << 28u},
    {X86Feature::Avx2, 3, Word::Leaf7Ebx, 1u << 5u},
    {X86Feature::Bmi1, 3, Word::Leaf7Ebx, 1u << 3u},
    {X86Feature::Bmi2, 3, Word::Leaf7Ebx, 1u << 8u},
    {X86Feature::F16c, 3, Word::Leaf1Ecx, 1u << 29u},
    {X86Feature::Fma, 3, Word::Leaf1Ecx, 1u << 12u},
    {X86Feature::Lzcnt, 3, Word::Ext1Ecx, 1u << 5u},
    {X86Feature::Movbe, 3, Word::Leaf1Ecx, 1u << 22u},
    {X86Feature::Osxsave, 3, Word::Leaf1Ecx, 1u << 27u},
    {X86Feature::OsAvxState, 3, Word::Xcr0, 0x6u},   // XMM | YMM
    {X86Feature::Avx512f, 4, Word::Leaf7Ebx, 1u << 16u},
    {X86Feature::Avx512bw, 4, Word::Leaf7Ebx, 1u << 30u},
    {X86Feature::Avx512cd, 4, Word::Leaf7Ebx, 1u << 28u},
    {X86Feature::Avx512dq, 4, Word::Leaf7Ebx, 1u << 17u},
    {X86Feature::Avx512vl, 4, Word::Leaf7Ebx, 1u << 31u},
    {X86Feature::OsAvx512State, 4, Word::Xcr0, 0xe0u}, // opmask | ZMM_Hi256 | Hi16_ZMM
}};

constexpr std::array<const char*, static_cast<size_t>(X86Feature::Count)> FeatureNames {
    "CX16", "LAHF-SAHF", "POPCNT", "SSE3",  "SSE4.1",  "SSE4.2",  "SSSE3",   "AVX",
    "AVX2", "BMI1",      "BMI2",   "F16C",  "FMA",     "LZCNT",   "MOVBE",   "OSXSAVE",
    "OS-AVX-state", "AVX512F", "AVX512BW", "AVX512CD", "AVX512DQ", "AVX512VL",
    "OS-AVX512-state",
};

constexpr bool RequirementsInOrder() {
	for (size_t i = 0; i < Requirements.size(); i++) {
		if (static_cast<size_t>(Requirements[i].feature) != i) {
			return false;
		}
	}
	return true;
}
static_assert(RequirementsInOrder(), "Requirements must be indexed by X86Feature");
static_assert(static_cast<size_t>(X86Feature::Count) <= 32, "missing-feature mask is 32 bits");

uint64_t Select(const X86CpuidWords& words, Word word) {
	switch (word) {
		case Word::Leaf1Ecx: return words.leaf1_ecx;
		case Word::Leaf7Ebx: return words.leaf7_ebx;
		case Word::Ext1Ecx: return words.ext1_ecx;
		case Word::Xcr0: return words.xcr0;
	}
	return 0;
}

const char* LevelSummary(uint32_t level) {
	switch (level) {
		case 2: return "SSE4.2/POPCNT";
		case 3: return "AVX2/BMI2/FMA";
		default: return "AVX-512";
	}
}

#if defined(KYTY_CPU_CHECK_X86_MSVC)
void Cpuid(uint32_t leaf, uint32_t subleaf, std::array<uint32_t, 4>& regs) {
	std::array<int, 4> out {};
	__cpuidex(out.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
	for (size_t i = 0; i < regs.size(); i++) {
		regs[i] = static_cast<uint32_t>(out[i]);
	}
}

// Only reached once CPUID reports OSXSAVE. The target attribute lets clang-cl expand the
// intrinsic in this one function without raising the whole file above the baseline ISA.
#if defined(__clang__)
__attribute__((target("xsave")))
#endif
uint64_t ReadXcr0() {
	return _xgetbv(0);
}
#elif defined(KYTY_CPU_CHECK_X86_GNU)
void Cpuid(uint32_t leaf, uint32_t subleaf, std::array<uint32_t, 4>& regs) {
	__cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
}

// Only reached once CPUID reports OSXSAVE.
uint64_t ReadXcr0() {
	uint32_t low  = 0;
	uint32_t high = 0;
	__asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0u));
	return (static_cast<uint64_t>(high) << 32u) | low;
}
#endif

#if defined(KYTY_CPU_CHECK_LEVEL) && KYTY_CPU_CHECK_LEVEL > 0
void CheckHostCpuOrExit() {
	const uint32_t missing = MissingX86Features(KYTY_CPU_CHECK_LEVEL, ReadHostX86CpuidWords());
	if (missing == 0) {
		return;
	}
	std::array<char, 512> message {};
	FormatMissingX86Features(KYTY_CPU_CHECK_LEVEL, missing, message.data(), message.size());
	std::fputs(message.data(), stderr);
	std::fflush(stderr);
	// No static destructors: nothing else has been constructed, and nothing may run that could
	// execute the instructions this CPU lacks.
	std::_Exit(1);
}
#endif

} // namespace

uint32_t MissingX86Features(uint32_t level, const X86CpuidWords& words) noexcept {
	uint32_t missing = 0;
	for (const auto& requirement: Requirements) {
		if (requirement.level <= level &&
		    (Select(words, requirement.word) & requirement.mask) != requirement.mask) {
			missing |= 1u << static_cast<uint32_t>(requirement.feature);
		}
	}
	return missing;
}

const char* X86FeatureName(X86Feature feature) noexcept {
	const auto index = static_cast<size_t>(feature);
	return index < FeatureNames.size() ? FeatureNames[index] : "?";
}

size_t FormatMissingX86Features(uint32_t level, uint32_t missing, char* buffer,
                                size_t size) noexcept {
	if (buffer == nullptr || size == 0) {
		return 0;
	}
	size_t     length = 0;
	const auto append = [&](const char* text) {
		const int written = std::snprintf(buffer + length, size - length, "%s", text);
		if (written > 0) {
			length = std::min(length + static_cast<size_t>(written), size - 1);
		}
	};
	char head[128] {};
	std::snprintf(head, sizeof(head), "kyty_emulator was built for x86-64-v%u (%s); this CPU lacks:",
	              level, LevelSummary(level));
	append(head);
	for (uint32_t i = 0; i < static_cast<uint32_t>(X86Feature::Count); i++) {
		if ((missing & (1u << i)) != 0) {
			append(" ");
			append(X86FeatureName(static_cast<X86Feature>(i)));
		}
	}
	append("\nRebuild with -DKYTY_MARCH= (empty) for a build that runs on this CPU.\n");
	return length;
}

X86CpuidWords ReadHostX86CpuidWords() noexcept {
	X86CpuidWords words {};
#if defined(KYTY_CPU_CHECK_X86_MSVC) || defined(KYTY_CPU_CHECK_X86_GNU)
	std::array<uint32_t, 4> regs {};
	Cpuid(0, 0, regs);
	const uint32_t max_leaf = regs[0];
	if (max_leaf >= 1u) {
		Cpuid(1, 0, regs);
		words.leaf1_ecx = regs[2];
	}
	if (max_leaf >= 7u) {
		Cpuid(7, 0, regs);
		words.leaf7_ebx = regs[1];
	}
	Cpuid(0x80000000u, 0, regs);
	if (regs[0] >= 0x80000001u) {
		Cpuid(0x80000001u, 0, regs);
		words.ext1_ecx = regs[2];
	}
	constexpr uint32_t OsxsaveBit = 1u << 27u;
	if ((words.leaf1_ecx & OsxsaveBit) != 0) {
		words.xcr0 = ReadXcr0();
	}
#endif
	return words;
}

} // namespace Common::CpuCheck

// Registration: ahead of every ordinary static initialiser of the executable, whose code is
// compiled for KYTY_MARCH.
#if defined(KYTY_CPU_CHECK_LEVEL) && KYTY_CPU_CHECK_LEVEL > 0
#if defined(_MSC_VER)
// .CRT$XCB sorts before .CRT$XCU, where the compiler puts C++ dynamic initialisers; the C
// runtime (stdio included) is already initialised by then (.CRT$XI*).
#pragma section(".CRT$XCB", read)
extern "C" {
#if defined(__clang__)
__attribute__((used))
#endif
__declspec(allocate(".CRT$XCB")) void (*kyty_cpu_check_initializer)() =
    Common::CpuCheck::CheckHostCpuOrExit;
}
#elif defined(__GNUC__) || defined(__clang__)
// Priority 101 is the first one not reserved for the implementation.
__attribute__((constructor(101))) static void KytyCpuCheckInitializer() {
	Common::CpuCheck::CheckHostCpuOrExit();
}
#endif
#endif
