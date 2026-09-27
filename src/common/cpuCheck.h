#ifndef KYTY_COMMON_CPU_CHECK_H_
#define KYTY_COMMON_CPU_CHECK_H_

#include <cstddef>
#include <cstdint>

// Startup check that the host CPU implements the x86-64 psABI level the emulator was compiled
// for (KYTY_MARCH, cmake/KytyOptimization.cmake). Without it a CPU below that level dies with a
// silent SIGILL the first time the compiler used, say, an AVX2 or BMI2 instruction. cpuCheck.cpp
// itself is compiled at the baseline ISA and runs before any other static initialiser, so the
// check executes before any code that could need the missing instructions.
namespace Common::CpuCheck {

// The CPUID/XGETBV words the level checks read. A leaf the CPU does not implement reads as zero,
// which reports its features as missing.
struct X86CpuidWords {
	uint32_t leaf1_ecx = 0; // CPUID.01H:ECX
	uint32_t leaf7_ebx = 0; // CPUID.(EAX=07H,ECX=0):EBX
	uint32_t ext1_ecx  = 0; // CPUID.80000001H:ECX
	uint64_t xcr0      = 0; // XGETBV(0); zero when CPUID.01H:ECX.OSXSAVE is clear
};

// The features x86-64-v2..v4 require, in reporting order. OsAvxState/OsAvx512State are the XCR0
// bits by which the OS declares it saves the YMM/ZMM registers: without them AVX/AVX-512
// instructions fault even on a CPU that implements them.
enum class X86Feature : uint32_t {
	Cx16,
	LahfSahf,
	Popcnt,
	Sse3,
	Sse41,
	Sse42,
	Ssse3,
	Avx,
	Avx2,
	Bmi1,
	Bmi2,
	F16c,
	Fma,
	Lzcnt,
	Movbe,
	Osxsave,
	OsAvxState,
	Avx512f,
	Avx512bw,
	Avx512cd,
	Avx512dq,
	Avx512vl,
	OsAvx512State,
	Count,
};

// Bit (1 << X86Feature) set for every feature x86-64-v`level` needs that `words` lack. Levels
// below 2 need nothing beyond the baseline and return 0.
[[nodiscard]] uint32_t MissingX86Features(uint32_t level, const X86CpuidWords& words) noexcept;

[[nodiscard]] const char* X86FeatureName(X86Feature feature) noexcept;

// Writes the one-line diagnostic for `missing` (a MissingX86Features result), always
// NUL-terminated. Returns the length written.
size_t FormatMissingX86Features(uint32_t level, uint32_t missing, char* buffer,
                                size_t size) noexcept;

// The host's words; all zero on a non-x86-64 build.
[[nodiscard]] X86CpuidWords ReadHostX86CpuidWords() noexcept;

} // namespace Common::CpuCheck

#endif /* KYTY_COMMON_CPU_CHECK_H_ */
