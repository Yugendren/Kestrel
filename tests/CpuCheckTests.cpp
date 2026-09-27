#include "common/cpuCheck.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace {

using Common::CpuCheck::FormatMissingX86Features;
using Common::CpuCheck::MissingX86Features;
using Common::CpuCheck::X86CpuidWords;
using Common::CpuCheck::X86Feature;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "CpuCheckTests: failed: %s\n", message);
    std::abort();
  }
}

uint32_t Mask(std::initializer_list<X86Feature> features) {
  uint32_t mask = 0;
  for (const auto feature : features) {
    mask |= 1u << static_cast<uint32_t>(feature);
  }
  return mask;
}

// Nehalem (qemu -cpu Nehalem): x86-64-v2 complete, nothing of v3.
constexpr X86CpuidWords Nehalem{
    .leaf1_ecx = (1u << 0u) | (1u << 9u) | (1u << 13u) | (1u << 19u) | (1u << 20u) | (1u << 23u),
    .leaf7_ebx = 0,
    .ext1_ecx = 1u << 0u,
    .xcr0 = 0,
};

// Haswell/Zen 2 with an OS that saves YMM state: x86-64-v3 complete, no AVX-512.
constexpr X86CpuidWords Haswell{
    .leaf1_ecx = Nehalem.leaf1_ecx | (1u << 12u) | (1u << 22u) | (1u << 27u) | (1u << 28u) |
                 (1u << 29u),
    .leaf7_ebx = (1u << 3u) | (1u << 5u) | (1u << 8u),
    .ext1_ecx = Nehalem.ext1_ecx | (1u << 5u),
    .xcr0 = 0x7,
};

void TestLevels() {
  Check(MissingX86Features(0, {}) == 0 && MissingX86Features(1, {}) == 0,
        "baseline levels need nothing");
  Check(MissingX86Features(2, Nehalem) == 0, "Nehalem is x86-64-v2");
  Check(MissingX86Features(3, Nehalem) ==
            Mask({X86Feature::Avx, X86Feature::Avx2, X86Feature::Bmi1, X86Feature::Bmi2,
                  X86Feature::F16c, X86Feature::Fma, X86Feature::Lzcnt, X86Feature::Movbe,
                  X86Feature::Osxsave, X86Feature::OsAvxState}),
        "Nehalem lacks exactly the v3 additions");
  Check(MissingX86Features(3, Haswell) == 0, "Haswell is x86-64-v3");
  Check(MissingX86Features(4, Haswell) ==
            Mask({X86Feature::Avx512f, X86Feature::Avx512bw, X86Feature::Avx512cd,
                  X86Feature::Avx512dq, X86Feature::Avx512vl, X86Feature::OsAvx512State}),
        "Haswell lacks exactly the v4 additions");

  // A v3 CPU whose OS does not enable YMM state still faults on AVX.
  auto no_ymm = Haswell;
  no_ymm.xcr0 = 0x3;
  Check(MissingX86Features(3, no_ymm) == Mask({X86Feature::OsAvxState}),
        "missing YMM state is reported");
  // A v3 check also covers v2 (a Haswell without POPCNT does not exist, but the check is total).
  auto no_popcnt = Haswell;
  no_popcnt.leaf1_ecx &= ~(1u << 23u);
  Check(MissingX86Features(3, no_popcnt) == Mask({X86Feature::Popcnt}),
        "v3 includes the v2 features");
}

void TestMessage() {
  char buffer[512];
  const auto missing = MissingX86Features(3, Nehalem);
  const auto length = FormatMissingX86Features(3, missing, buffer, sizeof(buffer));
  Check(length == std::strlen(buffer), "length matches the text");
  const std::string text(buffer);
  Check(text.starts_with("kyty_emulator was built for x86-64-v3 (AVX2/BMI2/FMA); this CPU "
                         "lacks: AVX AVX2 BMI1 BMI2 F16C FMA LZCNT MOVBE OSXSAVE OS-AVX-state\n"),
        "v3 message text");

  char tiny[16];
  const auto cut = FormatMissingX86Features(3, missing, tiny, sizeof(tiny));
  Check(cut == sizeof(tiny) - 1 && tiny[cut] == '\0', "a short buffer is truncated and terminated");
}

void TestHost() {
  // The machine running the tests runs the emulator, so it must pass the level it was built for.
  [[maybe_unused]] const auto words = Common::CpuCheck::ReadHostX86CpuidWords();
#if defined(KYTY_CPU_CHECK_LEVEL) && KYTY_CPU_CHECK_LEVEL > 0
  Check(MissingX86Features(KYTY_CPU_CHECK_LEVEL, words) == 0, "host passes its build level");
#endif
#if defined(__x86_64__) || defined(_M_X64)
  Check(words.leaf1_ecx != 0, "host CPUID was read");
#endif
}

} // namespace

int main() {
  TestLevels();
  TestMessage();
  TestHost();
  std::puts("cpu check tests passed");
  return 0;
}
