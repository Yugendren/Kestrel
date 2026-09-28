// Wave reductions recovered from guest lane-exchange chains (ir/passes/WaveReduction.h): IR built
// the way translation emits the PS5 compiler's DPP / V_PERMLANEX16 scans, checked for the exact
// lane sets the pass proves and for the chains it must leave alone.

#include "graphics/shader/recompiler/ir/LaneAddressing.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/WaveReduction.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;

// The shared lane addressing (also what the SPIR-V emitter decodes).
static_assert(DecodeDppControl(0x111, false).kind == DppControlKind::RowShiftRight);
static_assert(DecodeDppControl(0x153, false).kind == DppControlKind::RowShare &&
              DecodeDppControl(0x153, false).operand == 3);
static_assert(DecodeDppControl(0x142, false).kind == DppControlKind::Unsupported);
static_assert(DppSourceLaneOf(DecodeDppControl(0x153, false), 37).lane == 35);
static_assert(!DppSourceLaneOf(DecodeDppControl(0x112, false), 17).valid);
static_assert(DppSourceLaneOf(DecodeDppControl(0x112, false), 18).lane == 16);
static_assert(!DppSourceLaneOf(DecodeDppControl(0x101, false), 15).valid);
static_assert(DppSourceLaneOf(DecodeDppControl(0x121, false), 1).lane == 0 &&
              DppSourceLaneOf(DecodeDppControl(0x121, false), 0).lane == 15);
static_assert(DppSourceLaneOf(DecodeDppControl(0x140, false), 33).lane == 46);
static_assert(DppSourceLaneOf(DecodeDppControl(0x141, false), 9).lane == 14);
static_assert(DppSourceLaneOf(DecodeDppControl(0xb1, false), 6).lane == 7);
static_assert(DppSourceLaneOf(DecodeDppControl(0x165, false), 16).lane == 21);
// DPP8: three selector bits per lane of each group of eight (0x249249: every lane reads lane 1).
static_assert(DecodeDppControl(0x249249, true).kind == DppControlKind::Dpp8);
static_assert(DppSourceLaneOf(DecodeDppControl(0x249249, true), 46).lane == 41);
static_assert(Permlane16SourceLane(0, ~0u, ~0u, true) == 31);
static_assert(Permlane16SourceLane(40, ~0u, ~0u, true) == 63);
static_assert(Permlane16SourceLane(9, 0x76543210u, 0xfedcba98u, false) == 9);
static_assert(DppRowBankEnabled(0x2, 0xf, 16) && !DppRowBankEnabled(0x2, 0xf, 15));
static_assert(!DppRowBankEnabled(0xf, 0xe, 3) && DppRowBankEnabled(0xf, 0xe, 4));

int g_failures = 0;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    g_failures++;
    std::cerr << "  FAIL: " << message << '\n';
  }
}

std::string Hex(uint64_t value) {
  char text[32];
  std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
  return text;
}

template <typename T> uint64_t Bits(const T &flags) {
  uint64_t bits = 0;
  std::memcpy(&bits, &flags, sizeof(flags));
  return bits;
}

Value U32(uint32_t value) { return Value(value); }

constexpr uint64_t Lanes(uint32_t first, uint32_t last) {
  uint64_t mask = 0;
  for (uint32_t lane = first; lane <= last; lane++) {
    mask |= uint64_t{1} << lane;
  }
  return mask;
}

// One pixel-shader block, written the way translation writes the guest code:
//   S_ORN2_SAVEEXEC_B64 vcc, exec      -> exec = !x | x    (all_lanes)
//   V_CNDMASK_B32 v, 0, data, saved    -> Select(x, data, 0) under that exec (the masked leaf)
//   V_OP v, v.dpp(ctrl), v             -> DppUpdate(op(DppMove(v), v), v)
//   V_PERMLANEX16_B32 t, v, -1, -1     -> Select(exec, Permlane16(v, ~0, ~0), t)
struct Chain {
  Program program;
  Block *block = nullptr;
  Value pixels;    // x: the lanes that have a pixel (the saved, pre-WQM exec)
  Value all_lanes; // !x | x
  Value leaf;      // Select(x, data, 0)

  explicit Chain(uint32_t wave_size, uint32_t identity = 0) {
    program.stage = Libs::Graphics::ShaderType::Pixel;
    program.wave_size = wave_size;
    program.block_storage.push_back(std::make_unique<Block>());
    block = program.block_storage.back().get();
    program.blocks.push_back(block);
    program.block_info.push_back({.id = 0});

    const auto exec_bits = Undef();
    pixels = Value(Emit(ValueOpcode::INotEqual32, {exec_bits, U32(0)}));
    all_lanes = Value(Emit(ValueOpcode::LogicalOr, {Value(Emit(ValueOpcode::LogicalNot, {pixels})), pixels}));
    const auto data = Undef();
    leaf = Value(Emit(ValueOpcode::SelectU32, {pixels, data, U32(identity)}));
  }

  Inst *Emit(ValueOpcode op, std::initializer_list<Value> args, uint64_t flags = 0) {
    return &block->AppendNewInst(op, args, flags);
  }

  Value Undef() { return Value(Emit(ValueOpcode::UndefU32, {})); }

  // A full-wave VGPR write: the exec-merge select WriteOperand emits.
  Value Write(Value value, Value old) { return Write(value, old, all_lanes); }
  Value Write(Value value, Value old, Value exec) {
    return Value(Emit(ValueOpcode::SelectU32, {exec, value, old}));
  }

  // The first write of the scan register: V_CNDMASK_B32 v, 0, data, saved.
  Value Start() { return Write(leaf, Undef()); }

  // V_OP v, v.dpp(control), v.
  Value DppStep(ValueOpcode op, Value v, uint16_t control, bool bound_ctrl) {
    return DppStep(op, v, control, bound_ctrl, all_lanes);
  }
  Value DppStep(ValueOpcode op, Value v, uint16_t control, bool bound_ctrl, Value exec,
                uint8_t row_mask = 0xf, uint8_t bank_mask = 0xf) {
    const DppMoveFlags flags{.control = control,
                             .row_mask = row_mask,
                             .bank_mask = bank_mask,
                             .fetch_inactive = false,
                             .bound_control = bound_ctrl};
    const auto moved = Value(Emit(ValueOpcode::DppMoveU32, {v, exec}, Bits(flags)));
    const auto combined = Value(Emit(op, {moved, v}));
    return Value(Emit(ValueOpcode::DppUpdateU32, {combined, v, exec}, Bits(flags)));
  }

  // V_PERMLANEX16_B32 t.opsel(hi=1), v, -1, -1 followed by V_OP v, v, t.
  Value CrossRows(ValueOpcode op, Value v) {
    const PermlaneFlags flags{.x16 = true, .fetch_inactive = false, .bound_control = true};
    const auto other = Write(
        Value(Emit(ValueOpcode::Permlane16U32, {v, U32(~0u), U32(~0u), all_lanes}, Bits(flags))),
        Undef());
    return Write(Value(Emit(op, {v, other})), v);
  }

  Inst *ReadLane(Value v, uint32_t lane) { return ReadLane(v, U32(lane)); }
  Inst *ReadLane(Value v, Value lane) { return Emit(ValueOpcode::ReadLane, {v, lane}); }

  // The compiler's WaveActive* scan: row_shr 1, 2, 4, 8, then the other row's lane 15.
  Value RowScan(ValueOpcode op, bool bound_ctrl) {
    auto v = Start();
    for (const uint16_t control : {0x111, 0x112, 0x114, 0x118}) {
      v = DppStep(op, v, control, bound_ctrl);
    }
    return CrossRows(op, v);
  }
};

// Checks that `read` was replaced by WaveReduceU32(leaf, lanes) with `op`.
void CheckRewritten(const Chain &chain, const Inst *read, WaveReduceOp op, uint64_t lanes,
                    const std::string &name) {
  if (read->GetOpcode() != ValueOpcode::Identity) {
    Check(false, name + ": read was not rewritten");
    return;
  }
  const auto *reduce = read->Arg(0).Resolve().TryInstruction();
  if (reduce == nullptr || reduce->GetOpcode() != ValueOpcode::WaveReduceU32) {
    Check(false, name + ": read does not forward to a WaveReduceU32");
    return;
  }
  Check(reduce->Arg(0).Resolve() == chain.leaf, name + ": reduced value is not the leaf");
  Check(reduce->Flags<WaveReduceFlags>().op == op, name + ": combining op");
  const auto low = reduce->Arg(1).Resolve();
  const auto high = reduce->Arg(2).Resolve();
  Check(low.IsImmediate() && high.IsImmediate(), name + ": lane mask is not immediate");
  const auto mask = uint64_t{low.U32()} | (uint64_t{high.U32()} << 32u);
  Check(mask == lanes, name + ": lanes " + Hex(mask) + ", expected " + Hex(lanes));
}

void CheckUntouched(const Chain &chain, const Inst *read, const std::string &name) {
  Check(read->GetOpcode() == ValueOpcode::ReadLane, name + ": read was rewritten");
  Check(!FindWaveReduction(*read, chain.program.wave_size).has_value(),
        name + ": analysis proved a reduction");
}

// (1) The exact wave64 sequence of PS 0078b10cd (and 2b185b1e, 1ebb3213): WaveActiveBitOr.
void TestWave64BitOr() {
  Chain chain(64);
  const auto total = chain.RowScan(ValueOpcode::BitwiseOr32, false);
  auto *low = chain.ReadLane(total, 31);
  auto *high = chain.ReadLane(total, 63);
  const auto stats = RecoverWaveReductions(chain.program, 64);
  Check(stats.rewritten_reads == 2, "wave64 or: rewritten " + std::to_string(stats.rewritten_reads));
  CheckRewritten(chain, low, WaveReduceOp::BitwiseOr, Lanes(0, 31), "wave64 or lane 31");
  CheckRewritten(chain, high, WaveReduceOp::BitwiseOr, Lanes(32, 63), "wave64 or lane 63");
}

// (2) The wave32 variant: one readlane of lane 31.
void TestWave32BitOr() {
  Chain chain(32);
  const auto total = chain.RowScan(ValueOpcode::BitwiseOr32, false);
  auto *read = chain.ReadLane(total, 31);
  const auto stats = RecoverWaveReductions(chain.program, 32);
  Check(stats.rewritten_reads == 1, "wave32 or: rewritten " + std::to_string(stats.rewritten_reads));
  CheckRewritten(chain, read, WaveReduceOp::BitwiseOr, Lanes(0, 31), "wave32 or lane 31");
}

// (3) WaveActiveSum has the same shape; every step adds disjoint lanes.
void TestWave64Sum() {
  Chain chain(64);
  const auto total = chain.RowScan(ValueOpcode::IAdd32, false);
  auto *low = chain.ReadLane(total, 31);
  auto *high = chain.ReadLane(total, 63);
  const auto stats = RecoverWaveReductions(chain.program, 64);
  Check(stats.rewritten_reads == 2, "wave64 sum: rewritten " + std::to_string(stats.rewritten_reads));
  CheckRewritten(chain, low, WaveReduceOp::IAdd, Lanes(0, 31), "wave64 sum lane 31");
  CheckRewritten(chain, high, WaveReduceOp::IAdd, Lanes(32, 63), "wave64 sum lane 63");
}

// (4) row_mirror twice: lane 15 combines {0, 15} with its mirror {15, 0}. Idempotent Or is still
// the reduction over {0, 15}; IAdd would count both lanes twice.
void TestDoubleCount() {
  {
    Chain chain(64);
    auto v = chain.Start();
    v = chain.DppStep(ValueOpcode::IAdd32, v, 0x140, false);
    v = chain.DppStep(ValueOpcode::IAdd32, v, 0x140, false);
    auto *read = chain.ReadLane(v, 15);
    const auto stats = RecoverWaveReductions(chain.program, 64);
    Check(stats.rewritten_reads == 0, "double-counted sum: rewritten");
    CheckUntouched(chain, read, "double-counted sum");
  }
  {
    Chain chain(64);
    auto v = chain.Start();
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x140, false);
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x140, false);
    auto *read = chain.ReadLane(v, 15);
    const auto stats = RecoverWaveReductions(chain.program, 64);
    Check(stats.rewritten_reads == 1, "mirrored or: not rewritten");
    CheckRewritten(chain, read, WaveReduceOp::BitwiseOr, (uint64_t{1} << 0) | (uint64_t{1} << 15),
                   "mirrored or lane 15");
  }
}

// (5) A lane exchange under an exec that is not provably all lanes.
void TestExecNotAllLanes() {
  {
    // The pixel exec itself (no S_ORN2_SAVEEXEC): lanes without a pixel would not relay.
    Chain chain(64);
    auto v = chain.Start();
    for (const uint16_t control : {0x111, 0x112, 0x114, 0x118}) {
      v = chain.DppStep(ValueOpcode::BitwiseOr32, v, control, false, chain.pixels);
    }
    auto *read = chain.ReadLane(v, 15);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 0, "pixel exec: rewritten");
    CheckUntouched(chain, read, "pixel exec");
  }
  {
    // !x | y with y != x.
    Chain chain(64);
    const auto other = Value(chain.Emit(ValueOpcode::INotEqual32, {chain.Undef(), U32(0)}));
    const auto almost = Value(chain.Emit(
        ValueOpcode::LogicalOr, {Value(chain.Emit(ValueOpcode::LogicalNot, {chain.pixels})), other}));
    auto v = chain.Start();
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x111, false, almost);
    auto *read = chain.ReadLane(v, 15);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 0, "!x | y exec: rewritten");
    CheckUntouched(chain, read, "!x | y exec");
  }
  {
    // The immediate true (S_OR_SAVEEXEC ..., -1 / S_MOV_B64 exec, -1) is all lanes; the operand
    // order of x | !x does not matter.
    Chain chain(64);
    const auto swapped = Value(chain.Emit(
        ValueOpcode::LogicalOr, {chain.pixels, Value(chain.Emit(ValueOpcode::LogicalNot, {chain.pixels}))}));
    auto v = chain.Start();
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x111, false, Value(true));
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x112, false, swapped);
    auto *read = chain.ReadLane(v, 15);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 1, "true exec: not rewritten");
    CheckRewritten(chain, read, WaveReduceOp::BitwiseOr, Lanes(12, 15), "true exec lane 15");
  }
}

// (6) Reads the pass must not touch: a runtime lane, a lone lane, a lane past the wave.
void TestReadsLeftAlone() {
  Chain chain(64);
  const auto total = chain.RowScan(ValueOpcode::BitwiseOr32, false);
  auto *runtime = chain.ReadLane(total, chain.Undef());
  auto *lone = chain.ReadLane(chain.leaf, 5);
  Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 0, "left alone: rewritten");
  CheckUntouched(chain, runtime, "runtime lane");
  CheckUntouched(chain, lone, "lone lane");

  Chain wave32(32);
  auto *past = wave32.ReadLane(wave32.RowScan(ValueOpcode::BitwiseOr32, false), 40);
  Check(RecoverWaveReductions(wave32.program, 32).rewritten_reads == 0, "lane past wave32: rewritten");
  CheckUntouched(wave32, past, "lane past wave32");

  // A chain longer than any scan the compiler emits.
  Chain long_chain(64);
  auto v = long_chain.Start();
  for (int step = 0; step < 30; step++) {
    v = long_chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x111, false);
  }
  auto *deep = long_chain.ReadLane(v, 15);
  Check(RecoverWaveReductions(long_chain.program, 64).rewritten_reads == 0, "long chain: rewritten");
  CheckUntouched(long_chain, deep, "long chain");
}

// (7) bound_ctrl = 1: lanes past a row edge receive a literal zero. That is Or's identity, so it
// adds nothing; for UMin it is a value (zero wins), so a lane it reaches is not a UMin reduction.
// row_shr:1 then row_shl:1 puts the zero into lane 15.
void TestBoundControl() {
  const auto shift_both_ways = [](Chain &chain, ValueOpcode op) {
    auto v = chain.Start();
    v = chain.DppStep(op, v, 0x111, true);
    v = chain.DppStep(op, v, 0x101, true);
    return chain.ReadLane(v, 15);
  };
  {
    Chain chain(64);
    auto *read = shift_both_ways(chain, ValueOpcode::BitwiseOr32);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 1, "bc or: not rewritten");
    CheckRewritten(chain, read, WaveReduceOp::BitwiseOr, Lanes(14, 15), "bc or lane 15");
  }
  {
    Chain chain(64, 0xffffffffu);
    auto *read = shift_both_ways(chain, ValueOpcode::UMin32);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 0, "bc umin: rewritten");
    CheckUntouched(chain, read, "bc umin");
  }
  // In the row scan with bound_ctrl = 1 the zeros land only in lanes the last lane of each row
  // never reads again, so lane 63 is still an exact UMin reduction, while lane 7 (past the row
  // edge of the last row_shr:8) has taken a zero. With bound_ctrl = 0 those lanes keep their old
  // value and lane 7 is the reduction over its prefix and the other row.
  for (const bool bound_ctrl : {false, true}) {
    Chain chain(64, 0xffffffffu);
    const auto total = chain.RowScan(ValueOpcode::UMin32, bound_ctrl);
    auto *read = chain.ReadLane(total, 63);
    auto *prefix = chain.ReadLane(total, 7);
    const auto name = std::string(bound_ctrl ? "bc " : "") + "umin scan";
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == (bound_ctrl ? 1u : 2u),
          name + ": rewritten count");
    CheckRewritten(chain, read, WaveReduceOp::UMin, Lanes(32, 63), name + " lane 63");
    if (bound_ctrl) {
      CheckUntouched(chain, prefix, name + " lane 7");
    } else {
      CheckRewritten(chain, prefix, WaveReduceOp::UMin, Lanes(0, 7) | Lanes(16, 31),
                     name + " lane 7");
    }
  }
  {
    // A row mask that leaves row 3 unwritten: lane 63 never receives the rest of its row.
    Chain chain(64);
    auto v = chain.Start();
    for (const uint16_t control : {0x111, 0x112, 0x114, 0x118}) {
      v = chain.DppStep(ValueOpcode::BitwiseOr32, v, control, false, chain.all_lanes, 0x7);
    }
    auto *read = chain.ReadLane(v, 63);
    auto *kept = chain.ReadLane(v, 47);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 1, "row mask: rewritten count");
    CheckUntouched(chain, read, "row mask lane 63");
    CheckRewritten(chain, kept, WaveReduceOp::BitwiseOr, Lanes(32, 47), "row mask lane 47");
  }
}

// (8) Butterflies: every lane ends with the whole row, then the other row joins.
void TestButterflies() {
  {
    Chain chain(64);
    auto v = chain.Start();
    for (const uint16_t control : {0x161, 0x162, 0x164, 0x168}) {
      v = chain.DppStep(ValueOpcode::BitwiseOr32, v, control, false);
    }
    v = chain.CrossRows(ValueOpcode::BitwiseOr32, v);
    auto *low = chain.ReadLane(v, 31);
    auto *high = chain.ReadLane(v, 40);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 2, "xmask or: rewritten count");
    CheckRewritten(chain, low, WaveReduceOp::BitwiseOr, Lanes(0, 31), "xmask or lane 31");
    CheckRewritten(chain, high, WaveReduceOp::BitwiseOr, Lanes(32, 63), "xmask or lane 40");
  }
  {
    // quad_perm [1,0,3,2] and [2,3,0,1], then xmask 4 and 8: disjoint at every step.
    Chain chain(32);
    auto v = chain.Start();
    for (const uint16_t control : {0xb1, 0x4e, 0x164, 0x168}) {
      v = chain.DppStep(ValueOpcode::IAdd32, v, control, false);
    }
    v = chain.CrossRows(ValueOpcode::IAdd32, v);
    auto *read = chain.ReadLane(v, 31);
    Check(RecoverWaveReductions(chain.program, 32).rewritten_reads == 1, "quad sum: rewritten count");
    CheckRewritten(chain, read, WaveReduceOp::IAdd, Lanes(0, 31), "quad sum lane 31");
  }
  {
    // One op per chain: an Or step inside a sum is not a reduction.
    Chain chain(64);
    auto v = chain.Start();
    v = chain.DppStep(ValueOpcode::IAdd32, v, 0x161, false);
    v = chain.DppStep(ValueOpcode::BitwiseOr32, v, 0x162, false);
    auto *read = chain.ReadLane(v, 3);
    Check(RecoverWaveReductions(chain.program, 64).rewritten_reads == 0, "mixed ops: rewritten");
    CheckUntouched(chain, read, "mixed ops");
  }
}

// The wave32 form (PS 7f2b4ea6, 828d0301): S_ORN2_SAVEEXEC_B32 vcc_lo, exec_lo writes EXEC_LO as
// ~x | x and reaches the U1 exec through Translator::ThreadBit; there is no permlane, and the
// halves are read at lanes 15 and 31.
Value ThreadBitExec(Chain &chain, Value word) {
  const auto lane = Value(chain.Emit(ValueOpcode::LaneId, {}));
  const auto index = Value(chain.Emit(ValueOpcode::BitwiseAnd32, {lane, U32(31)}));
  const auto shifted = Value(chain.Emit(ValueOpcode::ShiftRightLogical32, {word, index}));
  const auto bit = Value(chain.Emit(ValueOpcode::BitwiseAnd32, {shifted, U32(1)}));
  return Value(chain.Emit(ValueOpcode::INotEqual32, {bit, U32(0)}));
}

Value RowScanOnly(Chain &chain, ValueOpcode op) {
  auto v = chain.Start();
  for (const uint16_t control : {0x111, 0x112, 0x114, 0x118}) {
    v = chain.DppStep(op, v, control, false);
  }
  return v;
}

void TestWave32ExecWord() {
  Chain chain(32);
  const auto exec_lo = chain.Undef();
  const auto not_exec = Value(chain.Emit(ValueOpcode::BitwiseNot32, {exec_lo}));
  chain.all_lanes =
      ThreadBitExec(chain, Value(chain.Emit(ValueOpcode::BitwiseOr32, {not_exec, exec_lo})));
  const auto total = RowScanOnly(chain, ValueOpcode::BitwiseOr32);
  auto *low = chain.ReadLane(total, 15);
  auto *high = chain.ReadLane(total, 31);
  const auto stats = RecoverWaveReductions(chain.program, 32);
  Check(stats.rewritten_reads == 2,
        "wave32 exec word: rewritten " + std::to_string(stats.rewritten_reads));
  CheckRewritten(chain, low, WaveReduceOp::BitwiseOr, Lanes(0, 15), "wave32 exec word lane 15");
  CheckRewritten(chain, high, WaveReduceOp::BitwiseOr, Lanes(16, 31), "wave32 exec word lane 31");
}

// ~x | y with two different words is not all ones: the scan is left alone.
void TestWave32ExecWordMismatch() {
  Chain chain(32);
  const auto not_exec = Value(chain.Emit(ValueOpcode::BitwiseNot32, {chain.Undef()}));
  chain.all_lanes =
      ThreadBitExec(chain, Value(chain.Emit(ValueOpcode::BitwiseOr32, {not_exec, chain.Undef()})));
  const auto total = RowScanOnly(chain, ValueOpcode::BitwiseOr32);
  auto *read = chain.ReadLane(total, 31);
  const auto stats = RecoverWaveReductions(chain.program, 32);
  Check(stats.rewritten_reads == 0,
        "wave32 exec word mismatch: rewritten " + std::to_string(stats.rewritten_reads));
  CheckUntouched(chain, read, "wave32 exec word mismatch");
}

} // namespace

int main() {
  TestWave64BitOr();
  TestWave32BitOr();
  TestWave64Sum();
  TestDoubleCount();
  TestExecNotAllLanes();
  TestReadsLeftAlone();
  TestBoundControl();
  TestButterflies();
  TestWave32ExecWord();
  TestWave32ExecWordMismatch();
  if (g_failures != 0) {
    std::cerr << "wave reduction tests: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "wave reduction tests passed\n";
  return 0;
}

// The full emulator supplies these assertion hooks through common; this focused target links
// only fmt.
namespace Common {
int DbgExitHandler(const char *, int, std::string_view text) {
  std::cerr << "fatal: " << text << '\n';
  std::abort();
}

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view text) {
  std::cerr << "fatal: " << text << '\n';
  std::abort();
}

int DbgExitIfHandler(const char *expression, const char *file, int line) {
  std::cerr << "typed IR assertion: " << expression << " at " << file << ':' << line << '\n';
  std::abort();
}

int DbgNotImplementedHandler(const char *expression, const char *file, int line) {
  std::cerr << "typed IR not implemented: " << expression << " at " << file << ':' << line
            << '\n';
  std::abort();
}

void DbgExit(int) { std::abort(); }
} // namespace Common

// Keep this focused standalone target self-contained by amalgamating its small typed-IR
// implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
