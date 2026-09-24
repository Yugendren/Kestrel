#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  program.block_info.push_back({.id = 0});
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan MixedSamplerPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count, uint32_t first) {
    DescriptorSource source;
    source.dword_count = dword_count;
    source.dwords[0] = Value(first);
    for (uint32_t i = 1; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8, 0);
  const auto image1 = AddSource(8, 0);
  const auto sampler0 = AddSource(4, 0x11111111u);
  const auto sampler1 = AddSource(4, 0x22222222u);
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D,
       .conversion_format =
           Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return ExtractResourcePlan(program);
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestMixedSamplerDuplicatesTheCorrectSnapshot() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = MixedSamplerPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "mixed sampler materialization failed");
  Check(snapshot.samplers.size() == 3,
        "mixed sampler materialization appended unrelated samplers");
  Check(snapshot.samplers[2] == snapshot.samplers[1] &&
            snapshot.samplers[2] != snapshot.samplers[0],
        "point sampler variant duplicated the wrong runtime descriptor");
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// CompiledSrtPlan equivalence. SrtWalker is the oracle: every case below runs
// the queries MaterializeResources makes through SrtRefresh without and with
// the compiled plan and requires identical success, sources, flat SRT, active
// flags and uniform values.
// ---------------------------------------------------------------------------

namespace ir = Libs::Graphics::ShaderRecompiler::IR;
using ir::ResourceBlock;
using ir::Value;
using ir::ValueOpcode;

uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

// Guest memory behind the reader callbacks. The specialization ("clean")
// reader returns the same dwords XORed with CleanXor so a test can tell which
// reader produced a word.
struct FakeMemory {
  static constexpr uint64_t Base = 0x00007a0000001000ull;
  static constexpr uint32_t BaseLow = 0x00001000u;
  static constexpr uint32_t BaseHigh = 0x00007a00u;
  static constexpr uint32_t CleanXor = 0x5a5a5a5au;

  std::vector<uint32_t> words = std::vector<uint32_t>(64);
  bool fail_clean = false;

  bool Lookup(uint64_t address, std::span<uint32_t> values) const {
    for (auto &value : values) {
      if (address < Base || (address - Base) % 4u != 0u ||
          (address - Base) / 4u >= words.size()) {
        return false;
      }
      value = words[(address - Base) / 4u];
      address += 4u;
    }
    return true;
  }

  static bool Raw(void *userdata, uint64_t address,
                  std::span<uint32_t> values) {
    return static_cast<const FakeMemory *>(userdata)->Lookup(address, values);
  }

  static bool Clean(void *userdata, uint64_t address,
                    std::span<uint32_t> values) {
    const auto *memory = static_cast<const FakeMemory *>(userdata);
    if (memory->fail_clean || !memory->Lookup(address, values)) {
      return false;
    }
    for (auto &value : values) {
      value ^= CleanXor;
    }
    return true;
  }
};

ir::SrtRuntime FakeRuntime(std::span<const uint32_t> user_data,
                           FakeMemory &memory, bool clean_reader = true) {
  return {.user_data = user_data,
          .shader_base = 0x0000123456789abcull,
          .read_memory = FakeMemory::Raw,
          .userdata = &memory,
          .read_specialization_memory =
              clean_reader ? FakeMemory::Clean : nullptr};
}

// Hand-built ResourcePlan whose values live in its own value_storage. Never
// moved, so every Value stays valid.
struct TestPlan {
  ir::ResourcePlan plan;
  ir::Value srt;

  TestPlan() {
    plan.stage = Libs::Graphics::ShaderType::Compute;
    plan.srt_plan_complete = true;
    plan.resource_tracking_complete = true;
  }

  ir::Value Emit(ir::ValueOpcode opcode, std::span<const ir::Value> args,
                 uint64_t flags = 0) {
    auto &inst = plan.value_storage.emplace_back(opcode, flags);
    for (size_t index = 0; index < args.size(); index++) {
      inst.SetArg(index, args[index]);
    }
    return ir::Value(&inst);
  }

  ir::Value Emit(ir::ValueOpcode opcode,
                 std::initializer_list<ir::Value> args = {},
                 uint64_t flags = 0) {
    return Emit(opcode, std::span(args.begin(), args.size()), flags);
  }

  ir::Value Phi(std::initializer_list<ir::Value> args) {
    auto &inst = plan.value_storage.emplace_back(ir::ValueOpcode::Phi);
    for (const auto arg : args) {
      inst.AddPhiOperand(nullptr, arg);
    }
    return ir::Value(&inst);
  }

  ir::Value UserData(uint32_t reg) {
    return Emit(ir::ValueOpcode::GetUserData,
                {ir::Value(static_cast<ir::ScalarReg>(reg))});
  }

  uint64_t Memory(ir::ResourceKind kind, int32_t immediate) {
    ir::MemoryInfo memory;
    memory.kind = kind;
    memory.offset = static_cast<uint32_t>(immediate);
    memory.planning_only = true;
    plan.memory_info.push_back(memory);
    return std::bit_cast<uint64_t>(ir::MemoryFlags{
        .index = static_cast<uint32_t>(plan.memory_info.size() - 1u)});
  }

  ir::Value AddressRead(ir::Value low, ir::Value high, ir::Value offset,
                        int32_t immediate) {
    const auto handle =
        Emit(ir::ValueOpcode::GetAddressResource, {low, high});
    return Emit(ir::ValueOpcode::LoadAddressU32,
                {handle, offset, ir::Value(0u), ir::Value(true)},
                Memory(ir::ResourceKind::ScalarAddress, immediate));
  }

  ir::Value BufferRead(ir::Value low, ir::Value high, ir::Value records,
                       ir::Value word3, ir::Value offset, int32_t immediate) {
    const auto handle = Emit(ir::ValueOpcode::GetBufferResource,
                             {low, high, records, word3});
    return Emit(ir::ValueOpcode::ReadConstBuffer, {handle, offset},
                Memory(ir::ResourceKind::ScalarBuffer, immediate));
  }

  // A scalar-buffer read whose handle carries no size dwords.
  ir::Value ShortBufferRead(ir::Value low, ir::Value high, ir::Value offset,
                            int32_t immediate) {
    const auto handle =
        Emit(ir::ValueOpcode::GetAddressResource, {low, high});
    return Emit(ir::ValueOpcode::ReadConstBuffer, {handle, offset},
                Memory(ir::ResourceKind::ScalarBuffer, immediate));
  }

  ir::Value ReadConst(ir::Value slot) {
    if (srt.IsEmpty()) {
      srt = Emit(ir::ValueOpcode::GetSrtResource);
    }
    return Emit(ir::ValueOpcode::ReadConst, {srt, slot});
  }

  ir::Value ReadConst(uint32_t slot) { return ReadConst(ir::Value(slot)); }

  ir::Value Extract(ir::ValueOpcode opcode, ir::Value composite,
                    uint32_t component) {
    return Emit(opcode, {composite, ir::Value(component)});
  }

  uint32_t SrtRead(ir::Value value, uint32_t flat_offset, bool clean = false) {
    plan.srt_reads.push_back({value, flat_offset});
    plan.clean_flat_slots.push_back(clean ? 1u : 0u);
    return static_cast<uint32_t>(plan.srt_reads.size() - 1u);
  }

  uint32_t Source(std::span<const ir::Value> dwords, bool materialize = true) {
    ir::DescriptorSource source;
    for (size_t index = 0; index < dwords.size(); index++) {
      source.dwords[index] = dwords[index];
    }
    source.dword_count = static_cast<uint32_t>(dwords.size());
    plan.descriptor_sources.push_back(source);
    const auto index =
        static_cast<uint32_t>(plan.descriptor_sources.size() - 1u);
    if (materialize) {
      Materialize(index);
    }
    return index;
  }

  // Materialised like a buffer: evaluated through the ordinary walk whenever
  // its block is active.
  void Materialize(uint32_t source) {
    ir::BufferResource buffer;
    buffer.source = source;
    plan.info.buffers.push_back(buffer);
  }

  uint32_t Source(std::initializer_list<ir::Value> dwords) {
    return Source(std::span(dwords.begin(), dwords.size()));
  }

  void Uniform(std::initializer_list<ir::Value> values) {
    plan.uniform_fill.fill.words = static_cast<uint32_t>(values.size());
    std::ranges::copy(values, plan.uniform_fill.values.begin());
  }
};

struct SrtOutcome {
  bool evaluated = false;
  std::vector<ir::DescriptorValue> sources;
  std::vector<uint32_t> flat;
  std::vector<uint8_t> active;
  bool uniform_evaluated = false;
  std::array<uint32_t, 4> uniform{};
};

// The queries MaterializeResources makes, in its order, through the walker pair
// of one refresh.
SrtOutcome Walk(const ir::ResourcePlan &plan, const ir::CompiledSrtPlan *compiled,
                const ir::SrtRuntime &runtime) {
  SrtOutcome outcome;
  ir::SrtRefresh refresh(plan, runtime, compiled);
  auto &clean = refresh.Clean();
  auto &walker = refresh.Main();
  const auto active = clean.FindActiveSources();
  outcome.active.assign(active.begin(), active.end());
  if (outcome.active.empty()) {
    outcome.active.assign(plan.descriptor_sources.size(), 1u);
  }
  if (!walker.RefreshFlatBuffer(outcome.flat)) {
    return outcome;
  }
  const auto words = plan.uniform_fill.fill.words;
  outcome.uniform_evaluated = words != 0;
  for (uint32_t index = 0; index < words && outcome.uniform_evaluated; index++) {
    outcome.uniform_evaluated =
        clean.Evaluate(plan.uniform_fill.values[index], outcome.uniform[index]);
  }
  for (const auto &buffer : plan.info.buffers) {
    ir::DescriptorValue value;
    if (!active.empty() && active[buffer.source] == 0u) {
      value.dword_count = plan.descriptor_sources[buffer.source].dword_count;
    } else if (!walker.EvaluateDescriptor(buffer.source, value)) {
      return outcome;
    }
    outcome.sources.push_back(value);
  }
  outcome.evaluated = true;
  return outcome;
}

bool SameOutcome(const SrtOutcome &expected, const SrtOutcome &actual,
                 uint32_t words) {
  if (expected.evaluated != actual.evaluated) {
    return false;
  }
  if (!expected.evaluated) {
    return true;
  }
  return expected.sources == actual.sources && expected.flat == actual.flat &&
         expected.active == actual.active &&
         expected.uniform_evaluated == actual.uniform_evaluated &&
         (!expected.uniform_evaluated ||
          std::equal(expected.uniform.begin(), expected.uniform.begin() + words,
                     actual.uniform.begin()));
}

void PrintOutcome(const char *label, const SrtOutcome &outcome,
                  uint32_t words) {
  std::fprintf(stderr, "  %s: evaluated=%d\n", label, outcome.evaluated);
  if (!outcome.evaluated) {
    return;
  }
  for (size_t index = 0; index < outcome.sources.size(); index++) {
    const auto &source = outcome.sources[index];
    std::fprintf(stderr, "    source[%zu] count=%u:", index,
                 source.dword_count);
    for (const auto dword : source.dwords) {
      std::fprintf(stderr, " %08x", dword);
    }
    std::fprintf(stderr, "\n");
  }
  std::fprintf(stderr, "    flat:");
  for (const auto dword : outcome.flat) {
    std::fprintf(stderr, " %08x", dword);
  }
  std::fprintf(stderr, "\n    active:");
  for (const auto active : outcome.active) {
    std::fprintf(stderr, " %u", active);
  }
  std::fprintf(stderr, "\n    uniform_evaluated=%d:", outcome.uniform_evaluated);
  for (uint32_t index = 0; index < words; index++) {
    std::fprintf(stderr, " %08x", outcome.uniform[index]);
  }
  std::fprintf(stderr, "\n");
}

void PrintRuntime(const ir::SrtRuntime &runtime) {
  std::fprintf(stderr, "  runtime: user_data[%zu]:", runtime.user_data.size());
  for (const auto dword : runtime.user_data) {
    std::fprintf(stderr, " %08x", dword);
  }
  std::fprintf(stderr, " clean_reader=%d\n",
               runtime.read_specialization_memory != nullptr);
}

// Evaluates `plan` through the walker and through `compiled` (three refreshes
// in a row, as a renderer refreshes a program draw after draw) and aborts on
// any difference. Returns the walker's outcome.
SrtOutcome ExpectCompiledMatchesInterpreter(const ir::ResourcePlan &plan,
                                            const ir::CompiledSrtPlan &compiled,
                                            const ir::SrtRuntime &runtime,
                                            const char *what) {
  if (!compiled.Usable()) {
    std::fprintf(stderr, "ResourceMaterializationTests: compiled plan unusable: %s\n",
                 what);
    std::abort();
  }
  const auto words = plan.uniform_fill.fill.words;
  const auto expected = Walk(plan, nullptr, runtime);
  for (size_t run = 0; run < 3; run++) {
    const auto actual = Walk(plan, &compiled, runtime);
    if (!SameOutcome(expected, actual, words)) {
      std::fprintf(stderr,
                   "ResourceMaterializationTests: failed: compiled SRT "
                   "evaluation (run %zu) diverged from the walker: %s\n",
                   run, what);
      PrintRuntime(runtime);
      PrintOutcome("walker", expected, words);
      PrintOutcome("compiled", actual, words);
      std::abort();
    }
  }
  return expected;
}

SrtOutcome ExpectCompiledMatchesInterpreter(const ir::ResourcePlan &plan,
                                            const ir::SrtRuntime &runtime,
                                            const char *what) {
  const ir::CompiledSrtPlan compiled(plan);
  return ExpectCompiledMatchesInterpreter(plan, compiled, runtime, what);
}

void ExpectBothFail(const ir::ResourcePlan &plan, const ir::SrtRuntime &runtime,
                    const char *what) {
  Check(!ExpectCompiledMatchesInterpreter(plan, runtime, what).evaluated,
        what);
}

SrtOutcome ExpectBothSucceed(const ir::ResourcePlan &plan,
                             const ir::SrtRuntime &runtime, const char *what) {
  auto outcome = ExpectCompiledMatchesInterpreter(plan, runtime, what);
  Check(outcome.evaluated, what);
  return outcome;
}

struct PureOpcode {
  ir::ValueOpcode opcode;
  uint32_t arity;
};

// Every opcode SrtWalker's PureOpArity accepts.
constexpr std::array<PureOpcode, 41> PureOpcodes = {{
    {ir::ValueOpcode::ConvertF32U32, 1},
    {ir::ValueOpcode::ConvertU32F32, 1},
    {ir::ValueOpcode::FPTrunc32, 1},
    {ir::ValueOpcode::FPIsNan32, 1},
    {ir::ValueOpcode::BitwiseNot32, 1},
    {ir::ValueOpcode::BitCount32, 1},
    {ir::ValueOpcode::FindILsb32, 1},
    {ir::ValueOpcode::FindUMsb32, 1},
    {ir::ValueOpcode::LogicalNot, 1},
    {ir::ValueOpcode::CompositeConstructU64, 2},
    {ir::ValueOpcode::IAdd32, 2},
    {ir::ValueOpcode::IAdd64, 2},
    {ir::ValueOpcode::ISub32, 2},
    {ir::ValueOpcode::ISub64, 2},
    {ir::ValueOpcode::IMul32, 2},
    {ir::ValueOpcode::IMul64, 2},
    {ir::ValueOpcode::UMin32, 2},
    {ir::ValueOpcode::FPMul32, 2},
    {ir::ValueOpcode::FPOrdLessThanEqual32, 2},
    {ir::ValueOpcode::FPOrdGreaterThanEqual32, 2},
    {ir::ValueOpcode::BitwiseAnd32, 2},
    {ir::ValueOpcode::BitwiseAnd64, 2},
    {ir::ValueOpcode::BitwiseOr32, 2},
    {ir::ValueOpcode::BitwiseXor32, 2},
    {ir::ValueOpcode::ShiftLeftLogical32, 2},
    {ir::ValueOpcode::ShiftLeftLogical64, 2},
    {ir::ValueOpcode::ShiftRightLogical32, 2},
    {ir::ValueOpcode::ShiftRightLogical64, 2},
    {ir::ValueOpcode::ShiftRightArithmetic32, 2},
    {ir::ValueOpcode::ShiftRightArithmetic64, 2},
    {ir::ValueOpcode::IEqual32, 2},
    {ir::ValueOpcode::INotEqual32, 2},
    {ir::ValueOpcode::ULessThan32, 2},
    {ir::ValueOpcode::UGreaterThan32, 2},
    {ir::ValueOpcode::SGreaterThanEqual32, 2},
    {ir::ValueOpcode::LogicalAnd, 2},
    {ir::ValueOpcode::LogicalOr, 2},
    {ir::ValueOpcode::LogicalXor, 2},
    {ir::ValueOpcode::BitFieldUExtract, 3},
    {ir::ValueOpcode::BitFieldSExtract, 3},
    {ir::ValueOpcode::BitFieldInsert, 4},
}};

const std::array<uint32_t, 20> InterestingWords = {
    0u,          1u,          2u,          3u,
    8u,          16u,         31u,         32u,
    33u,         0xffffffffu, 0x80000000u, 0x7fffffffu,
    0xabcd1234u, FloatBits(1.5f), FloatBits(-1.0f), FloatBits(NAN),
    FloatBits(INFINITY), FloatBits(3.0e9f), FloatBits(5.0e9f), FloatBits(0.5f)};

void TestCompiledUserDataAndShaderBase() {
  using namespace ir;
  FakeMemory memory;
  const std::vector<uint32_t> registers = {10, 11, 12, 13, 14, 15};
  {
    TestPlan t;
    t.plan.user_data_base = 2;
    const auto base = t.Emit(ValueOpcode::GetShaderBase);
    t.Source({t.UserData(5),
              t.Extract(ValueOpcode::CompositeExtractU64, base, 0),
              t.Extract(ValueOpcode::CompositeExtractU64, base, 1), base});
    for (size_t size = 0; size <= registers.size(); size++) {
      const auto runtime =
          FakeRuntime(std::span(registers).first(size), memory);
      const auto outcome = ExpectCompiledMatchesInterpreter(
          t.plan, runtime, "user data relative to user_data_base");
      Check(outcome.evaluated == (size >= 4u),
            "user data register outside the runtime span did not fail");
      if (outcome.evaluated) {
        Check(outcome.sources[0].dwords[0] == 13u &&
                  outcome.sources[0].dwords[1] == 0x56789abcu &&
                  outcome.sources[0].dwords[2] == 0x1234u &&
                  outcome.sources[0].dwords[3] == 0x56789abcu,
              "user data / shader base evaluated to the wrong value");
      }
    }
  }
  {
    TestPlan t;
    t.plan.user_data_base = 2;
    t.Source({t.UserData(1)});
    for (size_t size = 0; size <= registers.size(); size++) {
      ExpectBothFail(t.plan,
                     FakeRuntime(std::span(registers).first(size), memory),
                     "user data register below user_data_base");
    }
  }
}

// One plan per opcode and operand width; operands are user data so the same
// compiled plan is evaluated over many operand values.
void TestCompiledPureOps() {
  using namespace ir;
  std::mt19937 rng(0x5eed0001u);
  FakeMemory memory;
  uint32_t evaluated = 0;
  uint32_t failed = 0;
  for (const auto &[opcode, arity] : PureOpcodes) {
    for (const bool wide : {false, true}) {
      TestPlan t;
      std::array<Value, 4> args;
      for (uint32_t index = 0; index < arity; index++) {
        args[index] =
            wide ? t.Emit(ValueOpcode::CompositeConstructU64,
                          {t.UserData(2 * index), t.UserData(2 * index + 1)})
                 : t.UserData(index);
      }
      const auto result = t.Emit(opcode, std::span(args).first(arity));
      t.Source({t.Extract(ValueOpcode::CompositeExtractU64, result, 0),
                t.Extract(ValueOpcode::CompositeExtractU64, result, 1),
                result});
      const CompiledSrtPlan compiled(t.plan);
      std::vector<uint32_t> registers(8);
      for (uint32_t iteration = 0; iteration < 300; iteration++) {
        for (auto &reg : registers) {
          const auto choice = rng() % 3u;
          reg = choice == 0u   ? rng()
                : choice == 1u ? rng() % 40u
                               : InterestingWords[rng() % InterestingWords.size()];
        }
        const auto size = iteration % 50u == 49u ? (wide ? 2 * arity - 1 : arity - 1)
                                                  : registers.size();
        const auto outcome = ExpectCompiledMatchesInterpreter(
            t.plan, compiled, FakeRuntime(std::span(registers).first(size), memory),
            "pure opcode");
        (outcome.evaluated ? evaluated : failed)++;
      }
    }
  }
  Check(evaluated > failed, "pure opcode sweep mostly failed");

  const auto Pure = [&memory](ValueOpcode opcode,
                              std::vector<uint32_t> operands) {
    TestPlan t;
    std::vector<Value> args;
    for (uint32_t index = 0; index < operands.size(); index++) {
      args.push_back(t.UserData(index));
    }
    t.Source({t.Emit(opcode, args)});
    const auto outcome = ExpectCompiledMatchesInterpreter(
        t.plan, FakeRuntime(operands, memory), "pure opcode edge case");
    return outcome.evaluated ? std::optional(outcome.sources[0].dwords[0])
                             : std::nullopt;
  };
  Check(!Pure(ValueOpcode::ConvertU32F32, {FloatBits(-1.0f)}),
        "negative float converted to unsigned");
  Check(!Pure(ValueOpcode::ConvertU32F32, {FloatBits(NAN)}),
        "NaN converted to unsigned");
  Check(!Pure(ValueOpcode::ConvertU32F32, {FloatBits(INFINITY)}),
        "infinity converted to unsigned");
  Check(!Pure(ValueOpcode::ConvertU32F32, {FloatBits(5.0e9f)}),
        "out-of-range float converted to unsigned");
  Check(Pure(ValueOpcode::ConvertU32F32, {FloatBits(3.75f)}) == 3u,
        "float to unsigned conversion is wrong");
  Check(!Pure(ValueOpcode::BitFieldUExtract, {0xabcd1234u, 33u, 0u}),
        "unsigned extract accepted offset 33");
  Check(!Pure(ValueOpcode::BitFieldUExtract, {0xabcd1234u, 20u, 13u}),
        "unsigned extract accepted a field past bit 31");
  Check(Pure(ValueOpcode::BitFieldUExtract, {0xabcd1234u, 4u, 8u}) == 0x23u,
        "unsigned extract is wrong");
  Check(Pure(ValueOpcode::BitFieldUExtract, {0xabcd1234u, 32u, 0u}) == 0u,
        "empty unsigned extract at offset 32 is wrong");
  Check(!Pure(ValueOpcode::BitFieldSExtract, {0x80u, 1u, 32u}),
        "signed extract accepted a field past bit 31");
  Check(Pure(ValueOpcode::BitFieldSExtract, {0x80u, 4u, 4u}) == 0xfffffff8u,
        "signed extract did not sign-extend");
  Check(!Pure(ValueOpcode::BitFieldInsert, {0u, 0u, 30u, 3u}),
        "bitfield insert accepted a field past bit 31");
  Check(Pure(ValueOpcode::BitFieldInsert, {0xffffffffu, 0u, 4u, 8u}) ==
            0xfffff00fu,
        "bitfield insert is wrong");
  Check(Pure(ValueOpcode::SelectU32, {0u, 5u, 6u}) == 6u &&
            Pure(ValueOpcode::SelectU32, {2u, 5u, 6u}) == 5u,
        "select is wrong");
  Check(Pure(ValueOpcode::ShiftRightArithmetic32, {0x80000000u, 35u}) ==
            0xf0000000u,
        "arithmetic shift does not mask its count");
}

void TestCompiledComposites() {
  using namespace ir;
  FakeMemory memory;
  TestPlan t;
  const auto a = t.UserData(0);
  const auto b = t.UserData(1);
  const auto c = t.UserData(2);
  const auto d = t.UserData(3);
  const auto u64 = t.Emit(ValueOpcode::CompositeConstructU64, {a, b});
  const auto x2 = t.Emit(ValueOpcode::CompositeConstructU32x2, {a, b});
  const auto x4 = t.Emit(ValueOpcode::CompositeConstructU32x4, {a, b, c, d});
  const auto carry = t.Emit(ValueOpcode::IAddCarry32, {a, b});
  const auto E64 = ValueOpcode::CompositeExtractU64;
  const auto E2 = ValueOpcode::CompositeExtractU32x2;
  const auto E4 = ValueOpcode::CompositeExtractU32x4;
  t.Source({t.Extract(E64, u64, 0), t.Extract(E64, u64, 1),
            t.Extract(E2, x2, 0), t.Extract(E2, x2, 1), t.Extract(E4, x4, 0),
            t.Extract(E4, x4, 1), t.Extract(E4, x4, 2), t.Extract(E4, x4, 3)});
  t.Source({t.Extract(E2, carry, 0), t.Extract(E2, carry, 1),
            t.Extract(E4, carry, 0), t.Extract(E4, carry, 3),
            t.Extract(E2, x4, 1)});
  const CompiledSrtPlan compiled(t.plan);
  const std::vector<std::vector<uint32_t>> cases = {
      {1, 2, 3, 4},
      {0xffffffffu, 1, 7, 8},
      {0x80000000u, 0x80000000u, 0, 0},
      {0xdeadbeefu, 0x12345678u, 0xcafef00du, 0x0badf00du},
      {1, 2, 3}};
  for (const auto &registers : cases) {
    const auto outcome = ExpectCompiledMatchesInterpreter(
        t.plan, compiled, FakeRuntime(registers, memory), "composite extracts");
    Check(outcome.evaluated == (registers.size() == 4u),
          "composite extract success is wrong");
    if (registers[0] == 0xffffffffu) {
      Check(outcome.sources[1].dwords[0] == 0u &&
                outcome.sources[1].dwords[1] == 1u &&
                outcome.sources[1].dwords[3] == 1u,
            "add-with-carry extract lost the carry");
    }
    if (registers[0] == 1u && outcome.evaluated) {
      const std::array<uint32_t, 8> expected = {1, 2, 1, 2, 1, 2, 3, 4};
      Check(outcome.sources[0].dwords == expected,
            "composite extract returned the wrong component");
    }
  }

  const auto Failing = [&memory](auto build, const char *what) {
    TestPlan f;
    f.Source({build(f)});
    const std::vector<uint32_t> registers = {1, 2, 3, 4, 1};
    ExpectBothFail(f.plan, FakeRuntime(registers, memory), what);
  };
  Failing(
      [](TestPlan &f) {
        return f.Extract(ValueOpcode::CompositeExtractU64,
                         f.Emit(ValueOpcode::CompositeConstructU64,
                                {f.UserData(0), f.UserData(1)}),
                         2);
      },
      "64-bit extract of component 2");
  Failing(
      [](TestPlan &f) {
        return f.Emit(ValueOpcode::CompositeExtractU64,
                      {f.Emit(ValueOpcode::CompositeConstructU64,
                              {f.UserData(0), f.UserData(1)}),
                       Value(uint64_t{0})});
      },
      "64-bit extract with a 64-bit index");
  Failing(
      [](TestPlan &f) {
        return f.Emit(ValueOpcode::CompositeExtractU32x2,
                      {f.Emit(ValueOpcode::CompositeConstructU32x2,
                              {f.UserData(0), f.UserData(1)}),
                       f.UserData(4)});
      },
      "extract with a runtime index");
  Failing(
      [](TestPlan &f) {
        return f.Extract(ValueOpcode::CompositeExtractU32x4,
                         f.Emit(ValueOpcode::CompositeConstructU32x4,
                                {f.UserData(0), f.UserData(1), f.UserData(2),
                                 f.UserData(3)}),
                         4);
      },
      "x4 extract of component 4");
  Failing(
      [](TestPlan &f) {
        return f.Extract(ValueOpcode::CompositeExtractU32x2,
                         f.Emit(ValueOpcode::CompositeConstructU32x2,
                                {f.UserData(0), f.UserData(1)}),
                         2);
      },
      "x2 extract of component 2");
  Failing(
      [](TestPlan &f) {
        return f.Extract(ValueOpcode::CompositeExtractU32x2, f.UserData(0), 0);
      },
      "extract of a non-composite");
  // Only the selected component of a construct is evaluated.
  TestPlan lazy;
  lazy.Source({lazy.Extract(ValueOpcode::CompositeExtractU32x2,
                            lazy.Emit(ValueOpcode::CompositeConstructU32x2,
                                      {lazy.UserData(0), lazy.UserData(9)}),
                            0)});
  const std::vector<uint32_t> short_registers = {1, 2};
  ExpectBothSucceed(lazy.plan, FakeRuntime(short_registers, memory),
                    "extract of a construct whose other component fails");
}

void TestCompiledBitCastAndPhi() {
  using namespace ir;
  FakeMemory memory;
  TestPlan t;
  const auto ud0 = t.UserData(0);
  const auto first = t.Emit(ValueOpcode::IAdd32, {ud0, Value(1u)});
  const auto second = t.Emit(ValueOpcode::IAdd32, {ud0, Value(1u)});
  auto &self = t.plan.value_storage.emplace_back(ValueOpcode::Phi);
  self.AddPhiOperand(nullptr, ud0);
  self.AddPhiOperand(nullptr, Value(&self));
  const auto doubled = t.Emit(
      ValueOpcode::BitCastF32U32,
      {t.Emit(ValueOpcode::FPMul32,
              {t.Emit(ValueOpcode::BitCastU32F32, {ud0}), Value::F32(2.0f)})});
  t.Source({t.Emit(ValueOpcode::BitCastU32F32, {ud0}), doubled,
            t.Phi({ud0, ud0}), t.Phi({first, second}),
            t.Phi({Value(7u), Value(7u)}), t.Phi({t.Phi({first, first}), second}),
            Value(&self)});
  const CompiledSrtPlan compiled(t.plan);
  for (const auto word : InterestingWords) {
    const std::vector<uint32_t> registers = {word, 1};
    const auto outcome = ExpectCompiledMatchesInterpreter(
        t.plan, compiled, FakeRuntime(registers, memory), "bitcast and phi");
    Check(outcome.evaluated && outcome.sources[0].dwords[0] == word &&
              outcome.sources[0].dwords[3] == word + 1u &&
              outcome.sources[0].dwords[4] == 7u &&
              outcome.sources[0].dwords[6] == word,
          "invariant phi / bitcast evaluated wrongly");
  }
  ExpectBothFail(t.plan, FakeRuntime({}, memory), "phi over missing user data");

  TestPlan varying;
  varying.Source({varying.Phi({varying.UserData(0), varying.UserData(1)})});
  const std::vector<uint32_t> registers = {3, 3};
  ExpectBothFail(varying.plan, FakeRuntime(registers, memory),
                 "non-invariant phi");
}

// Reader == nullptr: the evaluators memcpy from the computed address, so the
// base points into a buffer this test owns.
void TestCompiledRawReadsFromHostMemory() {
  using namespace ir;
  std::vector<uint32_t> host(64);
  for (uint32_t index = 0; index < host.size(); index++) {
    host[index] = 0x01010101u * index + 0x1000u;
  }
  const auto address = reinterpret_cast<uint64_t>(host.data() + 32);
  const auto low = static_cast<uint32_t>(address);
  const auto high = static_cast<uint32_t>(address >> 32u);

  TestPlan t;
  const auto l = t.UserData(0);
  const auto h = t.UserData(1);
  const auto offset = t.UserData(2);
  const auto records = t.UserData(3);
  const auto word3 = t.UserData(4);
  const auto strided_high = t.UserData(5);
  t.Source({t.AddressRead(l, h, offset, 0), t.AddressRead(l, h, offset, 8),
            t.AddressRead(l, h, offset, -16), t.AddressRead(l, h, Value(0u), -128),
            t.AddressRead(l, h, offset, 5)});
  t.Source({t.BufferRead(l, strided_high, records, word3, offset, 4)});
  const CompiledSrtPlan compiled(t.plan);

  TestPlan negative;
  negative.Source({negative.BufferRead(
      negative.UserData(0), negative.UserData(5), negative.UserData(3),
      negative.UserData(4), negative.UserData(2), -4)});
  const CompiledSrtPlan negative_compiled(negative.plan);

  uint32_t evaluated = 0;
  for (const uint32_t off : {0u, 4u, 8u, 13u, 60u}) {
    for (const uint32_t count : {0u, 8u, 16u, 64u, 1000u}) {
      for (const uint32_t stride : {0u, 4u, 16u}) {
        for (const bool null_base : {false, true}) {
          const std::vector<uint32_t> registers = {
              null_base ? 0u : low, null_base ? 0u : high, off, count, 0x1234u,
              (null_base ? 0u : high) | (stride << 16u)};
          const SrtRuntime runtime{.user_data = registers};
          const auto outcome = ExpectCompiledMatchesInterpreter(
              t.plan, compiled, runtime, "host memory reads");
          ExpectCompiledMatchesInterpreter(negative.plan, negative_compiled,
                                           runtime,
                                           "negative scalar-buffer immediate");
          const auto size = stride == 0u ? count : stride * count;
          const bool in_bounds = ((off + 4u) & ~3u) + 4u <= size;
          Check(outcome.evaluated == (null_base || in_bounds),
                "scalar-buffer bounds check is wrong");
          if (!outcome.evaluated) {
            continue;
          }
          evaluated++;
          const auto word = [&](int32_t delta) {
            return null_base ? 0u : host[32 + delta];
          };
          const auto base_word = static_cast<int32_t>((off & ~3u) / 4u);
          Check(outcome.sources[0].dwords[0] == word(base_word) &&
                    outcome.sources[0].dwords[1] == word(base_word + 2) &&
                    outcome.sources[0].dwords[2] == word(base_word - 4) &&
                    outcome.sources[0].dwords[3] == word(-32) &&
                    outcome.sources[0].dwords[4] == word(base_word + 1),
                "scalar-address read returned the wrong dword");
          Check(outcome.sources[1].dwords[0] ==
                    word(static_cast<int32_t>((off + 4u) / 4u)),
                "scalar-buffer read returned the wrong dword");
        }
      }
    }
  }
  Check(evaluated > 0, "no host memory read evaluated");

  // The same plan through a reader callback; offsets past the fake memory
  // make the reader itself fail.
  FakeMemory memory;
  for (uint32_t index = 0; index < memory.words.size(); index++) {
    memory.words[index] = 0x7700u + index;
  }
  for (const uint32_t off : {16u, 64u, 200u, 240u, 4096u}) {
    const std::vector<uint32_t> registers = {
        FakeMemory::BaseLow + 0x80u, FakeMemory::BaseHigh, off, 0x10000u, 0,
        FakeMemory::BaseHigh | (4u << 16u)};
    ExpectCompiledMatchesInterpreter(t.plan, compiled,
                                     FakeRuntime(registers, memory),
                                     "reader callback reads");
  }
}

// A scalar-buffer read through a null base never evaluates its size dwords.
void TestCompiledLazyBufferSizes() {
  using namespace ir;
  FakeMemory memory;
  for (uint32_t index = 0; index < memory.words.size(); index++) {
    memory.words[index] = 0x100u + index;
  }
  const std::vector<uint32_t> null_base = {0, 0, 64, 0, 0};
  const std::vector<uint32_t> mapped = {FakeMemory::BaseLow,
                                        FakeMemory::BaseHigh, 64, 0, 0};
  const std::vector<uint32_t> inner_mapped = {0, 0, FakeMemory::BaseLow,
                                              FakeMemory::BaseHigh, 64};
  {
    TestPlan t;
    const auto missing = t.UserData(9);
    t.Source({t.BufferRead(t.UserData(0), t.UserData(1), missing, missing,
                           Value(0u), 0)});
    const auto outcome = ExpectBothSucceed(
        t.plan, FakeRuntime(null_base, memory), "lazy size dwords, null base");
    Check(outcome.sources[0].dwords[0] == 0u, "null-base read was not zero");
    ExpectBothFail(t.plan, FakeRuntime(mapped, memory),
                   "failing size dwords, mapped base");
  }
  {
    TestPlan t;
    t.Source({t.ShortBufferRead(t.UserData(0), t.UserData(1), Value(0u), 0)});
    const auto outcome = ExpectBothSucceed(
        t.plan, FakeRuntime(null_base, memory), "two-dword handle, null base");
    Check(outcome.sources[0].dwords[0] == 0u, "null-base read was not zero");
    ExpectBothFail(t.plan, FakeRuntime(mapped, memory),
                   "two-dword handle, mapped base");
  }
  {
    // The size node first appears inside the skippable region and is used
    // again by the same root afterwards.
    TestPlan t;
    const auto records =
        t.Emit(ValueOpcode::IAdd32, {t.UserData(2), Value(0u)});
    const auto read = t.BufferRead(t.UserData(0), t.UserData(1), records,
                                   records, Value(4u), 0);
    t.Source({t.Emit(ValueOpcode::IAdd32, {read, records}), records});
    const auto skipped = ExpectBothSucceed(
        t.plan, FakeRuntime(null_base, memory), "size node reused after skip");
    Check(skipped.sources[0].dwords[0] == 64u,
          "size node reused after a skipped read is wrong");
    const auto taken = ExpectBothSucceed(t.plan, FakeRuntime(mapped, memory),
                                         "size node reused after read");
    Check(taken.sources[0].dwords[0] == 0x101u + 64u,
          "size node reused after a read is wrong");
  }
  {
    // A failing size node skipped by one root and reached by a second root.
    TestPlan t;
    const auto records =
        t.Emit(ValueOpcode::IAdd32, {t.UserData(9), Value(1u)});
    const auto read = t.BufferRead(t.UserData(0), t.UserData(1), records,
                                   Value(0u), Value(0u), 0);
    t.Source({read, records});
    ExpectBothFail(t.plan, FakeRuntime(null_base, memory),
                   "failing size node reached from a second root");
    TestPlan flat;
    const auto flat_records =
        flat.Emit(ValueOpcode::IAdd32, {flat.UserData(9), Value(1u)});
    flat.Source({flat.BufferRead(flat.UserData(0), flat.UserData(1),
                                 flat_records, Value(0u), Value(0u), 0)});
    flat.SrtRead(flat_records, 0);
    ExpectBothFail(flat.plan, FakeRuntime(null_base, memory),
                   "failing size node reached from a flat read");
  }
  {
    // Nested guards: the outer read's size dwords are themselves a read.
    TestPlan t;
    const auto inner = t.BufferRead(t.UserData(2), t.UserData(3),
                                    t.UserData(4), t.UserData(4), Value(0u), 0);
    const auto outer = t.BufferRead(t.UserData(0), t.UserData(1), inner, inner,
                                    Value(0u), 0);
    t.Source({outer});
    TestPlan both;
    const auto both_inner =
        both.BufferRead(both.UserData(2), both.UserData(3), both.UserData(4),
                        both.UserData(4), Value(0u), 0);
    both.Source({both.BufferRead(both.UserData(0), both.UserData(1), both_inner,
                                 both_inner, Value(0u), 0),
                 both_inner});
    const std::vector<std::vector<uint32_t>> cases = {
        {0, 0, 0, 0, 0},
        {0, 0, FakeMemory::BaseLow, FakeMemory::BaseHigh, 0},
        {0, 0, FakeMemory::BaseLow, FakeMemory::BaseHigh, 64},
        {FakeMemory::BaseLow, FakeMemory::BaseHigh, 0, 0, 0},
        {FakeMemory::BaseLow, FakeMemory::BaseHigh, FakeMemory::BaseLow,
         FakeMemory::BaseHigh, 0},
        {FakeMemory::BaseLow, FakeMemory::BaseHigh, FakeMemory::BaseLow,
         FakeMemory::BaseHigh, 64},
        inner_mapped};
    for (const auto &registers : cases) {
      ExpectCompiledMatchesInterpreter(t.plan, FakeRuntime(registers, memory),
                                       "nested lazy reads");
      ExpectCompiledMatchesInterpreter(both.plan,
                                       FakeRuntime(registers, memory),
                                       "nested lazy reads, inner also a root");
    }
    ExpectBothSucceed(t.plan,
                      FakeRuntime(std::vector<uint32_t>{0, 0, FakeMemory::BaseLow,
                                                        FakeMemory::BaseHigh, 0},
                                  memory),
                      "null outer base skips a failing inner read");
  }
}

void TestCompiledCleanFlatSlots() {
  using namespace ir;
  FakeMemory memory;
  memory.words[0] = 0x11111111u;
  memory.words[1] = 0x22222222u;
  const std::vector<uint32_t> registers = {FakeMemory::BaseLow,
                                           FakeMemory::BaseHigh};
  {
    TestPlan t;
    const auto base_low = t.UserData(0);
    const auto base_high = t.UserData(1);
    t.SrtRead(t.AddressRead(base_low, base_high, Value(0u), 0), 0);
    t.SrtRead(t.AddressRead(base_low, base_high, Value(0u), 4), 1, true);
    t.Source({t.ReadConst(0), t.ReadConst(1),
              t.Emit(ValueOpcode::IAdd32, {t.ReadConst(1), Value(1u)})});
    t.Source({t.Emit(ValueOpcode::ReadFirstLane, {t.ReadConst(1), Value(true)})});
    const auto outcome = ExpectBothSucceed(
        t.plan, FakeRuntime(registers, memory), "clean flat slot redirect");
    const auto clean = 0x22222222u ^ FakeMemory::CleanXor;
    Check(outcome.flat == std::vector<uint32_t>{0x11111111u, clean},
          "clean flat slot did not use the specialization reader");
    Check(outcome.sources[0].dwords[0] == 0x11111111u &&
              outcome.sources[0].dwords[1] == clean &&
              outcome.sources[0].dwords[2] == clean + 1u &&
              outcome.sources[1].dwords[0] == clean,
          "ReadConst of a clean slot did not use the specialization reader");
    ExpectBothFail(t.plan, FakeRuntime(registers, memory, false),
                   "clean slot without a specialization reader");
    memory.fail_clean = true;
    ExpectBothFail(t.plan, FakeRuntime(registers, memory),
                   "clean slot whose specialization read fails");
    memory.fail_clean = false;
  }
  {
    // Flat offsets permuted: the flat walk keys cleanliness by flat offset,
    // ReadConst by slot index.
    TestPlan t;
    const auto base_low = t.UserData(0);
    const auto base_high = t.UserData(1);
    t.SrtRead(t.AddressRead(base_low, base_high, Value(0u), 0), 1, true);
    t.SrtRead(t.AddressRead(base_low, base_high, Value(0u), 4), 0);
    t.Source({t.ReadConst(0), t.ReadConst(1)});
    ExpectBothSucceed(t.plan, FakeRuntime(registers, memory),
                      "permuted clean flat offsets");
  }
  {
    TestPlan t;
    t.SrtRead(t.UserData(0), 0);
    t.Source({t.ReadConst(5)});
    ExpectBothFail(t.plan, FakeRuntime(registers, memory),
                   "ReadConst slot out of range");
    TestPlan dynamic;
    dynamic.SrtRead(dynamic.UserData(0), 0);
    dynamic.Source({dynamic.ReadConst(dynamic.UserData(1))});
    ExpectBothFail(dynamic.plan, FakeRuntime(registers, memory),
                   "ReadConst with a runtime slot");
  }
}

void TestCompiledReadFirstLaneMask() {
  using namespace ir;
  FakeMemory memory;
  TestPlan t;
  const auto mask = t.Emit(ValueOpcode::INotEqual32, {t.UserData(2), Value(0u)});
  const auto other = t.Emit(ValueOpcode::IEqual32, {t.UserData(3), Value(0u)});
  const auto select =
      t.Emit(ValueOpcode::SelectU32, {mask, t.UserData(0), t.UserData(1)});
  const auto other_select =
      t.Emit(ValueOpcode::SelectU32, {other, t.UserData(1), t.UserData(0)});
  const auto rfl = [&t](Value value, Value lane_mask) {
    return t.Emit(ValueOpcode::ReadFirstLane, {value, lane_mask});
  };
  t.Source({rfl(select, mask), select,
            rfl(t.Emit(ValueOpcode::IAdd32, {select, Value(1u)}), mask),
            rfl(rfl(t.Emit(ValueOpcode::IAdd32, {select, other_select}), other),
                mask),
            rfl(select, other)});
  for (const uint32_t lane : {0u, 1u}) {
    for (const uint32_t other_lane : {0u, 5u}) {
      const std::vector<uint32_t> registers = {100, 200, lane, other_lane};
      const auto outcome = ExpectBothSucceed(
          t.plan, FakeRuntime(registers, memory), "read-first-lane mask");
      const auto &dwords = outcome.sources[0].dwords;
      Check(dwords[0] == 100u && dwords[1] == (lane != 0u ? 100u : 200u) &&
                dwords[2] == 101u,
            "read-first-lane did not take the active select operand");
    }
  }

  // The inactive operand fails: only the masked evaluation succeeds.
  TestPlan inactive;
  const auto inactive_mask =
      inactive.Emit(ValueOpcode::INotEqual32, {inactive.UserData(0), Value(0u)});
  const auto inactive_select = inactive.Emit(
      ValueOpcode::SelectU32,
      {inactive_mask, inactive.UserData(1), inactive.UserData(9)});
  inactive.Source({inactive.Emit(ValueOpcode::ReadFirstLane,
                                 {inactive_select, inactive_mask})});
  const std::vector<uint32_t> registers = {1, 42};
  const auto outcome = ExpectBothSucceed(
      inactive.plan, FakeRuntime(registers, memory),
      "read-first-lane skips the inactive operand");
  Check(outcome.sources[0].dwords[0] == 42u,
        "read-first-lane took the wrong operand");
  // Outside it the select still evaluates only the operand it chooses.
  inactive.Source({inactive_select});
  ExpectBothSucceed(inactive.plan, FakeRuntime(registers, memory),
                    "select outside read-first-lane skips the other operand");

  TestPlan immediate;
  const auto immediate_select = immediate.Emit(
      ValueOpcode::SelectU32,
      {Value(true), immediate.UserData(0), immediate.UserData(9)});
  immediate.Source({immediate.Emit(ValueOpcode::ReadFirstLane,
                                   {immediate_select, Value(true)})});
  ExpectBothSucceed(immediate.plan, FakeRuntime(registers, memory),
                    "read-first-lane with an immediate mask");
  immediate.SrtRead(immediate_select, 0);
  ExpectBothSucceed(immediate.plan, FakeRuntime(registers, memory),
                    "immediate-mask select outside read-first-lane");
}

void TestCompiledControlFlow() {
  using namespace ir;
  FakeMemory memory;
  memory.words[0] = FakeMemory::CleanXor;       // clean value 0
  memory.words[1] = FakeMemory::CleanXor ^ 7u;  // clean value 7
  TestPlan t;
  const auto base_low = t.UserData(0);
  const auto base_high = t.UserData(1);
  const auto condition = t.AddressRead(base_low, base_high, t.UserData(2), 0);
  const auto s0 = t.Source({t.UserData(3)});
  const auto s1 = t.Source({t.Emit(ValueOpcode::IAdd32, {t.UserData(3), Value(1u)})});
  const auto s2 = t.Source({t.UserData(9), Value(5u)});
  const auto s3 = t.Source({Value(9u)});
  const auto s4 = t.Source({t.Emit(ValueOpcode::IAdd32, {condition, Value(0u)})});
  // Condition nodes shared with a clean flat slot and a uniform value.
  t.SrtRead(condition, 0, true);
  t.Uniform({condition});
  t.plan.control_flow.push_back(
      {.condition = condition, .successors = {1, 2}, .sources = {s0}});
  t.plan.control_flow.push_back({.successors = {3}, .sources = {s1}});
  t.plan.control_flow.push_back({.successors = {}, .sources = {s2}});
  t.plan.control_flow.push_back({.successors = {0}, .sources = {s3}});
  (void)s4;
  const CompiledSrtPlan compiled(t.plan);

  const auto Run = [&](uint32_t offset, bool long_user_data, bool clean_reader) {
    std::vector<uint32_t> registers = {FakeMemory::BaseLow,
                                       FakeMemory::BaseHigh, offset, 50};
    if (long_user_data) {
      registers.resize(10, 77u);
    }
    return ExpectCompiledMatchesInterpreter(
        t.plan, compiled, FakeRuntime(registers, memory, clean_reader),
        "control flow");
  };
  // Condition true (clean value 7): blocks 0, 1, 3; block 2 inactive.
  for (const bool long_user_data : {false, true}) {
    const auto taken = Run(4, long_user_data, true);
    Check(taken.evaluated &&
              taken.active == std::vector<uint8_t>{1, 1, 0, 1, 1} &&
              taken.sources[s2].dwords[0] == 0u &&
              taken.sources[s2].dwords[1] == 0u &&
              taken.sources[s2].dword_count == 2u,
          "true condition did not deactivate the false successor");
    Check(taken.uniform_evaluated && taken.uniform[0] == 7u,
          "clean uniform value is wrong");
  }
  // Condition false (clean value 0): blocks 0, 2.
  Check(!Run(0, false, true).evaluated,
        "false condition did not activate the failing source");
  const auto not_taken = Run(0, true, true);
  Check(not_taken.evaluated &&
            not_taken.active == std::vector<uint8_t>{1, 0, 1, 0, 1},
        "false condition activated the wrong blocks");
  // Condition fails (address past memory): every successor.
  Check(!Run(4096, true, true).evaluated,
        "failing clean condition should make the clean flat read fail");
  // No specialization reader: every successor, and the clean slot fails.
  Check(!Run(4, true, false).evaluated,
        "clean slot without a specialization reader succeeded");

  // Without the clean slot, a failing or unreadable condition takes both
  // successors and evaluation succeeds.
  TestPlan open;
  const auto open_condition = open.AddressRead(open.UserData(0), open.UserData(1),
                                               open.UserData(2), 0);
  const auto o0 = open.Source({open.UserData(3)});
  const auto o1 = open.Source({open.UserData(4)});
  const auto o2 = open.Source({open.UserData(5)});
  open.Uniform({open_condition, open.UserData(3)});
  open.plan.control_flow.push_back(
      {.condition = open_condition, .successors = {1, 2}, .sources = {o0}});
  open.plan.control_flow.push_back({.successors = {0}, .sources = {o1}});
  open.plan.control_flow.push_back({.successors = {2}, .sources = {o2}});
  for (const uint32_t offset : {0u, 4u, 4096u}) {
    for (const bool clean_reader : {true, false}) {
      const std::vector<uint32_t> registers = {
          FakeMemory::BaseLow, FakeMemory::BaseHigh, offset, 3, 4, 5};
      const auto outcome = ExpectBothSucceed(
          open.plan, FakeRuntime(registers, memory, clean_reader),
          "control flow without clean slots");
      const bool both = offset == 4096u || !clean_reader;
      const std::vector<uint8_t> expected = {
          1, static_cast<uint8_t>(both || offset == 4u),
          static_cast<uint8_t>(both || offset == 0u)};
      Check(outcome.active == expected, "control-flow active set is wrong");
      Check(outcome.uniform_evaluated == !both,
            "uniform values evaluated through a failing clean read");
    }
  }

  // A 64-bit condition is truncated to its low dword.
  TestPlan wide;
  const auto wide_condition = wide.Emit(ValueOpcode::CompositeConstructU64,
                                        {Value(0u), wide.UserData(0)});
  const auto w0 = wide.Source({wide.UserData(0)});
  const auto w1 = wide.Source({wide.UserData(0)});
  wide.plan.control_flow.push_back(
      {.condition = wide_condition, .successors = {1, 0}, .sources = {w0}});
  wide.plan.control_flow.push_back({.successors = {}, .sources = {w1}});
  const std::vector<uint32_t> wide_registers = {1};
  const auto wide_outcome = ExpectBothSucceed(
      wide.plan, FakeRuntime(wide_registers, memory), "64-bit condition");
  Check(wide_outcome.active == std::vector<uint8_t>{1, 0},
        "64-bit condition was not truncated");
}

void TestCompiledUniformValues() {
  using namespace ir;
  FakeMemory memory;
  memory.words[0] = 0x33u ^ FakeMemory::CleanXor;
  for (uint32_t words = 1; words <= 4; words++) {
    TestPlan t;
    const auto clean = t.AddressRead(t.UserData(1), t.UserData(2), Value(0u), 0);
    const std::array<Value, 4> values = {
        t.UserData(0), clean, t.Emit(ValueOpcode::IAdd32, {t.UserData(0), clean}),
        t.UserData(9)};
    t.plan.uniform_fill.fill.words = words;
    std::ranges::copy(values, t.plan.uniform_fill.values.begin());
    t.Source({t.UserData(0)});
    const CompiledSrtPlan compiled(t.plan);
    const std::vector<std::vector<uint32_t>> cases = {
        {5, FakeMemory::BaseLow, FakeMemory::BaseHigh},
        {5, FakeMemory::BaseLow, FakeMemory::BaseHigh, 0, 0, 0, 0, 0, 0, 9},
        {5, 0, 0},
        {}};
    for (const auto &registers : cases) {
      for (const bool clean_reader : {true, false}) {
        const auto outcome = ExpectCompiledMatchesInterpreter(
            t.plan, compiled, FakeRuntime(registers, memory, clean_reader),
            "uniform fill values");
        if (outcome.evaluated && registers[1] != 0u && words <= 3u) {
          Check(outcome.uniform_evaluated == clean_reader || words == 1u,
                "uniform clean read ignored the specialization reader");
        }
      }
    }
    const std::vector<uint32_t> registers = {5, FakeMemory::BaseLow,
                                             FakeMemory::BaseHigh};
    const auto outcome = ExpectBothSucceed(
        t.plan, FakeRuntime(registers, memory), "uniform fill value contents");
    if (words <= 3u) {
      Check(outcome.uniform_evaluated && outcome.uniform[0] == 5u &&
                (words < 2u || outcome.uniform[1] == 0x33u) &&
                (words < 3u || outcome.uniform[2] == 0x38u),
            "uniform fill values are wrong");
    } else {
      Check(!outcome.uniform_evaluated,
            "failing uniform fill value was reported evaluated");
    }
  }
}

void TestCompiledSharedSubexpressions() {
  using namespace ir;
  FakeMemory memory;
  for (uint32_t index = 0; index < memory.words.size(); index++) {
    memory.words[index] = 0x9000u + index * 3u;
  }
  TestPlan t;
  const auto read =
      t.AddressRead(t.UserData(0), t.UserData(1), t.UserData(2), 0);
  const auto n = t.Emit(ValueOpcode::IAdd32, {read, t.UserData(3)});
  const auto m = t.Emit(ValueOpcode::IMul32, {n, n});
  const auto f =
      t.Emit(ValueOpcode::BitFieldUExtract, {n, t.UserData(4), t.UserData(5)});
  t.Source({n, m, t.Emit(ValueOpcode::IAdd32, {m, n}), n});
  t.Source({t.Emit(ValueOpcode::ShiftLeftLogical32, {n, Value(1u)}), m});
  t.SrtRead(m, 0);
  t.SrtRead(n, 1);
  t.Source({t.ReadConst(0), t.ReadConst(1), read});
  t.Source({t.Emit(ValueOpcode::IAdd32, {f, Value(1u)})});
  t.SrtRead(t.Emit(ValueOpcode::BitwiseXor32, {f, n}), 2);
  const CompiledSrtPlan compiled(t.plan);
  uint32_t evaluated = 0;
  for (const uint32_t offset : {0u, 8u, 252u, 256u}) {
    for (const uint32_t bit_offset : {0u, 4u, 30u}) {
      for (const uint32_t width : {0u, 2u, 8u}) {
        const std::vector<uint32_t> registers = {
            FakeMemory::BaseLow, FakeMemory::BaseHigh, offset, 11, bit_offset,
            width};
        const auto outcome = ExpectCompiledMatchesInterpreter(
            t.plan, compiled, FakeRuntime(registers, memory),
            "shared subexpressions");
        const bool expected = offset < 256u && bit_offset + width <= 32u;
        Check(outcome.evaluated == expected,
              "shared failing node did not fail every root");
        if (outcome.evaluated) {
          evaluated++;
          const auto value = memory.words[offset / 4u] + 11u;
          Check(outcome.sources[0].dwords[1] == value * value &&
                    outcome.sources[2].dwords[0] == value * value &&
                    outcome.flat[1] == value,
                "shared subexpression evaluated wrongly");
        }
      }
    }
  }
  Check(evaluated > 0, "no shared-subexpression case evaluated");

  // A clean-context node that fails as a control-flow condition (tolerated)
  // and again as a clean flat read (fatal).
  TestPlan condition;
  const auto failing = condition.Emit(
      ValueOpcode::BitFieldUExtract,
      {condition.UserData(0), condition.UserData(1), Value(40u)});
  const auto c0 = condition.Source({condition.UserData(0)});
  const auto c1 = condition.Source({condition.UserData(1)});
  condition.plan.control_flow.push_back(
      {.condition = failing, .successors = {1, 1}, .sources = {c0}});
  condition.plan.control_flow.push_back({.successors = {}, .sources = {c1}});
  const std::vector<uint32_t> registers = {3, 0};
  ExpectBothSucceed(condition.plan, FakeRuntime(registers, memory),
                    "failing condition alone");
  condition.SrtRead(failing, 0, true);
  ExpectBothFail(condition.plan, FakeRuntime(registers, memory),
                 "failing condition reached again by a clean flat read");
}

// Random expression DAGs over user data, constants, pure ops, raw reads,
// extracts, selects, ReadConst and ReadFirstLane, placed in descriptor
// sources, flat SRT reads, uniform values and control-flow conditions.
class RandomSrtPlan {
public:
  static constexpr uint32_t Registers = 12;

  RandomSrtPlan(std::mt19937 &rng, TestPlan &t) : m_rng(rng), m_t(t) {}

  void Build() {
    auto &plan = m_t.plan;
    plan.user_data_base = Chance(20) ? 1u : 0u;
    for (uint32_t reg = 0; reg < Registers + 2u; reg++) {
      m_user_data.push_back(m_t.UserData(reg));
    }
    m_small = {Value(0u),       Value(1u),       Value(4u),
               Value(8u),       m_user_data[6],  m_user_data[7],
               m_user_data[8]};
    m_pool = {m_user_data[6], m_user_data[7],  m_user_data[8],
              m_user_data[9], m_user_data[10], m_user_data[11],
              Value(3u),      Value(0x10u),    Value(0xffffffffu),
              Value::F32(2.5f)};
    m_masks = {Value(true),
               m_t.Emit(ValueOpcode::INotEqual32, {m_user_data[9], Value(0u)})};
    const auto nodes = 12u + Pick(24);
    for (uint32_t node = 0; node < nodes; node++) {
      m_pool.push_back(Node());
      if (Chance(12)) {
        m_t.SrtRead(Any(), 0);
      }
    }
    for (auto &read : plan.srt_reads) {
      read.flat_offset = Pick(static_cast<uint32_t>(plan.srt_reads.size()));
    }
    for (auto &clean : plan.clean_flat_slots) {
      clean = Chance(25) ? 1u : 0u;
    }
    const auto sources = 1u + Pick(4);
    for (uint32_t source = 0; source < sources; source++) {
      std::vector<Value> dwords(1u + Pick(8));
      for (auto &dword : dwords) {
        dword = Chance(70) ? Recent() : Any();
      }
      m_t.Source(dwords, !Chance(10));
    }
    if (Chance(20)) {
      m_t.Materialize(Pick(sources));
    }
    if (Chance(35)) {
      const auto blocks = 2u + Pick(3);
      for (uint32_t index = 0; index < blocks; index++) {
        ResourceBlock block;
        if (Chance(60)) {
          block.condition = Chance(50) ? m_masks[Pick(m_masks.size())] : Any();
          block.successors = {Pick(blocks), Pick(blocks)};
        } else {
          for (uint32_t edge = Pick(3); edge > 0; edge--) {
            block.successors.push_back(Pick(blocks));
          }
        }
        plan.control_flow.push_back(block);
      }
      for (uint32_t source = 0; source < sources; source++) {
        if (Chance(50)) {
          plan.control_flow[Pick(blocks)].sources.push_back(source);
        }
      }
    }
    if (Chance(40)) {
      plan.uniform_fill.fill.words = 1u + Pick(4);
      for (uint32_t index = 0; index < plan.uniform_fill.fill.words; index++) {
        plan.uniform_fill.values[index] = Any();
      }
    }
  }

  // Register values for one runtime: 0/1 the mapped base (sometimes null),
  // 2/3 a second base (usually null), 4 a record count, 5 the base high dword
  // with stride bits, 6-8 small, 9-11 anything.
  static std::vector<uint32_t> Runtime(std::mt19937 &rng) {
    const auto pick = [&rng](uint32_t count) { return rng() % count; };
    std::vector<uint32_t> registers(Registers);
    const bool null_base = pick(10) == 0u;
    registers[0] = null_base ? 0u : FakeMemory::BaseLow;
    registers[1] = null_base ? 0u : FakeMemory::BaseHigh;
    const bool second = pick(10) < 4u;
    registers[2] = second ? FakeMemory::BaseLow + 0x40u : 0u;
    registers[3] = second ? FakeMemory::BaseHigh : 0u;
    registers[4] = pick(300);
    const std::array<uint32_t, 3> strides = {0u, 4u, 16u};
    registers[5] = registers[1] | (strides[pick(3)] << 16u);
    for (uint32_t reg = 6; reg < 9; reg++) {
      registers[reg] = pick(33);
    }
    for (uint32_t reg = 9; reg < Registers; reg++) {
      registers[reg] = pick(3) == 0u ? rng()
                                     : pick(2) == 0u ? pick(64)
                                                     : InterestingWords[pick(
                                                           InterestingWords.size())];
    }
    return registers;
  }

private:
  uint32_t Pick(size_t count) {
    return std::uniform_int_distribution<uint32_t>(
        0, static_cast<uint32_t>(count - 1u))(m_rng);
  }
  bool Chance(uint32_t percent) { return Pick(100) < percent; }

  Value Recent() {
    const auto window = std::min<size_t>(m_pool.size(), 8u);
    return m_pool[m_pool.size() - 1u - Pick(window)];
  }
  Value Any() { return Chance(60) ? Recent() : m_pool[Pick(m_pool.size())]; }
  Value Small() { return m_small[Pick(m_small.size())]; }
  Value Mask() { return m_masks[Pick(m_masks.size())]; }

  Value Offset() {
    if (Chance(50)) {
      return Value(Pick(16) * 4u);
    }
    if (Chance(50)) {
      return Small();
    }
    return m_t.Emit(ValueOpcode::BitwiseAnd32, {Any(), Value(0x3cu)});
  }

  std::pair<Value, Value> Base() {
    const auto choice = Pick(10);
    if (choice < 6u) {
      return {m_user_data[0], m_user_data[1]};
    }
    if (choice < 9u || Chance(70)) {
      return {m_user_data[2], m_user_data[3]};
    }
    return {Any(), Any()};
  }

  int32_t Immediate() {
    return Chance(80) ? static_cast<int32_t>(Pick(16) * 4u)
                      : static_cast<int32_t>(Pick(40)) - 20;
  }

  Value Node() {
    switch (Pick(15)) {
    case 0:
    case 1:
    case 2: {
      const auto &[opcode, arity] = PureOpcodes[Pick(PureOpcodes.size())];
      const bool bitfield = opcode == ValueOpcode::BitFieldUExtract ||
                            opcode == ValueOpcode::BitFieldSExtract ||
                            opcode == ValueOpcode::BitFieldInsert;
      std::array<Value, 4> args;
      for (uint32_t index = 0; index < arity; index++) {
        const bool field_operand = bitfield && index + 2u >= arity;
        args[index] = field_operand ? Small()
                      : Chance(15)  ? m_t.Emit(ValueOpcode::CompositeConstructU64,
                                               {Any(), Any()})
                                    : Any();
      }
      return m_t.Emit(opcode, std::span(args).first(arity));
    }
    case 3:
      return m_t.Emit(ValueOpcode::SelectU32,
                      {Chance(50) ? Mask() : Any(), Any(), Any()});
    case 4: {
      const auto [low, high] = Base();
      return m_t.AddressRead(low, high, Offset(), Immediate());
    }
    case 5: {
      const auto [low, high] = Base();
      const auto strided = low == m_user_data[0] && Chance(50) ? m_user_data[5]
                                                               : high;
      const auto records = Chance(80)
                               ? (Chance(50) ? m_user_data[4] : Value(Pick(512)))
                               : Any();
      const auto word3 = Chance(95) ? m_user_data[10] : m_user_data[12];
      return m_t.BufferRead(low, strided, records, word3, Offset(), Immediate());
    }
    case 6: {
      const auto [low, high] = Base();
      return m_t.ShortBufferRead(low, high, Offset(), Immediate());
    }
    case 7:
      return Extract();
    case 8:
      if (!m_t.plan.srt_reads.empty()) {
        return m_t.ReadConst(Chance(95)
                                 ? Pick(m_t.plan.srt_reads.size())
                                 : 1000u);
      }
      return Any();
    case 9:
      return m_t.Emit(ValueOpcode::ReadFirstLane, {Any(), Mask()});
    case 10:
      return Chance(50) ? m_t.Emit(ValueOpcode::BitCastU32F32, {Any()})
                        : m_t.Emit(ValueOpcode::GetShaderBase);
    case 11: {
      const auto value = Any();
      if (Chance(10)) {
        return m_t.Phi({value, Any()});
      }
      return Chance(50) ? m_t.Phi({value, value})
                        : m_t.Phi({m_t.Phi({value, value}), value});
    }
    case 12: {
      const std::array<ValueOpcode, 3> compares = {ValueOpcode::INotEqual32,
                                                   ValueOpcode::IEqual32,
                                                   ValueOpcode::ULessThan32};
      const auto mask = m_t.Emit(compares[Pick(3)], {Any(), Small()});
      m_masks.push_back(mask);
      return mask;
    }
    case 13:
      return m_t.UserData(Pick(Chance(90) ? Registers : Registers + 2u));
    default:
      return m_t.Emit(ValueOpcode::IAdd32, {Any(), Small()});
    }
  }

  Value Extract() {
    const auto index = [this](uint32_t count) {
      return Chance(95) ? Pick(count) : count;
    };
    switch (Pick(4)) {
    case 0:
      return m_t.Extract(ValueOpcode::CompositeExtractU64,
                         Chance(50) ? m_t.Emit(ValueOpcode::CompositeConstructU64,
                                               {Any(), Any()})
                                    : Any(),
                         index(2));
    case 1:
      return m_t.Extract(
          ValueOpcode::CompositeExtractU32x2,
          m_t.Emit(ValueOpcode::CompositeConstructU32x2, {Any(), Any()}),
          index(2));
    case 2:
      return m_t.Extract(ValueOpcode::CompositeExtractU32x4,
                         m_t.Emit(ValueOpcode::CompositeConstructU32x4,
                                  {Any(), Any(), Any(), Any()}),
                         index(4));
    default:
      return m_t.Extract(Chance(50) ? ValueOpcode::CompositeExtractU32x2
                                    : ValueOpcode::CompositeExtractU32x4,
                         m_t.Emit(ValueOpcode::IAddCarry32, {Any(), Any()}),
                         index(2));
    }
  }

  std::mt19937 &m_rng;
  TestPlan &m_t;
  std::vector<Value> m_user_data;
  std::vector<Value> m_small;
  std::vector<Value> m_pool;
  std::vector<Value> m_masks;
};

void TestCompiledRandomDifferential() {
  using namespace ir;
  std::mt19937 rng(0x5a17c0deu);
  FakeMemory memory;
  uint32_t evaluated = 0;
  uint32_t failed = 0;
  uint32_t uniform_evaluated = 0;
  uint32_t with_inactive = 0;
  constexpr uint32_t Plans = 400;
  constexpr uint32_t RuntimesPerPlan = 8;
  for (uint32_t plan_index = 0; plan_index < Plans; plan_index++) {
    TestPlan t;
    RandomSrtPlan(rng, t).Build();
    const CompiledSrtPlan compiled(t.plan);
    for (uint32_t run = 0; run < RuntimesPerPlan; run++) {
      for (auto &word : memory.words) {
        word = rng() % 4u == 0u ? 0u : rng() % 3u == 0u ? rng() : rng() % 64u;
      }
      memory.fail_clean = rng() % 10u == 0u;
      const auto registers = RandomSrtPlan::Runtime(rng);
      const uint32_t size =
          rng() % 8u == 0u ? static_cast<uint32_t>(rng() % RandomSrtPlan::Registers)
                           : RandomSrtPlan::Registers;
      const auto first = std::min(t.plan.user_data_base, size);
      const auto runtime =
          FakeRuntime(std::span(registers).subspan(first, size - first), memory,
                      rng() % 10u != 0u);
      char what[96];
      std::snprintf(what, sizeof(what), "random plan %u runtime %u", plan_index,
                    run);
      const auto outcome =
          ExpectCompiledMatchesInterpreter(t.plan, compiled, runtime, what);
      if (outcome.evaluated) {
        evaluated++;
        uniform_evaluated += outcome.uniform_evaluated ? 1u : 0u;
        with_inactive += std::ranges::count(outcome.active, uint8_t{0}) != 0 ? 1u : 0u;
      } else {
        failed++;
      }
    }
  }
  std::printf("ResourceMaterializationTests: random SRT differential: %u "
              "runtimes, %u evaluated (%u with uniform values, %u with "
              "inactive sources), %u failed in both paths\n",
              evaluated + failed, evaluated, uniform_evaluated, with_inactive,
              failed);
  std::fflush(stdout);
  Check(evaluated + failed >= 2000u, "too few random runtimes");
  Check(evaluated * 5u >= evaluated + failed,
        "random differential rarely evaluated successfully");
  Check(failed * 20u >= evaluated + failed,
        "random differential rarely exercised failures");
}

void ExpectSameMaterialization(const ir::ResourcePlan &plan,
                               const ir::CompiledSrtPlan &compiled,
                               const ir::SrtRuntime &runtime,
                               const char *what) {
  using namespace ir;
  Check(compiled.Usable(), what);
  ResourceSnapshot expected_snapshot;
  ResourceSpecialization expected_specialization;
  const bool expected = MaterializeResources(
      plan, runtime, expected_snapshot, expected_specialization);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const bool actual = MaterializeResources(plan, &compiled, runtime, snapshot,
                                           specialization);
  Check(expected == actual, what);
  Check(snapshot.buffers == expected_snapshot.buffers &&
            snapshot.images == expected_snapshot.images &&
            snapshot.samplers == expected_snapshot.samplers &&
            snapshot.flattened_srt == expected_snapshot.flattened_srt &&
            snapshot.user_data == expected_snapshot.user_data &&
            snapshot.uniform_fill == expected_snapshot.uniform_fill &&
            specialization == expected_specialization,
        what);
}

void TestCompiledMaterializeMatchesInterpreter() {
  using namespace ir;
  const uint32_t dword = 0x12345678u;
  auto srt = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  auto user_data = UserDataBufferPlan();
  auto mixed = MixedSamplerPlan();
  auto uniform = UserDataBufferPlan();
  auto &fill_value = uniform.value_storage.emplace_back(ValueOpcode::GetUserData);
  fill_value.SetArg(0, Value(static_cast<ScalarReg>(1)));
  uniform.uniform_fill.fill.kind = UniformFillKind::Buffer;
  uniform.uniform_fill.fill.words = 2;
  uniform.uniform_fill.values = {Value(&fill_value), Value(7u)};
  const std::array<const ResourcePlan *, 4> plans = {&srt, &user_data, &mixed,
                                                     &uniform};
  std::vector<std::unique_ptr<CompiledSrtPlan>> compiled;
  for (const auto *plan : plans) {
    compiled.push_back(std::make_unique<CompiledSrtPlan>(*plan));
  }
  const std::vector<std::vector<uint32_t>> registers = {
      {}, {0x1000u, 7u}, {0x2000u, 8u, 3u}, {0u, 7u, 0u, 0u}};
  for (uint32_t round = 0; round < 2; round++) {
    for (const auto &values : registers) {
      for (size_t index = 0; index < plans.size(); index++) {
        const SrtRuntime runtime{.user_data = values};
        ExpectSameMaterialization(*plans[index], *compiled[index], runtime,
                                  "compiled materialization diverged");
      }
    }
  }
  // The compiled path's uniform fill: equal words yield a fill value.
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const std::vector<uint32_t> equal = {0x1000u, 7u};
  Check(MaterializeResources(uniform, compiled[3].get(), {.user_data = equal},
                             snapshot, specialization) &&
            snapshot.uniform_fill.words == 2u &&
            snapshot.uniform_fill.value == 7u,
        "compiled materialization lost the uniform fill");
}

// Upstream af3a3f4b: a select evaluates its predicate in the strict walk and
// only the operand it chooses.
void TestCompiledSelectIsLazyWithStrictPredicate() {
  using namespace ir;
  FakeMemory memory;
  TestPlan t;
  const auto predicate =
      t.AddressRead(t.UserData(0), t.UserData(1), Value(0u), 0);
  // Choosing through the ordinary reader would take the missing register 9.
  const auto select =
      t.Emit(ValueOpcode::SelectU32, {predicate, t.UserData(2), t.UserData(9)});
  t.Source({select, t.Emit(ValueOpcode::IAdd32, {select, Value(1u)})});
  const std::vector<uint32_t> registers = {FakeMemory::BaseLow,
                                           FakeMemory::BaseHigh, 40};
  memory.words[0] = 0u; // ordinary reader: 0, strict reader: nonzero
  const auto outcome = ExpectBothSucceed(
      t.plan, FakeRuntime(registers, memory), "lazy select, strict predicate");
  Check(outcome.sources[0].dwords[0] == 40u &&
            outcome.sources[0].dwords[1] == 41u,
        "select did not take the operand its strict predicate chose");
  memory.words[0] = FakeMemory::CleanXor; // strict reader: 0
  ExpectBothFail(t.plan, FakeRuntime(registers, memory),
                 "select choosing a failing operand");
  memory.words[0] = 0u;
  ExpectBothFail(t.plan, FakeRuntime(registers, memory, false),
                 "select predicate without a strict reader");
}

// Upstream: below a ReadFirstLane a clean flat slot is read by a strict walk
// under the same mask, so a masked select inside the slot takes its active
// operand there too.
void TestCompiledReadFirstLaneRedirectsToMaskedStrictWalk() {
  using namespace ir;
  FakeMemory memory;
  memory.words[0] = 0x44u ^ FakeMemory::CleanXor;
  TestPlan t;
  const auto mask = t.Emit(ValueOpcode::INotEqual32, {t.UserData(2), Value(0u)});
  const auto read = t.AddressRead(t.UserData(0), t.UserData(1), Value(0u), 0);
  t.SrtRead(t.Emit(ValueOpcode::SelectU32, {mask, read, t.UserData(9)}), 0,
            true);
  const auto source = t.Source(
      {t.Emit(ValueOpcode::ReadFirstLane, {t.ReadConst(0), mask})});
  const CompiledSrtPlan compiled(t.plan);
  Check(compiled.Usable(), "masked redirect plan did not compile");
  const std::vector<uint32_t> registers = {FakeMemory::BaseLow,
                                           FakeMemory::BaseHigh, 0u};
  const auto runtime = FakeRuntime(registers, memory);
  for (const auto *plan :
       {static_cast<const CompiledSrtPlan *>(nullptr), &compiled}) {
    SrtRefresh refresh(t.plan, runtime, plan);
    DescriptorValue value;
    Check(refresh.Main().EvaluateDescriptor(source, value) &&
              value.dwords[0] == 0x44u,
          "clean slot below ReadFirstLane was not read by the masked strict "
          "walk");
  }
}

// Anything the compiler does not mirror keeps the program on SrtWalker.
void TestCompiledUnsupportedPlansFallBack() {
  using namespace ir;
  FakeMemory memory;
  const std::vector<uint32_t> registers = {1, 2};
  const auto runtime = FakeRuntime(registers, memory);
  {
    TestPlan t;
    t.Source({t.Emit(ValueOpcode::CompositeConstructU32x2,
                     {t.UserData(0), t.UserData(1)})});
    const CompiledSrtPlan compiled(t.plan);
    Check(!compiled.Usable(),
          "a plan with an opcode the compiler does not know compiled");
    Check(SameOutcome(Walk(t.plan, nullptr, runtime),
                      Walk(t.plan, &compiled, runtime), 0),
          "an unusable plan did not fall back to the walker");
  }
  {
    // A cycle without a phi: SrtWalker fails whichever node it reaches again,
    // which depends on the order it meets them in.
    TestPlan t;
    auto &a = t.plan.value_storage.emplace_back(ValueOpcode::IAdd32);
    const auto b = t.Emit(ValueOpcode::IAdd32, {Value(&a), Value(1u)});
    a.SetArg(0, b);
    a.SetArg(1, t.UserData(0));
    t.Source({Value(&a), b});
    const CompiledSrtPlan compiled(t.plan);
    Check(!compiled.Usable(), "a cyclic plan compiled");
    const auto expected = Walk(t.plan, nullptr, runtime);
    Check(!expected.evaluated, "a cyclic plan evaluated");
    Check(SameOutcome(expected, Walk(t.plan, &compiled, runtime), 0),
          "a cyclic plan did not fall back to the walker");
  }
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestMappedSrtUsesDirectReaderByDefault();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestUnbasedFlatCacheHitMaterializes();
  TestFailedMaterializationRejectsStage();
  TestMixedSamplerDuplicatesTheCorrectSnapshot();
  TestCompiledUserDataAndShaderBase();
  TestCompiledPureOps();
  TestCompiledComposites();
  TestCompiledBitCastAndPhi();
  TestCompiledRawReadsFromHostMemory();
  TestCompiledLazyBufferSizes();
  TestCompiledCleanFlatSlots();
  TestCompiledReadFirstLaneMask();
  TestCompiledSelectIsLazyWithStrictPredicate();
  TestCompiledReadFirstLaneRedirectsToMaskedStrictWalk();
  TestCompiledControlFlow();
  TestCompiledUniformValues();
  TestCompiledSharedSubexpressions();
  TestCompiledUnsupportedPlansFallBack();
  TestCompiledRandomDifferential();
  TestCompiledMaterializeMatchesInterpreter();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
