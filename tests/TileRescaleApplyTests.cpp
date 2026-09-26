#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/TileRescale.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <span>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

// Emission checks for the tile-rescale rewrite (ir/passes/TileRescale.h): ApplyTileRescale on a
// translated tile program, the control-word binding, and the SPIR-V invocation-remap prologue.
// The pass is applied with a hand-made plan, so these hold whatever AnalyzeTileRescale accepts.

namespace {

using namespace Libs::Graphics;
namespace IR = ShaderRecompiler::IR;
using u32 = uint32_t;

constexpr const char *Name = "TileRescaleEmission";

[[noreturn]] void Fail(const char *stage, const std::string &message) {
  std::fprintf(stderr, "TileRescaleApplyTests: %s failed at %s: %s\n", Name, stage,
               message.c_str());
  std::abort();
}

void Require(const char *stage, bool value, const std::string &message) {
  if (!value) {
    Fail(stage, message);
  }
}

void EnsureConfigInitialized() {
  static Common::Subsystems subsystems;
  Common::InitializeThreads();
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions options;
  options.printf_direction = Config::LogDirection::Silent;
  Config::Load(options);
  subsystems.Initialize<Log::Lifecycle>();
  ShaderInit();
}

// RDNA2 encodings, as in ShaderRecompilerComputeTests.cpp.
constexpr u32 InlineU32(u32 value) { return 128u + value; }

constexpr u32 Vgpr(u32 reg) { return 256u + reg; }

constexpr u32 EncodeSop2(u32 opcode, u32 dst, u32 src0, u32 src1) {
  return 0x80000000u | ((opcode & 0x7fu) << 23u) | ((dst & 0x7fu) << 16u) |
         ((src1 & 0xffu) << 8u) | (src0 & 0xffu);
}

constexpr u32 EncodeVop1(u32 opcode, u32 dst, u32 src0) {
  return (0x3fu << 25u) | ((dst & 0xffu) << 17u) | ((opcode & 0xffu) << 9u) |
         (src0 & 0x1ffu);
}

constexpr u32 EncodeVop2(u32 opcode, u32 dst, u32 src0, u32 src1) {
  return ((opcode & 0x3fu) << 25u) | ((dst & 0xffu) << 17u) |
         ((src1 & 0xffu) << 9u) | (src0 & 0x1ffu);
}

constexpr u32 EncodeSmem0(u32 opcode, u32 dst, u32 sbase) {
  return (0x3du << 26u) | ((opcode & 0xffu) << 18u) | ((dst & 0x7fu) << 6u) |
         (sbase & 0x3fu);
}

constexpr u32 EncodeSmem1(u32 offset, u32 soffset) {
  return (offset & 0x1fffffu) | ((soffset & 0x7fu) << 25u);
}

constexpr u32 EncodeMimg0(u32 opcode, u32 dmask) {
  return (0x3cu << 26u) | ((opcode >> 7u) & 0x1u) | (1u << 3u) |
         ((dmask & 0xfu) << 8u) | ((opcode & 0x7fu) << 18u);
}

constexpr u32 EncodeMimg1(u32 vdata, u32 vaddr) {
  return ((vdata & 0xffu) << 8u) | (vaddr & 0xffu);
}

bool ReadZeroMemory(void *, uint64_t, std::span<uint32_t> values) {
  std::fill(values.begin(), values.end(), 0u);
  return true;
}

size_t Count(const IR::Program &program, IR::ValueOpcode opcode) {
  size_t result = 0;
  for (const auto *block : program.blocks) {
    result += std::ranges::count_if(
        *block, [&](const IR::Inst &inst) { return inst.GetOpcode() == opcode; });
  }
  return result;
}

// A 16x16 wave32 tile program: the tile origin comes from a constant buffer indexed by the
// workgroup, and the program loads and stores its own pixel (origin << 4) + LocalInvocationID,
// storing float(x) in one component.
class TileProgram {
public:
  TileProgram() {
    m_code.push_back(EncodeSop2(0x1e, 13, 12, InlineU32(3))); // s13 = group.x * 8
    m_code.push_back(EncodeSmem0(0x09, 16, 4)); // s[16:17] = origin[group.x]
    m_code.push_back(EncodeSmem1(0, 13));
    m_code.push_back(EncodeSop2(0x1e, 18, 16, InlineU32(4)));
    m_code.push_back(EncodeSop2(0x1e, 19, 17, InlineU32(4)));
    m_code.push_back(EncodeVop2(0x25, 2, 18, 0)); // v2 = (origin.x << 4) + x
    m_code.push_back(EncodeVop2(0x25, 3, 19, 1)); // v3 = (origin.y << 4) + y
    m_code.push_back(EncodeMimg0(0x00, 0xf));     // image_load v[4:7], v[2:3]
    m_code.push_back(EncodeMimg1(4, 2));
    m_code.push_back(EncodeVop1(0x06, 4, Vgpr(2))); // v_cvt_f32_u32 v4, v2
    m_code.push_back(EncodeMimg0(0x08, 0xf));       // image_store v[4:7], v[2:3]
    m_code.push_back(EncodeMimg1(4, 2));
    m_code.push_back(0xbf810000u); // s_endpgm

    // s[0:7]: an RGBA32F 2D image; s[8:11]: the tile-origin constant buffer.
    m_user_data[0] = 0x1000u;
    m_user_data[1] =
        static_cast<u32>(Prospero::BufferFormat::k32_32_32_32Float) << 20u;
    m_user_data[3] = (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u) |
                     DstSel(4, 5, 6, 7);
    m_user_data[8] = 0x2000u;
    m_user_data[10] = 256u;
    m_user_data[11] = DstSel(4, 5, 6, 7) | (1u << 24u);

    m_compute.lds_size_dwords = 1024;
    m_compute.threads_num[0] = 16;
    m_compute.threads_num[1] = 16;
    m_compute.threads_num[2] = 1;
    m_compute.wave_size = 32;
    m_compute.host_subgroup_size = 32;
    m_compute.thread_ids_num = 2;
    m_compute.group_id[0] = true;
    m_compute.workgroup_register = 12;

    m_options.stage = ShaderType::Compute;
    m_options.wave_size = 32;
    m_options.input_info.compute = &m_compute;
    m_options.user_data = m_user_data;
  }

  [[nodiscard]] const ShaderRecompiler::CompileOptions &Options() const {
    return m_options;
  }

  // Translates and specialises one copy, as CompileProgram does up to the tile-rescale point.
  [[nodiscard]] IR::Program Specialise() const {
    auto translated = ShaderRecompiler::TranslateProgram(m_code, m_options);
    const auto plan = IR::ExtractResourcePlan(translated.program);
    IR::ResourceSnapshot resources;
    IR::ResourceSpecialization specialization;
    const IR::SrtRuntime runtime{
        .user_data = m_options.user_data,
        .shader_base = reinterpret_cast<uint64_t>(m_code.data()),
        .read_memory = ReadZeroMemory,
    };
    Require("resource materialization",
            IR::MaterializeResources(plan, runtime, resources, specialization),
            "translated resources could not be materialized");
    auto program = std::move(translated.program);
    IR::ApplyResourceSpecialization(program, specialization);
    IR::DiscardResourcePlanningInputs(program);
    IR::RemoveIdentities(program.blocks);
    IR::EliminateDeadCode(program.blocks);
    return program;
  }

  // Applies an accepted k = 2 plan whose own-coordinate conversion is the float(x).
  static void Rescale(IR::Program &program) {
    IR::TileRescalePlan plan;
    plan.accepted = true;
    plan.scale_log2 = 1;
    plan.cross_lane = IR::CrossLaneClass::LaneSerialized;
    for (const auto *block : program.blocks) {
      for (const auto &inst : *block) {
        if (inst.GetOpcode() == IR::ValueOpcode::ConvertF32U32) {
          plan.own_coordinate_conversions.push_back(&inst);
        }
      }
    }
    Require("plan", plan.own_coordinate_conversions.size() == 1,
            "expected one own-coordinate conversion:\n" + IR::ProgramToString(program));
    IR::ApplyTileRescale(program, plan);
    IR::EliminateDeadCode(program.blocks);
  }

private:
  std::vector<u32> m_code;
  std::array<u32, 64> m_user_data{};
  ShaderComputeInputInfo m_compute{};
  ShaderRecompiler::CompileOptions m_options;
};

// Every effect of the prologue -- the three remapped builtins and the early return -- is keyed
// on the remap bit, so a zero control word leaves it inert.
void CheckPrologueKeyedOnRemapBit(const std::string &text) {
  std::smatch match;
  const std::regex decode(R"((%\w+) = OpShiftRightLogical %uint %\w+ %uint_31\n)");
  Require("prologue remap bit", std::regex_search(text, match, decode),
          "no remap-bit decode");
  const std::regex test("(%\\w+) = OpINotEqual %bool " + match[1].str() + " %uint_0\n");
  Require("prologue remap test", std::regex_search(text, match, test), "no remap-bit test");
  const auto remap = match[1].str();
  const std::regex select("= OpSelect %uint " + remap + " ");
  const auto selects = std::distance(
      std::sregex_iterator(text.begin(), text.end(), select), std::sregex_iterator());
  Require("prologue builtins", selects == 3,
          "expected three remap selects, found " + std::to_string(selects));
  const std::regex exit("(%\\w+) = OpLogicalAnd %bool " + remap + " ");
  Require("prologue exit", std::regex_search(text, match, exit),
          "the early return is not keyed on the remap bit");
  Require("prologue branch",
          text.find("OpBranchConditional " + match[1].str() + " ") != std::string::npos,
          "the early return does not branch on the remap test");
}

void CheckRescaledEmission(const TileProgram &tile) {
  auto program = tile.Specialise();
  TileProgram::Rescale(program);
  // The read and the write each get their own rewritten address, and each decodes the word.
  Require("texel rewrite",
          Count(program, IR::ValueOpcode::MakeImageAddress) == 2 &&
              Count(program, IR::ValueOpcode::GetRescaleControl) == 3 &&
              Count(program, IR::ValueOpcode::FPAdd32) == 1 &&
              Count(program, IR::ValueOpcode::SelectF32) == 1,
          "unexpected rewritten IR:\n" + IR::ProgramToString(program));
  IR::CollectShaderInfo(program, tile.Options().input_info);
  IR::AllocateBindings(program);
  const auto &bindings = program.bindings;
  Require("program metadata",
          program.tile_rescale.Enabled() && program.tile_rescale.scale_log2 == 1 &&
              program.tile_rescale.NeedsRuntimeCheck(),
          "tile_rescale was not recorded");
  Require("control word",
          bindings.HasRescaleControl() &&
              bindings.rescale_control_dword == bindings.MemoryOffsetEndDword() &&
              bindings.ShaderDataDwords() == bindings.rescale_control_dword + 1u,
          "the control word was not placed after the memory offsets");
  Require("prologue inputs",
          std::ranges::any_of(program.info.inputs,
                              [](const auto &input) {
                                return input.kind == IR::StageInputKind::NumWorkgroups;
                              }),
          "NumWorkgroups was not declared");

  const auto spirv = ShaderRecompiler::Spirv::EmitProgram(program, tile.Options().input_info);
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
  std::string messages;
  tools.SetMessageConsumer([&messages](spv_message_level_t, const char *,
                                       const spv_position_t &position,
                                       const char *message) {
    messages += std::to_string(position.index) + ": " + message + "\n";
  });
  Require("SPIR-V validation (vulkan1.3)", tools.Validate(spirv), messages);
  std::string text;
  Require("SPIR-V disassembly", tools.Disassemble(spirv, &text), "failed to disassemble");
  Require("SPIR-V builtins", text.find("BuiltIn NumWorkgroups") != std::string::npos,
          "NumWorkgroups is not bound");
  CheckPrologueKeyedOnRemapBit(text);
}

// Folding a zero control word into the rewritten IR leaves every texel address unshifted and
// removes the centre offset.
void CheckZeroWordIsInert(const TileProgram &tile) {
  auto program = tile.Specialise();
  TileProgram::Rescale(program);
  for (auto *block : program.blocks) {
    for (auto &inst : *block) {
      if (inst.GetOpcode() == IR::ValueOpcode::GetRescaleControl) {
        inst.ReplaceUsesWith(IR::Value(0u));
      }
    }
  }
  IR::ConstantPropagationPass(program.blocks);
  IR::RemoveIdentities(program.blocks);
  IR::EliminateDeadCode(program.blocks);
  const auto unshifted = [](IR::Value value) {
    const auto *inst = value.Resolve().TryInstruction();
    if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::ShiftRightLogical32) {
      return true;
    }
    const auto shift = inst->Arg(1).Resolve();
    return shift.IsImmediate() && shift.U32() == 0u;
  };
  size_t accesses = 0;
  for (const auto *block : program.blocks) {
    for (const auto &inst : *block) {
      if (inst.GetOpcode() != IR::ValueOpcode::ImageRead &&
          inst.GetOpcode() != IR::ValueOpcode::ImageWrite) {
        continue;
      }
      accesses++;
      const auto *address = inst.Arg(1).Resolve().TryInstruction();
      Require("zero-word texel address",
              address != nullptr && unshifted(address->Arg(0)) && unshifted(address->Arg(1)),
              "a zero control word still shifts a texel address:\n" +
                  IR::ProgramToString(program));
    }
  }
  Require("zero-word accesses", accesses == 2, "expected one image read and one write");
  Require("zero-word centre",
          Count(program, IR::ValueOpcode::FPAdd32) == 0 &&
              Count(program, IR::ValueOpcode::SelectF32) == 0,
          "a zero control word still offsets the coordinate:\n" +
              IR::ProgramToString(program));
}

} // namespace

int main() {
  EnsureConfigInitialized();
  const TileProgram tile;
  CheckRescaledEmission(tile);
  CheckZeroWordIsInert(tile);
  std::printf("TileRescaleApplyTests: passed\n");
  return 0;
}
