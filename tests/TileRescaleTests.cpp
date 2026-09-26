// Tile-rescale analysis (ir/passes/TileRescale.h): the offline detector's synthetic cases, its
// golden verdicts over the Astro's Playroom IR dumps, and one program built through the IR.
//
// The dumps and the synthetic cases are .irx text, the format the measurement branch
// (meas/remap-detector) wrote and the Python detector (remapdet.py) reads. The parser below is
// test-only: it rebuilds a RescaleGraph from that text, mapping the dumps' enum numbering
// through explicit tables so a renumbered enum cannot silently change a verdict.
//
// Golden run: KYTY_TILE_RESCALE_DUMPS=<dir with dump-*/ and rd-*.txt> tile_rescale_tests

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/TileRescale.h"
#include "graphics/shader/recompiler/ir/passes/TileRescaleGraph.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;

int g_failures = 0;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    g_failures++;
    std::cerr << "  FAIL: " << message << '\n';
  }
}

// ---------------------------------------------------------------------------
// .irx parser
// ---------------------------------------------------------------------------

std::vector<std::string_view> Split(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(separator, start);
    if (end == std::string_view::npos) {
      parts.push_back(text.substr(start));
      break;
    }
    parts.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

std::vector<std::string_view> Words(std::string_view text) {
  std::vector<std::string_view> words;
  for (auto word : Split(text, ' ')) {
    if (!word.empty()) {
      words.push_back(word);
    }
  }
  return words;
}

uint64_t ParseInt(std::string_view text, int base = 10) {
  if (base == 16 && text.starts_with("0x")) {
    text.remove_prefix(2);
  }
  uint64_t value = 0;
  std::from_chars(text.data(), text.data() + text.size(), value, base);
  return value;
}

// key=value words after the first `skip` words.
std::map<std::string_view, std::string_view>
KeyValues(const std::vector<std::string_view> &words, size_t skip) {
  std::map<std::string_view, std::string_view> result;
  for (size_t i = skip; i < words.size(); i++) {
    const auto eq = words[i].find('=');
    if (eq != std::string_view::npos) {
      result[words[i].substr(0, eq)] = words[i].substr(eq + 1);
    }
  }
  return result;
}

// The dumps' numbering of the enums they store as integers (pre-sync ShaderIR.h and
// ShaderDecoder.h), mapped by name.
constexpr std::array LegacyStageInputKinds = {
    StageInputKind::VertexIndex,          StageInputKind::InstanceIndex,
    StageInputKind::FragCoord,            StageInputKind::FrontFacing,
    StageInputKind::PackedAncillary,      StageInputKind::Layer,
    StageInputKind::SampleId,             StageInputKind::BaryCoordSmooth,
    StageInputKind::BaryCoordNoPerspective, StageInputKind::WorkgroupId,
    StageInputKind::LocalInvocationId,    StageInputKind::LocalInvocationIndex,
    StageInputKind::GlobalInvocationId,   StageInputKind::Parameter,
    StageInputKind::NumWorkgroups,
};
constexpr std::array LegacyResourceKinds = {
    ResourceKind::None,    ResourceKind::ScalarBuffer, ResourceKind::ScalarAddress,
    ResourceKind::Buffer,  ResourceKind::Flat,         ResourceKind::Global,
    ResourceKind::Scratch, ResourceKind::Lds,          ResourceKind::Gds,
    ResourceKind::Image,   ResourceKind::Sampler,
};
constexpr std::array LegacyImageDimensions = {
    Decoder::ImageDimension::Unknown,    Decoder::ImageDimension::Dim1D,
    Decoder::ImageDimension::Dim1DArray, Decoder::ImageDimension::Dim2D,
    Decoder::ImageDimension::Dim3D,      Decoder::ImageDimension::Dim2DArray,
    Decoder::ImageDimension::Dim2DMsaa,  Decoder::ImageDimension::Dim2DMsaaArray,
};
constexpr std::array LegacyMipModes = {ImageMipMode::None,
                                       ImageMipMode::DynamicStorage};

template <typename T, size_t N>
T FromLegacy(const std::array<T, N> &table, uint64_t value) {
  return value < N ? table[value] : table[0];
}

const std::unordered_map<std::string, ValueOpcode> &OpcodesByName() {
  static const auto map = [] {
    std::unordered_map<std::string, ValueOpcode> result;
    for (size_t i = 0; i < static_cast<size_t>(ValueOpcode::Count); i++) {
      const auto op = static_cast<ValueOpcode>(i);
      result.emplace(std::string(ValueOpcodeName(op)), op);
    }
    return result;
  }();
  return map;
}

const std::unordered_map<std::string, Type> &TypesByName() {
  static const auto map = [] {
    std::unordered_map<std::string, Type> result{{"Void", Type::Void}};
    for (uint32_t bit = 0; bit < 20; bit++) {
      const auto type = static_cast<Type>(1u << bit);
      result.emplace(TypeName(type), type);
    }
    return result;
  }();
  return map;
}

struct IrxProgram {
  std::string hash;
  TileRescaleShape shape;
  RescaleGraph graph;
  bool ok = true;
  std::string error;
};

class IrxParser {
public:
  IrxProgram Parse(std::string_view text) {
    for (auto line : Split(text, '\n')) {
      if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
      }
      if (line.empty()) {
        continue;
      }
      if (line.starts_with("P ")) {
        ParseProgram(Words(line));
      } else if (line.starts_with("I ")) {
        ParseImage(Words(line));
      } else if (line.starts_with("B ")) {
        continue;
      } else if (line.starts_with("M ")) {
        ParseMemory(Words(line));
      } else if (line.starts_with("K ")) {
        ParseBlock(Words(line));
      } else {
        ParseNodeHead(line);
      }
    }
    for (auto &block : result.graph.blocks) {
      block.num_nodes = 0;
    }
    for (size_t i = 0; i < result.graph.nodes.size(); i++) {
      auto &block = result.graph.blocks[result.graph.nodes[i].block];
      if (block.num_nodes++ == 0) {
        block.first_node = static_cast<uint32_t>(i);
      }
    }
    for (size_t i = 0; i < result.graph.nodes.size(); i++) {
      ParseOperands(result.graph.nodes[i], arg_texts[i]);
    }
    for (size_t b = 0; b < block_conditions.size(); b++) {
      result.graph.blocks[b].condition = Operand(block_conditions[b]);
    }
    for (auto &node : result.graph.nodes) {
      node.indexed_register_write = IsLadderRung(node);
    }
    result.graph.LinkUsers();
    return std::move(result);
  }

private:
  void Fail(std::string message) {
    if (result.ok) {
      result.ok = false;
      result.error = std::move(message);
    }
  }

  void ParseProgram(const std::vector<std::string_view> &words) {
    auto kv = KeyValues(words, 1);
    result.hash = std::string(kv["hash"]);
    result.shape.wave_size = static_cast<uint32_t>(ParseInt(kv["wave"]));
    result.shape.host_subgroup_size =
        static_cast<uint32_t>(ParseInt(kv["host_sg"]));
    const auto threads = Split(kv["threads"], ',');
    for (size_t i = 0; i < 3 && i < threads.size(); i++) {
      result.shape.threads[i] = static_cast<uint32_t>(ParseInt(threads[i]));
    }
    result.shape.tg_size_en = kv["tg_size_en"] == "1";
    result.shape.scale_log2 = 1;
    result.graph.dispatcher_fallback = kv["fallback"] == "1";
  }

  void ParseImage(const std::vector<std::string_view> &words) {
    const auto index = ParseInt(words[1]);
    auto kv = KeyValues(words, 2);
    if (result.graph.images.size() <= index) {
      result.graph.images.resize(index + 1);
    }
    result.graph.images[index] = {
        FromLegacy(LegacyImageDimensions, ParseInt(kv["dim"])),
        FromLegacy(LegacyMipModes, ParseInt(kv["mipmode"]))};
  }

  void ParseMemory(const std::vector<std::string_view> &words) {
    const auto index = ParseInt(words[1]);
    auto kv = KeyValues(words, 2);
    if (result.graph.memory.size() <= index) {
      result.graph.memory.resize(index + 1);
    }
    auto &memory = result.graph.memory[index];
    memory.kind = FromLegacy(LegacyResourceKinds, ParseInt(kv["kind"]));
    memory.resource = static_cast<uint32_t>(ParseInt(kv["res"]));
    memory.address_components = static_cast<uint32_t>(ParseInt(kv["comps"]));
    const auto x = Split(kv["x"], ':');
    const auto y = Split(kv["y"], ':');
    memory.x_offset = static_cast<uint32_t>(ParseInt(x[0]));
    memory.x_width = static_cast<uint32_t>(ParseInt(x.at(1)));
    memory.y_offset = static_cast<uint32_t>(ParseInt(y[0]));
    memory.y_width = static_cast<uint32_t>(ParseInt(y.at(1)));
  }

  void ParseBlock(const std::vector<std::string_view> &words) {
    auto kv = KeyValues(words, 2);
    RescaleBlock block;
    if (kv["succ"] != "-") {
      for (auto s : Split(kv["succ"], ',')) {
        block.successors.push_back(static_cast<uint32_t>(ParseInt(s)));
      }
    }
    result.graph.blocks.push_back(std::move(block));
    block_conditions.push_back(kv["cond"]);
  }

  void ParseNodeHead(std::string_view line) {
    const auto split = line.find(" a=");
    const auto head = line.substr(0, split);
    const auto words = Words(head);
    if (words.size() < 3 || result.graph.blocks.empty()) {
      Fail("malformed instruction line: " + std::string(line));
      return;
    }
    RescaleNode node;
    node.id = static_cast<uint32_t>(ParseInt(words[0]));
    const auto op = OpcodesByName().find(std::string(words[1]));
    const auto type = TypesByName().find(std::string(words[2]));
    if (op == OpcodesByName().end() || type == TypesByName().end()) {
      Fail("unknown opcode or type: " + std::string(line));
      return;
    }
    node.opcode = op->second;
    node.type = type->second;
    auto kv = KeyValues(words, 3);
    node.flags = ParseInt(kv["f"], 16);
    node.buffer_access = static_cast<BufferAccess>(ParseInt(kv["ba"]));
    node.shared_access = static_cast<SharedAccess>(ParseInt(kv["sa"]));
    node.address_access = static_cast<AddressAccess>(ParseInt(kv["aa"]));
    node.image_access = static_cast<ImageAccess>(ParseInt(kv["ia"]));
    node.block = static_cast<uint32_t>(result.graph.blocks.size() - 1);
    node_of_id[node.id] = static_cast<uint32_t>(result.graph.nodes.size());
    result.graph.nodes.push_back(node);
    arg_texts.push_back(split == std::string_view::npos
                            ? std::string_view{}
                            : line.substr(split + 3));
  }

  RescaleOperand Operand(std::string_view text) {
    if (text.starts_with('%')) {
      const auto it = node_of_id.find(static_cast<uint32_t>(ParseInt(text.substr(1))));
      if (it == node_of_id.end()) {
        Fail("operand names an unknown instruction: " + std::string(text));
        return {};
      }
      return RescaleOperand::ForNode(it->second);
    }
    const auto colon = text.find(':');
    if (colon != std::string_view::npos) {
      const auto tag = text.substr(0, colon);
      const auto value = text.substr(colon + 1);
      if (tag == "u1") return RescaleOperand::ForImmediate(Type::U1, ParseInt(value));
      if (tag == "u8") return RescaleOperand::ForImmediate(Type::U8, ParseInt(value));
      if (tag == "u16") return RescaleOperand::ForImmediate(Type::U16, ParseInt(value));
      if (tag == "u32") return RescaleOperand::ForImmediate(Type::U32, ParseInt(value, 16));
      if (tag == "u64") return RescaleOperand::ForImmediate(Type::U64, ParseInt(value, 16));
      if (tag == "f16") return RescaleOperand::ForImmediate(Type::F16, ParseInt(value, 16));
      if (tag == "f32") {
        const float f = std::strtof(std::string(value).c_str(), nullptr);
        return RescaleOperand::ForImmediate(Type::F32, std::bit_cast<uint32_t>(f));
      }
      return {};
    }
    if (text.size() > 1 && (text[0] == 's' || text[0] == 'v') &&
        std::isdigit(static_cast<unsigned char>(text[1]))) {
      return RescaleOperand::ForImmediate(
          text[0] == 's' ? Type::ScalarReg : Type::VectorReg, ParseInt(text.substr(1)));
    }
    return {}; // null
  }

  void ParseOperands(RescaleNode &node, std::string_view text) {
    if (text.empty()) {
      return;
    }
    for (auto arg : Split(text, '|')) {
      uint32_t phi_block = 0;
      if (node.opcode == ValueOpcode::Phi) {
        const auto at = arg.rfind('@');
        phi_block = static_cast<uint32_t>(ParseInt(arg.substr(at + 1)));
        arg = arg.substr(0, at);
      }
      auto operand = Operand(arg);
      // GetBuiltin's kind is an immediate of the dump's StageInputKind numbering.
      if (node.opcode == ValueOpcode::GetBuiltin && node.num_args == 0 &&
          operand.IsImmediate(Type::U32)) {
        operand.bits = static_cast<uint32_t>(FromLegacy(LegacyStageInputKinds, operand.bits));
      }
      result.graph.AddOperand(node, operand, phi_block);
    }
  }

  // Legacy dumps carry no SelectFlags: recognise a V_MOVRELD rung the way the offline detector
  // does, cond = exec & ((m0 & 0xff) == K).
  bool IsLadderRung(const RescaleNode &node) const {
    const auto &graph = result.graph;
    if (node.opcode != ValueOpcode::SelectU32) {
      return false;
    }
    const auto *c = graph.ArgNode(node, 0);
    if (c == nullptr || c->opcode != ValueOpcode::LogicalAnd) {
      return false;
    }
    for (size_t i = 0; i < 2; i++) {
      const auto *e = graph.ArgNode(*c, i);
      if (e != nullptr && e->opcode == ValueOpcode::IEqual32 && e->num_args > 1 &&
          graph.Args(*e)[1].ImmU32()) {
        const auto *m = graph.ArgNode(*e, 0);
        if (m != nullptr && m->opcode == ValueOpcode::BitwiseAnd32 && m->num_args > 1 &&
            graph.Args(*m)[1].ImmU32() == 0xffu) {
          return true;
        }
      }
    }
    return false;
  }

  IrxProgram result;
  std::unordered_map<uint32_t, uint32_t> node_of_id;
  std::vector<std::string_view> arg_texts;
  std::vector<std::string_view> block_conditions;
};

IrxProgram ParseIrx(std::string_view text) { return IrxParser().Parse(text); }

// ---------------------------------------------------------------------------
// Synthetic cases (test_remapdet.py): a 16x16 wave32 tile shader that loads a tile origin from a
// tile list indexed by WorkGroupID.x, forms (x, y) = (origin << 4) + LocalID and stores one texel
// -- the Astro tile-lighting shape -- with one mutation per case.
// ---------------------------------------------------------------------------

std::string Header(uint32_t wave, uint32_t tx, uint32_t ty, uint32_t rw, uint32_t bw) {
  char text[1024];
  std::snprintf(
      text, sizeof(text),
      "P hash=%016x wave=%u host_sg=32 threads=%u,%u,1 tg_size_en=0 fallback=0 blocks=1 "
      "images=2 buffers=1 mem=3\n"
      "I 0 dim=3 class=1 read=1 written=0 texel=1 atomic=0 mips=1 mipmode=0 cube=0 r128=0 "
      "indirect=0\n"
      "I 1 dim=3 class=2 read=%u written=1 texel=1 atomic=0 mips=1 mipmode=0 cube=0 r128=0 "
      "indirect=0\n"
      "B 0 read=1 written=%u atomic=0 scalar=1 alias=0\n"
      "M 0 kind=1 res=0 dim=0 comps=0 x=0:32 y=32:32 sflags=0x0\n"
      "M 1 kind=9 res=0 dim=3 comps=2 x=0:32 y=32:32 sflags=0x0\n"
      "M 2 kind=9 res=1 dim=3 comps=2 x=0:32 y=32:32 sflags=0x0\n"
      "K 0 succ=- preds=- cond=null\n",
      0u, wave, tx, ty, rw, bw);
  return text;
}

std::string Line(int id, const char *op, const char *type, std::vector<std::string> args,
                 int ia = 0, int ba = 0, int sa = 0, int flags = 0) {
  std::string joined;
  for (const auto &arg : args) {
    joined += (joined.empty() ? "" : "|") + arg;
  }
  char head[256];
  std::snprintf(head, sizeof(head), "%d %s %s f=0x%x ba=%d sa=%d aa=0 ia=%d a=", id, op,
                type, flags, ba, sa, ia);
  return head + joined + "\n";
}

std::vector<std::string> Zeros(size_t count) {
  return std::vector<std::string>(count, "u32:0x0");
}

std::vector<std::string> Concat(std::vector<std::string> a, const std::vector<std::string> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

struct BaseOptions {
  std::string x_expr;
  std::string y_expr;
  std::string extra;
  std::string store_x = "%20";
  std::string store_y = "%21";
  std::string pred = "u1:1";
};

std::string Base(const BaseOptions &o) {
  std::string b;
  b += Line(1, "GetBuiltin", "U32", {"u32:0xa", "u32:0x0"}); // Lx
  b += Line(2, "GetBuiltin", "U32", {"u32:0xa", "u32:0x1"}); // Ly
  b += Line(3, "GetBuiltin", "U32", {"u32:0x9", "u32:0x0"}); // WorkGroupID.x
  b += Line(4, "GetBufferResource", "BufferResource", Zeros(4));
  b += Line(5, "ShiftLeftLogical32", "U32", {"%3", "u32:0x2"});
  b += Line(6, "ReadConstBuffer", "U32", {"%4", "%5"}); // packed tile (x << 16 | y)
  b += Line(7, "ShiftRightLogical32", "U32", {"%6", "u32:0x10"});
  b += Line(8, "BitwiseAnd32", "U32", {"%6", "u32:0xffff"});
  b += Line(9, "ShiftLeftLogical32", "U32", {"%7", "u32:0x4"});
  b += Line(10, "ShiftLeftLogical32", "U32", {"%8", "u32:0x4"});
  b += o.x_expr.empty() ? Line(20, "IAdd32", "U32", {"%9", "%1"}) : o.x_expr;
  b += o.y_expr.empty() ? Line(21, "IAdd32", "U32", {"%10", "%2"}) : o.y_expr;
  b += Line(30, "GetImageResource", "ImageResource", Zeros(8));
  b += Line(31, "MakeImageAddress", "ImageAddress", Concat({"%20", "%21"}, Zeros(11)));
  b += Line(32, "ImageRead", "U32x4", {"%30", "%31", "u1:1"}, 1, 0, 0, 1);
  b += o.extra;
  b += Line(40, "GetImageResource", "ImageResource", Concat({"u32:0x1"}, Zeros(7)));
  b += Line(41, "MakeImageAddress", "ImageAddress", Concat({o.store_x, o.store_y}, Zeros(11)));
  b += Line(42, "ImageWrite", "Void", {"%40", "%41", "%32", o.pred}, 2, 0, 0, 2);
  return b;
}

struct ProgramOptions {
  uint32_t wave = 32;
  uint32_t tx = 16;
  uint32_t ty = 16;
  uint32_t rw = 0;
  uint32_t bw = 0;
};

std::string MakeProgram(const std::string &body, ProgramOptions o = {}) {
  return Header(o.wave, o.tx, o.ty, o.rw, o.bw) + body;
}

struct SyntheticCase {
  const char *name;
  bool expected;
  std::string text;
};

std::vector<SyntheticCase> SyntheticCases() {
  const auto base = [](BaseOptions o = {}) { return Base(o); };
  return {
      {"own-pixel tile store (Astro shape)", true, MakeProgram(base())},
      {"tile origin OR'd instead of added", true,
       MakeProgram(base({.x_expr = Line(20, "BitwiseOr32", "U32", {"%9", "%1"})}))},
      {"bounds check against uniform extent", true,
       MakeProgram(base({.extra = Line(33, "ULessThan32", "U1", {"%20", "%6"}), .pred = "%33"}))},
      {"wave64 guest", false, MakeProgram(base(), {.wave = 64})},
      {"8x8 workgroup (slot below one warp)", false, MakeProgram(base(), {.tx = 8, .ty = 8})},
      {"store at neighbour pixel x+1", false,
       MakeProgram(base({.extra = Line(33, "IAdd32", "U32", {"%20", "u32:0x1"}), .store_x = "%33"}))},
      {"store at a uniform texel", false, MakeProgram(base({.store_x = "%9", .store_y = "%10"}))},
      {"store at a loaded (data-dependent) texel", false,
       MakeProgram(base({.extra = Line(33, "CompositeExtractU32x4", "U32", {"%32", "u32:0x0"}),
                     .store_x = "%33"}))},
      {"tile origin not a multiple of 2", false,
       MakeProgram(base({.x_expr = Line(19, "BitwiseOr32", "U32", {"%9", "u32:0x1"}) +
                               Line(20, "IAdd32", "U32", {"%19", "%1"})}))},
      {"checkerboard (x ^ y) & 1 selects the result", false,
       MakeProgram(base({.extra = Line(33, "BitwiseXor32", "U32", {"%20", "%21"}) +
                              Line(34, "BitwiseAnd32", "U32", {"%33", "u32:0x1"}) +
                              Line(35, "IEqual32", "U1", {"%34", "u32:0x0"}),
                     .pred = "%35"}))},
      {"x & 1 parity", false,
       MakeProgram(base({.extra = Line(33, "BitwiseAnd32", "U32", {"%20", "u32:0x1"}) +
                              Line(34, "INotEqual32", "U1", {"%33", "u32:0x0"}),
                     .pred = "%34"}))},
      {"special work only at x == 15 (odd edge lane)", false,
       MakeProgram(base({.extra = Line(33, "IEqual32", "U1", {"%1", "u32:0xf"}), .pred = "%33"}))},
      {"LDS + barrier", false,
       MakeProgram(base({.extra = Line(33, "WriteSharedU32", "Void", {"%1", "%20", "u1:1"}, 0, 0, 2) +
                              Line(34, "Barrier", "Void", {})}))},
      {"buffer store", false,
       MakeProgram(base({.extra = Line(33, "StoreBufferU32", "Void",
                                   {"%4", "%20", "u32:0x0", "u32:0x0", "%21", "u1:1"}, 0, 2)}),
               {.bw = 1})},
      {"reads GlobalInvocationID", false,
       MakeProgram(base({.extra = Line(33, "GetBuiltin", "U32", {"u32:0xc", "u32:0x0"})}))},
      {"image size query", false,
       MakeProgram(base({.extra = Line(33, "ImageQueryDimensions", "U32x4", {"%30", "%31"}, 1, 0, 0, 1)}))},
      {"reads lane 5 of a varying value", false,
       MakeProgram(base({.extra = Line(33, "CompositeExtractU32x4", "U32", {"%32", "u32:0x0"}) +
                              Line(34, "ReadLane", "U32", {"%33", "u32:0x5"})}))},
      {"mbcnt-style prefix count of a ballot", false,
       MakeProgram(base({.extra = Line(33, "Ballot", "U32x4", {"u1:1"}) +
                              Line(34, "CompositeExtractU32x4", "U32", {"%33", "u32:0x0"}) +
                              Line(35, "BitCount32", "U32", {"%34"})}))},
      {"image atomic", false,
       MakeProgram(base({.extra = Line(33, "ImageAtomicIAdd32", "U32", {"%40", "%31", "u32:0x1", "u1:1"},
                                   3, 0, 0, 2)}))},
  };
}

void TestSyntheticCases() {
  const auto cases = SyntheticCases();
  int passed = 0;
  for (const auto &test : cases) {
    const auto program = ParseIrx(test.text);
    Check(program.ok, std::string(test.name) + ": parse: " + program.error);
    const auto report = AnalyzeRescaleGraph(program.graph, program.shape);
    const bool ok = report.plan.accepted == test.expected;
    Check(ok, std::string(test.name) + ": expected " + (test.expected ? "ACCEPT" : "REJECT") +
                  (report.plan.reasons.empty() ? "" : " (" + report.plan.reasons[0] + ")"));
    Check(report.plan.accepted == report.plan.reasons.empty(),
          std::string(test.name) + ": accepted with reasons");
    passed += ok;
  }
  std::cout << "synthetic: " << passed << "/" << cases.size() << " cases as expected\n";
}

// ---------------------------------------------------------------------------
// Golden dumps: every .irx under dump-*/ against the Python detector's rd-*.txt.
// ---------------------------------------------------------------------------

using Counts = std::map<std::string, uint32_t>;

// A Python dict repr of str -> int, e.g. {'own': 6, 'data-dependent': 2}.
Counts ParsePythonCounts(std::string_view text) {
  Counts counts;
  size_t i = 0;
  while (true) {
    i = text.find_first_of("'\"", i);
    if (i == std::string_view::npos) {
      break;
    }
    const auto end = text.find(text[i], i + 1);
    const auto key = std::string(text.substr(i + 1, end - i - 1));
    auto colon = text.find(": ", end);
    auto stop = text.find_first_of(",}", colon);
    counts[key] = static_cast<uint32_t>(ParseInt(text.substr(colon + 2, stop - colon - 2)));
    i = stop;
  }
  return counts;
}

Counts ToCounts(const TileRescaleReport::Counts &counts) {
  return {counts.begin(), counts.end()};
}

std::string Between(std::string_view line, std::string_view from, std::string_view to) {
  const auto start = line.find(from);
  if (start == std::string_view::npos) {
    return {};
  }
  const auto begin = start + from.size();
  const auto end = to.empty() ? line.size() : line.find(to, begin);
  return std::string(line.substr(begin, end - begin));
}

struct Verdict {
  bool accept = false;
  Counts loads;
  uint32_t store_align = 0;
  uint32_t nb = 0;
  std::string cls;
  Counts parity;
  Counts cross_lane;
  std::vector<std::string> reasons;
};

std::map<std::string, Verdict> ParseVerdicts(const std::filesystem::path &path) {
  std::map<std::string, Verdict> verdicts;
  std::ifstream in(path);
  std::string line;
  Verdict *current = nullptr;
  while (std::getline(in, line)) {
    const auto words = Words(line);
    if (words.size() > 2 && words[0].size() == 16 &&
        (words[1] == "ACCEPT" || words[1] == "REJECT")) {
      current = &verdicts[std::string(words[0])];
      current->accept = words[1] == "ACCEPT";
      current->loads = ParsePythonCounts(Between(line, "loads=", " store_align="));
      current->store_align = static_cast<uint32_t>(ParseInt(Between(line, "store_align=", " ")));
      current->nb = static_cast<uint32_t>(ParseInt(Between(line, " nb=", " ")));
    } else if (current != nullptr && line.starts_with("    - ")) {
      if (!line.starts_with("    - ... ")) {
        current->reasons.push_back(line.substr(6));
      }
    } else if (current != nullptr && line.starts_with("    class=")) {
      current->cls = Between(line, "class=", " ");
      current->parity = ParsePythonCounts(Between(line, "parity: ", " | cross-lane: "));
      current->cross_lane = ParsePythonCounts(Between(line, "cross-lane: ", ""));
    }
  }
  return verdicts;
}

// The Python prints reasons with the instruction id removed: "why [ Op]".
std::string StripId(const std::string &reason) {
  const auto at = reason.find(" [%");
  if (at == std::string::npos) {
    return reason;
  }
  auto end = at + 3;
  while (end < reason.size() && std::isdigit(static_cast<unsigned char>(reason[end]))) {
    end++;
  }
  return reason.substr(0, at + 2) + reason.substr(end);
}

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

const char *ClassName(CrossLaneClass cls) {
  switch (cls) {
  case CrossLaneClass::ExecMaskOnly: return "C0";
  case CrossLaneClass::Waterfall: return "C1";
  case CrossLaneClass::LaneSerialized: return "C2";
  }
  return "?";
}

void TestGoldenDumps() {
  const char *root_env = std::getenv("KYTY_TILE_RESCALE_DUMPS");
  if (root_env == nullptr) {
    std::cout << "golden: SKIPPED (set KYTY_TILE_RESCALE_DUMPS)\n";
    return;
  }
  namespace fs = std::filesystem;
  const fs::path root(root_env);
  const std::set<std::string> accepted_family = {"d0c04d63", "1a33cd8c", "6f082767", "162f3174",
                                                 "ae5cc42b", "10e28006", "8e91b5d8", "91187dcd"};
  std::vector<fs::path> scenes;
  std::error_code error;
  for (const auto &entry : fs::directory_iterator(root, error)) {
    if (entry.is_directory() && entry.path().filename().string().starts_with("dump-")) {
      scenes.push_back(entry.path());
    }
  }
  std::sort(scenes.begin(), scenes.end());
  Check(!scenes.empty(), "golden: no dump-* directories under " + root.string());

  std::set<std::string> all_hashes;
  double max_ms = 0;
  std::string max_program;
  size_t programs_analysed = 0;
  size_t hashes_agreeing = 0;
  for (const auto &scene : scenes) {
    // dump-pre-plaza-moving -> rd-pre-moving.txt
    auto name = scene.filename().string().substr(5);
    if (const auto at = name.find("-plaza"); at != std::string::npos) {
      name.erase(at, 6);
    }
    const auto verdicts = ParseVerdicts(root / ("rd-" + name + ".txt"));
    Check(!verdicts.empty(), "golden: no verdicts for " + scene.string());

    std::vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(scene, error)) {
      if (entry.path().extension() == ".irx") {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());

    struct HashResult {
      bool accept = true;
      bool first = true;
      TileRescaleReport report; // first variant, as the Python prints
      std::set<std::string> reasons;
      int variants = 0;
    };
    std::map<std::string, HashResult> results;
    for (const auto &file : files) {
      const auto program = ParseIrx(ReadFile(file));
      Check(program.ok, "golden: parse " + file.string() + ": " + program.error);
      const auto start = std::chrono::steady_clock::now();
      auto report = AnalyzeRescaleGraph(program.graph, program.shape);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      programs_analysed++;
      if (ms > max_ms) {
        max_ms = ms;
        max_program = file.filename().string() + " (" +
                      std::to_string(program.graph.nodes.size()) + " insts)";
      }
      auto &result = results[program.hash];
      result.variants++;
      result.accept &= report.plan.accepted;
      for (const auto &reason : report.plan.reasons) {
        result.reasons.insert(StripId(reason));
      }
      // Every accepted member of the lighting family: C2, tile origins aligned to 16.
      if (accepted_family.contains(program.hash.substr(0, 8)) && report.plan.accepted) {
        Check(report.plan.cross_lane == CrossLaneClass::LaneSerialized,
              "golden: " + file.string() + " class " + ClassName(report.plan.cross_lane));
        Check(report.plan.store_align == 4,
              "golden: " + file.string() + " store_align " + std::to_string(report.plan.store_align));
      }
      if (result.first) {
        result.first = false;
        result.report = std::move(report);
      }
    }

    int accepted = 0;
    int agreeing = 0;
    for (const auto &[hash, result] : results) {
      all_hashes.insert(hash);
      const auto it = verdicts.find(hash);
      if (it == verdicts.end()) {
        Check(false, "golden: " + name + " " + hash + " missing from rd file");
        continue;
      }
      const auto &expected = it->second;
      const auto where = "golden: " + name + " " + hash;
      bool same = expected.accept == result.accept;
      Check(same, where + " verdict " + (result.accept ? "ACCEPT" : "REJECT") + " vs Python " +
                      (expected.accept ? "ACCEPT" : "REJECT") + " -- " + result.report.plan.summary +
                      (result.reasons.empty() ? "" : " first: " + *result.reasons.begin()));
      const auto &report = result.report;
      const bool details = ToCounts(report.load_classes) == expected.loads &&
                           report.plan.store_align == expected.store_align &&
                           report.max_neighbour == expected.nb;
      Check(details, where + " load classes/store_align/nb differ: " + report.plan.summary);
      same &= details;
      if (expected.accept && result.accept) {
        const bool notes = ClassName(report.plan.cross_lane) == expected.cls &&
                           ToCounts(report.parity_notes) == expected.parity &&
                           ToCounts(report.cross_lane_notes) == expected.cross_lane;
        Check(notes, where + " class/notes differ: " + report.plan.summary);
        same &= notes;
        Check(accepted_family.contains(hash.substr(0, 8)), where + " accepted outside the family");
      }
      for (const auto &reason : expected.reasons) {
        const bool found = result.reasons.contains(reason);
        Check(found, where + " missing Python reason: " + reason);
        same &= found;
      }
      accepted += result.accept;
      agreeing += same;
    }
    Check(results.size() == verdicts.size(),
          "golden: " + name + " analysed " + std::to_string(results.size()) + " hashes, rd has " +
              std::to_string(verdicts.size()));
    hashes_agreeing += agreeing;
    std::cout << "golden " << name << ": programs=" << results.size() << " accepted=" << accepted
              << " rejected=" << results.size() - accepted << " agree=" << agreeing << "/"
              << results.size() << " (files " << files.size() << ")\n";
  }
  Check(all_hashes.size() == 36,
        "golden: " + std::to_string(all_hashes.size()) + " distinct hashes, expected 36");
  std::cout << "golden: " << all_hashes.size() << " distinct programs, " << programs_analysed
            << " dumps, " << hashes_agreeing << " scene verdicts agree; max analysis " << max_ms << " ms (" << max_program << ")\n";
  Check(max_ms < 10.0, "golden: max analysis time " + std::to_string(max_ms) + " ms >= 10 ms");
}

// ---------------------------------------------------------------------------
// A program built through the IR: the accept shape, with its x coordinate carried through a
// tagged indexed-register-write rung whose condition does not look like the legacy ladder
// pattern. Only the tag (assumption A1) lets the store be proven.
// ---------------------------------------------------------------------------

struct BuiltProgram {
  Libs::Graphics::ShaderRecompiler::IR::Program program;
  Inst *rung = nullptr;
  Inst *own_conversion = nullptr;
  Inst *origin_conversion = nullptr;
};

void BuildTileProgram(BuiltProgram &built, bool tag_rung) {
  auto &program = built.program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.block_storage.push_back(std::make_unique<Block>());
  auto *block = program.block_storage.back().get();
  program.blocks.push_back(block);
  program.block_info.push_back({.id = 0});

  MemoryInfo tile_list;
  tile_list.kind = ResourceKind::ScalarBuffer;
  MemoryInfo input;
  input.kind = ResourceKind::Image;
  input.resource = 0;
  input.image_dimension = Decoder::ImageDimension::Dim2D;
  input.image_address_components = 2;
  MemoryInfo output = input;
  output.resource = 1;
  program.memory_info = {tile_list, input, output};
  ImageResource image;
  image.dimension = Decoder::ImageDimension::Dim2D;
  image.read = true;
  image.texel_addressed = true;
  ImageResource written = image;
  written.read = false;
  written.written = true;
  program.info.images = {image, written};

  const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args, uint64_t flags = 0) {
    return &block->AppendNewInst(op, args, flags);
  };
  const auto memory = [](uint32_t index) {
    uint64_t bits = 0;
    const MemoryFlags flags{index, 0};
    std::memcpy(&bits, &flags, sizeof(flags));
    return bits;
  };
  const auto u32 = [](uint32_t v) { return Value(v); };
  const auto builtin = [&](StageInputKind kind, uint32_t component) {
    return Value(emit(ValueOpcode::GetBuiltin, {u32(static_cast<uint32_t>(kind)), u32(component)}));
  };
  const auto lx = builtin(StageInputKind::LocalInvocationId, 0);
  const auto ly = builtin(StageInputKind::LocalInvocationId, 1);
  const auto group = builtin(StageInputKind::WorkgroupId, 0);
  const auto buffer = Value(emit(ValueOpcode::GetBufferResource, {u32(0), u32(0), u32(0), u32(0)}, memory(0)));
  const auto offset = Value(emit(ValueOpcode::ShiftLeftLogical32, {group, u32(2)}));
  const auto tile = Value(emit(ValueOpcode::ReadConstBuffer, {buffer, offset}, memory(0)));
  const auto tile_x = Value(emit(ValueOpcode::ShiftRightLogical32, {tile, u32(16)}));
  const auto tile_y = Value(emit(ValueOpcode::BitwiseAnd32, {tile, u32(0xffff)}));
  const auto origin_x = Value(emit(ValueOpcode::ShiftLeftLogical32, {tile_x, u32(4)}));
  const auto origin_y = Value(emit(ValueOpcode::ShiftLeftLogical32, {tile_y, u32(4)}));
  const auto x = Value(emit(ValueOpcode::IAdd32, {origin_x, lx}));
  const auto y = Value(emit(ValueOpcode::IAdd32, {origin_y, ly}));

  const auto address = [&](Value x_value, Value y_value) {
    return Value(emit(ValueOpcode::MakeImageAddress,
                      {x_value, y_value, u32(0), u32(0), u32(0), u32(0), u32(0), u32(0), u32(0),
                       u32(0), u32(0), u32(0), u32(0)}));
  };
  const auto input_image = Value(emit(ValueOpcode::GetImageResource,
                                      {u32(0), u32(0), u32(0), u32(0), u32(0), u32(0), u32(0), u32(0)},
                                      memory(1)));
  const auto load_address = address(x, y);
  const auto texel = Value(emit(ValueOpcode::ImageRead, {input_image, load_address, Value(true)}, memory(1)));

  // An indexed register write of loaded data at a runtime index (M0 from user data): the rung
  // for the register holding x keeps x unless the index names it.
  const auto m0 = Value(emit(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))}));
  const auto names_x = Value(emit(ValueOpcode::IEqual32, {m0, u32(0x1c)}));
  const auto data = Value(emit(ValueOpcode::CompositeExtractU32x4, {texel, u32(0)}));
  uint64_t rung_flags = 0;
  if (tag_rung) {
    const SelectFlags flags{.indexed_register_write = true};
    std::memcpy(&rung_flags, &flags, sizeof(flags));
  }
  built.rung = emit(ValueOpcode::SelectU32, {names_x, data, x}, rung_flags);
  const auto x_after = Value(built.rung);

  built.own_conversion = emit(ValueOpcode::ConvertF32U32, {x_after});
  built.origin_conversion = emit(ValueOpcode::ConvertF32U32, {origin_x});
  const auto output_image = Value(emit(ValueOpcode::GetImageResource,
                                       {u32(1), u32(0), u32(0), u32(0), u32(0), u32(0), u32(0), u32(0)},
                                       memory(2)));
  const auto store_address = address(x_after, y);
  emit(ValueOpcode::ImageWrite, {output_image, store_address, texel, Value(true)}, memory(2));
}

void TestProgramWithTaggedRung() {
  const TileRescaleShape shape{.threads = {16, 16, 1}, .wave_size = 32, .host_subgroup_size = 32};

  BuiltProgram tagged;
  BuildTileProgram(tagged, true);
  const auto plan = AnalyzeTileRescale(tagged.program, shape);
  Check(plan.accepted, "program: tagged rung rejected: " +
                           (plan.reasons.empty() ? std::string() : plan.reasons[0]));
  Check(plan.scale_log2 == 1, "program: scale_log2");
  Check(plan.store_align == 4, "program: store_align " + std::to_string(plan.store_align));
  Check(plan.cross_lane == CrossLaneClass::ExecMaskOnly, "program: class");
  Check(plan.own_coordinate_conversions.size() == 1 &&
            plan.own_coordinate_conversions[0] == tagged.own_conversion,
        "program: own coordinate conversions");

  // The same rung untagged is an exec-merge select the store predicate cannot decide.
  BuiltProgram untagged;
  BuildTileProgram(untagged, false);
  const auto refused = AnalyzeTileRescale(untagged.program, shape);
  Check(!refused.accepted, "program: untagged rung accepted");
  std::cout << "program: tagged rung " << (plan.accepted ? "ACCEPT" : "REJECT")
            << ", untagged " << (refused.accepted ? "ACCEPT" : "REJECT") << " -- "
            << plan.summary << '\n';
}

} // namespace

int main() {
  TestSyntheticCases();
  TestProgramWithTaggedRung();
  TestGoldenDumps();
  if (g_failures != 0) {
    std::cerr << "tile rescale tests: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "tile rescale tests passed\n";
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
