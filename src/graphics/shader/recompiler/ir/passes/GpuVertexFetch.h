#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_GPUVERTEXFETCH_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_GPUVERTEXFETCH_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// V# dword 3 bits a vertex program is compiled against: destination select and the
// data/number format. Stride, base and record count stay dynamic.
inline constexpr uint32_t VertexFetchCheckMask = (0x7fu << 12u) | 0xfffu;

// Guest page that no allocation can occupy; a vertex program touches it when the runtime V#
// no longer matches the format it was compiled against, so the fault pass reports the drift.
inline constexpr uint64_t VertexFetchDriftAddress = (uint64_t {1} << 40u) - (uint64_t {1} << 14u);

// Turns the marked attribute-table reads and formatted vertex loads of a vertex program into
// raw address loads through the BDA page table, so no descriptor has to exist before the draw.
// Returns how many vertex loads were lowered.
uint32_t LowerGpuVertexFetch(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif
