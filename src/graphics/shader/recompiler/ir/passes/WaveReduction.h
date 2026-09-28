#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Wave reductions recovered from guest lane-exchange chains.
//
// The PS5 shader compiler lowers wave intrinsics (WaveActiveBitOr, WaveActiveSum, ...) to a scan
// that assumes every lane of the guest wave exists: it turns on all lanes (S_ORN2_SAVEEXEC /
// S_OR_SAVEEXEC ..., -1), masks the operand of the lanes that were off to the operation's
// identity (V_CNDMASK with the saved exec), combines lanes through DPP row shifts / xmasks and
// V_PERMLANEX16, and reads the total with V_READLANE_B32 from the last lane of each half. On the
// host a lane without a pixel (or beyond the host subgroup) has no invocation, so it cannot relay
// partial results through the scan, and a read of a lane that does not exist is undefined.
//
// FindWaveReduction proves, for one ReadLane(X, L), that on the guest (all W lanes present, as
// on RDNA2) X at lane L is  op over the lanes in S of B  for one leaf value B, one integer op and a
// lane set S. RecoverWaveReductions replaces such reads with WaveReduceU32(B, S) that combines B
// over the host invocations whose guest lane id is in S.
//
// Soundness. The chain is evaluated exactly as the emitter executes it (DPP and permlane lane
// addressing from ir/LaneAddressing.h; every lane exchange must run under a provably all-true
// exec), so for a full guest wave the recovered set is exact, and since integer op is
// associative and commutative the host reduction over the same lanes is bit-identical to what
// the scan computes today. For a lane without a host invocation, compiler-generated code has
// already masked B to op's identity (the saved-exec V_CNDMASK), so leaving it out does not
// change the result: every host invocation gets the value the guest lane would have read.
// Where the host runs one guest wave as several subgroups (a wave64 pixel shader on a 32-lane
// device), each subgroup reduces over its own guest lanes -- the same partition every other
// translated wave operation already sees -- instead of reading a lane that does not exist.
//
// Compute programs are not rewritten: their subgroups are pinned full, where the scan is exact.

struct WaveReduction {
	Value        leaf;
	WaveReduceOp op    = WaveReduceOp::BitwiseOr;
	uint64_t     lanes = 0;
};

// The reduction ReadLane `read` computes on a W-lane guest wave (W = wave_size, 32 or 64), if
// its source is a lane-exchange chain the analysis proves; nullopt otherwise. The result always
// combines at least two lanes: a read of a lone lane is not a reduction.
[[nodiscard]] std::optional<WaveReduction> FindWaveReduction(const Inst& read, uint32_t wave_size);

struct WaveReductionStats {
	uint32_t rewritten_reads = 0;
};

[[nodiscard]] WaveReductionStats RecoverWaveReductions(Program& program, uint32_t wave_size);

} // namespace Libs::Graphics::ShaderRecompiler::IR
