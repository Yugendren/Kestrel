#pragma once

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Lane addressing of the RDNA2 cross-lane operations as the recompiler implements them
// (DppMoveU32, DppUpdateU32, Permlane16U32). The SPIR-V emitter evaluates these functions per
// invocation on the subgroup lane id; IR analyses evaluate them on constant lanes. Both take the
// control decoding from here, so an analysis proves facts about exactly the lane exchange the
// emitter performs. Lanes are guest wave lane ids in [0, 64).

enum class DppControlKind {
	Dpp8,
	QuadPerm,
	RowShiftLeft,
	RowShiftRight,
	RowRotateRight,
	RowMirror,
	RowHalfMirror,
	RowShare,
	RowXmask,
	Unsupported,
};

// operand: the 24-bit selector (DPP8: three bits per lane of each group of eight), the 8-bit
// selector (quad_perm), the amount 1..15 (row_shl, row_shr, row_ror), the
// shared lane 0..15 (row_share) or the xor mask 0..15 (row_xmask); unused otherwise.
struct DppControl {
	DppControlKind kind    = DppControlKind::Unsupported;
	uint32_t       operand = 0;
};

// `dpp8` is DppMoveFlags::dpp8: a DPP8 encoding carries a lane selector, not a DPP16 control.
[[nodiscard]] constexpr DppControl DecodeDppControl(uint32_t control, bool dpp8) {
	if (dpp8) {
		return {DppControlKind::Dpp8, control & 0xffffffu};
	}
	if (control <= 0xffu) {
		return {DppControlKind::QuadPerm, control};
	}
	if (control >= 0x101u && control <= 0x10fu) {
		return {DppControlKind::RowShiftLeft, control & 0xfu};
	}
	if (control >= 0x111u && control <= 0x11fu) {
		return {DppControlKind::RowShiftRight, control & 0xfu};
	}
	if (control >= 0x121u && control <= 0x12fu) {
		return {DppControlKind::RowRotateRight, control & 0xfu};
	}
	if (control == 0x140u) {
		return {DppControlKind::RowMirror, 0};
	}
	if (control == 0x141u) {
		return {DppControlKind::RowHalfMirror, 0};
	}
	if (control >= 0x150u && control <= 0x15fu) {
		return {DppControlKind::RowShare, control & 0xfu};
	}
	if (control >= 0x160u && control <= 0x16fu) {
		return {DppControlKind::RowXmask, control & 0xfu};
	}
	return {};
}

// The lane a DPP source operand reads for destination `lane`. `valid` is false where a row shift
// runs past the edge of its 16-lane row: the hardware reads nothing there, and `lane` is then
// meaningless. An unsupported control reads the lane itself, as the emitter does.
struct DppSourceLane {
	uint32_t lane  = 0;
	bool     valid = true;
};

[[nodiscard]] constexpr DppSourceLane DppSourceLaneOf(DppControl control, uint32_t lane) {
	const uint32_t row    = lane & ~15u;
	const uint32_t in_row = lane & 15u;
	switch (control.kind) {
		case DppControlKind::Dpp8:
			return {(lane & ~7u) | ((control.operand >> ((lane & 7u) * 3u)) & 7u)};
		case DppControlKind::QuadPerm:
			return {(lane & ~3u) | ((control.operand >> ((lane & 3u) * 2u)) & 3u)};
		case DppControlKind::RowShiftLeft:
			return {row | (in_row + control.operand), in_row < 16u - control.operand};
		case DppControlKind::RowShiftRight:
			return {row | (in_row - control.operand), in_row >= control.operand};
		case DppControlKind::RowRotateRight:
			return {row | (in_row >= control.operand ? in_row - control.operand
			                                         : in_row + 16u - control.operand)};
		case DppControlKind::RowMirror: return {row | (15u - in_row)};
		case DppControlKind::RowHalfMirror: return {(lane & ~7u) | (7u - (lane & 7u))};
		case DppControlKind::RowShare: return {row | control.operand};
		case DppControlKind::RowXmask: return {lane ^ control.operand};
		case DppControlKind::Unsupported: break;
	}
	return {lane};
}

// DPP row_mask/bank_mask: whether the destination `lane` may be written at all. Rows are 16 lanes
// (bits 0-3 of row_mask), banks the four lanes of each quarter row (bits 0-3 of bank_mask).
[[nodiscard]] constexpr bool DppRowBankEnabled(uint32_t row_mask, uint32_t bank_mask,
                                               uint32_t lane) {
	return ((row_mask >> ((lane >> 4u) & 3u)) & 1u) != 0u &&
	       ((bank_mask >> ((lane >> 2u) & 3u)) & 1u) != 0u;
}

// V_PERMLANE16_B32 / V_PERMLANEX16_B32: the lane read for `lane`. Lanes 0-7 of each row take
// their 4-bit selector from sel_lo, lanes 8-15 from sel_hi; x16 reads the other row of the same
// 32-lane half.
[[nodiscard]] constexpr uint32_t Permlane16SourceLane(uint32_t lane, uint32_t sel_lo,
                                                      uint32_t sel_hi, bool x16) {
	const uint32_t row      = (lane & ~15u) ^ (x16 ? 16u : 0u);
	const uint32_t in_row   = lane & 15u;
	const uint32_t selector = in_row >= 8u ? sel_hi : sel_lo;
	return row | ((selector >> ((in_row & 7u) * 4u)) & 15u);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
