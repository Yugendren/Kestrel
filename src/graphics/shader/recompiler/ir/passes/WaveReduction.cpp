#include "graphics/shader/recompiler/ir/passes/WaveReduction.h"

#include "graphics/shader/recompiler/ir/LaneAddressing.h"

#include <array>
#include <bit>
#include <map>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// Chains longer than this are not scans the compiler emits (a wave64 Kogge-Stone scan with its
// exec-merge selects is about 20 nodes); give up rather than walk arbitrary dataflow.
constexpr size_t MaxChainNodes = 64;

// A chain value, lane by lane: lanes[l] is the set of leaf lanes whose values are combined into
// lane l. A bit in `unknown` marks a lane whose value is not such a combination -- it received a
// literal zero that is not the op's identity, a fetch past a row edge, or (under a
// non-idempotent op) some leaf lane twice. An unknown lane only matters if it reaches the lane
// that is read.
struct LaneTable {
	std::array<uint64_t, 64> lanes {};
	uint64_t                 unknown = 0;

	[[nodiscard]] bool IsUnknown(uint32_t lane) const { return ((unknown >> lane) & 1u) != 0u; }

	void SetUnknown(uint32_t lane) {
		lanes[lane] = 0;
		unknown |= uint64_t {1} << lane;
	}

	void CopyLane(uint32_t lane, const LaneTable& source, uint32_t source_lane) {
		lanes[lane] = source.lanes[source_lane];
		if (source.IsUnknown(source_lane)) {
			unknown |= uint64_t {1} << lane;
		}
	}
};

std::optional<WaveReduceOp> CombineOpOf(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::BitwiseOr32: return WaveReduceOp::BitwiseOr;
		case ValueOpcode::BitwiseAnd32: return WaveReduceOp::BitwiseAnd;
		case ValueOpcode::BitwiseXor32: return WaveReduceOp::BitwiseXor;
		case ValueOpcode::IAdd32: return WaveReduceOp::IAdd;
		case ValueOpcode::UMin32: return WaveReduceOp::UMin;
		case ValueOpcode::UMax32: return WaveReduceOp::UMax;
		case ValueOpcode::SMin32: return WaveReduceOp::SMin;
		case ValueOpcode::SMax32: return WaveReduceOp::SMax;
		default: return std::nullopt;
	}
}

// x op x == x: a lane may reach a total along several paths without changing it.
bool IsIdempotent(WaveReduceOp op) {
	return op != WaveReduceOp::BitwiseXor && op != WaveReduceOp::IAdd;
}

bool IsNegationOf(Value negated, Value value) {
	const auto* inst = negated.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot &&
	       inst->Arg(0).Resolve() == value;
}

// Whether a U1 lane mask is true in every lane of the guest wave: the immediate true (what
// S_OR_SAVEEXEC ..., -1 folds to), or x | !x in either order (S_ORN2_SAVEEXEC_B64 s, exec, which
// translation leaves as LogicalOr(LogicalNot(exec), exec)).
bool IsAllLanesTrue(Value mask) {
	mask = mask.Resolve();
	if (mask.IsImmediate()) {
		return mask.GetType() == Type::U1 && mask.U1();
	}
	const auto* inst = mask.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LogicalOr) {
		return false;
	}
	const auto lhs = inst->Arg(0).Resolve();
	const auto rhs = inst->Arg(1).Resolve();
	return IsNegationOf(lhs, rhs) || IsNegationOf(rhs, lhs);
}

bool ImmediateU32(Value value, uint32_t& result) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) {
		return false;
	}
	result = value.U32();
	return true;
}

// Evaluates the lane tables of one ReadLane's source chain. Every node kind mirrors what the
// SPIR-V emitter executes for it, restricted to an all-true exec:
//  - op(A, C), one integer op for the whole chain: lane l combines A[l] and C[l];
//  - DppMoveU32(A): lane l reads A at its DPP source lane; past a row edge it is a literal zero
//    (an empty set when zero is op's identity);
//  - DppUpdateU32(N, Old): lanes the DPP write enables (row/bank mask, bound_ctrl or a valid
//    source) take N, the others keep Old;
//  - Permlane16U32(A, sel_lo, sel_hi) with immediate selects: lane l reads A at its source lane;
//  - SelectU32(all-true, a, b): the exec merge of a full-wave write, which is a.
// Any other value is the leaf; all leaves must be the same value, and not a phi (the chain is
// straight-line code; a phi would mean a loop-carried or merged value this analysis does not
// follow).
class ChainEvaluator {
public:
	explicit ChainEvaluator(uint32_t wave_size): wave_size(wave_size) {
		for (uint32_t lane = 0; lane < wave_size; lane++) {
			leaf_table.lanes[lane] = uint64_t {1} << lane;
		}
	}

	std::optional<WaveReduction> Evaluate(Value source, uint32_t lane) {
		const auto* table = Table(source);
		if (table == nullptr || !op || table->IsUnknown(lane) ||
		    std::popcount(table->lanes[lane]) < 2) {
			return std::nullopt;
		}
		return WaveReduction {.leaf = leaf, .op = *op, .lanes = table->lanes[lane]};
	}

private:
	[[nodiscard]] static bool IsChainNode(const Inst& inst) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::DppMoveU32:
			case ValueOpcode::DppUpdateU32:
			case ValueOpcode::Permlane16U32: return true;
			case ValueOpcode::SelectU32: return IsAllLanesTrue(inst.Arg(0));
			default: return CombineOpOf(inst.GetOpcode()).has_value();
		}
	}

	// The table of `value`, or nullptr when the chain is not one this analysis models. Tables
	// live in a node-based map, so returned pointers stay valid while others are added; `nodes`
	// counts every node entered, which also bounds the recursion depth.
	const LaneTable* Table(Value value) {
		value      = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || !IsChainNode(*inst)) {
			if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
				return nullptr;
			}
			if (leaf.IsEmpty()) {
				leaf = value;
			}
			return leaf == value ? &leaf_table : nullptr;
		}
		if (const auto found = tables.find(inst); found != tables.end()) {
			return &found->second;
		}
		if (++nodes > MaxChainNodes) {
			return nullptr;
		}
		auto table = Build(*inst);
		if (!table) {
			return nullptr;
		}
		return &tables.emplace(inst, *table).first->second;
	}

	std::optional<LaneTable> Build(const Inst& inst) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::DppMoveU32: return BuildDppMove(inst);
			case ValueOpcode::DppUpdateU32: return BuildDppUpdate(inst);
			case ValueOpcode::Permlane16U32: return BuildPermlane16(inst);
			case ValueOpcode::SelectU32: {
				const auto* selected = Table(inst.Arg(1));
				return selected != nullptr ? std::optional {*selected} : std::nullopt;
			}
			default: return BuildCombine(inst);
		}
	}

	std::optional<LaneTable> BuildCombine(const Inst& inst) {
		const auto inst_op = CombineOpOf(inst.GetOpcode());
		if (!op) {
			op = inst_op;
		} else if (op != inst_op) {
			return std::nullopt;
		}
		const auto* lhs = Table(inst.Arg(0));
		const auto* rhs = lhs != nullptr ? Table(inst.Arg(1)) : nullptr;
		if (rhs == nullptr) {
			return std::nullopt;
		}
		const bool idempotent = IsIdempotent(*op);
		LaneTable  table;
		for (uint32_t lane = 0; lane < wave_size; lane++) {
			if (lhs->IsUnknown(lane) || rhs->IsUnknown(lane) ||
			    (!idempotent && (lhs->lanes[lane] & rhs->lanes[lane]) != 0u)) {
				table.SetUnknown(lane);
			} else {
				table.lanes[lane] = lhs->lanes[lane] | rhs->lanes[lane];
			}
		}
		return table;
	}

	std::optional<LaneTable> BuildDppMove(const Inst& inst) {
		const auto flags   = inst.Flags<DppMoveFlags>();
		const auto control = DecodeDppControl(flags.control, flags.dpp8);
		if (control.kind == DppControlKind::Unsupported || !IsAllLanesTrue(inst.Arg(1))) {
			return std::nullopt;
		}
		const auto* source = Table(inst.Arg(0));
		if (source == nullptr) {
			return std::nullopt;
		}
		// The emitter writes a literal zero where the source lane is out of its row (and fetches
		// an undefined lane with fetch_inactive); zero is an empty combination only if it is the
		// op's identity.
		const bool zero_is_empty =
		    !flags.fetch_inactive && op && WaveReduceIdentity(*op) == 0u;
		LaneTable table;
		for (uint32_t lane = 0; lane < wave_size; lane++) {
			const auto from = DppSourceLaneOf(control, lane);
			if (from.valid && from.lane < wave_size) {
				table.CopyLane(lane, *source, from.lane);
			} else if (from.valid || !zero_is_empty) {
				table.SetUnknown(lane);
			}
		}
		return table;
	}

	std::optional<LaneTable> BuildDppUpdate(const Inst& inst) {
		const auto flags   = inst.Flags<DppMoveFlags>();
		const auto control = DecodeDppControl(flags.control, flags.dpp8);
		if (control.kind == DppControlKind::Unsupported || !IsAllLanesTrue(inst.Arg(2))) {
			return std::nullopt;
		}
		const auto* written = Table(inst.Arg(0));
		const auto* old     = written != nullptr ? Table(inst.Arg(1)) : nullptr;
		if (old == nullptr) {
			return std::nullopt;
		}
		LaneTable table;
		for (uint32_t lane = 0; lane < wave_size; lane++) {
			const bool write = DppRowBankEnabled(flags.row_mask, flags.bank_mask, lane) &&
			                   (flags.bound_control || DppSourceLaneOf(control, lane).valid);
			table.CopyLane(lane, write ? *written : *old, lane);
		}
		return table;
	}

	std::optional<LaneTable> BuildPermlane16(const Inst& inst) {
		const auto flags  = inst.Flags<PermlaneFlags>();
		uint32_t   sel_lo = 0;
		uint32_t   sel_hi = 0;
		if (!ImmediateU32(inst.Arg(1), sel_lo) || !ImmediateU32(inst.Arg(2), sel_hi) ||
		    !IsAllLanesTrue(inst.Arg(3))) {
			return std::nullopt;
		}
		const auto* source = Table(inst.Arg(0));
		if (source == nullptr) {
			return std::nullopt;
		}
		LaneTable table;
		for (uint32_t lane = 0; lane < wave_size; lane++) {
			const auto from = Permlane16SourceLane(lane, sel_lo, sel_hi, flags.x16);
			if (from < wave_size) {
				table.CopyLane(lane, *source, from);
			} else {
				table.SetUnknown(lane);
			}
		}
		return table;
	}

	uint32_t                         wave_size;
	LaneTable                        leaf_table;
	Value                            leaf;
	std::optional<WaveReduceOp>      op;
	std::map<const Inst*, LaneTable> tables;
	size_t                           nodes = 0;
};

} // namespace

std::optional<WaveReduction> FindWaveReduction(const Inst& read, uint32_t wave_size) {
	uint32_t lane = 0;
	if ((wave_size != 32u && wave_size != 64u) || read.GetOpcode() != ValueOpcode::ReadLane ||
	    !ImmediateU32(read.Arg(1), lane) || lane >= wave_size) {
		return std::nullopt;
	}
	return ChainEvaluator(wave_size).Evaluate(read.Arg(0), lane);
}

WaveReductionStats RecoverWaveReductions(Program& program, uint32_t wave_size) {
	WaveReductionStats stats;
	for (auto* block: program.blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			const auto reduction = FindWaveReduction(*inst, wave_size);
			if (!reduction) {
				continue;
			}
			const auto reduce = block->PrependNewInst(
			    inst, ValueOpcode::WaveReduceU32,
			    {reduction->leaf, Value(static_cast<uint32_t>(reduction->lanes)),
			     Value(static_cast<uint32_t>(reduction->lanes >> 32u))});
			reduce->SetFlags(WaveReduceFlags {.op = reduction->op});
			inst->ReplaceUsesWith(Value(&*reduce));
			stats.rewritten_reads++;
		}
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
