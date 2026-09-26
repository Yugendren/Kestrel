#include "common/assert.h"
#include "graphics/shader/recompiler/ir/IREmitter.h"
#include "graphics/shader/recompiler/ir/passes/TileRescale.h"

#include <algorithm>
#include <utility>
#include <vector>

// The rewriting half of the tile-rescale pass (TileRescale.h). Everything it adds follows the
// runtime RescaleControl word, so a program dispatched with a zero word computes exactly what
// it did before the rewrite; the renderer decides per dispatch whether the scaled bindings and
// the invocation remap are in effect.

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

U32 Imm(uint32_t value) {
	return U32(Value(value));
}

Block::iterator IteratorOf(Block& block, const Inst& inst) {
	const auto found =
	    std::ranges::find_if(block, [&](const Inst& candidate) { return &candidate == &inst; });
	EXIT_IF(found == block.end());
	return found;
}

U1 ControlBitSet(IREmitter& ir, uint32_t bit) {
	const auto control = U32(ir.Emit(ValueOpcode::GetRescaleControl));
	return ir.INotEqual(ir.BitwiseAnd(ir.ShiftRightLogical(control, Imm(bit)), Imm(1u)), Imm(0u));
}

// Points one texel access at the scaled image when the renderer bound it scaled. The address is
// rebuilt rather than edited because the translator may share one MakeImageAddress between
// several accesses to different images.
void RescaleTexelAddress(Program& program, Block& block, Inst& access, uint32_t scale_log2) {
	const auto index = access.Flags<MemoryFlags>().index;
	EXIT_IF(index >= program.memory_info.size());
	const auto resource = program.memory_info[index].resource;
	if (resource >= RescaleControl::ImageMaskBits) {
		// The control word cannot mark this image scaled, so it is always bound at guest size.
		return;
	}
	auto* address = access.Arg(1).Resolve().TryInstruction();
	EXIT_IF(address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress);

	IREmitter  ir(&block, IteratorOf(block, access));
	const auto shift = ir.Select(ControlBitSet(ir, resource), Imm(scale_log2), Imm(0u));
	const auto x     = ir.ShiftRightLogical(U32(address->Arg(0)), shift);
	const auto y     = ir.ShiftRightLogical(U32(address->Arg(1)), shift);
	const auto rescaled =
	    ir.Emit(ValueOpcode::MakeImageAddress,
	            {x, y, address->Arg(2), address->Arg(3), address->Arg(4), address->Arg(5),
	             address->Arg(6), address->Arg(7), address->Arg(8), address->Arg(9),
	             address->Arg(10), address->Arg(11), address->Arg(12)},
	            address->Flags<uint64_t>());
	access.SetArg(1, rescaled);
}

// The representative of a k x k block is its top-left guest pixel; moving a float pixel
// position by 0.5*(k-1) puts the shading point at the centre of the scaled texel instead, which
// is where normalised samples of the scaled inputs are centred.
void CentreCoordinateConversion(Inst& conversion, uint32_t scale_log2) {
	const auto op = conversion.GetOpcode();
	EXIT_IF(op != ValueOpcode::ConvertF32U32 && op != ValueOpcode::ConvertF32S32);
	auto&      block    = *conversion.Parent();
	IREmitter  ir(&block, std::next(IteratorOf(block, conversion)));
	const auto original = Value(&conversion);
	const auto offset   = Value::F32(0.5F * static_cast<float>((1u << scale_log2) - 1u));
	const auto centre   = ControlBitSet(ir, RescaleControl::CentreBit);
	const auto shifted  = ir.Emit(ValueOpcode::FPAdd32, {original, offset});
	const auto selected = ir.Emit(ValueOpcode::SelectF32, {centre, shifted, original});
	const auto uses     = conversion.Uses();
	for (const auto& use: uses) {
		if (use.user != shifted.Instruction() && use.user != selected.Instruction()) {
			use.user->SetArg(use.operand, selected);
		}
	}
}

} // namespace

void ApplyTileRescale(Program& program, const TileRescalePlan& plan) {
	if (!plan.accepted) {
		return;
	}
	EXIT_IF(plan.scale_log2 == 0u || plan.scale_log2 >= (1u << RescaleControl::ScaleBits));

	// Collected first: the rewrite inserts instructions into the blocks being walked.
	std::vector<std::pair<Block*, Inst*>> accesses;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			// Sampling and LOD queries take normalised coordinates, which do not depend on the
			// extent of the bound image.
			if (inst.GetOpcode() == ValueOpcode::ImageRead ||
			    inst.GetOpcode() == ValueOpcode::ImageWrite) {
				accesses.emplace_back(block, &inst);
			}
		}
	}
	for (const auto& [block, access]: accesses) {
		RescaleTexelAddress(program, *block, *access, plan.scale_log2);
	}
	for (const auto* conversion: plan.own_coordinate_conversions) {
		CentreCoordinateConversion(*const_cast<Inst*>(conversion), plan.scale_log2);
	}
	program.tile_rescale = {.scale_log2 = plan.scale_log2, .cross_lane = plan.cross_lane};
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
