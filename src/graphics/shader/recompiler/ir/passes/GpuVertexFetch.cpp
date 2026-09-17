#include "graphics/shader/recompiler/ir/passes/GpuVertexFetch.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/Block.h"

#include <algorithm>
#include <bit>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct Address {
	Value low;
	Value high;
};

template <typename Iterator>
Value Emit(Block& block, Iterator at, ValueOpcode opcode, std::initializer_list<Value> args) {
	return Value(&*block.PrependNewInst(at, opcode, args));
}

// 48-bit base plus a 32-bit byte offset, carried into the high word.
template <typename Iterator>
Address AddOffset(Block& block, Iterator at, Value base_low, Value base_high, Value offset) {
	const auto low     = Emit(block, at, ValueOpcode::IAdd32, {base_low, offset});
	const auto wrapped = Emit(block, at, ValueOpcode::ULessThan32, {low, base_low});
	const auto carry   = Emit(block, at, ValueOpcode::SelectU32, {wrapped, Value(1u), Value(0u)});
	const auto high    = Emit(block, at, ValueOpcode::IAdd32, {base_high, carry});
	return {low, high};
}

MemoryFlags AddFlatMemory(Program& program, const MemoryInfo& source, uint32_t pc) {
	auto info            = source;
	info.kind            = ResourceKind::Flat;
	info.address_is_full = true;
	info.data_dwords     = 1u;
	info.sampler         = 0u;
	// The emitter unpacks the attribute's component_index from the record; a raw or table
	// read is one element whose position is already folded into the address.
	if (info.formatted && info.typed) {
		info.component_count = info.component_index + 1u;
	} else {
		info.component_index = 0u;
		info.component_count = 1u;
	}
	info.offset        = 0u;
	info.planning_only = false;
	const MemoryFlags flags {.index = static_cast<uint32_t>(program.memory_info.size()), .pc = pc};
	program.memory_info.push_back(info);
	return flags;
}

template <typename Iterator>
Value FlatLoad(Program& program, Block& block, Iterator at, ValueOpcode opcode,
               const Address& address, Value active, const MemoryInfo& source, uint32_t pc) {
	const auto handle =
	    Emit(block, at, ValueOpcode::GetAddressResource, {address.low, address.high});
	return Value(&*block.PrependNewInst(at, opcode, {handle, address.low, address.high, active},
	                                    std::bit_cast<uint64_t>(AddFlatMemory(program, source, pc))));
}

bool IsTableLoad(const Program& program, const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::LoadAddressU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	return index < program.memory_info.size() &&
	       program.memory_info[index].vertex_fetch == VertexFetchRole::Table;
}

// A descriptor word that is computed, however indirectly, from a table read only exists on the
// GPU, so every read through it has to go through the page table as well.
bool ReachesTableLoad(const Program& program, Value value, uint32_t depth) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr || depth > 24u) {
		return false;
	}
	if (IsTableLoad(program, *inst)) {
		return true;
	}
	if (inst->GetOpcode() == ValueOpcode::LoadAddressU32 ||
	    inst->GetOpcode() == ValueOpcode::ReadConstBuffer) {
		return false;
	}
	for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
		if (ReachesTableLoad(program, inst->Arg(arg), depth + 1u)) {
			return true;
		}
	}
	return false;
}

bool DescriptorReachesTable(const Program& program, const Inst* handle) {
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
	    handle->NumArgs() != 4u) {
		return false;
	}
	for (uint32_t dword = 0; dword < 4u; dword++) {
		if (ReachesTableLoad(program, handle->Arg(dword), 0u)) {
			return true;
		}
	}
	return false;
}

// A scalar table read is a ScalarAddress load: {GetAddressResource(low, high), byte offset,
// 0, active} plus an immediate in its MemoryInfo. Rebase it onto the full guest address so the
// SRT planner leaves it in the shader.
bool LowerTableLoad(Program& program, Inst& inst) {
	const auto  flags  = inst.Flags<MemoryFlags>();
	const auto& memory = program.memory_info[flags.index];
	if (memory.kind != ResourceKind::ScalarAddress || inst.NumArgs() < 4u) {
		return false;
	}
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
	    handle->NumArgs() != 2u) {
		return false;
	}
	auto&      block = *inst.Parent();
	const auto at = std::ranges::find_if(block, [&](const Inst& other) { return &other == &inst; });
	auto offset = Emit(block, at, ValueOpcode::BitwiseAnd32, {inst.Arg(1), Value(~3u)});
	if ((memory.offset & ~3u) != 0u) {
		offset = Emit(block, at, ValueOpcode::IAdd32, {offset, Value(memory.offset & ~3u)});
	}
	const auto address = AddOffset(block, at, handle->Arg(0), handle->Arg(1), offset);
	inst.ReplaceUsesWith(FlatLoad(program, block, at, ValueOpcode::LoadAddressU32, address,
	                              inst.Arg(3), memory, flags.pc));
	return true;
}

// A scalar buffer read through a table-derived V#: {GetBufferResource(v0..v3), dword offset}.
bool LowerScalarBufferRead(Program& program, Inst& inst) {
	const auto  flags  = inst.Flags<MemoryFlags>();
	const auto& memory = program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	auto&       block  = *inst.Parent();
	const auto at = std::ranges::find_if(block, [&](const Inst& other) { return &other == &inst; });
	const auto base_high = Emit(block, at, ValueOpcode::BitwiseAnd32, {handle->Arg(1), Value(0xffffu)});
	auto       offset    = inst.Arg(1);
	if (memory.offset != 0u) {
		offset = Emit(block, at, ValueOpcode::IAdd32, {offset, Value(memory.offset)});
	}
	const auto address = AddOffset(block, at, handle->Arg(0), base_high, offset);
	MemoryInfo raw;
	raw.data_bits = 32u;
	inst.ReplaceUsesWith(FlatLoad(program, block, at, ValueOpcode::LoadAddressU32, address,
	                              Value(true), raw, flags.pc));
	return true;
}

// A vector buffer load through a table-derived V#: {GetBufferResource(v0..v3), index, offset,
// soffset, active}. Reproduce the hardware address: base + index * stride + offset + soffset,
// records bounded by NUM_RECORDS (stride 0 counts bytes), out of range reads zero. A load the
// translator marked as a vertex attribute carries its baked format; a typed load carries its
// own; anything else is read raw dword by dword.
bool LowerBufferLoad(Program& program, Inst& inst) {
	const auto  flags  = inst.Flags<MemoryFlags>();
	const auto& memory = program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	if (inst.NumArgs() != 5u) {
		return false;
	}
	if (memory.formatted && !memory.typed && memory.vertex_fetch != VertexFetchRole::Attribute) {
		EXIT("GPU vertex fetch: formatted buffer load at pc 0x%08x takes its format from a "
		     "descriptor the CPU never sees\n",
		     flags.pc);
	}
	auto&      block = *inst.Parent();
	const auto at = std::ranges::find_if(block, [&](const Inst& other) { return &other == &inst; });
	const auto emit  = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
		return Emit(block, at, opcode, args);
	};
	const auto dword1    = handle->Arg(1);
	const auto dword3    = handle->Arg(3);
	const auto base_low  = handle->Arg(0);
	const auto base_high = emit(ValueOpcode::BitwiseAnd32, {dword1, Value(0xffffu)});
	const auto stride    = emit(ValueOpcode::BitwiseAnd32,
	                            {emit(ValueOpcode::ShiftRightLogical32, {dword1, Value(16u)}),
	                             Value(0x3fffu)});
	const auto records   = handle->Arg(2);
	const auto index     = inst.Arg(1);
	auto       byte_offset = emit(ValueOpcode::IAdd32, {inst.Arg(2), inst.Arg(3)});
	if (memory.offset != 0u) {
		byte_offset = emit(ValueOpcode::IAdd32, {byte_offset, Value(memory.offset)});
	}

	const auto components = memory.data_dwords;
	const auto format     = memory.formatted && memory.typed
	                            ? Format::GetFormatInfo(Format::DecodeTBufferFormat(
	                                  memory.data_format, memory.number_format))
	                            : Format::BufferFormatInfo {};
	// A typed format the decoder does not know reads as raw dwords, like the descriptor path.
	const bool formatted = format.type != Format::ComponentType::Unknown;
	const auto extent    = formatted ? format.byte_size : components * (memory.data_bits / 8u);
	// RDNA2 range checking follows OOB_SELECT (V# dword 3 bits 28-29): 0 and 1 bound the record
	// index, 2 only requires a non-empty buffer, 3 bounds the byte offset. A zero stride is
	// always a raw byte check. The offset-within-record clause of mode 0 is not applied, so a
	// read the hardware would reject as straddling its record still returns data.
	const auto structured = emit(ValueOpcode::INotEqual32, {stride, Value(0u)});
	const auto index_ok   = emit(ValueOpcode::ULessThan32, {index, records});
	const auto raw_end    = emit(ValueOpcode::IAdd32, {byte_offset, Value(std::max(extent, 1u))});
	const auto raw_ok     = emit(ValueOpcode::LogicalOr,
	                             {emit(ValueOpcode::ULessThan32, {raw_end, records}),
	                              emit(ValueOpcode::IEqual32, {raw_end, records})});
	const auto non_empty  = emit(ValueOpcode::INotEqual32, {records, Value(0u)});
	const auto oob_select = emit(ValueOpcode::BitwiseAnd32,
	                             {emit(ValueOpcode::ShiftRightLogical32, {dword3, Value(28u)}),
	                              Value(3u)});
	const auto oob_raw    = emit(ValueOpcode::IEqual32, {oob_select, Value(3u)});
	const auto oob_any    = emit(ValueOpcode::IEqual32, {oob_select, Value(2u)});
	const auto structured_ok = emit(ValueOpcode::SelectU1,
	                                {oob_raw, raw_ok,
	                                 emit(ValueOpcode::SelectU1, {oob_any, non_empty, index_ok})});
	const auto in_bounds  = emit(ValueOpcode::SelectU1, {structured, structured_ok, raw_ok});
	const auto active     = emit(ValueOpcode::LogicalAnd, {inst.Arg(4), in_bounds});
	const auto element    = emit(ValueOpcode::IAdd32,
	                             {emit(ValueOpcode::IMul32, {index, stride}), byte_offset});
	const auto address    = AddOffset(block, at, base_low, base_high, element);

	if (memory.vertex_fetch == VertexFetchRole::Attribute) {
		// The program was compiled against one format and destination select; if the guest
		// hands the shader a different V# (or a swizzled one), touch the drift page so the
		// fault pass can report it and the program gets recompiled against the new table.
		const auto check   = emit(ValueOpcode::BitwiseAnd32, {dword3, Value(VertexFetchCheckMask)});
		const auto swizzle = emit(ValueOpcode::ShiftRightLogical32, {dword1, Value(31u)});
		const auto drift   = emit(ValueOpcode::LogicalOr,
		                          {emit(ValueOpcode::INotEqual32, {check, Value(memory.vertex_check)}),
		                           emit(ValueOpcode::INotEqual32, {swizzle, Value(0u)})});
		const auto    drift_active = emit(ValueOpcode::LogicalAnd, {inst.Arg(4), drift});
		MemoryInfo    drift_memory;
		drift_memory.data_bits = 32u;
		const Address drift_address {Value(static_cast<uint32_t>(VertexFetchDriftAddress)),
		                             Value(static_cast<uint32_t>(VertexFetchDriftAddress >> 32u))};
		const auto drift_load = FlatLoad(program, block, at, ValueOpcode::LoadAddressU32,
		                                 drift_address, drift_active, drift_memory, flags.pc);
		block.PrependNewInst(at, ValueOpcode::ReferenceU32, {drift_load});
	}

	std::vector<Value> loaded;
	loaded.reserve(components);
	for (uint32_t component = 0; component < components; component++) {
		auto element_memory = memory;
		if (formatted) {
			// One formatted component per load; the emitter adds its byte offset within the
			// record from the format.
			element_memory.component_index =
			    memory.vertex_fetch == VertexFetchRole::Attribute ? memory.component_index
			                                                      : component;
			loaded.push_back(FlatLoad(program, block, at, ValueOpcode::LoadAddressU32, address,
			                          active, element_memory, flags.pc));
			continue;
		}
		element_memory.formatted = false;
		element_memory.typed     = false;
		const auto opcode = memory.data_bits == 8u    ? ValueOpcode::LoadAddressU8
		                    : memory.data_bits == 16u ? ValueOpcode::LoadAddressU16
		                                              : ValueOpcode::LoadAddressU32;
		const auto element_address =
		    component == 0u ? address
		                    : AddOffset(block, at, address.low, address.high, Value(component * 4u));
		loaded.push_back(FlatLoad(program, block, at, opcode, element_address, active,
		                          element_memory, flags.pc));
	}
	Value result = loaded.front();
	if (components == 2u) {
		result = emit(ValueOpcode::CompositeConstructU32x2, {loaded[0], loaded[1]});
	} else if (components == 3u) {
		result = emit(ValueOpcode::CompositeConstructU32x3, {loaded[0], loaded[1], loaded[2]});
	} else if (components == 4u) {
		result = emit(ValueOpcode::CompositeConstructU32x4,
		              {loaded[0], loaded[1], loaded[2], loaded[3]});
	}
	inst.ReplaceUsesWith(result);
	return true;
}

bool IsBufferLoad(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::LoadBufferU8:
		case ValueOpcode::LoadBufferU16:
		case ValueOpcode::LoadBufferU32:
		case ValueOpcode::LoadBufferU32x2:
		case ValueOpcode::LoadBufferU32x3:
		case ValueOpcode::LoadBufferU32x4: return true;
		default: return false;
	}
}

} // namespace

uint32_t LowerGpuVertexFetch(Program& program) {
	std::vector<Inst*> tables;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (IsTableLoad(program, inst)) {
				tables.push_back(&inst);
			}
		}
	}
	if (tables.empty()) {
		return 0;
	}
	for (auto* inst: tables) {
		LowerTableLoad(program, *inst);
	}
	std::vector<Inst*> loads;
	std::vector<Inst*> scalar_reads;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			if (opcode != ValueOpcode::ReadConstBuffer && !IsBufferLoad(opcode)) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				continue;
			}
			const auto* handle = inst.Arg(0).Resolve().TryInstruction();
			const bool  marked = program.memory_info[index].vertex_fetch == VertexFetchRole::Attribute;
			if (!marked && !DescriptorReachesTable(program, handle)) {
				continue;
			}
			if (opcode == ValueOpcode::ReadConstBuffer) {
				scalar_reads.push_back(&inst);
			} else {
				loads.push_back(&inst);
			}
		}
	}
	uint32_t lowered = 0;
	for (auto* inst: scalar_reads) {
		if (LowerScalarBufferRead(program, *inst)) {
			lowered++;
		}
	}
	for (auto* inst: loads) {
		if (LowerBufferLoad(program, *inst)) {
			lowered++;
		}
	}
	program.info.uses_dma = true;
	return lowered;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
