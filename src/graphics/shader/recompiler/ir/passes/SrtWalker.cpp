#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

// KYTY_SRT_DIAG=1 dumps the opcode tree of a descriptor dword that failed bind-time evaluation
// (the "a descriptor source did not evaluate" drop). Zero cost unless the env var is set.
// True when an address expression is carried around a loop, so it has no value before the draw.
bool AddressReachesPhi(Value value, std::vector<const Inst*>& visited, uint32_t depth) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr || depth > 64u) {
		return false;
	}
	if (inst->GetOpcode() == ValueOpcode::Phi) {
		return true;
	}
	if (std::ranges::find(visited, inst) != visited.end()) {
		return false;
	}
	visited.push_back(inst);
	for (size_t index = 0; index < inst->NumArgs(); index++) {
		if (AddressReachesPhi(inst->Arg(index), visited, depth + 1u)) {
			return true;
		}
	}
	return false;
}

bool SrtDiagEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SRT_DIAG");
		return v != nullptr && v[0] != '0';
	}();
	return on;
}

std::string DescribeValueTree(const ResourcePlan& program, Value value, uint32_t depth = 0) {
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return value.IsImmediate() && value.GetType() == Type::U32
		           ? fmt::format("0x{:08x}", value.U32())
		           : std::string("opaque");
	}
	const auto op   = inst->GetOpcode();
	auto       text = std::string(ValueOpcodeName(op));
	if (op == ValueOpcode::GetUserData && inst->NumArgs() == 1 &&
	    inst->Arg(0).GetType() == Type::ScalarReg) {
		return text + fmt::format(" s{}", RegIndex(inst->Arg(0).ScalarRegister()));
	}
	if (op == ValueOpcode::ReadConst && inst->NumArgs() == 2 &&
	    inst->Arg(1).Resolve().IsImmediate()) {
		return text + fmt::format(" slot={}", inst->Arg(1).Resolve().U32());
	}
	if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
		const auto flags = inst->Flags<MemoryFlags>();
		text += fmt::format(" pc=0x{:08x}", flags.pc);
		if (flags.index < program.memory_info.size()) {
			text += fmt::format(" offset=0x{:x}", program.memory_info[flags.index].offset);
		}
	}
	if (inst->NumArgs() == 0 || depth >= 5u) {
		return text;
	}
	text += '(';
	for (size_t index = 0; index < inst->NumArgs(); index++) {
		text += index == 0 ? "" : ", ";
		text += DescribeValueTree(program, inst->Arg(index), depth + 1u);
	}
	return text + ')';
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::BitCount32:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

	const std::string& Reason() const { return m_reason; }

private:
	bool Reject(std::string reason) {
		if (m_reason.empty()) {
			m_reason = std::move(reason);
		}
		return false;
	}

	std::string Describe(Value value, uint32_t depth = 0) const {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return value.IsImmediate() && value.GetType() == Type::U32
			           ? fmt::format("0x{:08x}", value.U32())
			           : std::string("opaque");
		}
		const auto op   = inst->GetOpcode();
		auto       text = std::string(ValueOpcodeName(op));
		if (op == ValueOpcode::GetUserData && inst->NumArgs() == 1 &&
		    inst->Arg(0).GetType() == Type::ScalarReg) {
			return text + fmt::format(" s{}", RegIndex(inst->Arg(0).ScalarRegister()));
		}
		if (op == ValueOpcode::ReadConst && inst->NumArgs() == 2 &&
		    inst->Arg(1).Resolve().IsImmediate()) {
			return text + fmt::format(" slot={}", inst->Arg(1).Resolve().U32());
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto flags = inst->Flags<MemoryFlags>();
			text += fmt::format(" pc=0x{:08x}", flags.pc);
			if (flags.index < m_program.memory_info.size()) {
				text += fmt::format(" offset=0x{:x}", m_program.memory_info[flags.index].offset);
			}
			return text;
		}
		if (inst->NumArgs() == 0 || depth >= 3u) {
			return text;
		}
		text += '(';
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			text += index == 0 ? "" : ", ";
			text += Describe(inst->Arg(index), depth + 1u);
		}
		return text + ')';
	}

	std::string DescribePhi(Value value) const {
		std::vector<Value>              leaves;
		std::vector<Value>              pending {value};
		std::unordered_set<const Inst*> seen;
		while (!pending.empty()) {
			const auto current = pending.back().Resolve();
			pending.pop_back();
			const auto* inst = current.TryInstruction();
			if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
				if (seen.insert(inst).second) {
					for (size_t index = 0; index < inst->NumArgs(); index++) {
						pending.push_back(inst->Arg(index));
					}
				}
				continue;
			}
			if (std::ranges::none_of(leaves, [&](Value known) {
				    return EquivalentValue(m_program, known, current);
			    })) {
				leaves.push_back(current);
			}
		}
		auto text = fmt::format("Phi merges {} unequal values:", leaves.size());
		for (size_t index = 0; index < leaves.size() && index < 6u; index++) {
			text += fmt::format(" [{}] {}", index, Describe(leaves[index]));
		}
		return text;
	}

	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return Reject("operand is not an integer");
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			if (require_uniform) {
				Reject(fmt::format("{} is cyclic", ValueOpcodeName(inst->GetOpcode())));
			}
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			if (!valid && require_uniform) {
				Reject(fmt::format("{} cannot be evaluated before the draw",
				                   ValueOpcodeName(inst->GetOpcode())));
			}
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				Reject(DescribePhi(value));
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x4) {
			// A 4-dword descriptor (V#/S#) the shader assembles in registers and then indexes,
			// e.g. Team Asobi compute shaders building a sampler from CompositeConstructU32x4.
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 4u ||
			    source->GetOpcode() != ValueOpcode::CompositeConstructU32x4) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
	std::string                     m_reason;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		// A flat slot is read once before the draw, so it only works when the whole read can be
		// evaluated then. An immediate offset is not enough: if the base address is carried
		// around a loop -- a pointer walked by the shader, as ray-tracing kernels do over a BVH
		// -- there is no address to read from yet, and giving it a slot only guarantees the
		// evaluation fails later and the dispatch is dropped. Leave those as runtime reads.
		std::vector<const Inst*> visited;
		if (!offset.IsImmediate() || offset.GetType() != Type::U32 ||
		    AddressReachesPhi(inst->Arg(0), visited, 0u)) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		m_patches.push_back({inst, slot, true});
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
};

float Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

uint64_t Float32Bits(float value) {
	return std::bit_cast<uint32_t>(value);
}

// Operand count of an opcode whose result depends on nothing but its operand values, all of them
// evaluated in the walk the instruction is in, or zero. SrtWalker and CompiledSrtPlan both apply
// these through ApplyPureOp, so the two cannot drift apart arithmetically.
uint32_t PureOpArity(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::LogicalNot: return 1;
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalXor: return 2;
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: return 3;
		case ValueOpcode::BitFieldInsert: return 4;
		default: return 0;
	}
}

// Applies a PureOpArity opcode to already-evaluated operands. False means the operation has no
// defined value for these operands (for example an out-of-range bitfield).
bool ApplyPureOp(ValueOpcode op, const uint64_t* args, uint64_t& result) {
	const uint64_t a = args[0];
	const uint64_t b = args[1];
	const uint64_t c = args[2];
	const uint64_t d = args[3];
	switch (op) {
		case ValueOpcode::CompositeConstructU64:
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::IMul64: result = a * b; return true;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = Float32Bits(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			const auto value = Float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32: result = Float32Bits(Float32(a) * Float32(b)); return true;
		case ValueOpcode::FPTrunc32: result = Float32Bits(std::trunc(Float32(a))); return true;
		case ValueOpcode::FPIsNan32: result = std::isnan(Float32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32: result = Float32(a) <= Float32(b); return true;
		case ValueOpcode::FPOrdGreaterThanEqual32: result = Float32(a) >= Float32(b); return true;
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::BitCount32:
			result = static_cast<uint32_t>(std::popcount(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::FindILsb32: {
			const auto bits = static_cast<uint32_t>(a);
			result = bits == 0u ? UINT32_MAX : static_cast<uint32_t>(std::countr_zero(bits));
			return true;
		}
		case ValueOpcode::FindUMsb32: {
			const auto bits = static_cast<uint32_t>(a);
			result = bits == 0u ? UINT32_MAX : static_cast<uint32_t>(31 - std::countl_zero(bits));
			return true;
		}
		case ValueOpcode::ShiftLeftLogical32: result = static_cast<uint32_t>(a) << (b & 31u); return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32: result = static_cast<uint32_t>(a) >> (b & 31u); return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32:
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::INotEqual32:
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThan32:
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32:
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::LogicalAnd: result = (a != 0u) && (b != 0u); return true;
		case ValueOpcode::LogicalOr: result = (a != 0u) || (b != 0u); return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		default: return false;
	}
}

// Base address of a raw scalar read. A V#/T# whose base pointer resolved to null is an unbound
// (optional) descriptor slot; on real hardware a read through it returns all-zero rather than
// faulting, so both evaluators resolve it to 0 instead of failing the whole materialisation (which
// would drop the dispatch). Correct emulation, not a soft-ladder.
uint64_t RawReadBase(uint64_t low, uint64_t high) {
	return ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
}

// Guest address of a scalar-buffer (ReadConstBuffer) dword for a non-null base, or false when the
// dword lies outside the V#'s records.
bool ScalarBufferReadAddress(uint64_t base, uint64_t high, int64_t immediate, uint64_t offset,
                             uint64_t records, uint64_t& address) {
	if (immediate < 0) {
		return false;
	}
	const auto byte_offset = static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
	const auto aligned     = byte_offset & ~uint64_t {3};
	const auto stride      = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
	const auto size        = stride == 0u
	                             ? static_cast<uint64_t>(static_cast<uint32_t>(records))
	                             : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
	if (aligned > size || size - aligned < sizeof(uint32_t)) {
		return false;
	}
	address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	return true;
}

// Guest address of a scalar-address (LoadAddressU32) dword for a non-null base.
bool ScalarAddressReadAddress(uint64_t base, int64_t immediate, uint64_t offset, uint64_t& address) {
	const auto relative =
	    (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
	return AddSignedAddress(base & ~uint64_t {3}, relative, address);
}

bool ReadSrtWord(SrtMemoryReader reader, void* userdata, uint64_t address, uint64_t& result) {
	uint32_t word = 0;
	if (reader != nullptr) {
		if (!reader(userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {}

SrtWalker::~SrtWalker() { --m_program.evaluation_depth; }

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	if (m_compiled != nullptr) {
		if (const auto* root = m_compiled->FindValueRoot(value, m_compiled_clean)) {
			return m_compiled->Run(*root, m_runtime, result);
		}
	}
	return Interpret(value, result);
}

bool SrtWalker::Interpret(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = Float32Bits(value.F32Value()); return true;
			default: return false;
		}
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		if (m_diag_first_fail == nullptr) {
			m_diag_first_fail = inst;
		}
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	const auto max_component =
	    inst.GetOpcode() == ValueOpcode::CompositeExtractU32x4 ? 4u : 2u;
	if (component >= max_component) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2 ||
	    source->GetOpcode() == ValueOpcode::CompositeConstructU32x4) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base = RawReadBase(low, high);
	if (base == 0) {
		result = 0;
		return true;
	}
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (!ScalarBufferReadAddress(base, high, immediate, offset, records, address)) {
			return false;
		}
	} else if (!ScalarAddressReadAddress(base, immediate, offset, address)) {
		return false;
	}
	return ReadSrtWord(m_runtime.read_memory, m_runtime.userdata, address, result);
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	if (const auto arity = PureOpArity(inst.GetOpcode()); arity != 0u) {
		uint64_t args[4] = {};
		for (uint32_t index = 0; index < arity; index++) {
			if (!Arg(inst, index, args[index])) {
				return false;
			}
		}
		return ApplyPureOp(inst.GetOpcode(), args, result);
	}
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x4: return EvaluateExtract(inst, result);
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			if (IsRawRead(m_program, inst)) {
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			uint64_t predicate_value = 0;
			auto&    predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), predicate_value)) {
				return Arg(inst, predicate_value != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}

bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	const auto* roots =
	    m_compiled != nullptr ? m_compiled->DescriptorRoots(source, m_compiled_clean) : nullptr;
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		m_diag_first_fail = nullptr;
		const bool evaluated =
		    roots != nullptr ? m_compiled->Run(roots[index], m_runtime, result.dwords[index])
		                     : Interpret(descriptor.dwords[index], result.dwords[index]);
		if (!evaluated) {
			if (SrtDiagEnabled()) {
				if (roots != nullptr) {
					// A compiled run keeps no failure path; the interpreter finds the same failure.
					uint32_t ignored = 0;
					Interpret(descriptor.dwords[index], ignored);
				}
				LogDescriptorDiagnostic(source, index);
			}
			return false;
		}
	}
	return true;
}

void SrtWalker::LogDescriptorDiagnostic(uint32_t source, uint32_t dword) {
	const auto& descriptor = m_program.descriptor_sources[source];
	std::string fail_node  = "(none)";
	if (m_diag_first_fail != nullptr) {
		fail_node = DescribeValueTree(m_program, Value(const_cast<Inst*>(m_diag_first_fail)));
	}
	std::string probe;
	const Inst* failed = m_diag_first_fail;
	if (failed != nullptr && failed->GetOpcode() == ValueOpcode::Phi) {
		for (size_t edge = 0; edge < failed->NumArgs(); edge++) {
			SrtWalker  edge_walker(m_program, m_runtime, m_clean_flat_slots, m_clean_evaluator);
			uint32_t   value = 0;
			const bool ok    = edge_walker.Interpret(failed->Arg(edge), value);
			probe += fmt::format("\n  phi-edge[{}] const={} value=0x{:08x} {}", edge,
			                     ok ? "YES" : "no", value,
			                     DescribeValueTree(m_program, failed->Arg(edge), 1));
		}
	}
	std::fprintf(stderr, "SRT-DIAG: hash=0x%016" PRIx64 " stage=%s source=%u/%u dword=%u/%u "
	     "srt_slots=%u buffers=%u images=%u samplers=%u indirect_img=%d "
	     "did not evaluate:\n  tree: %s\n  first-fail: %s%s\n",
	     m_program.shader_hash, StageName(m_program.stage), source,
	     static_cast<uint32_t>(m_program.descriptor_sources.size()), dword,
	     descriptor.dword_count, static_cast<uint32_t>(m_program.srt_reads.size()),
	     static_cast<uint32_t>(m_program.info.buffers.size()),
	     static_cast<uint32_t>(m_program.info.images.size()),
	     static_cast<uint32_t>(m_program.info.samplers.size()),
	     descriptor.indirect_image.has_value() ? 1 : 0,
	     DescribeValueTree(m_program, descriptor.dwords[dword]).c_str(), fail_node.c_str(),
	     probe.c_str());
}

std::span<const uint8_t> SrtWalker::FindActiveSources() {
	if (m_program.control_flow.empty()) {
		return {};
	}
	// The compiled conditions belong to the strict walk.
	const auto* compiled = m_compiled_clean ? m_compiled : nullptr;
	auto& active = m_program.active_sources;
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) {
			active.at(source) = 0u;
		}
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) {
			continue;
		}
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		uint32_t condition = 0;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    (compiled != nullptr ? compiled->Run(compiled->m_conditions[index], m_runtime, condition)
		                         : Interpret(block.condition, condition))) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return active;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) {
		return false;
	}
	// The compiled flat roots belong to the ordinary walk, clean-slot redirect included.
	const auto* compiled = m_compiled != nullptr && !m_compiled_clean ? m_compiled : nullptr;
	flat.resize(m_program.srt_reads.size());
	for (size_t index = 0; index < m_program.srt_reads.size(); index++) {
		const auto& read  = m_program.srt_reads[index];
		const bool  clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		if (read.flat_offset >= flat.size()) {
			return false;
		}
		auto&      value = flat[read.flat_offset];
		const bool evaluated =
		    compiled != nullptr ? compiled->Run(compiled->m_flat_roots[index], m_runtime, value)
		    : clean             ? m_clean_evaluator->Evaluate(read.value, value)
		                        : Interpret(read.value, value);
		if (!evaluated) {
			return false;
		}
	}
	return true;
}

// Lowers a ResourcePlan into a CompiledSrtPlan, mirroring SrtWalker instruction by instruction.
// Each SrtWalker a refresh can create becomes an explicit walk index: the ordinary walk, the strict
// walk and, below a ReadFirstLane, either of them with that active-lane mask. Anything SrtWalker
// resolves without looking at runtime values (identities, invariant phis, component extracts of
// constructs, ReadFirstLane, masked selects, clean-slot redirects) is resolved here and costs
// nothing per refresh.
class SrtPlanCompiler {
public:
	SrtPlanCompiler(const ResourcePlan& plan, CompiledSrtPlan& out): m_plan(plan), m_out(out) {}

	bool Run() {
		const auto& plan = m_plan;
		if (!plan.srt_plan_complete ||
		    plan.uniform_fill.fill.words > plan.uniform_fill.values.size()) {
			return false;
		}
		m_walks.push_back({.clean = false, .redirect = true});
		m_walks.push_back({.clean = true, .redirect = false});

		for (const auto& block: plan.control_flow) {
			if (!block.condition.IsEmpty() && block.successors.size() < 2u) {
				return false;
			}
			for (const auto successor: block.successors) {
				if (successor >= plan.control_flow.size()) {
					return false;
				}
			}
			for (const auto source: block.sources) {
				if (source >= plan.descriptor_sources.size()) {
					return false;
				}
			}
			m_out.m_conditions.push_back(block.condition.IsEmpty()
			                                 ? Root {}
			                                 : MakeRoot(Lower(block.condition, CleanWalk)));
		}

		for (const auto& read: plan.srt_reads) {
			const bool clean = read.flat_offset < plan.clean_flat_slots.size() &&
			                   plan.clean_flat_slots[read.flat_offset] != 0u;
			m_out.m_flat_roots.push_back(MakeRoot(Lower(read.value, clean ? CleanWalk : MainWalk)));
		}

		// Exactly the queries MaterializeResources makes of each walker.
		m_out.m_main_sources.assign(plan.descriptor_sources.size(), NoRoot);
		m_out.m_clean_sources.assign(plan.descriptor_sources.size(), NoRoot);
		for (const auto& buffer: plan.info.buffers) {
			CompileSource(buffer.source, MainWalk);
			if (buffer.written) {
				CompileSource(buffer.source, CleanWalk);
			}
		}
		for (const auto& sampler: plan.info.samplers) {
			CompileSource(sampler.source, MainWalk);
		}
		for (const auto& image: plan.info.images) {
			if (image.source >= plan.descriptor_sources.size()) {
				continue;
			}
			const auto& indirect = plan.descriptor_sources[image.source].indirect_image;
			if (!indirect.has_value()) {
				CompileSource(image.source, MainWalk);
				continue;
			}
			if (indirect->material_source != UINT32_MAX) {
				CompileSource(indirect->material_source, CleanWalk);
			}
			CompileSource(indirect->table_source, CleanWalk);
			CompileValue(indirect->key_count);
			CompileValue(indirect->selector_mask);
		}
		for (uint32_t index = 0; index < plan.uniform_fill.fill.words; index++) {
			CompileValue(plan.uniform_fill.values[index]);
		}
		return !m_unsupported;
	}

private:
	using Code = CompiledSrtPlan::Code;
	using Op   = CompiledSrtPlan::Op;
	using Root = CompiledSrtPlan::Root;

	struct Walk {
		bool  clean    = false;
		bool  redirect = false;
		Value mask;
	};

	static constexpr uint32_t MainWalk  = 0;
	static constexpr uint32_t CleanWalk = 1;
	static constexpr uint32_t NoRoot    = CompiledSrtPlan::NoRoot;

	uint32_t WalkIndex(const Walk& walk) {
		for (uint32_t index = 0; index < m_walks.size(); index++) {
			const auto& known = m_walks[index];
			if (known.clean == walk.clean && known.redirect == walk.redirect &&
			    known.mask == walk.mask) {
				return index;
			}
		}
		m_walks.push_back(walk);
		return static_cast<uint32_t>(m_walks.size() - 1u);
	}

	// The walk SrtWalker hands clean flat slots and select predicates to: the strict walk under the
	// same active-lane mask, or the walk itself when it is already strict.
	uint32_t StrictWalk(uint32_t walk) {
		if (m_walks[walk].clean) {
			return walk;
		}
		return WalkIndex({.clean = true, .redirect = false, .mask = m_walks[walk].mask});
	}

	void CompileSource(uint32_t source, uint32_t walk) {
		if (source >= m_plan.descriptor_sources.size()) {
			return;
		}
		auto& first = (walk == CleanWalk ? m_out.m_clean_sources : m_out.m_main_sources)[source];
		if (first != NoRoot) {
			return;
		}
		const auto& descriptor = m_plan.descriptor_sources[source];
		if (descriptor.dword_count > descriptor.dwords.size()) {
			m_unsupported = true;
			return;
		}
		first = static_cast<uint32_t>(m_out.m_dword_roots.size());
		for (uint32_t dword = 0; dword < descriptor.dword_count; dword++) {
			m_out.m_dword_roots.push_back(MakeRoot(Lower(descriptor.dwords[dword], walk)));
		}
	}

	// A value the strict walker is asked for directly. Immediates cost the walker nothing.
	void CompileValue(Value value) {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || m_out.FindValueRoot(value, true) != nullptr) {
			return;
		}
		m_out.m_value_roots.push_back(
		    {.inst = inst, .clean = true, .root = MakeRoot(Lower(value, CleanWalk))});
	}

	uint32_t AddOp(const Op& op, uint64_t initial = 0, uint8_t state = CompiledSrtPlan::Unset) {
		m_out.m_ops.push_back(op);
		m_out.m_initial_values.push_back(initial);
		m_out.m_initial_states.push_back(state);
		return static_cast<uint32_t>(m_out.m_ops.size() - 1u);
	}

	uint32_t Constant(uint64_t value) {
		if (const auto found = m_constants.find(value); found != m_constants.end()) {
			return found->second;
		}
		const auto slot = AddOp({.code = Code::Constant}, value, CompiledSrtPlan::Done);
		m_constants.emplace(value, slot);
		return slot;
	}

	uint32_t Fail() {
		if (m_fail == NoRoot) {
			m_fail = AddOp({.code = Code::Fail});
		}
		return m_fail;
	}

	// Mirrors SrtWalker::EvaluateWide.
	uint32_t Lower(Value value, uint32_t walk) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: return Constant(value.U1());
				case Type::U8: return Constant(value.U8());
				case Type::U16: return Constant(value.U16());
				case Type::U32: return Constant(value.U32());
				case Type::U64: return Constant(value.U64());
				case Type::F32: return Constant(Float32Bits(value.F32Value()));
				default: return Fail();
			}
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return Fail();
		}
		const auto mask = m_walks[walk].mask;
		if (!mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == mask) {
			return Lower(inst->Arg(1), walk);
		}
		const auto key = std::make_pair(inst, walk);
		if (const auto found = m_nodes.find(key); found != m_nodes.end()) {
			return found->second;
		}
		// SrtWalker fails an instruction reached again while it is being evaluated, which depends
		// on the order roots are evaluated in. A shared slot cannot reproduce that, so a program
		// whose graph has such a cycle stays on the walker.
		if (std::ranges::find(m_visiting, key) != m_visiting.end()) {
			m_unsupported = true;
			return Fail();
		}
		m_visiting.push_back(key);
		const auto slot = LowerInst(*inst, walk);
		m_visiting.pop_back();
		m_nodes.emplace(key, slot);
		return slot;
	}

	// Mirrors SrtWalker::EvaluateInst. An opcode SrtWalker would evaluate but this does not know
	// makes the whole plan unusable rather than being guessed at.
	uint32_t LowerInst(const Inst& inst, uint32_t walk) {
		const auto opcode = inst.GetOpcode();
		if (const auto arity = PureOpArity(opcode); arity != 0u) {
			Op op {.code = Code::Pure, .pure = opcode};
			for (uint32_t index = 0; index < arity; index++) {
				op.args[index] = Lower(inst.Arg(index), walk);
			}
			return AddOp(op);
		}
		switch (opcode) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_plan.user_data_base) {
					return Fail();
				}
				return AddOp({.code = Code::UserData, .immediate = reg - m_plan.user_data_base});
			}
			case ValueOpcode::GetShaderBase: return AddOp({.code = Code::ShaderBase});
			case ValueOpcode::Phi: {
				const auto invariant = ResolveInvariantPhi(m_plan, Value(const_cast<Inst*>(&inst)));
				return invariant.IsEmpty() ? Fail() : Lower(invariant, walk);
			}
			case ValueOpcode::ReadFirstLane: {
				auto lane = m_walks[walk];
				lane.mask = inst.Arg(1).Resolve();
				return Lower(inst.Arg(0), WalkIndex(lane));
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return Lower(inst.Arg(0), walk);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::CompositeExtractU32x4: return LowerExtract(inst, walk);
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_plan.srt_reads.size()) {
					return Fail();
				}
				const bool clean = m_walks[walk].redirect &&
				                   slot.U32() < m_plan.clean_flat_slots.size() &&
				                   m_plan.clean_flat_slots[slot.U32()] != 0u;
				return Lower(m_plan.srt_reads[slot.U32()].value, clean ? StrictWalk(walk) : walk);
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				return IsRawRead(m_plan, inst) ? LowerRawRead(inst, walk) : Fail();
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				Op op {.code = Code::Select};
				op.args[0] = Lower(inst.Arg(0), StrictWalk(walk));
				op.args[1] = Lower(inst.Arg(1), walk);
				op.args[2] = Lower(inst.Arg(2), walk);
				return AddOp(op);
			}
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64: return Fail();
			default: m_unsupported = true; return Fail();
		}
	}

	// Mirrors SrtWalker::EvaluateExtract.
	uint32_t LowerExtract(const Inst& inst, uint32_t walk) {
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return Fail();
		}
		const auto component = index.U32();
		const auto max_component =
		    inst.GetOpcode() == ValueOpcode::CompositeExtractU32x4 ? 4u : 2u;
		if (component >= max_component) {
			return Fail();
		}
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			Op op {.code = Code::ExtractU64, .immediate = component};
			op.args[0] = Lower(inst.Arg(0), walk);
			return AddOp(op);
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return Fail();
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2 ||
		    source->GetOpcode() == ValueOpcode::CompositeConstructU32x4) {
			return Lower(source->Arg(component), walk);
		}
		if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
			Op op {.code = component == 0u ? Code::AddCarryLow : Code::AddCarryHigh};
			op.args[0] = Lower(source->Arg(0), walk);
			op.args[1] = Lower(source->Arg(1), walk);
			return AddOp(op);
		}
		return Fail();
	}

	// Mirrors SrtWalker::EvaluateRawRead.
	uint32_t LowerRawRead(const Inst& inst, uint32_t walk) {
		const auto  flags  = inst.Flags<MemoryFlags>();
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (flags.index >= m_plan.memory_info.size() || handle == nullptr) {
			return Fail();
		}
		const auto immediate =
		    static_cast<int64_t>(static_cast<int32_t>(m_plan.memory_info[flags.index].offset));
		Op op {.clean = m_walks[walk].clean, .immediate = static_cast<uint64_t>(immediate)};
		op.args[0] = Lower(handle->Arg(0), walk);
		op.args[1] = Lower(handle->Arg(1), walk);
		op.args[2] = Lower(inst.Arg(1), walk);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32) {
			op.code = Code::AddressRead;
			return AddOp(op);
		}
		op.code = Code::BufferRead;
		// A handle without size dwords still reads a null base as zero and fails otherwise.
		op.args[3] = handle->NumArgs() == 4u ? Lower(handle->Arg(2), walk) : NoRoot;
		op.args[4] = handle->NumArgs() == 4u ? Lower(handle->Arg(3), walk) : NoRoot;
		return AddOp(op);
	}

	Root MakeRoot(uint32_t slot) {
		for (const auto emitted: m_emitted_list) {
			m_emitted[emitted] = 0u;
		}
		m_emitted_list.clear();
		const auto begin = static_cast<uint32_t>(m_out.m_sequence.size());
		Emit(slot);
		return {begin, static_cast<uint32_t>(m_out.m_sequence.size()), slot};
	}

	void MarkEmitted(uint32_t slot) {
		if (m_emitted.size() <= slot) {
			m_emitted.resize(m_out.m_ops.size());
		}
		m_emitted[slot] = 1u;
		m_emitted_list.push_back(slot);
	}

	// Emits `slot` as a region the run may skip. Anything first emitted inside it may be left
	// unevaluated, so it is forgotten afterwards and emitted again by later users. Returns the
	// region's length.
	uint32_t EmitSkippable(uint32_t slot) {
		const auto begin = m_out.m_sequence.size();
		const auto mark  = m_emitted_list.size();
		Emit(slot);
		for (auto index = mark; index < m_emitted_list.size(); index++) {
			m_emitted[m_emitted_list[index]] = 0u;
		}
		m_emitted_list.resize(mark);
		return static_cast<uint32_t>(m_out.m_sequence.size() - begin);
	}

	uint32_t AddMarker(Code code) {
		const auto marker = AddOp({.code = code});
		m_out.m_sequence.push_back(marker);
		return marker;
	}

	// Appends the post-order walk of `slot` that SrtWalker performs, skipping what this root has
	// already evaluated unconditionally.
	void Emit(uint32_t slot) {
		if (slot < m_emitted.size() && m_emitted[slot] != 0u) {
			return;
		}
		const auto op = m_out.m_ops[slot];
		switch (op.code) {
			case Code::Constant: return;
			case Code::Pure:
				for (uint32_t index = 0; index < PureOpArity(op.pure); index++) {
					Emit(op.args[index]);
				}
				break;
			case Code::ExtractU64: Emit(op.args[0]); break;
			case Code::AddCarryLow:
			case Code::AddCarryHigh:
			case Code::AddressRead:
				for (uint32_t index = 0; index < (op.code == Code::AddressRead ? 3u : 2u); index++) {
					Emit(op.args[index]);
				}
				break;
			case Code::BufferRead: {
				Emit(op.args[0]);
				Emit(op.args[1]);
				Emit(op.args[2]);
				// SrtWalker evaluates the size dwords only for a non-null base.
				const auto guard = AddMarker(Code::BufferReadGuard);
				auto       size_length = 0u;
				if (op.args[3] != NoRoot) {
					size_length += EmitSkippable(op.args[3]);
					size_length += EmitSkippable(op.args[4]);
				}
				m_out.m_ops[guard].args[0]   = op.args[0];
				m_out.m_ops[guard].args[1]   = op.args[1];
				m_out.m_ops[guard].args[2]   = slot;
				m_out.m_ops[guard].immediate = size_length + 1u;
				break;
			}
			case Code::Select: {
				// SrtWalker evaluates the predicate, then only the operand it chooses.
				Emit(op.args[0]);
				const auto guard        = AddMarker(Code::SelectGuard);
				const auto true_length  = EmitSkippable(op.args[1]);
				const auto otherwise    = AddMarker(Code::SelectElse);
				const auto false_length = EmitSkippable(op.args[2]);
				m_out.m_ops[guard].args[0]     = op.args[0];
				m_out.m_ops[guard].args[2]     = slot;
				m_out.m_ops[guard].args[3]     = true_length + 1u + false_length;
				m_out.m_ops[guard].immediate   = true_length + 1u;
				m_out.m_ops[otherwise].immediate = false_length;
				break;
			}
			default: break;
		}
		m_out.m_sequence.push_back(slot);
		MarkEmitted(slot);
	}

	const ResourcePlan&                                  m_plan;
	CompiledSrtPlan&                                     m_out;
	std::vector<Walk>                                    m_walks;
	std::unordered_map<uint64_t, uint32_t>               m_constants;
	std::map<std::pair<const Inst*, uint32_t>, uint32_t> m_nodes;
	std::vector<std::pair<const Inst*, uint32_t>>        m_visiting;
	std::vector<uint8_t>                                 m_emitted;
	std::vector<uint32_t>                                m_emitted_list;
	uint32_t                                             m_fail        = NoRoot;
	bool                                                 m_unsupported = false;
};

CompiledSrtPlan::CompiledSrtPlan(const ResourcePlan& plan) {
	if (!SrtPlanCompiler(plan, *this).Run()) {
		*this = {};
		return;
	}
	m_plan   = &plan;
	m_usable = true;
}

void CompiledSrtPlan::Begin() const {
	m_values.assign(m_initial_values.begin(), m_initial_values.end());
	m_states.assign(m_initial_states.begin(), m_initial_states.end());
}

const CompiledSrtPlan::Root* CompiledSrtPlan::FindValueRoot(Value value, bool clean) const {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) {
		return nullptr;
	}
	for (const auto& entry: m_value_roots) {
		if (entry.inst == inst && entry.clean == clean) {
			return &entry.root;
		}
	}
	return nullptr;
}

const CompiledSrtPlan::Root* CompiledSrtPlan::DescriptorRoots(uint32_t source, bool clean) const {
	const auto& firsts = clean ? m_clean_sources : m_main_sources;
	if (source >= firsts.size() || firsts[source] == NoRoot) {
		return nullptr;
	}
	return m_dword_roots.data() + firsts[source];
}

bool CompiledSrtPlan::Execute(const Op& op, const SrtRuntime& runtime, uint64_t& result) const {
	const auto* values = m_values.data();
	switch (op.code) {
		case Code::UserData:
			if (op.immediate >= runtime.user_data.size()) {
				return false;
			}
			result = runtime.user_data[op.immediate];
			return true;
		case Code::ShaderBase: result = runtime.shader_base; return true;
		case Code::Pure: {
			const uint64_t args[4] = {values[op.args[0]], values[op.args[1]], values[op.args[2]],
			                          values[op.args[3]]};
			return ApplyPureOp(op.pure, args, result);
		}
		case Code::Select:
			result = values[op.args[0]] != 0u ? values[op.args[1]] : values[op.args[2]];
			return true;
		case Code::ExtractU64:
			result = static_cast<uint32_t>(values[op.args[0]] >> (op.immediate * 32u));
			return true;
		case Code::AddCarryLow:
		case Code::AddCarryHigh: {
			const auto sum = static_cast<uint64_t>(static_cast<uint32_t>(values[op.args[0]])) +
			                 static_cast<uint32_t>(values[op.args[1]]);
			result = op.code == Code::AddCarryLow ? static_cast<uint32_t>(sum)
			                                      : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case Code::AddressRead:
		case Code::BufferRead: {
			const auto high = values[op.args[1]];
			const auto base = RawReadBase(values[op.args[0]], high);
			if (base == 0) {
				result = 0;
				return true;
			}
			const auto immediate = static_cast<int64_t>(op.immediate);
			const auto offset    = values[op.args[2]];
			uint64_t   address   = 0;
			if (op.code == Code::AddressRead) {
				if (!ScalarAddressReadAddress(base, immediate, offset, address)) {
					return false;
				}
			} else if (op.args[3] == NoRoot ||
			           !ScalarBufferReadAddress(base, high, immediate, offset, values[op.args[3]],
			                                    address)) {
				return false;
			}
			// The strict walk reads as CleanRuntime does: never through the ordinary reader.
			const auto reader = op.clean ? CleanRuntime(runtime).read_memory : runtime.read_memory;
			return ReadSrtWord(reader, runtime.userdata, address, result);
		}
		default: return false;
	}
}

bool CompiledSrtPlan::Run(const Root& root, const SrtRuntime& runtime, uint32_t& result) const {
	auto* values = m_values.data();
	auto* states = m_states.data();
	if (states[root.slot] != Done) {
		for (auto position = root.begin; position < root.end; position++) {
			const auto  index = m_sequence[position];
			const auto& op    = m_ops[index];
			switch (op.code) {
				case Code::BufferReadGuard: {
					const auto read = op.args[2];
					if (states[read] == Failed) {
						return false;
					}
					if (states[read] == Unset) {
						if (RawReadBase(values[op.args[0]], values[op.args[1]]) != 0u) {
							continue;
						}
						values[read] = 0;
						states[read] = Done;
					}
					position += static_cast<uint32_t>(op.immediate) - 1u;
					continue;
				}
				case Code::SelectGuard: {
					const auto select = op.args[2];
					if (states[select] == Failed) {
						return false;
					}
					if (states[select] == Done) {
						position += op.args[3];
					} else if (values[op.args[0]] == 0u) {
						position += static_cast<uint32_t>(op.immediate);
					}
					continue;
				}
				case Code::SelectElse: position += static_cast<uint32_t>(op.immediate); continue;
				default: break;
			}
			auto& state = states[index];
			if (state == Done) {
				continue;
			}
			if (state == Failed || !Execute(op, runtime, values[index])) {
				state = Failed;
				return false;
			}
			state = Done;
		}
	}
	result = static_cast<uint32_t>(values[root.slot]);
	return true;
}

SrtRefresh::SrtRefresh(const ResourcePlan& program, const SrtRuntime& runtime,
                       const CompiledSrtPlan* compiled)
    : m_clean(program, CleanRuntime(runtime)),
      m_main(program, runtime, program.clean_flat_slots, &m_clean) {
	if (compiled == nullptr || !compiled->Usable()) {
		return;
	}
	EXIT_IF(compiled->m_plan != &program);
	compiled->Begin();
	m_clean.m_compiled       = compiled;
	m_clean.m_compiled_clean = true;
	m_main.m_compiled        = compiled;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type,
                          std::string* reason) {
	RuntimeValidator validator(program, type);
	if (validator.Run(value)) {
		return true;
	}
	if (reason != nullptr) {
		*reason = validator.Reason();
	}
	return false;
}

void BuildSrtPlan(Program& program) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program).Run();
	program.srt_plan_complete = true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
