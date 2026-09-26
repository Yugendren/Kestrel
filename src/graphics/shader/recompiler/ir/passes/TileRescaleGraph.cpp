#include "graphics/shader/recompiler/ir/passes/TileRescaleGraph.h"

#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"

#include <bit>
#include <cstring>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

RescaleOperand OperandOf(Value value, const std::unordered_map<const Inst*, uint32_t>& node_of) {
	if (value.IsEmpty()) {
		return {};
	}
	if (const auto* inst = value.TryInstruction(); inst != nullptr) {
		const auto it = node_of.find(inst);
		return it == node_of.end() ? RescaleOperand {} : RescaleOperand::ForNode(it->second);
	}
	const auto type = value.GetType();
	switch (type) {
		case Type::U1: return RescaleOperand::ForImmediate(type, value.U1() ? 1u : 0u);
		case Type::U8: return RescaleOperand::ForImmediate(type, value.U8());
		case Type::U16: return RescaleOperand::ForImmediate(type, value.U16());
		case Type::U32: return RescaleOperand::ForImmediate(type, value.U32());
		case Type::U64: return RescaleOperand::ForImmediate(type, value.U64());
		case Type::F16: return RescaleOperand::ForImmediate(type, value.F16Bits());
		case Type::F32:
			return RescaleOperand::ForImmediate(type, std::bit_cast<uint32_t>(value.F32Value()));
		case Type::ScalarReg:
			return RescaleOperand::ForImmediate(type, RegIndex(value.ScalarRegister()));
		case Type::VectorReg:
			return RescaleOperand::ForImmediate(type, RegIndex(value.VectorRegister()));
		default: return RescaleOperand::ForImmediate(type, 0);
	}
}

} // namespace

uint32_t RescaleNode::MemoryIndex() const {
	MemoryFlags memory_flags {};
	std::memcpy(&memory_flags, &flags, sizeof(memory_flags));
	return memory_flags.index;
}

void RescaleGraph::AddOperand(RescaleNode& node, RescaleOperand operand, uint32_t phi_block) {
	if (node.num_args == 0) {
		node.first_arg = static_cast<uint32_t>(operands.size());
	}
	operands.push_back(operand);
	phi_blocks.push_back(phi_block);
	node.num_args++;
}

void RescaleGraph::LinkUsers() {
	std::vector<uint32_t> counts(nodes.size(), 0);
	for (const auto& node: nodes) {
		for (const auto& arg: Args(node)) {
			if (arg.IsNode()) {
				counts[arg.Node()]++;
			}
		}
	}
	uint32_t cursor = 0;
	for (size_t index = 0; index < nodes.size(); index++) {
		nodes[index].first_use = cursor;
		nodes[index].num_uses  = 0;
		cursor += counts[index];
	}
	uses.assign(cursor, {});
	for (uint32_t user = 0; user < nodes.size(); user++) {
		const auto args = Args(nodes[user]);
		for (uint32_t operand = 0; operand < args.size(); operand++) {
			if (args[operand].IsNode()) {
				auto& used = nodes[args[operand].Node()];
				uses[used.first_use + used.num_uses++] = {user, operand};
			}
		}
	}
}

const RescaleMemory* RescaleGraph::MemoryOf(const RescaleNode& node) const {
	const auto index = node.MemoryIndex();
	return index < memory.size() ? &memory[index] : nullptr;
}

RescaleGraph BuildRescaleGraph(const Program& program) {
	RescaleGraph                               graph;
	std::unordered_map<const Inst*, uint32_t>  node_of;
	std::unordered_map<const Block*, uint32_t> block_of;
	graph.dispatcher_fallback = program.dispatcher_fallback;

	for (uint32_t b = 0; b < program.blocks.size(); b++) {
		const auto* block = program.blocks[b];
		block_of.emplace(block, b);
		RescaleBlock rescale_block;
		rescale_block.first_node = static_cast<uint32_t>(graph.nodes.size());
		for (const auto& inst: *block) {
			const auto  op    = inst.GetOpcode();
			const auto  index = static_cast<uint32_t>(graph.nodes.size());
			RescaleNode node;
			node.opcode         = op;
			node.type           = inst.GetType();
			node.flags          = inst.Flags<uint64_t>();
			node.buffer_access  = BufferAccessOf(op);
			node.shared_access  = SharedAccessOf(op);
			node.address_access = AddressOpcodeInfoOf(op).access;
			node.image_access   = ImageOpcodeInfoOf(op).access;
			node.indexed_register_write =
			    op == ValueOpcode::SelectU32 && inst.Flags<SelectFlags>().indexed_register_write;
			node.block = b;
			node.id    = index + 1;
			node.inst  = &inst;
			node_of.emplace(&inst, index);
			graph.nodes.push_back(node);
		}
		rescale_block.num_nodes =
		    static_cast<uint32_t>(graph.nodes.size()) - rescale_block.first_node;
		graph.blocks.push_back(std::move(rescale_block));
	}

	for (auto& node: graph.nodes) {
		const auto& inst  = *node.inst;
		const bool  phi   = inst.GetOpcode() == ValueOpcode::Phi;
		for (size_t i = 0; i < inst.NumArgs(); i++) {
			uint32_t phi_block = 0;
			if (phi) {
				const auto it = block_of.find(inst.PhiBlock(i));
				phi_block     = it == block_of.end() ? UINT32_MAX : it->second;
			}
			graph.AddOperand(node, OperandOf(inst.Arg(i), node_of), phi_block);
		}
	}
	for (uint32_t b = 0; b < program.blocks.size(); b++) {
		for (const auto* successor: program.blocks[b]->ImmSuccessors()) {
			const auto it = block_of.find(successor);
			if (it != block_of.end()) {
				graph.blocks[b].successors.push_back(it->second);
			}
		}
		if (b < program.block_info.size()) {
			graph.blocks[b].condition = OperandOf(program.block_info[b].condition, node_of);
		}
	}

	graph.images.reserve(program.info.images.size());
	for (const auto& image: program.info.images) {
		graph.images.push_back({image.dimension, image.mip_mode});
	}
	graph.memory.reserve(program.memory_info.size());
	for (const auto& info: program.memory_info) {
		const auto x = Decoder::ImageAddressComponentLayout(info.image_sample_flags, 0);
		const auto y = Decoder::ImageAddressComponentLayout(info.image_sample_flags, 1);
		graph.memory.push_back({info.kind, info.resource, info.image_address_components,
		                        x.bit_offset, x.bit_width, y.bit_offset, y.bit_width});
	}
	graph.LinkUsers();
	return graph;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
