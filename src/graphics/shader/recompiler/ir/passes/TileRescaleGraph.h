#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/TileRescale.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The program as the tile-rescale analysis (TileRescale.h) sees it: a compact, read-only graph.
// Why not the IR itself: the analysis also has to run on the text dumps it was validated against
// (tests), so both sources reduce to this one form. Nodes are the instructions in program order,
// block by block; operands name nodes by index. Users are listed in program order, operand by
// operand -- the proof searches visit them in that order and their budgets make it observable.

namespace Libs::Graphics::ShaderRecompiler::IR {

struct RescaleOperand {
	enum class Kind : uint8_t { Empty, Node, Immediate };

	Kind     kind = Kind::Empty;
	// The immediate's type (ScalarReg/VectorReg for register operands); Void otherwise.
	Type     type = Type::Void;
	// Node index, or the immediate's bits (F32/F16 as bit patterns, registers as their index).
	uint64_t bits = 0;

	[[nodiscard]] static RescaleOperand ForNode(uint32_t node) {
		return {Kind::Node, Type::Void, node};
	}
	[[nodiscard]] static RescaleOperand ForImmediate(Type type, uint64_t bits) {
		return {Kind::Immediate, type, bits};
	}

	[[nodiscard]] bool     IsNode() const { return kind == Kind::Node; }
	[[nodiscard]] uint32_t Node() const { return static_cast<uint32_t>(bits); }
	[[nodiscard]] bool     IsImmediate(Type immediate_type) const {
		return kind == Kind::Immediate && type == immediate_type;
	}
	[[nodiscard]] std::optional<uint32_t> ImmU32() const {
		return IsImmediate(Type::U32) ? std::optional<uint32_t>(static_cast<uint32_t>(bits))
		                              : std::nullopt;
	}

	bool operator==(const RescaleOperand& other) const = default;
};

struct RescaleUse {
	uint32_t user    = 0;
	uint32_t operand = 0;
};

struct RescaleNode {
	ValueOpcode   opcode         = ValueOpcode::Void;
	Type          type           = Type::Void;
	// Raw instruction flags (MemoryFlags for memory and image operations).
	uint64_t      flags          = 0;
	BufferAccess  buffer_access  = BufferAccess::None;
	SharedAccess  shared_access  = SharedAccess::None;
	AddressAccess address_access = AddressAccess::None;
	ImageAccess   image_access   = ImageAccess::None;
	// A SelectU32 rung of an indexed register write (SelectFlags::indexed_register_write).
	bool          indexed_register_write = false;
	uint32_t      block                  = 0;
	// Stable number used in reasons; ascends in program order.
	uint32_t      id                     = 0;
	uint32_t      first_arg              = 0;
	uint32_t      num_args               = 0;
	uint32_t      first_use              = 0;
	uint32_t      num_uses               = 0;
	// The instruction this node was built from; null for a graph parsed from text.
	const Inst*   inst                   = nullptr;

	[[nodiscard]] uint32_t MemoryIndex() const;
};

struct RescaleBlock {
	uint32_t              first_node = 0;
	uint32_t              num_nodes  = 0;
	// ImmSuccessors order: a conditional block lists its true successor first.
	std::vector<uint32_t> successors;
	RescaleOperand        condition;
};

struct RescaleImage {
	Decoder::ImageDimension dimension = Decoder::ImageDimension::Unknown;
	ImageMipMode            mip_mode  = ImageMipMode::None;
};

struct RescaleMemory {
	ResourceKind kind               = ResourceKind::None;
	uint32_t     resource           = 0;
	uint32_t     address_components = 0;
	// Bit layout of the x and y texel address components.
	uint32_t     x_offset           = 0;
	uint32_t     x_width            = 0;
	uint32_t     y_offset           = 0;
	uint32_t     y_width            = 0;
};

class RescaleGraph {
public:
	std::vector<RescaleNode>    nodes;
	std::vector<RescaleBlock>   blocks;
	std::vector<RescaleImage>   images;
	std::vector<RescaleMemory>  memory;
	bool                        dispatcher_fallback = false;

	// Appends a node's operands; phi_block is the predecessor block of a Phi operand.
	void AddOperand(RescaleNode& node, RescaleOperand operand, uint32_t phi_block = 0);
	// Builds the user lists once every node and operand is in place.
	void LinkUsers();

	[[nodiscard]] std::span<const RescaleOperand> Args(const RescaleNode& node) const {
		return {operands.data() + node.first_arg, node.num_args};
	}
	[[nodiscard]] uint32_t PhiBlock(const RescaleNode& node, size_t index) const {
		return phi_blocks[node.first_arg + index];
	}
	[[nodiscard]] std::span<const RescaleUse> Users(const RescaleNode& node) const {
		return {uses.data() + node.first_use, node.num_uses};
	}
	// The node an operand names, or null for immediates.
	[[nodiscard]] const RescaleNode* NodeOf(const RescaleOperand& operand) const {
		return operand.IsNode() ? &nodes[operand.Node()] : nullptr;
	}
	[[nodiscard]] const RescaleNode* ArgNode(const RescaleNode& node, size_t index) const {
		return index < node.num_args ? NodeOf(Args(node)[index]) : nullptr;
	}
	[[nodiscard]] uint32_t IndexOf(const RescaleNode& node) const {
		return static_cast<uint32_t>(&node - nodes.data());
	}
	[[nodiscard]] const RescaleMemory* MemoryOf(const RescaleNode& node) const;

private:
	std::vector<RescaleOperand> operands;
	std::vector<uint32_t>       phi_blocks;
	std::vector<RescaleUse>     uses;
};

[[nodiscard]] RescaleGraph BuildRescaleGraph(const Program& program);

// AnalyzeTileRescale's result with the breakdown behind TileRescalePlan::summary.
struct TileRescaleReport {
	using Counts = std::vector<std::pair<std::string, uint32_t>>;

	TileRescalePlan       plan;
	// In first-seen order.
	Counts                load_classes;
	Counts                parity_notes;
	Counts                cross_lane_notes;
	std::vector<uint32_t> written_images;
	// Largest |constant offset| of a neighbour load.
	uint32_t              max_neighbour = 0;
};

[[nodiscard]] TileRescaleReport AnalyzeRescaleGraph(const RescaleGraph&     graph,
                                                    const TileRescaleShape& shape);

} // namespace Libs::Graphics::ShaderRecompiler::IR
