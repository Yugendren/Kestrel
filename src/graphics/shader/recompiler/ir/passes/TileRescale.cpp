#include "graphics/shader/recompiler/ir/passes/TileRescale.h"

#include "graphics/shader/recompiler/ir/passes/TileRescaleGraph.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <functional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// The design and every rule below come from the offline detector that was validated against all
// 36 Astro's Playroom compute programs (research/remap-detector-2026-09-26.md, remapdet.py). This
// is a faithful port: where a rule looks arbitrary, it is the detector's, and the golden tests
// check the two agree program by program.

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint32_t CountTrailingZeros(uint32_t value) {
	return value == 0 ? 32u : static_cast<uint32_t>(std::countr_zero(value));
}

constexpr int64_t Signed(uint32_t value) {
	return static_cast<int32_t>(value);
}

bool IsOneOf(ValueOpcode op, std::initializer_list<ValueOpcode> ops) {
	return std::find(ops.begin(), ops.end(), op) != ops.end();
}

// Insertion-ordered counters, the shape of the verdict's notes.
class NoteCounter {
public:
	void Add(std::string_view note) {
		for (auto& [key, count]: counts) {
			if (key == note) {
				count++;
				return;
			}
		}
		counts.emplace_back(std::string(note), 1u);
	}
	[[nodiscard]] std::string ToString() const {
		std::string text = "{";
		for (const auto& [key, count]: counts) {
			text += fmt::format("{}{}: {}", text.size() > 1 ? ", " : "", key, count);
		}
		return text + "}";
	}

	TileRescaleReport::Counts counts;
};

class ReasonLog {
public:
	void Reject(std::string_view why, const RescaleNode* node = nullptr) {
		if (node == nullptr) {
			reasons.emplace_back(why);
		} else {
			reasons.push_back(fmt::format("{} [%{} {}]", why, node->id, ValueOpcodeName(node->opcode)));
		}
	}

	std::vector<std::string> reasons;
};

// ---------------------------------------------------------------------------------------------
// AffineForm: ax*Lx + ay*Ly + sum(coef*atom) + c, all mod 2^32. Atoms are workgroup-uniform
// values that are not linear in anything simpler; they are kept sorted by node, never with a
// zero coefficient.
// ---------------------------------------------------------------------------------------------
class AffineForm {
public:
	struct Atom {
		uint32_t node = 0;
		uint32_t coef = 0;
		bool     operator==(const Atom& other) const = default;
	};

	[[nodiscard]] static AffineForm Constant(uint32_t c) {
		AffineForm form;
		form.c = c;
		return form;
	}
	[[nodiscard]] static AffineForm LocalX() {
		AffineForm form;
		form.ax = 1;
		return form;
	}
	[[nodiscard]] static AffineForm LocalY() {
		AffineForm form;
		form.ay = 1;
		return form;
	}
	[[nodiscard]] static AffineForm OfAtom(uint32_t node) {
		AffineForm form;
		form.atoms.push_back({node, 1});
		return form;
	}

	[[nodiscard]] bool Uniform() const { return ax == 0 && ay == 0; }
	[[nodiscard]] bool Const() const { return Uniform() && atoms.empty(); }

	// this + sign*other, sign being 1 or -1 (mod 2^32).
	[[nodiscard]] AffineForm Add(const AffineForm& other, uint32_t sign = 1) const {
		AffineForm result;
		result.ax = ax + sign * other.ax;
		result.ay = ay + sign * other.ay;
		result.c  = c + sign * other.c;
		result.atoms.reserve(atoms.size() + other.atoms.size());
		size_t i = 0;
		size_t j = 0;
		while (i < atoms.size() || j < other.atoms.size()) {
			if (j == other.atoms.size() || (i < atoms.size() && atoms[i].node < other.atoms[j].node)) {
				result.atoms.push_back(atoms[i++]);
			} else if (i == atoms.size() || other.atoms[j].node < atoms[i].node) {
				result.atoms.push_back({other.atoms[j].node, sign * other.atoms[j].coef});
				j++;
			} else {
				const auto coef = atoms[i].coef + sign * other.atoms[j].coef;
				if (coef != 0) {
					result.atoms.push_back({atoms[i].node, coef});
				}
				i++;
				j++;
			}
		}
		return result;
	}

	[[nodiscard]] AffineForm Mul(uint32_t m) const {
		AffineForm result;
		result.ax = ax * m;
		result.ay = ay * m;
		result.c  = c * m;
		for (const auto& atom: atoms) {
			if (atom.coef * m != 0) {
				result.atoms.push_back({atom.node, atom.coef * m});
			}
		}
		return result;
	}

	[[nodiscard]] std::string ToString(const RescaleGraph& graph) const {
		const auto scaled = [](uint32_t coef, std::string_view term) {
			return coef == 1 ? std::string(term) : fmt::format("{}*{}", Signed(coef), term);
		};
		std::vector<std::string> terms;
		if (ax != 0) {
			terms.push_back(scaled(ax, "Lx"));
		}
		if (ay != 0) {
			terms.push_back(scaled(ay, "Ly"));
		}
		for (const auto& atom: atoms) {
			terms.push_back(scaled(atom.coef, fmt::format("u{}", graph.nodes[atom.node].id)));
		}
		if (c != 0 || terms.empty()) {
			terms.push_back(fmt::format("{}", Signed(c)));
		}
		std::string text;
		for (const auto& term: terms) {
			text += text.empty() ? term : "+" + term;
		}
		return text;
	}

	bool operator==(const AffineForm& other) const = default;

	uint32_t          ax = 0;
	uint32_t          ay = 0;
	std::vector<Atom> atoms;
	uint32_t          c = 0;
};

// The S2 fact of a value. U32 values are Lin, Wave or Var; other types only carry a uniformity
// class (Uniform, WaveUniform, Varying). Unknown: not evaluated yet (optimistic).
enum class FactKind : uint8_t { Unknown, Lin, Wave, Var, Uniform, WaveUniform, Varying };
enum class Uniformity : uint8_t { Unknown, Uniform, Wave, Varying };

struct Fact {
	FactKind   kind = FactKind::Unknown;
	AffineForm lin;

	[[nodiscard]] static Fact Of(FactKind kind) { return {kind, {}}; }
	[[nodiscard]] static Fact Of(AffineForm lin) { return {FactKind::Lin, std::move(lin)}; }

	[[nodiscard]] bool IsLin() const { return kind == FactKind::Lin; }
	[[nodiscard]] bool IsVarying() const {
		return kind == FactKind::Var || kind == FactKind::Varying;
	}
	[[nodiscard]] Uniformity Class() const {
		switch (kind) {
			case FactKind::Unknown: return Uniformity::Unknown;
			case FactKind::Lin: return lin.Uniform() ? Uniformity::Uniform : Uniformity::Varying;
			case FactKind::Wave:
			case FactKind::WaveUniform: return Uniformity::Wave;
			case FactKind::Var:
			case FactKind::Varying: return Uniformity::Varying;
			case FactKind::Uniform: return Uniformity::Uniform;
		}
		return Uniformity::Unknown;
	}
	[[nodiscard]] std::string ToString(const RescaleGraph& graph) const {
		switch (kind) {
			case FactKind::Unknown: return "None";
			case FactKind::Lin: return lin.ToString(graph);
			case FactKind::Wave: return "WAVE";
			case FactKind::Var: return "VAR";
			case FactKind::Uniform: return "U";
			case FactKind::WaveUniform: return "W";
			case FactKind::Varying: return "V";
		}
		return "?";
	}

	bool operator==(const Fact& other) const {
		return kind == other.kind && (kind != FactKind::Lin || lin == other.lin);
	}
};

// ---------------------------------------------------------------------------------------------
// PredicateLogic: syntactic proofs over the translator's SIMT predicates. implies(p, c) walks
// LogicalAnd/LogicalOr and phis (coinductively: a pair met again on the current path counts as
// proven, loop-carried exec masks re-enter the header with the same predicate) and per-lane mask
// bits ((M >> lane) & 1) != 0, with mask_subset(M, M') through And/Or/phi. Each top-level query
// has a step budget and a depth limit; running out means "not proven". Results are memoised.
// ---------------------------------------------------------------------------------------------
class PredicateLogic {
public:
	explicit PredicateLogic(const RescaleGraph& graph_): graph(graph_) {}

	[[nodiscard]] bool Implies(const RescaleOperand& p, const RescaleOperand& c) {
		return ImpliesAt(p, c, 0, nullptr);
	}

	// If u1 is the per-lane bit of a wave mask, ((M >> x) & 1) != 0, returns M.
	[[nodiscard]] std::optional<RescaleOperand> MaskBitOf(const RescaleOperand& u1) const {
		const auto* test = graph.NodeOf(u1);
		if (test == nullptr || test->opcode != ValueOpcode::INotEqual32 ||
		    Arg(*test, 1).ImmU32() != 0u) {
			return std::nullopt;
		}
		const auto* bit = graph.ArgNode(*test, 0);
		if (bit == nullptr || bit->opcode != ValueOpcode::BitwiseAnd32 || Arg(*bit, 1).ImmU32() != 1u) {
			return std::nullopt;
		}
		const auto* shift = graph.ArgNode(*bit, 0);
		if (shift == nullptr || shift->opcode != ValueOpcode::ShiftRightLogical32) {
			return std::nullopt;
		}
		return Arg(*shift, 0);
	}

private:
	using Pair = std::pair<RescaleOperand, RescaleOperand>;

	struct PairHash {
		size_t operator()(const Pair& pair) const {
			const auto mix = [](const RescaleOperand& operand) {
				return std::hash<uint64_t> {}(operand.bits * 0x9e3779b97f4a7c15ull ^
				                              (static_cast<uint64_t>(operand.type) << 3u) ^
				                              static_cast<uint64_t>(operand.kind));
			};
			return mix(pair.first) * 31u ^ mix(pair.second);
		}
	};

	static constexpr uint32_t MaxDepth    = 24;
	static constexpr int32_t  StepBudget  = 3000;

	[[nodiscard]] RescaleOperand Arg(const RescaleNode& node, size_t index) const {
		return index < node.num_args ? graph.Args(node)[index] : RescaleOperand {};
	}

	// The pairs under proof on the current path, with a one-word filter so the common miss
	// skips the scan.
	struct Path {
		std::vector<Pair> pairs;
		uint64_t          filter = 0;

		static uint64_t Bit(const Pair& pair) { return uint64_t {1} << (PairHash {}(pair) & 63u); }
		[[nodiscard]] bool Contains(const Pair& pair) const {
			return (filter & Bit(pair)) != 0 && std::find(pairs.begin(), pairs.end(), pair) != pairs.end();
		}
		void Clear() {
			pairs.clear();
			filter = 0;
		}
	};

	// Proves `rules` with `pair` assumed; restores the path afterwards.
	template <typename Rules>
	static bool WithAssumption(Path& path, const Pair& pair, Rules&& rules) {
		const auto saved = path.filter;
		path.pairs.push_back(pair);
		path.filter |= Path::Bit(pair);
		const bool result = rules();
		path.pairs.pop_back();
		path.filter = saved;
		return result;
	}

	bool ImpliesAt(const RescaleOperand& p, const RescaleOperand& c, uint32_t depth, Path* path) {
		if (p == c || (c.IsImmediate(Type::U1) && c.bits == 1) ||
		    (p.IsImmediate(Type::U1) && p.bits == 0)) {
			return true;
		}
		if (depth > MaxDepth) {
			return false;
		}
		if (path == nullptr) {
			const Pair key {p, c};
			if (const auto it = implies_memo.find(key); it != implies_memo.end()) {
				return it->second;
			}
			// Provisional while computing: breaks cycles through the memo.
			implies_memo[key] = false;
			implies_budget    = StepBudget;
			implies_path.Clear();
			const bool result = ImpliesAt(p, c, depth, &implies_path);
			implies_memo[key] = result;
			return result;
		}
		if (--implies_budget <= 0) {
			return false;
		}
		const Pair pair {p, c};
		if (path->Contains(pair)) {
			return true;
		}
		return WithAssumption(*path, pair, [&] { return ImpliesRules(p, c, depth, *path); });
	}

	bool ImpliesRules(const RescaleOperand& p, const RescaleOperand& c, uint32_t depth, Path& path) {
		const auto* pi = graph.NodeOf(p);
		const auto* ci = graph.NodeOf(c);
		const auto  any_arg = [&](const RescaleNode& node, auto&& test) {
			for (const auto& arg: graph.Args(node)) {
				if (test(arg)) {
					return true;
				}
			}
			return false;
		};
		const auto all_args = [&](const RescaleNode& node, auto&& test) {
			for (const auto& arg: graph.Args(node)) {
				if (!test(arg)) {
					return false;
				}
			}
			return true;
		};
		const auto from_arg = [&](const RescaleOperand& x) { return ImpliesAt(x, c, depth + 1, &path); };
		const auto to_arg   = [&](const RescaleOperand& x) { return ImpliesAt(p, x, depth + 1, &path); };
		if (pi != nullptr && pi->opcode == ValueOpcode::LogicalAnd && any_arg(*pi, from_arg)) {
			return true;
		}
		if (ci != nullptr && ci->opcode == ValueOpcode::LogicalAnd && all_args(*ci, to_arg)) {
			return true;
		}
		if (ci != nullptr && ci->opcode == ValueOpcode::LogicalOr && any_arg(*ci, to_arg)) {
			return true;
		}
		if (ci != nullptr && ci->opcode == ValueOpcode::Phi && all_args(*ci, to_arg)) {
			return true;
		}
		if (pi != nullptr && pi->opcode == ValueOpcode::Phi && all_args(*pi, from_arg)) {
			return true;
		}
		const auto mp = MaskBitOf(p);
		const auto mc = MaskBitOf(c);
		if (mp && mc && MaskSubset(*mp, *mc)) {
			return true;
		}
		// bit(Ballot(q)) == q
		if (mp && ci != nullptr &&
		    !IsOneOf(ci->opcode, {ValueOpcode::Phi, ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) &&
		    MaskIsBallotOf(*mp, c)) {
			return true;
		}
		return false;
	}

	[[nodiscard]] bool MaskIsBallotOf(const RescaleOperand& m, const RescaleOperand& q) const {
		const auto* extract = graph.NodeOf(m);
		if (extract == nullptr || extract->opcode != ValueOpcode::CompositeExtractU32x4) {
			return false;
		}
		const auto* ballot = graph.ArgNode(*extract, 0);
		return ballot != nullptr && ballot->opcode == ValueOpcode::Ballot && Arg(*ballot, 0) == q;
	}

	bool MaskSubset(const RescaleOperand& a, const RescaleOperand& b) {
		return MaskSubsetAt(a, b, 0, nullptr);
	}

	// Wave mask a is a subset of wave mask b.
	bool MaskSubsetAt(const RescaleOperand& a, const RescaleOperand& b, uint32_t depth, Path* path) {
		if (a == b) {
			return true;
		}
		if (depth > MaxDepth) {
			return false;
		}
		if (path == nullptr) {
			const Pair key {a, b};
			if (const auto it = subset_memo.find(key); it != subset_memo.end()) {
				return it->second;
			}
			subset_memo[key] = false;
			subset_budget    = StepBudget;
			subset_path.Clear();
			const bool result = MaskSubsetAt(a, b, depth, &subset_path);
			subset_memo[key]  = result;
			return result;
		}
		if (--subset_budget <= 0) {
			return false;
		}
		const Pair pair {a, b};
		if (path->Contains(pair)) {
			return true;
		}
		return WithAssumption(*path, pair, [&] { return MaskSubsetRules(a, b, depth, *path); });
	}

	bool MaskSubsetRules(const RescaleOperand& a, const RescaleOperand& b, uint32_t depth, Path& path) {
		const auto* ai = graph.NodeOf(a);
		const auto* bi = graph.NodeOf(b);
		const auto  from_arg = [&](const RescaleOperand& x) { return MaskSubsetAt(x, b, depth + 1, &path); };
		const auto to_arg = [&](const RescaleOperand& x) { return MaskSubsetAt(a, x, depth + 1, &path); };
		const auto any_of = [&](const RescaleNode& node, auto&& test) {
			const auto args = graph.Args(node);
			return std::any_of(args.begin(), args.end(), test);
		};
		const auto all_of = [&](const RescaleNode& node, auto&& test) {
			const auto args = graph.Args(node);
			return std::all_of(args.begin(), args.end(), test);
		};
		if (ai != nullptr && ai->opcode == ValueOpcode::BitwiseAnd32 && any_of(*ai, from_arg)) {
			return true;
		}
		if (bi != nullptr && bi->opcode == ValueOpcode::BitwiseOr32 && any_of(*bi, to_arg)) {
			return true;
		}
		if (ai != nullptr && ai->opcode == ValueOpcode::Phi && all_of(*ai, from_arg)) {
			return true;
		}
		if (bi != nullptr && bi->opcode == ValueOpcode::Phi && all_of(*bi, to_arg)) {
			return true;
		}
		return a.ImmU32() == 0u;
	}

	const RescaleGraph&                    graph;
	std::unordered_map<Pair, bool, PairHash> implies_memo;
	std::unordered_map<Pair, bool, PairHash> subset_memo;
	// The coinductive assumptions of the proof in progress (top-level queries never nest).
	Path                                   implies_path;
	Path                                   subset_path;
	int32_t                                implies_budget = 0;
	int32_t                                subset_budget  = 0;
};

// ---------------------------------------------------------------------------------------------
// Sccp: sparse conditional constant propagation. The translated IR keeps code the compile options
// made unreachable (BVH traversal under --rt-mode off, where every intersection is a constant
// miss); operations that can never execute must not cause a refusal. Exec-merge selects inside a
// LogicalAnd are folded under the other operand, which is all the predicate awareness needed.
// ---------------------------------------------------------------------------------------------
class Sccp {
public:
	Sccp(const RescaleGraph& graph_, PredicateLogic& logic_)
	    : graph(graph_), logic(logic_), values(graph_.nodes.size()), live_blocks(graph_.blocks.size()),
	      dirty(graph_.nodes.size(), true), dirty_in_block(graph_.blocks.size(), 0),
	      first_edge(graph_.blocks.size() + 1, 0), watch_head(graph_.nodes.size(), NoWatch),
	      visit_stamp(graph_.nodes.size(), 0), memo_slot(graph_.nodes.size(), 0) {
		for (const auto& node: graph.nodes) {
			dirty_in_block[node.block]++;
		}
		for (size_t b = 0; b < graph.blocks.size(); b++) {
			first_edge[b + 1] = first_edge[b] + static_cast<uint32_t>(graph.blocks[b].successors.size());
		}
		live_edges.assign(first_edge.back(), false);
	}

	// The detector's schedule: sweep the live blocks in index order (blocks found live during a
	// sweep join from the next one) until nothing changes. A node is only re-evaluated when an
	// input changed since its last evaluation: an operand's value, a new live edge into a phi's
	// block, or the constness of a value a LogicalAnd's ConstUnder walk consulted. Re-applying an
	// unchanged result never changes the state, so this skips work without changing the outcome.
	void Run() {
		if (graph.blocks.empty()) {
			return;
		}
		live_blocks[0] = true;
		std::vector<uint32_t> sweep;
		for (bool changed = true; changed;) {
			changed = false;
			sweep.clear();
			for (uint32_t b = 0; b < live_blocks.size(); b++) {
				if (live_blocks[b]) {
					sweep.push_back(b);
				}
			}
			for (const auto b: sweep) {
				if (dirty_in_block[b] != 0) {
					changed |= EvaluateBlock(graph.blocks[b]);
				}
				changed |= FollowEdges(b);
			}
		}
	}

	[[nodiscard]] bool Dead(const RescaleNode& node) const { return !live_blocks[node.block]; }

private:
	enum class State : uint8_t { Unknown, Const, Over };
	struct Lattice {
		State    state = State::Unknown;
		uint64_t value = 0;
		bool     operator==(const Lattice& other) const {
			return state == other.state && (state != State::Const || value == other.value);
		}
	};
	static constexpr Lattice Over() { return {State::Over, 0}; }
	static constexpr Lattice Const(uint64_t value) { return {State::Const, value}; }

	static constexpr uint32_t MaxConstUnderDepth = 10;

	bool EvaluateBlock(const RescaleBlock& block) {
		bool changed = false;
		for (uint32_t index = block.first_node; index < block.first_node + block.num_nodes; index++) {
			if (!dirty[index]) {
				continue;
			}
			dirty[index] = false;
			dirty_in_block[block_of(index)]--;
			const auto& node  = graph.nodes[index];
			auto        value = Evaluate(node, index);
			if (value.state == State::Unknown) {
				continue;
			}
			auto& old = values[index];
			if (old.state == State::Over || old == value) {
				continue;
			}
			if (old.state != State::Unknown) {
				value = Over();
			}
			// ConstUnder only tells a constant from anything else.
			const bool constness = old.state == State::Const || value.state == State::Const;
			old                  = value;
			changed              = true;
			for (const auto& use: graph.Users(node)) {
				MarkDirty(use.user);
			}
			if (constness) {
				for (auto link = watch_head[index]; link != NoWatch; link = watch_links[link].next) {
					MarkDirty(watch_links[link].watcher);
				}
				watch_head[index] = NoWatch;
			}
		}
		return changed;
	}

	bool FollowEdges(uint32_t b) {
		const auto& block      = graph.blocks[b];
		const auto& successors = block.successors;
		size_t      first      = 0;
		size_t      count      = successors.size();
		if (block.condition.kind != RescaleOperand::Kind::Empty && successors.size() == 2) {
			const auto condition = Get(block.condition);
			if (condition.state == State::Unknown) {
				count = 0;
			} else if (condition.state == State::Const) {
				first = condition.value != 0 ? 0 : 1;
				count = 1;
			}
		}
		bool changed = false;
		for (size_t i = first; i < first + count; i++) {
			const auto target = successors[i];
			if (!live_edges[first_edge[b] + i]) {
				live_edges[first_edge[b] + i] = true;
				changed                       = true;
				// The target's phis see one more incoming value.
				const auto& target_block = graph.blocks[target];
				for (uint32_t index = target_block.first_node;
				     index < target_block.first_node + target_block.num_nodes; index++) {
					if (graph.nodes[index].opcode == ValueOpcode::Phi) {
						MarkDirty(index);
					}
				}
			}
			if (!live_blocks[target]) {
				live_blocks[target] = true;
				changed             = true;
			}
		}
		return changed;
	}

	[[nodiscard]] uint32_t block_of(uint32_t index) const { return graph.nodes[index].block; }

	[[nodiscard]] bool IsLiveEdge(uint32_t from, uint32_t to) const {
		if (from >= graph.blocks.size()) {
			return false;
		}
		const auto& successors = graph.blocks[from].successors;
		for (size_t i = 0; i < successors.size(); i++) {
			if (successors[i] == to && live_edges[first_edge[from] + i]) {
				return true;
			}
		}
		return false;
	}

	void MarkDirty(uint32_t index) {
		if (!dirty[index]) {
			dirty[index] = true;
			dirty_in_block[block_of(index)]++;
		}
	}

	static bool IsSelect(ValueOpcode op) {
		return IsOneOf(op, {ValueOpcode::SelectU32, ValueOpcode::SelectU1, ValueOpcode::SelectF32});
	}
	static bool HasAccess(const RescaleNode& node) {
		return node.image_access != ImageAccess::None || node.buffer_access != BufferAccess::None ||
		       node.address_access != AddressAccess::None || node.shared_access != SharedAccess::None;
	}

	[[nodiscard]] Lattice Get(const RescaleOperand& operand) const {
		if (operand.IsNode()) {
			return values[operand.Node()];
		}
		if (operand.kind == RescaleOperand::Kind::Immediate &&
		    (operand.type == Type::U32 || operand.type == Type::U1 || operand.type == Type::F32 ||
		     operand.type == Type::U64)) {
			return Const(operand.bits);
		}
		return Over();
	}

	static std::optional<uint64_t> EvalConst(ValueOpcode op, std::span<const uint64_t> c) {
		constexpr uint64_t Mask = 0xffffffffull;
		const auto         need = [&](size_t n) { return c.size() >= n; };
		// The low word as a signed value.
		const auto s32 = [](uint64_t v) {
			return static_cast<int64_t>((v & 0x80000000u) != 0 ? v - (1ull << 32u) : v);
		};
		const auto flag = [](bool v) { return static_cast<uint64_t>(v ? 1 : 0); };
		switch (op) {
			case ValueOpcode::IAdd32: if (need(2)) return (c[0] + c[1]) & Mask; break;
			case ValueOpcode::ISub32: if (need(2)) return (c[0] - c[1]) & Mask; break;
			case ValueOpcode::IMul32: if (need(2)) return (c[0] * c[1]) & Mask; break;
			case ValueOpcode::ShiftLeftLogical32: if (need(2)) return (c[0] << (c[1] & 31u)) & Mask; break;
			case ValueOpcode::ShiftRightLogical32: if (need(2)) return c[0] >> (c[1] & 31u); break;
			case ValueOpcode::ShiftRightArithmetic32:
				if (need(2)) return static_cast<uint64_t>(s32(c[0]) >> (c[1] & 31u)) & Mask;
				break;
			case ValueOpcode::BitwiseAnd32: if (need(2)) return c[0] & c[1]; break;
			case ValueOpcode::BitwiseOr32: if (need(2)) return c[0] | c[1]; break;
			case ValueOpcode::BitwiseXor32: if (need(2)) return c[0] ^ c[1]; break;
			case ValueOpcode::BitwiseNot32: if (need(1)) return ~c[0] & Mask; break;
			case ValueOpcode::FindILsb32:
				if (need(1)) return c[0] != 0 ? CountTrailingZeros(static_cast<uint32_t>(c[0])) : Mask;
				break;
			case ValueOpcode::BitCount32: if (need(1)) return static_cast<uint64_t>(std::popcount(c[0])); break;
			case ValueOpcode::IEqual32: if (need(2)) return flag(c[0] == c[1]); break;
			case ValueOpcode::INotEqual32: if (need(2)) return flag(c[0] != c[1]); break;
			case ValueOpcode::ULessThan32: if (need(2)) return flag(c[0] < c[1]); break;
			case ValueOpcode::ULessThanEqual32: if (need(2)) return flag(c[0] <= c[1]); break;
			case ValueOpcode::UGreaterThan32: if (need(2)) return flag(c[0] > c[1]); break;
			case ValueOpcode::UGreaterThanEqual32: if (need(2)) return flag(c[0] >= c[1]); break;
			case ValueOpcode::SLessThan32: if (need(2)) return flag(s32(c[0]) < s32(c[1])); break;
			case ValueOpcode::SLessThanEqual32: if (need(2)) return flag(s32(c[0]) <= s32(c[1])); break;
			case ValueOpcode::SGreaterThan32: if (need(2)) return flag(s32(c[0]) > s32(c[1])); break;
			case ValueOpcode::SGreaterThanEqual32: if (need(2)) return flag(s32(c[0]) >= s32(c[1])); break;
			case ValueOpcode::LogicalAnd: if (need(2)) return c[0] & c[1]; break;
			case ValueOpcode::LogicalOr: if (need(2)) return c[0] | c[1]; break;
			case ValueOpcode::LogicalXor: if (need(2)) return c[0] ^ c[1]; break;
			case ValueOpcode::LogicalNot: if (need(1)) return 1u - c[0]; break;
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::Identity:
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane:
			case ValueOpcode::CompositeExtractU32x4:
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::AnyLane:
				if (need(1)) return c[0];
				break;
			case ValueOpcode::UMin32: if (need(2)) return std::min(c[0], c[1]); break;
			case ValueOpcode::UMax32: if (need(2)) return std::max(c[0], c[1]); break;
			default: break;
		}
		return std::nullopt;
	}

	static bool CanEvalConst(ValueOpcode op) {
		static constexpr std::array<uint64_t, 4> probe {};
		return EvalConst(op, probe).has_value();
	}

	static Lattice FromEval(ValueOpcode op, std::span<const uint64_t> c) {
		const auto value = EvalConst(op, c);
		return value ? Const(*value) : Over();
	}

	Lattice Evaluate(const RescaleNode& node, uint32_t index) {
		const auto op   = node.opcode;
		const auto args = graph.Args(node);
		if (op == ValueOpcode::Phi) {
			Lattice out;
			for (size_t i = 0; i < args.size(); i++) {
				if (!IsLiveEdge(graph.PhiBlock(node, i), node.block)) {
					continue;
				}
				const auto value = Get(args[i]);
				if (value.state == State::Unknown) {
					continue;
				}
				if (value.state == State::Over || (out.state != State::Unknown && !(out == value))) {
					return Over();
				}
				out = value;
			}
			return out;
		}
		if (IsSelect(op)) {
			const auto condition = Get(args[0]);
			if (condition.state == State::Unknown) {
				return {};
			}
			if (condition.state == State::Const) {
				return Get(condition.value != 0 ? args[1] : args[2]);
			}
			const auto a = Get(args[1]);
			const auto b = Get(args[2]);
			if (a.state == State::Const && a == b) {
				return a;
			}
			return Over();
		}
		if (op == ValueOpcode::LogicalAnd || op == ValueOpcode::LogicalOr) {
			const auto     a      = Get(args[0]);
			const auto     b      = Get(args[1]);
			const uint64_t absorb = op == ValueOpcode::LogicalAnd ? 0 : 1;
			if (a == Const(absorb) || b == Const(absorb)) {
				return Const(absorb);
			}
			// q is only observed where p holds: fold exec-merge selects inside q under p.
			if (op == ValueOpcode::LogicalAnd && FoldsToZeroUnderOther(args, index)) {
				return Const(0);
			}
		}
		if (op == ValueOpcode::Ballot) {
			const auto a = Get(args[0]);
			return a == Const(0) ? Const(0) : a.state == State::Unknown ? Lattice {} : Over();
		}
		if (op == ValueOpcode::WriteLane) {
			const auto a = Get(args[0]);
			const auto v = Get(args[1]);
			if (a.state == State::Unknown || v.state == State::Unknown) {
				return {};
			}
			return a.state == State::Const && a == v ? a : Over();
		}
		if (op == ValueOpcode::ReadLane || op == ValueOpcode::ReadFirstLane) {
			return Get(args[0]);
		}
		if (IsOneOf(op, {ValueOpcode::GetBuiltin, ValueOpcode::GetUserData, ValueOpcode::LaneId}) ||
		    node.type == Type::Void) {
			return Over();
		}
		if (args.empty() || HasAccess(node)) {
			return Over();
		}
		bool unknown = false;
		scratch.clear();
		for (const auto& arg: args) {
			const auto value = Get(arg);
			if (value.state == State::Over) {
				return Over();
			}
			unknown |= value.state == State::Unknown;
			scratch.push_back(value.value);
		}
		if (unknown) {
			return {};
		}
		return FromEval(op, scratch);
	}

	// LogicalAnd(p, q): ConstUnder(q, p) == 0 or ConstUnder(p, q) == 0. The walk registers the
	// node as a watcher of every value it consulted, so it is re-evaluated when one of them gains
	// or loses a constant.
	bool FoldsToZeroUnderOther(std::span<const RescaleOperand> args, uint32_t index) {
		watching = index;
		visit++;
		memo_pool.clear();
		return ConstUnder(args[1], args[0], 0, 0) == Const(0) || ConstUnder(args[0], args[1], 1, 0) == Const(0);
	}

	// The constant a has in the lanes where pred holds (pred_index names which LogicalAnd operand
	// pred is, for the memo).
	Lattice ConstUnder(const RescaleOperand& a, const RescaleOperand& pred, uint32_t pred_index, uint32_t depth) {
		if (a.IsNode() && visit_stamp[a.Node()] != visit) {
			visit_stamp[a.Node()] = visit;
			watch_links.push_back({watching, watch_head[a.Node()]});
			watch_head[a.Node()] = static_cast<uint32_t>(watch_links.size() - 1);
			memo_slot[a.Node()]  = static_cast<uint32_t>(memo_pool.size());
			memo_pool.emplace_back();
		}
		const auto known = Get(a);
		if (known.state == State::Const) {
			return known;
		}
		if (!a.IsNode() || depth > MaxConstUnderDepth) {
			return Over();
		}
		// The walk revisits shared subexpressions of the exec-mask algebra at every depth.
		auto* memo = &memo_pool[memo_slot[a.Node()]][depth * 2 + pred_index];
		if (memo->has_value()) {
			return **memo;
		}
		const auto value = ConstUnderNode(graph.nodes[a.Node()], pred, pred_index, depth);
		// The pool may have grown during the walk.
		memo_pool[memo_slot[a.Node()]][depth * 2 + pred_index] = value;
		return value;
	}

	Lattice ConstUnderNode(const RescaleNode& node, const RescaleOperand& pred, uint32_t pred_index,
	                       uint32_t depth) {
		const auto args = graph.Args(node);
		if (IsSelect(node.opcode)) {
			// Taken arm if pred decides the condition, else Over. The arms are pure, so they are
			// walked first: when every arm pred could select is Over, the proof is not needed.
			const auto  taken   = ConstUnder(args[1], pred, pred_index, depth + 1);
			const auto* negated = graph.ArgNode(node, 0);
			const bool  negation = negated != nullptr && negated->opcode == ValueOpcode::LogicalNot;
			const auto  other    = negation ? ConstUnder(args[2], pred, pred_index, depth + 1) : Over();
			if (taken.state == State::Over && other.state == State::Over) {
				return Over();
			}
			if (logic.Implies(pred, args[0])) {
				return taken;
			}
			if (negation && logic.Implies(pred, graph.Args(*negated)[0])) {
				return other;
			}
			return Over();
		}
		if (IsOneOf(node.opcode, {ValueOpcode::Phi, ValueOpcode::GetBuiltin, ValueOpcode::GetUserData,
		                          ValueOpcode::LaneId, ValueOpcode::Ballot}) ||
		    node.type == Type::Void || args.empty() || HasAccess(node)) {
			return Over();
		}
		// Only the result is observable (Implies is memoised and order-independent), so the
		// operand walk stops as soon as the result is decided: an op EvalConst cannot fold is
		// never constant, any other op is Over at its first Over operand, and a LogicalAnd is 0
		// at its first zero operand.
		const bool logical_and = node.opcode == ValueOpcode::LogicalAnd;
		if (!logical_and && !CanEvalConst(node.opcode)) {
			return Over();
		}
		std::array<uint64_t, 16> inline_constants;
		std::vector<uint64_t>    spilled;
		std::span<uint64_t>      constants(inline_constants.data(), args.size());
		if (args.size() > inline_constants.size()) {
			spilled.resize(args.size());
			constants = spilled;
		}
		bool over = false;
		for (size_t i = 0; i < args.size(); i++) {
			const auto value = ConstUnder(args[i], pred, pred_index, depth + 1);
			if (logical_and && value == Const(0)) {
				return Const(0);
			}
			if (value.state == State::Over) {
				if (!logical_and) {
					return Over();
				}
				over = true;
			}
			constants[i] = value.value;
		}
		if (over) {
			return Over();
		}
		return FromEval(node.opcode, constants);
	}

	const RescaleGraph&          graph;
	PredicateLogic&              logic;
	std::vector<Lattice>         values;
	std::vector<bool>            live_blocks;
	std::vector<bool>            dirty;
	std::vector<uint32_t>        dirty_in_block;
	// Live CFG edges, indexed first_edge[block] + successor position.
	std::vector<uint32_t>        first_edge;
	std::vector<bool>            live_edges;
	// Per value, a list of the LogicalAnd nodes whose ConstUnder walk consulted it.
	struct WatchLink {
		uint32_t watcher = 0;
		uint32_t next    = 0;
	};
	static constexpr uint32_t NoWatch = UINT32_MAX;
	std::vector<uint32_t>     watch_head;
	std::vector<WatchLink>    watch_links;
	// One walk's memo: per visited value, its result at each depth under either predicate.
	using MemoSlots = std::array<std::optional<Lattice>, 2 * (MaxConstUnderDepth + 1)>;
	std::vector<uint64_t>     visit_stamp;
	std::vector<uint32_t>     memo_slot;
	std::vector<MemoSlots>    memo_pool;
	uint64_t                  visit    = 0;
	uint32_t                  watching = 0;
	std::vector<uint64_t>                        scratch;
};

// ---------------------------------------------------------------------------------------------
// CoordinateFacts (S2): the affine fact of every value, by an optimistic fixpoint over the
// program in order, and the fact of an operand in the lanes where a predicate holds.
// ---------------------------------------------------------------------------------------------
class CoordinateFacts {
public:
	CoordinateFacts(const RescaleGraph& graph_, PredicateLogic& logic_, const std::array<uint32_t, 3>& threads_)
	    : graph(graph_), logic(logic_), threads(threads_), facts(graph_.nodes.size()),
	      align(graph_.nodes.size(), 0) {}

	// False when the fixpoint did not settle within the sweep limit.
	bool Solve() {
		for (uint32_t sweep = 0; sweep < MaxSweeps; sweep++) {
			bool changed = false;
			for (uint32_t index = 0; index < graph.nodes.size(); index++) {
				const auto& node = graph.nodes[index];
				if (node.type == Type::Void) {
					continue;
				}
				auto fact = Transfer(node, index);
				if (fact.kind == FactKind::Unknown) {
					continue;
				}
				auto& old = facts[index];
				if (old == fact) {
					continue;
				}
				// Monotone widening: a varying value never becomes more precise again (a Lin
				// may still widen to anything).
				if (!old.IsLin() && old.IsVarying() && !fact.IsVarying()) {
					continue;
				}
				old     = std::move(fact);
				changed = true;
			}
			if (!changed) {
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] const Fact& FactOf(uint32_t node) const { return facts[node]; }

	[[nodiscard]] Fact ValueFact(const RescaleOperand& operand) const {
		if (operand.IsNode()) {
			return facts[operand.Node()];
		}
		if (const auto value = operand.ImmU32()) {
			return Fact::Of(AffineForm::Constant(*value));
		}
		// Other immediates and registers: uniform.
		return Fact::Of(FactKind::Uniform);
	}

	// Known trailing zero bits of the uniform part (atoms and constant) of a form.
	[[nodiscard]] uint32_t AtomAlign(const AffineForm& lin) const {
		auto result = CountTrailingZeros(lin.c);
		for (const auto& atom: lin.atoms) {
			result = std::min(result, std::min(32u, align[atom.node] + CountTrailingZeros(atom.coef)));
		}
		return result;
	}

	// Upper bound of an L-only form with non-negative coefficients.
	[[nodiscard]] std::optional<int64_t> RangeMax(const AffineForm& lin) const {
		if (Signed(lin.ax) < 0 || Signed(lin.ay) < 0 || Signed(lin.c) < 0) {
			return std::nullopt;
		}
		return Signed(lin.ax) * (static_cast<int64_t>(threads[0]) - 1) +
		       Signed(lin.ay) * (static_cast<int64_t>(threads[1]) - 1) + Signed(lin.c);
	}

	// Peels exec-merge selects (v = exec ? new : old) that the consumer's predicate decides: the
	// consumer only reads the operand in lanes where pred holds.
	[[nodiscard]] RescaleOperand ResolveUnder(RescaleOperand a, const RescaleOperand& pred) {
		for (uint32_t depth = 0; a.IsNode() && depth < 64; depth++) {
			const auto& select = graph.nodes[a.Node()];
			if (select.opcode != ValueOpcode::SelectU32) {
				break;
			}
			const auto args = graph.Args(select);
			if (select.indexed_register_write) {
				// Assumption A1: the rung never really retargets this register.
				a = args[2];
			} else if (logic.Implies(pred, args[0])) {
				a = args[1];
			} else if (const auto* negated = graph.NodeOf(args[0]);
			           negated != nullptr && negated->opcode == ValueOpcode::LogicalNot &&
			           logic.Implies(pred, graph.Args(*negated)[0])) {
				a = args[2];
			} else {
				break;
			}
		}
		return a;
	}

	// The fact of a in the lanes where pred holds: selects peeled, and adds re-derived below them.
	[[nodiscard]] Fact FactUnder(const RescaleOperand& a, const RescaleOperand& pred) {
		const auto resolved = ResolveUnder(a, pred);
		if (const auto* node = graph.NodeOf(resolved);
		    node != nullptr && node->type == Type::U32 &&
		    (node->opcode == ValueOpcode::IAdd32 || node->opcode == ValueOpcode::ISub32)) {
			const auto args = graph.Args(*node);
			const auto f0   = FactUnder(args[0], pred);
			const auto f1   = FactUnder(args[1], pred);
			if (f0.IsLin() && f1.IsLin()) {
				return Fact::Of(f0.lin.Add(f1.lin, node->opcode == ValueOpcode::IAdd32 ? 1u : ~0u));
			}
		}
		return ValueFact(resolved);
	}

private:
	static constexpr uint32_t MaxSweeps = 64;

	Fact MakeAtom(uint32_t index, uint32_t atom_align) {
		align[index] = atom_align;
		return Fact::Of(AffineForm::OfAtom(index));
	}

	[[nodiscard]] const Fact& ArgFact(const RescaleOperand& operand, Fact& immediate) const {
		if (operand.IsNode()) {
			return facts[operand.Node()];
		}
		if (const auto value = operand.ImmU32()) {
			immediate = Fact::Of(AffineForm::Constant(*value));
		} else {
			immediate = Fact::Of(FactKind::Uniform);
		}
		return immediate;
	}

	static std::optional<uint32_t> Fold(ValueOpcode op, const std::vector<const Fact*>& a) {
		if (a.size() < 2) {
			return std::nullopt;
		}
		const auto x = a[0]->lin.c;
		const auto y = a[1]->lin.c;
		switch (op) {
			case ValueOpcode::IAdd32: return x + y;
			case ValueOpcode::ISub32: return x - y;
			case ValueOpcode::IMul32: return x * y;
			case ValueOpcode::ShiftLeftLogical32: return x << (y & 31u);
			case ValueOpcode::ShiftRightLogical32: return x >> (y & 31u);
			case ValueOpcode::BitwiseAnd32: return x & y;
			case ValueOpcode::BitwiseOr32: return x | y;
			case ValueOpcode::BitwiseXor32: return x ^ y;
			default: return std::nullopt;
		}
	}

	[[nodiscard]] uint32_t UniformAlign(ValueOpcode op, const std::vector<const Fact*>& a) const {
		const auto at = [&](size_t i) { return AtomAlign(a[i]->lin); };
		switch (op) {
			case ValueOpcode::BitwiseAnd32: return std::max(at(0), at(1));
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32: return std::min(at(0), at(1));
			case ValueOpcode::IMul32: return std::min(32u, at(0) + at(1));
			case ValueOpcode::ShiftRightLogical32:
				if (a[1]->lin.Const()) {
					const auto shift = a[1]->lin.c & 31u;
					return at(0) > shift ? at(0) - shift : 0u;
				}
				return 0;
			case ValueOpcode::SelectU32: return std::min(at(1), at(2));
			default: return 0;
		}
	}

	Fact Transfer(const RescaleNode& node, uint32_t index) {
		const auto op   = node.opcode;
		const auto args = graph.Args(node);
		immediates.resize(args.size());
		operand_facts.clear();
		for (size_t i = 0; i < args.size(); i++) {
			operand_facts.push_back(&ArgFact(args[i], immediates[i]));
		}
		const auto& f = operand_facts;

		if (op == ValueOpcode::Phi) {
			const Fact* first   = nullptr;
			bool        equal   = true;
			bool        varying = false;
			bool        uniform_or_wave = true;
			for (const auto* fact: f) {
				if (fact->kind == FactKind::Unknown) {
					continue;
				}
				if (first == nullptr) {
					first = fact;
				} else if (!(*fact == *first)) {
					equal = false;
				}
				const auto cls = fact->Class();
				varying |= cls == Uniformity::Varying;
				uniform_or_wave &= cls == Uniformity::Uniform || cls == Uniformity::Wave;
			}
			if (first == nullptr) {
				return {};
			}
			if (equal && first->IsLin()) {
				return *first;
			}
			// Differing uniform inputs merge behind a wave-level branch.
			if (node.type == Type::U32) {
				return Fact::Of(uniform_or_wave ? FactKind::Wave : FactKind::Var);
			}
			return Fact::Of(varying ? FactKind::Varying : FactKind::WaveUniform);
		}
		// An operand not known yet (a back edge on the first sweep): stay optimistic, the
		// fixpoint re-evaluates once it is known.
		if (std::any_of(f.begin(), f.end(), [](const Fact* fact) { return fact->kind == FactKind::Unknown; })) {
			return {};
		}
		const auto cls   = [&](size_t i) { return f[i]->Class(); };
		bool       all_u = true;
		bool       all_uw = true;
		for (size_t i = 0; i < f.size(); i++) {
			all_u &= cls(i) == Uniformity::Uniform;
			all_uw &= cls(i) == Uniformity::Uniform || cls(i) == Uniformity::Wave;
		}
		if (node.indexed_register_write) {
			return *f[2];
		}
		switch (op) {
			case ValueOpcode::GetBuiltin: {
				const auto kind      = args[0].ImmU32();
				const auto component = args[1].ImmU32();
				if (kind == static_cast<uint32_t>(StageInputKind::LocalInvocationId)) {
					return component == 0u   ? Fact::Of(AffineForm::LocalX())
					       : component == 1u ? Fact::Of(AffineForm::LocalY())
					                         : Fact::Of(FactKind::Var);
				}
				if (kind == static_cast<uint32_t>(StageInputKind::WorkgroupId)) {
					return MakeAtom(index, 0);
				}
				return Fact::Of(FactKind::Var);
			}
			case ValueOpcode::GetUserData:
			case ValueOpcode::GetShaderBase:
				return node.type == Type::U32 ? MakeAtom(index, 0) : Fact::Of(FactKind::Uniform);
			case ValueOpcode::LaneId: return Fact::Of(FactKind::Var);
			case ValueOpcode::Ballot:
			case ValueOpcode::AnyLane:
				return Fact::Of(cls(0) == Uniformity::Uniform ? FactKind::Uniform : FactKind::WaveUniform);
			case ValueOpcode::ReadLane:
			case ValueOpcode::ReadFirstLane:
				if (cls(0) == Uniformity::Uniform) {
					return f[0]->IsLin() ? *f[0] : MakeAtom(index, 0);
				}
				return Fact::Of(node.type == Type::U32 ? FactKind::Wave : FactKind::WaveUniform);
			default: break;
		}
		if (node.type == Type::U32 &&
		    std::all_of(f.begin(), f.end(), [](const Fact* fact) { return fact->IsLin(); })) {
			if (auto lin = LinearTransfer(node, f)) {
				return Fact::Of(std::move(*lin));
			}
			if (std::all_of(f.begin(), f.end(), [](const Fact* fact) { return fact->lin.Uniform(); })) {
				if (std::all_of(f.begin(), f.end(), [](const Fact* fact) { return fact->lin.Const(); })) {
					if (const auto folded = Fold(op, f)) {
						return Fact::Of(AffineForm::Constant(*folded));
					}
				}
				return MakeAtom(index, UniformAlign(op, f));
			}
			return Fact::Of(FactKind::Var);
		}
		if (all_u) {
			return node.type == Type::U32 ? MakeAtom(index, 0) : Fact::Of(FactKind::Uniform);
		}
		if (node.type == Type::U32) {
			return Fact::Of(all_uw ? FactKind::Wave : FactKind::Var);
		}
		return Fact::Of(all_uw ? FactKind::WaveUniform : FactKind::Varying);
	}

	// The linear transfer rules over all-Lin operands, or nullopt when none applies.
	std::optional<AffineForm> LinearTransfer(const RescaleNode& node, const std::vector<const Fact*>& f) const {
		if (f.size() < 2) {
			return std::nullopt;
		}
		const auto& a0 = f[0]->lin;
		const auto& a1 = f[1]->lin;
		switch (node.opcode) {
			case ValueOpcode::IAdd32: return a0.Add(a1);
			case ValueOpcode::ISub32: return a0.Add(a1, ~0u);
			case ValueOpcode::ShiftLeftLogical32:
				if (a1.Const()) {
					return a0.Mul(1u << (a1.c & 31u));
				}
				break;
			case ValueOpcode::IMul32:
				if (a1.Const()) {
					return a0.Mul(a1.c);
				}
				if (a0.Const()) {
					return a1.Mul(a0.c);
				}
				break;
			case ValueOpcode::BitwiseOr32:
				// An OR of bit-disjoint halves is an add: a uniform base aligned to 2^n OR'd
				// with a LocalInvocationID term known to stay below 2^n.
				for (const auto& [u, l]: {std::pair {&a0, &a1}, std::pair {&a1, &a0}}) {
					if (u->Uniform() && !l->Uniform() && l->atoms.empty()) {
						const auto hi = RangeMax(*l);
						if (hi && *hi < (int64_t {1} << AtomAlign(*u))) {
							return u->Add(*l);
						}
					}
				}
				break;
			case ValueOpcode::SelectU32:
				if (f.size() > 2 && f[1]->lin == f[2]->lin) {
					return f[1]->lin;
				}
				break;
			default: break;
		}
		return std::nullopt;
	}

	const RescaleGraph&         graph;
	PredicateLogic&             logic;
	std::array<uint32_t, 3>     threads;
	std::vector<Fact>           facts;
	// Known trailing zero bits per atom.
	std::vector<uint32_t>       align;
	std::vector<Fact>           immediates;
	std::vector<const Fact*>    operand_facts;
};

// ---------------------------------------------------------------------------------------------
// ProgramStructure: S0 (dispatch shape) and S1 (side effects).
// ---------------------------------------------------------------------------------------------
class ProgramStructure {
public:
	ProgramStructure(const RescaleGraph& graph_, const Sccp& sccp_, ReasonLog& log_)
	    : graph(graph_), sccp(sccp_), log(log_) {}

	void CheckShape(const TileRescaleShape& shape) {
		if (graph.dispatcher_fallback) {
			log.Reject("dispatcher fallback (unstructured CFG)");
		}
		const auto [w, h, d] = shape.threads;
		const auto pow2      = [](uint32_t v) { return v >= 2 && (v & (v - 1)) == 0; };
		if (d > 1 || !pow2(w) || !pow2(h)) {
			log.Reject(fmt::format("workgroup {}x{}x{} is not a power-of-two 2D tile", w, h, d));
		}
		// Every remapped slot must be whole warps: that keeps WorkGroupID uniform per warp.
		const uint64_t k2 = uint64_t {1} << (2u * shape.scale_log2);
		if (shape.wave_size != 32 || shape.host_subgroup_size != 32 ||
		    (static_cast<uint64_t>(w) * h) % (k2 * 32u) != 0) {
			log.Reject(fmt::format("wave{}/host subgroup {}/{} invocations: remapped slots are not "
			                       "whole warps",
			                       shape.wave_size, shape.host_subgroup_size, w * h));
		}
		if (shape.tg_size_en) {
			log.Reject("reads its wave index within the workgroup (tg_size_en)");
		}
	}

	void CheckSideEffects() {
		for (const auto& node: graph.nodes) {
			if (sccp.Dead(node)) {
				continue;
			}
			CheckEffect(node);
			CheckTexelAccess(node);
		}
	}

private:
	// WriteLane is classified by S6.
	static std::optional<std::string_view> RefusedOpReason(ValueOpcode op) {
		switch (op) {
			case ValueOpcode::DppMoveU32:
			case ValueOpcode::DppUpdateU32: return "DPP lane move";
			case ValueOpcode::WqmU32:
			case ValueOpcode::WqmU64: return "whole-quad mode";
			case ValueOpcode::Permlane16U32:
			case ValueOpcode::BpermuteU32: return "lane permute";
			case ValueOpcode::SwizzleU32: return "LDS swizzle (lane exchange)";
			case ValueOpcode::Barrier: return "workgroup barrier";
			case ValueOpcode::DataAppend: return "append counter";
			case ValueOpcode::DataConsume: return "consume counter";
			case ValueOpcode::ImageQueryDimensions:
				return "image size query (would need native-size patching)";
			default: return std::nullopt;
		}
	}

	static std::string_view DimensionName(Decoder::ImageDimension dimension) {
		switch (dimension) {
			case Decoder::ImageDimension::Unknown: return "unknown";
			case Decoder::ImageDimension::Dim1D: return "1d";
			case Decoder::ImageDimension::Dim1DArray: return "1darray";
			case Decoder::ImageDimension::Dim2D: return "2d";
			case Decoder::ImageDimension::Dim3D: return "3d";
			case Decoder::ImageDimension::Dim2DArray: return "2darray";
			case Decoder::ImageDimension::Dim2DMsaa: return "2dmsaa";
			case Decoder::ImageDimension::Dim2DMsaaArray: return "2dmsaaarray";
		}
		return "?";
	}

	void CheckEffect(const RescaleNode& node) {
		if (const auto reason = RefusedOpReason(node.opcode)) {
			log.Reject(*reason, &node);
		} else if (node.shared_access != SharedAccess::None) {
			log.Reject("workgroup-shared (LDS/GDS) access", &node);
		} else if (node.buffer_access == BufferAccess::Write || node.buffer_access == BufferAccess::Atomic) {
			log.Reject("buffer store or atomic", &node);
		} else if (node.address_access == AddressAccess::Write) {
			// Scratch is private to the invocation.
			const auto* memory = graph.MemoryOf(node);
			if (memory == nullptr || memory->kind != ResourceKind::Scratch) {
				log.Reject("global/flat store", &node);
			}
		} else if (node.image_access == ImageAccess::Atomic) {
			log.Reject("image atomic", &node);
		} else if (node.opcode == ValueOpcode::GetBuiltin) {
			const auto args      = graph.Args(node);
			const auto kind      = args[0].ImmU32();
			const auto component = args[1].ImmU32();
			if (kind == static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) {
				log.Reject("reads builtin GlobalInvocationID", &node);
			} else if (kind == static_cast<uint32_t>(StageInputKind::NumWorkgroups)) {
				log.Reject("reads builtin NumWorkgroups", &node);
			} else if (kind == static_cast<uint32_t>(StageInputKind::LocalInvocationIndex)) {
				log.Reject("reads builtin LocalInvocationIndex", &node);
			} else if (kind == static_cast<uint32_t>(StageInputKind::LocalInvocationId) &&
			           component != 0u && component != 1u) {
				log.Reject("reads LocalInvocationID.z", &node);
			}
		}
	}

	void CheckTexelAccess(const RescaleNode& node) {
		const auto op    = node.opcode;
		const bool texel = op == ValueOpcode::ImageRead || op == ValueOpcode::ImageWrite;
		if (node.image_access == ImageAccess::None && !texel) {
			return;
		}
		// Normalised coordinates need no change.
		if (IsOneOf(op, {ValueOpcode::ImageSampleRaw, ValueOpcode::ImageGatherRaw, ValueOpcode::ImageQueryLod})) {
			return;
		}
		if (!texel) {
			log.Reject("texel-addressed image op other than load/store", &node);
			return;
		}
		const auto* memory = graph.MemoryOf(node);
		if (memory == nullptr || memory->resource >= graph.images.size()) {
			log.Reject("image access without metadata", &node);
			return;
		}
		const auto& image = graph.images[memory->resource];
		if (image.dimension != Decoder::ImageDimension::Dim2D &&
		    image.dimension != Decoder::ImageDimension::Dim2DArray) {
			log.Reject(fmt::format("texel access into a {} image", DimensionName(image.dimension)), &node);
		}
		if (memory->x_offset != 0 || memory->x_width != 32 || memory->y_offset != 32 ||
		    memory->y_width != 32 || memory->address_components < 2) {
			log.Reject("packed/short texel address", &node);
		}
		if (image.mip_mode != ImageMipMode::None) {
			log.Reject("texel access with a dynamic mip level", &node);
		}
	}

	const RescaleGraph& graph;
	const Sccp&         sccp;
	ReasonLog&          log;
};

// ---------------------------------------------------------------------------------------------
// StoreOwnership: S3 (every image store at exactly (Lx + Ux, Ly + Uy) with a tile origin that is
// a multiple of k) and S4 (load classes; a load of a written image must be the store's own pixel).
// Addresses are read under the access's own predicate.
// ---------------------------------------------------------------------------------------------
class StoreOwnership {
public:
	StoreOwnership(const RescaleGraph& graph_, CoordinateFacts& facts_, ReasonLog& log_, uint32_t scale_log2_)
	    : graph(graph_), facts(facts_), log(log_), scale_log2(scale_log2_) {}

	void Check() {
		std::vector<const RescaleNode*> stores;
		for (const auto& node: graph.nodes) {
			if (node.opcode == ValueOpcode::ImageWrite) {
				if (const auto* memory = graph.MemoryOf(node)) {
					if (std::find(written.begin(), written.end(), memory->resource) == written.end()) {
						written.push_back(memory->resource);
					}
				}
				stores.push_back(&node);
			}
		}
		std::sort(written.begin(), written.end());
		for (const auto* store: stores) {
			CheckStore(*store);
		}
		for (const auto& node: graph.nodes) {
			if (node.opcode == ValueOpcode::ImageRead) {
				CheckLoad(node);
			}
		}
	}

	std::vector<uint32_t> written;
	uint32_t              store_align   = 32;
	uint32_t              max_neighbour = 0;
	NoteCounter           loads;

private:
	struct AddressFacts {
		Fact              x;
		Fact              y;
		std::vector<Fact> rest;
	};

	AddressFacts AddressFactsOf(const RescaleNode& access) {
		const auto* address = graph.ArgNode(access, 1);
		if (address == nullptr || address->opcode != ValueOpcode::MakeImageAddress) {
			return {};
		}
		const auto& pred = graph.Args(access)[access.opcode == ValueOpcode::ImageWrite ? 3 : 2];
		const auto  args = graph.Args(*address);
		AddressFacts result {facts.FactUnder(args[0], pred), facts.FactUnder(args[1], pred), {}};
		for (size_t i = 2; i < args.size(); i++) {
			result.rest.push_back(facts.FactUnder(args[i], pred));
		}
		return result;
	}

	static bool IsOwnForm(const Fact& x, const Fact& y) {
		return x.IsLin() && y.IsLin() && x.lin.ax == 1 && x.lin.ay == 0 && y.lin.ax == 0 && y.lin.ay == 1;
	}

	void CheckStore(const RescaleNode& store) {
		const auto address = AddressFactsOf(store);
		if (!IsOwnForm(address.x, address.y)) {
			log.Reject(fmt::format("image store not at (Lx + Ux, Ly + Uy): x={} y={}",
			                       address.x.ToString(graph), address.y.ToString(graph)),
			           &store);
			return;
		}
		auto       ux      = address.x.lin.Add(AffineForm::LocalX(), ~0u);
		auto       uy      = address.y.lin.Add(AffineForm::LocalY(), ~0u);
		const auto x_align = facts.AtomAlign(ux);
		const auto y_align = facts.AtomAlign(uy);
		store_align        = std::min({store_align, x_align, y_align});
		if (std::min(x_align, y_align) < scale_log2) {
			log.Reject(fmt::format("store tile origin not provably a multiple of {} (x align {}, y "
			                       "align {})",
			                       1u << scale_log2, x_align, y_align),
			           &store);
		}
		for (const auto& fact: address.rest) {
			if (fact.Class() != Uniformity::Uniform && !(fact.IsLin() && fact.lin.Const())) {
				log.Reject("store array/lod component not workgroup-uniform", &store);
				break;
			}
		}
		const std::pair base {std::move(ux), std::move(uy)};
		if (std::find(store_bases.begin(), store_bases.end(), base) == store_bases.end()) {
			store_bases.push_back(base);
		}
	}

	std::string_view Classify(const Fact& x, const Fact& y) {
		if (x.IsLin() && y.IsLin()) {
			if (x.lin.Uniform() && y.lin.Uniform()) {
				return "uniform";
			}
			if (IsOwnForm(x, y)) {
				if (x.lin.c == 0 && y.lin.c == 0) {
					return "own";
				}
				max_neighbour = std::max({max_neighbour, static_cast<uint32_t>(std::abs(Signed(x.lin.c))),
				                          static_cast<uint32_t>(std::abs(Signed(y.lin.c)))});
				return "neighbour";
			}
			return "affine-other";
		}
		if (x.kind == FactKind::Wave || y.kind == FactKind::Wave) {
			return "wave-uniform";
		}
		return "data-dependent";
	}

	void CheckLoad(const RescaleNode& load) {
		const auto* memory  = graph.MemoryOf(load);
		const auto  address = AddressFactsOf(load);
		const auto  cls     = Classify(address.x, address.y);
		loads.Add(cls);
		if (memory == nullptr ||
		    std::find(written.begin(), written.end(), memory->resource) == written.end()) {
			return;
		}
		// Anything but a read-modify-write of the own pixel reads texels another invocation writes.
		if (cls != "own") {
			log.Reject(fmt::format("reads a written image away from its own pixel ({})", cls), &load);
			return;
		}
		const std::pair base {address.x.lin.Add(AffineForm::LocalX(), ~0u),
		                      address.y.lin.Add(AffineForm::LocalY(), ~0u)};
		if (std::find(store_bases.begin(), store_bases.end(), base) == store_bases.end()) {
			log.Reject("reads a written image at a different tile origin", &load);
		}
	}

	const RescaleGraph&                           graph;
	CoordinateFacts&                              facts;
	ReasonLog&                                    log;
	uint32_t                                      scale_log2;
	std::vector<std::pair<AffineForm, AffineForm>> store_bases;
};

// ---------------------------------------------------------------------------------------------
// ParityCheck (S5): LocalInvocationID reaches integer consumers only through monotone operations,
// so the remap's representative pixels cannot see a checkerboard or an edge-lane special case.
// Taint: values with an L term, plus values derived by monotone ops; merges only when every
// non-constant input is tainted (exec-merge selects mix a coordinate register with unrelated
// values for inactive lanes).
// ---------------------------------------------------------------------------------------------
class ParityCheck {
public:
	ParityCheck(const RescaleGraph& graph_, const CoordinateFacts& facts_, const Sccp& sccp_, ReasonLog& log_,
	            uint32_t scale_log2_)
	    : graph(graph_), facts(facts_), sccp(sccp_), log(log_), scale_log2(scale_log2_) {}

	void Run() {
		std::vector<bool> tainted(graph.nodes.size(), false);
		for (uint32_t index = 0; index < graph.nodes.size(); index++) {
			const auto& fact = facts.FactOf(index);
			tainted[index]   = fact.IsLin() && !fact.lin.Uniform();
		}
		const auto is_tainted = [&](const RescaleOperand& a) { return a.IsNode() && tainted[a.Node()]; };
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t index = 0; index < graph.nodes.size(); index++) {
				const auto& node = graph.nodes[index];
				if (tainted[index] || node.type != Type::U32 || sccp.Dead(node)) {
					continue;
				}
				auto args = graph.Args(node);
				if (IsMonotone(node.opcode) && std::any_of(args.begin(), args.end(), is_tainted)) {
					tainted[index] = changed = true;
				} else if (node.opcode == ValueOpcode::Phi || node.opcode == ValueOpcode::SelectU32) {
					if (node.opcode == ValueOpcode::SelectU32) {
						args = node.indexed_register_write ? args.subspan(2, 1) : args.subspan(1);
					}
					bool any_node = false;
					bool all      = true;
					for (const auto& a: args) {
						if (a.IsNode()) {
							any_node = true;
							all &= is_tainted(a) || a.Node() == index;
						}
					}
					if (any_node && all) {
						tainted[index] = changed = true;
					}
				}
			}
		}
		for (uint32_t index = 0; index < graph.nodes.size(); index++) {
			if (tainted[index]) {
				CheckUsers(index);
			}
		}
	}

	NoteCounter notes;

private:
	static bool IsMonotone(ValueOpcode op) {
		return IsOneOf(op, {ValueOpcode::ShiftRightLogical32, ValueOpcode::ShiftRightArithmetic32,
		                    ValueOpcode::UDiv32, ValueOpcode::UMulHi, ValueOpcode::SMulHi,
		                    ValueOpcode::UMin32, ValueOpcode::UMax32, ValueOpcode::SMin32,
		                    ValueOpcode::SMax32, ValueOpcode::SMinTri32, ValueOpcode::UMinTri32,
		                    ValueOpcode::SMaxTri32, ValueOpcode::UMaxTri32, ValueOpcode::SMedTri32,
		                    ValueOpcode::UMedTri32, ValueOpcode::IAbs32, ValueOpcode::IAdd32,
		                    ValueOpcode::ISub32, ValueOpcode::IMul32, ValueOpcode::ShiftLeftLogical32});
	}

	static bool IsOrderedCompare(ValueOpcode op) {
		return IsOneOf(op, {ValueOpcode::ULessThan32, ValueOpcode::SLessThan32,
		                    ValueOpcode::ULessThanEqual32, ValueOpcode::SLessThanEqual32,
		                    ValueOpcode::UGreaterThan32, ValueOpcode::SGreaterThan32,
		                    ValueOpcode::UGreaterThanEqual32, ValueOpcode::SGreaterThanEqual32});
	}

	void CheckUsers(uint32_t source) {
		const auto& fact = facts.FactOf(source);
		for (const auto& use: graph.Users(graph.nodes[source])) {
			const auto& user = graph.nodes[use.user];
			const auto  op   = user.opcode;
			const auto  args = graph.Args(user);
			if (sccp.Dead(user) || IsMonotone(op) ||
			    IsOneOf(op, {ValueOpcode::Phi, ValueOpcode::SelectU32, ValueOpcode::MakeImageAddress})) {
				continue;
			}
			if (op == ValueOpcode::IndexedVectorLoad) {
				notes.Add("indexed-register candidate (A1)");
				continue;
			}
			if (IsOrderedCompare(op)) {
				notes.Add("ordered compare");
				continue;
			}
			if (op == ValueOpcode::ConvertF32U32 || op == ValueOpcode::ConvertF32S32) {
				notes.Add("to float");
				continue;
			}
			if (op == ValueOpcode::BitwiseOr32 && facts.FactOf(use.user).IsLin()) {
				continue; // proven disjoint add
			}
			if (op == ValueOpcode::BitwiseAnd32) {
				if (const auto mask = args[1 - use.operand].ImmU32()) {
					if ((*mask & ((1u << scale_log2) - 1u)) == 0) {
						notes.Add("quantising mask");
						continue;
					}
					if (fact.IsLin() && fact.lin.atoms.empty() && Signed(fact.lin.c) >= 0) {
						const auto hi = facts.RangeMax(fact.lin);
						if (hi) {
							const auto covered = (uint64_t {1} << std::bit_width(static_cast<uint64_t>(*hi))) - 1u;
							if ((*mask | covered) == *mask) {
								notes.Add("identity mask");
								continue;
							}
						}
					}
				}
				log.Reject("extracts low bits of a pixel coordinate", &user);
				continue;
			}
			if ((op == ValueOpcode::BitFieldUExtract || op == ValueOpcode::BitFieldSExtract) && use.operand == 0) {
				const auto offset = args[1].ImmU32();
				if (offset && *offset >= scale_log2) {
					notes.Add("high-bit extract");
					continue;
				}
				log.Reject("bit-field extract of a pixel coordinate's low bits", &user);
				continue;
			}
			if (op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32) {
				const auto& other = args[1 - use.operand];
				if (facts.ValueFact(other).Class() == Uniformity::Uniform && GuardsLoadsOnly(use.user)) {
					notes.Add("border guard of a texel load");
					continue;
				}
				log.Reject("equality test on a pixel coordinate (parity/edge-lane sensitive)", &user);
				continue;
			}
			if (user.buffer_access == BufferAccess::Read || user.address_access == AddressAccess::Read ||
			    op == ValueOpcode::ReadConstBuffer) {
				notes.Add("per-pixel buffer/global read");
				continue;
			}
			if (IsOneOf(op, {ValueOpcode::CompositeConstructU32x2, ValueOpcode::CompositeConstructU32x3,
			                 ValueOpcode::CompositeConstructU32x4})) {
				notes.Add("vector construct");
				continue;
			}
			if (op == ValueOpcode::BitCastF32U32) {
				log.Reject("pixel coordinate reinterpreted as float bits", &user);
				continue;
			}
			log.Reject(fmt::format("pixel coordinate reaches {}", ValueOpcodeName(op)), &user);
		}
	}

	// True when a predicate (typically `y != height-1`) only steers which texel an image LOAD
	// reads: it flows through predicate/exec-mask algebra into select conditions whose results
	// are texel-load coordinates. A representative pixel never meets an odd border, but its taps
	// stay in range either way. Anything that reaches a store, a store predicate or data is not.
	bool GuardsLoadsOnly(uint32_t predicate) const {
		static constexpr uint32_t MaxVisited = 400;
		std::unordered_set<uint32_t> seen;
		std::vector<uint32_t>        work {predicate};
		while (!work.empty()) {
			const auto index = work.back();
			work.pop_back();
			if (!seen.insert(index).second) {
				continue;
			}
			if (seen.size() > MaxVisited) {
				return false;
			}
			for (const auto& use: graph.Users(graph.nodes[index])) {
				const auto& user = graph.nodes[use.user];
				if (sccp.Dead(user)) {
					continue;
				}
				if (user.opcode == ValueOpcode::SelectU32 && use.operand == 0) {
					std::unordered_set<uint64_t> visited;
					if (!FeedsLoadAddress(use.user, 0, visited, false)) {
						return false;
					}
					continue;
				}
				if (IsOneOf(user.opcode,
				            {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr, ValueOpcode::LogicalNot,
				             ValueOpcode::LogicalXor, ValueOpcode::Ballot, ValueOpcode::CompositeExtractU32x4,
				             ValueOpcode::CompositeExtractU32x2, ValueOpcode::ShiftRightLogical32,
				             ValueOpcode::BitwiseAnd32, ValueOpcode::BitwiseOr32, ValueOpcode::BitwiseXor32,
				             ValueOpcode::BitwiseNot32, ValueOpcode::INotEqual32, ValueOpcode::IEqual32,
				             ValueOpcode::Phi, ValueOpcode::SelectU1})) {
					work.push_back(use.user);
					continue;
				}
				if (user.opcode == ValueOpcode::ImageRead && use.operand == 2) {
					continue;
				}
				return false;
			}
		}
		return true;
	}

	// The value (a guarded neighbour coordinate) is only consumed as a texel-LOAD address. Flows
	// into the `old` arm of a later exec-merge select are register reuse: the value survives there
	// only in lanes that skipped the new assignment, where the source variable is dead and the
	// register holds another one. Such stale flows are ignored unless they reach a store address.
	bool FeedsLoadAddress(uint32_t index, uint32_t depth, std::unordered_set<uint64_t>& seen, bool stale) const {
		static constexpr uint32_t MaxDepth = 48;
		if (depth > MaxDepth || !seen.insert((static_cast<uint64_t>(index) << 1u) | (stale ? 1u : 0u)).second) {
			return true;
		}
		for (const auto& use: graph.Users(graph.nodes[index])) {
			const auto& user = graph.nodes[use.user];
			if (sccp.Dead(user)) {
				continue;
			}
			const auto op = user.opcode;
			if (op == ValueOpcode::MakeImageAddress) {
				const auto consumers = graph.Users(user);
				if (std::any_of(consumers.begin(), consumers.end(), [&](const RescaleUse& consumer) {
					    return graph.nodes[consumer.user].opcode == ValueOpcode::ImageWrite;
				    })) {
					return false;
				}
				if (!stale && std::any_of(consumers.begin(), consumers.end(), [&](const RescaleUse& consumer) {
					    return graph.nodes[consumer.user].opcode != ValueOpcode::ImageRead;
				    })) {
					return false;
				}
				continue;
			}
			if (op == ValueOpcode::SelectU32) {
				if (use.operand == 0) {
					if (!stale) {
						return false;
					}
					continue;
				}
				if (!FeedsLoadAddress(use.user, depth + 1, seen, stale || use.operand == 2)) {
					return false;
				}
				continue;
			}
			if (IsOneOf(op, {ValueOpcode::IAdd32, ValueOpcode::ISub32, ValueOpcode::Phi, ValueOpcode::UMin32,
			                 ValueOpcode::UMax32, ValueOpcode::SMin32, ValueOpcode::SMax32})) {
				if (!FeedsLoadAddress(use.user, depth + 1, seen, stale)) {
					return false;
				}
				continue;
			}
			if (op == ValueOpcode::IndexedVectorLoad) {
				continue;
			}
			if (stale && user.image_access == ImageAccess::None && user.buffer_access == BufferAccess::None &&
			    user.address_access == AddressAccess::None) {
				continue;
			}
			return false;
		}
		return true;
	}

	const RescaleGraph&    graph;
	const CoordinateFacts& facts;
	const Sccp&            sccp;
	ReasonLog&             log;
	uint32_t               scale_log2;
};

// ---------------------------------------------------------------------------------------------
// CrossLaneClassifier (S6): every operation whose per-lane result can depend on which
// invocations share the wave. Ballot masks are tracked through mask algebra and may only reach
// compares, first-active-lane searches, per-lane bit tests and lane indices; the program's class
// is the worst one seen (C0 exec masks, C1 waterfalls, C2 lane-serialised wave code, X refuses).
// ---------------------------------------------------------------------------------------------
class CrossLaneClassifier {
public:
	static constexpr uint32_t Refused = 3;

	CrossLaneClassifier(const RescaleGraph& graph_, const CoordinateFacts& facts_, const Sccp& sccp_, ReasonLog& log_)
	    : graph(graph_), facts(facts_), sccp(sccp_), log(log_), mask(graph_.nodes.size()),
	      lane_index(graph_.nodes.size()), scratch(graph_.nodes.size()) {}

	void Run() {
		Propagate();
		for (uint32_t index = 0; index < graph.nodes.size(); index++) {
			if (mask[index] && !sccp.Dead(graph.nodes[index])) {
				CheckMaskUsers(index);
			}
		}
		for (const auto& node: graph.nodes) {
			if (sccp.Dead(node)) {
				continue;
			}
			switch (node.opcode) {
				case ValueOpcode::LaneId: CheckLaneId(node); break;
				case ValueOpcode::WriteLane: CheckWriteLane(node); break;
				case ValueOpcode::ReadLane:
				case ValueOpcode::ReadFirstLane: CheckReadLane(node); break;
				default: break;
			}
		}
	}

	uint32_t    worst = 0;
	NoteCounter notes;

private:
	[[nodiscard]] bool In(const std::vector<bool>& set, const RescaleOperand& a) const {
		return a.IsNode() && set[a.Node()];
	}
	[[nodiscard]] bool AnyIn(const std::vector<bool>& set, const RescaleNode& node) const {
		const auto args = graph.Args(node);
		return std::any_of(args.begin(), args.end(), [&](const RescaleOperand& a) { return In(set, a); });
	}
	// Uniform per wave, or a lane index.
	[[nodiscard]] bool WaveUniform(const RescaleOperand& a) const {
		const auto cls = facts.ValueFact(a).Class();
		return cls == Uniformity::Uniform || cls == Uniformity::Wave || In(lane_index, a);
	}
	void Worsen(uint32_t cls) { worst = std::max(worst, cls); }

	void Propagate() {
		for (uint32_t index = 0; index < graph.nodes.size(); index++) {
			const auto& node = graph.nodes[index];
			mask[index]      = node.opcode == ValueOpcode::Ballot && !sccp.Dead(node);
			scratch[index]   = node.opcode == ValueOpcode::WriteLane && !sccp.Dead(node);
		}
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t index = 0; index < graph.nodes.size(); index++) {
				const auto& node = graph.nodes[index];
				if (sccp.Dead(node)) {
					continue;
				}
				const auto op = node.opcode;
				if (!mask[index] &&
				    IsOneOf(op, {ValueOpcode::CompositeExtractU32x4, ValueOpcode::CompositeExtractU32x2,
				                 ValueOpcode::BitwiseAnd32, ValueOpcode::BitwiseOr32, ValueOpcode::BitwiseXor32,
				                 ValueOpcode::BitwiseNot32, ValueOpcode::Phi, ValueOpcode::SelectU32,
				                 ValueOpcode::CompositeConstructU64, ValueOpcode::CompositeConstructU32x2}) &&
				    AnyIn(mask, node)) {
					mask[index] = changed = true;
				}
				if (!lane_index[index] &&
				    ((op == ValueOpcode::FindILsb32 && node.num_args > 0 && In(mask, graph.Args(node)[0])) ||
				     (IsOneOf(op, {ValueOpcode::BitwiseAnd32, ValueOpcode::Phi, ValueOpcode::SelectU32,
				                   ValueOpcode::IAdd32, ValueOpcode::ISub32}) &&
				      AnyIn(lane_index, node)))) {
					lane_index[index] = changed = true;
				}
			}
		}
		// Lane-scratch registers: WriteLane chains and merges of them.
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t index = 0; index < graph.nodes.size(); index++) {
				const auto& node = graph.nodes[index];
				if (!scratch[index] && !sccp.Dead(node) &&
				    (node.opcode == ValueOpcode::Phi || node.opcode == ValueOpcode::SelectU32) &&
				    AnyIn(scratch, node)) {
					scratch[index] = changed = true;
				}
			}
		}
	}

	void CheckMaskUsers(uint32_t source) {
		for (const auto& use: graph.Users(graph.nodes[source])) {
			const auto& user = graph.nodes[use.user];
			const auto  op   = user.opcode;
			if (sccp.Dead(user) || mask[use.user] || lane_index[use.user]) {
				continue;
			}
			if (IsOneOf(op, {ValueOpcode::IEqual32, ValueOpcode::INotEqual32, ValueOpcode::IEqual64,
			                 ValueOpcode::INotEqual64, ValueOpcode::ULessThan32, ValueOpcode::UGreaterThan32})) {
				notes.Add("mask compare");
				continue;
			}
			if (op == ValueOpcode::FindILsb32) {
				continue;
			}
			if ((op == ValueOpcode::ShiftRightLogical32 || op == ValueOpcode::ShiftRightLogical64) &&
			    use.operand == 0) {
				notes.Add("per-lane bit test");
				continue;
			}
			if (IsOneOf(op, {ValueOpcode::ReadLane, ValueOpcode::ReadFirstLane, ValueOpcode::WriteLane})) {
				continue;
			}
			if (op == ValueOpcode::BitCount32 || op == ValueOpcode::BitCount64) {
				log.Reject("counts lanes of an exec/ballot mask (prefix/compaction)", &user);
			} else {
				log.Reject(fmt::format("exec/ballot mask reaches {}", ValueOpcodeName(op)), &user);
			}
			Worsen(Refused);
		}
	}

	void CheckLaneId(const RescaleNode& lane) {
		for (const auto& use: graph.Users(lane)) {
			const auto& user = graph.nodes[use.user];
			const auto  args = graph.Args(user);
			// A lane & 31 (or 63) is looked through to its own users.
			const bool wrapped = user.opcode == ValueOpcode::BitwiseAnd32 &&
			                     (args[1 - use.operand].ImmU32() == 31u || args[1 - use.operand].ImmU32() == 63u);
			const auto uses    = wrapped ? graph.Users(user) : std::span<const RescaleUse>(&use, 1);
			for (const auto& inner: uses) {
				const auto& x = graph.nodes[inner.user];
				if (sccp.Dead(x)) {
					continue;
				}
				if ((x.opcode == ValueOpcode::ShiftRightLogical32 || x.opcode == ValueOpcode::ShiftRightLogical64) &&
				    inner.operand == 1) {
					notes.Add("per-lane bit test");
					continue;
				}
				if (x.opcode == ValueOpcode::ShiftLeftLogical32 && inner.operand == 1) {
					notes.Add("lane bit");
					continue;
				}
				if ((x.opcode == ValueOpcode::IEqual32 || x.opcode == ValueOpcode::INotEqual32) &&
				    WaveUniform(graph.Args(x)[1 - inner.operand])) {
					notes.Add("is-selected-lane test");
					Worsen(2);
					continue;
				}
				log.Reject(fmt::format("lane position used as data ({})", ValueOpcodeName(x.opcode)), &x);
				Worsen(Refused);
			}
		}
	}

	void CheckWriteLane(const RescaleNode& write) {
		const auto args = graph.Args(write);
		if (WaveUniform(args[1]) && WaveUniform(args[2])) {
			notes.Add("lane-scratch write");
			Worsen(2);
		} else {
			log.Reject("writes a varying value or lane of a register", &write);
			Worsen(Refused);
		}
	}

	void CheckReadLane(const RescaleNode& read) {
		const auto args   = graph.Args(read);
		const auto source = args[0];
		if (facts.ValueFact(source).Class() == Uniformity::Uniform) {
			notes.Add("uniform broadcast");
			return;
		}
		const bool read_lane = read.opcode == ValueOpcode::ReadLane;
		if (In(scratch, source) && (!read_lane || WaveUniform(args[1]))) {
			notes.Add("lane-scratch read");
			Worsen(2);
			return;
		}
		if (read_lane && !In(lane_index, args[1])) {
			log.Reject("reads a fixed or computed lane of a varying value", &read);
			Worsen(Refused);
			return;
		}
		const auto users   = graph.Users(read);
		const bool guarded = std::any_of(users.begin(), users.end(), [&](const RescaleUse& use) {
			const auto& user = graph.nodes[use.user];
			return (user.opcode == ValueOpcode::IEqual32 || user.opcode == ValueOpcode::INotEqual32) &&
			       graph.Args(user)[1 - use.operand] == source;
		});
		if (guarded) {
			notes.Add("waterfall (C1)");
			Worsen(1);
		} else {
			notes.Add("first-active-lane read (C2)");
			Worsen(2);
		}
	}

	const RescaleGraph&    graph;
	const CoordinateFacts& facts;
	const Sccp&            sccp;
	ReasonLog&             log;
	std::vector<bool>      mask;
	std::vector<bool>      lane_index;
	std::vector<bool>      scratch;
};

constexpr std::string_view CrossLaneName(uint32_t cls) {
	constexpr std::array<std::string_view, 4> names {"C0", "C1", "C2", "X"};
	return names[std::min<uint32_t>(cls, 3)];
}

constexpr CrossLaneClass ToCrossLaneClass(uint32_t cls) {
	switch (cls) {
		case 0: return CrossLaneClass::ExecMaskOnly;
		case 1: return CrossLaneClass::Waterfall;
		default: return CrossLaneClass::LaneSerialized;
	}
}

// Float conversions of exactly the invocation's own pixel coordinate on one axis.
std::vector<const Inst*> OwnCoordinateConversions(const RescaleGraph& graph, const CoordinateFacts& facts,
                                                  const Sccp& sccp) {
	std::vector<const Inst*> result;
	for (const auto& node: graph.nodes) {
		if (node.inst == nullptr || sccp.Dead(node) ||
		    (node.opcode != ValueOpcode::ConvertF32U32 && node.opcode != ValueOpcode::ConvertF32S32)) {
			continue;
		}
		const auto fact = facts.ValueFact(graph.Args(node)[0]);
		if (fact.IsLin() && ((fact.lin.ax == 1 && fact.lin.ay == 0) || (fact.lin.ax == 0 && fact.lin.ay == 1))) {
			result.push_back(node.inst);
		}
	}
	return result;
}

} // namespace

TileRescaleReport AnalyzeRescaleGraph(const RescaleGraph& graph, const TileRescaleShape& shape) {
	const auto      s = shape.scale_log2;
	ReasonLog       log;
	PredicateLogic  logic(graph);
	Sccp            sccp(graph, logic);
	sccp.Run();

	ProgramStructure structure(graph, sccp, log);
	structure.CheckShape(shape);
	structure.CheckSideEffects();

	CoordinateFacts facts(graph, logic, shape.threads);
	if (!facts.Solve()) {
		log.Reject("affine analysis did not converge");
	}
	StoreOwnership ownership(graph, facts, log, s);
	ownership.Check();
	ParityCheck parity(graph, facts, sccp, log, s);
	parity.Run();
	CrossLaneClassifier cross_lane(graph, facts, sccp, log);
	cross_lane.Run();

	TileRescaleReport report;
	auto&             plan = report.plan;
	plan.accepted          = log.reasons.empty();
	plan.reasons           = std::move(log.reasons);
	plan.cross_lane        = ToCrossLaneClass(cross_lane.worst);
	plan.scale_log2        = plan.accepted ? s : 0;
	plan.store_align       = ownership.store_align;
	if (plan.accepted) {
		plan.own_coordinate_conversions = OwnCoordinateConversions(graph, facts, sccp);
	}
	plan.summary = fmt::format("{} loads={} store_align={} nb={} class={} parity: {} | cross-lane: {}",
	                           plan.accepted ? "ACCEPT" : "REJECT", ownership.loads.ToString(),
	                           ownership.store_align, ownership.max_neighbour,
	                           CrossLaneName(cross_lane.worst), parity.notes.ToString(),
	                           cross_lane.notes.ToString());
	report.load_classes     = std::move(ownership.loads.counts);
	report.parity_notes     = std::move(parity.notes.counts);
	report.cross_lane_notes = std::move(cross_lane.notes.counts);
	report.written_images   = std::move(ownership.written);
	report.max_neighbour    = ownership.max_neighbour;
	return report;
}

TileRescalePlan AnalyzeTileRescale(const Program& program, const TileRescaleShape& shape) {
	return AnalyzeRescaleGraph(BuildRescaleGraph(program), shape).plan;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
