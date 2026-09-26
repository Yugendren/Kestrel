#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;
class CompiledSrtPlan;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
// Makes a guest range readable (GPU-written data synchronised back) before it is read.
using SrtMemorySync = bool (*)(void* userdata, uint64_t address, uint64_t size);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	SrtMemorySync             sync_memory                = nullptr;
	// Answers an ordinary read (read_memory == nullptr) when the bytes are already current, or
	// returns false. Such a read otherwise loads straight through the guest mapping, and a page
	// that still holds GPU-written bytes faults and waits for their download first, even when
	// the table dwords being read are not among them.
	SrtMemoryReader           read_current_memory        = nullptr;
};

enum class RuntimeValueType { Any, Integer };

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot.
void BuildSrtPlan(Program& program);
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any,
                          std::string* reason = nullptr);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	std::span<const uint8_t> FindActiveSources();
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	friend class SrtRefresh;

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	bool Interpret(Value value, uint32_t& result);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);
	// KYTY_SRT_DIAG=1: explains why a descriptor dword did not evaluate.
	void LogDescriptorDiagnostic(uint32_t source, uint32_t dword);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
	// Deepest instruction whose evaluation failed during the current descriptor dword.
	const Inst* m_diag_first_fail = nullptr;
	// Bound by SrtRefresh: queries the compiled plan has a root for in this walker's walk (the
	// strict one when m_compiled_clean) run from that plan instead of being interpreted.
	const CompiledSrtPlan* m_compiled       = nullptr;
	bool                   m_compiled_clean = false;
};

// A ResourcePlan's per-refresh walk, lowered once into flat instruction sequences.
//
// Every draw refreshes its programs' resources by walking the same value graph with different user
// data and SRT memory. Interpreting that graph -- resolving identities, dispatching on opcodes,
// recursing through operands and memoising every instruction -- costs more than the reads it
// performs. The compiled form assigns every (instruction, walk) pair a slot once, and each root the
// refresh asks for (control-flow condition, flattened-SRT dword, descriptor dword, uniform fill or
// indirect-image value) gets the post-order sequence of slots SrtWalker would visit. A refresh then
// runs those sequences over a flat value array with a per-slot state byte in place of the memo.
//
// A walk is the interpreter's evaluation context: the ordinary walk (clean flat slots redirected to
// the strict walk), the strict walk, and below a ReadFirstLane either of them with that active-lane
// mask. The compiled run evaluates the same instructions in the same order as the walker,
// including its value-dependent laziness (only the chosen operand of a select, and never the size
// dwords of a scalar-buffer read through a null base), so each root succeeds exactly when the
// walker does and yields the same value. A plan reaching anything the compiler does not reproduce
// is not Usable() and its programs keep using SrtWalker.
class CompiledSrtPlan {
public:
	CompiledSrtPlan() = default;
	explicit CompiledSrtPlan(const ResourcePlan& plan);

	[[nodiscard]] bool Usable() const { return m_usable; }

private:
	friend class SrtPlanCompiler;
	friend class SrtWalker;
	friend class SrtRefresh;

	enum class Code : uint8_t {
		Constant,
		Fail,
		UserData,
		ShaderBase,
		Pure,
		Select,
		ExtractU64,
		AddCarryLow,
		AddCarryHigh,
		AddressRead,
		BufferRead,
		// Control markers rather than values. BufferReadGuard skips the size operands and the read
		// that follows when the read is already known or reads through a null base; SelectGuard
		// skips the operand a select does not choose, and SelectElse ends its first operand.
		BufferReadGuard,
		SelectGuard,
		SelectElse,
	};

	struct Op {
		Code        code      = Code::Fail;
		ValueOpcode pure      = ValueOpcode::Count;
		bool        clean     = false;
		uint32_t    args[5]   = {};
		uint64_t    immediate = 0;
	};

	// Per-slot evaluation state; constants start out Done.
	static constexpr uint8_t  Unset  = 0;
	static constexpr uint8_t  Done   = 1;
	static constexpr uint8_t  Failed = 2;
	static constexpr uint32_t NoRoot = UINT32_MAX;

	struct Root {
		uint32_t begin = 0;
		uint32_t end   = 0;
		uint32_t slot  = NoRoot;
	};

	struct ValueRoot {
		const Inst* inst  = nullptr;
		bool        clean = false;
		Root        root;
	};

	// Starts a refresh: forgets every value the previous one computed.
	void Begin() const;
	bool Run(const Root& root, const SrtRuntime& runtime, uint32_t& result) const;
	bool Execute(const Op& op, const SrtRuntime& runtime, uint64_t& result) const;
	[[nodiscard]] const Root* FindValueRoot(Value value, bool clean) const;
	// The roots of every dword of `source` in the given walk, or null when none were compiled.
	[[nodiscard]] const Root* DescriptorRoots(uint32_t source, bool clean) const;

	const ResourcePlan*    m_plan = nullptr;
	std::vector<Op>        m_ops;
	std::vector<uint64_t>  m_initial_values;
	std::vector<uint8_t>   m_initial_states;
	std::vector<uint32_t>  m_sequence;
	// Per control-flow block, in the strict walk; slot == NoRoot when the block has no condition.
	std::vector<Root>      m_conditions;
	// Per ResourcePlan::srt_reads element, with the ordinary walk's clean-slot redirect applied.
	std::vector<Root>      m_flat_roots;
	std::vector<Root>      m_dword_roots;
	// Per descriptor source: index of its first dword root in m_dword_roots, or NoRoot.
	std::vector<uint32_t>  m_main_sources;
	std::vector<uint32_t>  m_clean_sources;
	std::vector<ValueRoot> m_value_roots;
	bool                   m_usable = false;
	// Run state of the current refresh; like ResourcePlan's evaluation scratch it belongs to the one
	// thread refreshing this program.
	mutable std::vector<uint64_t> m_values;
	mutable std::vector<uint8_t>  m_states;
};

// The two walkers of one resource refresh: Clean() reads only through the strict reader and Main()
// redirects the clean flat slots to it. Given a usable plan compiled from the same program, both
// answer what the plan compiled from one shared compiled run and interpret everything else.
class SrtRefresh {
public:
	SrtRefresh(const ResourcePlan& program, const SrtRuntime& runtime,
	           const CompiledSrtPlan* compiled = nullptr);
	SrtRefresh(const SrtRefresh&)            = delete;
	SrtRefresh& operator=(const SrtRefresh&) = delete;

	SrtWalker& Clean() { return m_clean; }
	SrtWalker& Main() { return m_main; }

private:
	// Declared first: m_main refers to it, and the walkers' evaluation contexts nest in this order.
	SrtWalker m_clean;
	SrtWalker m_main;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
