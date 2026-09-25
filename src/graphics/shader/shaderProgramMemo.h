#ifndef EMULATOR_SRC_GRAPHICS_SHADER_SHADERPROGRAMMEMO_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_SHADERPROGRAMMEMO_H_

#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <span>

namespace Libs::Graphics {

// Consecutive draws overwhelmingly repeat the programs of the draw before, with only user-data
// SGPRs, draw arguments or unrelated registers written in between. Preparing a program reads a
// handful of registers, the registered shader and the vertex tables; the memos below remember
// the register part of the previous draw per stage by value, so a repeat skips the shader
// registry, the metadata decode and the program-cache key build and lookup. Everything that is
// per draw (user data, vertex table contents and addresses, SRT materialisation) still runs.

// KYTY_PROGRAM_MEMO_ORACLE=1: every memo hit (program preparation, program cache, graphics
// pipeline) also runs the path it skipped and EXITs on any difference, and hit counts are logged
// once per second. Off by default; read once.
bool ShaderProgramMemoOracleEnabled();

// A by-value key of fixed capacity, so building one per draw never allocates.
template <size_t Capacity>
class ShaderMemoKey {
public:
	void Clear() { m_size = 0; }
	void Push(uint32_t value) {
		EXIT_IF(m_size >= Capacity);
		m_words[m_size++] = value;
	}
	void Push64(uint64_t value) {
		Push(static_cast<uint32_t>(value));
		Push(static_cast<uint32_t>(value >> 32u));
	}
	void PushFloat(float value) { Push(std::bit_cast<uint32_t>(value)); }

	[[nodiscard]] bool operator==(const ShaderMemoKey& other) const {
		return m_size == other.m_size &&
		       std::equal(m_words.begin(), m_words.begin() + m_size, other.m_words.begin());
	}

private:
	std::array<uint32_t, Capacity> m_words {};
	uint32_t                       m_size = 0;
};

using VertexProgramKey = ShaderMemoKey<32>;
using PixelProgramKey  = ShaderMemoKey<64>;

// Which registration of a shader address a memo was built from. Every registration moves the
// registry generation; only then is the remembered address looked up again, so repeats between
// registrations never touch the registry, and registering some other shader does not drop the
// memo. A new registration at the same address (new code) is a miss.
class ShaderRegistrationStamp {
public:
	void Record(uint64_t address, uint64_t registration, uint64_t generation) {
		m_address          = address;
		m_registration     = registration;
		m_checked_generation = generation;
	}
	void Clear() { *this = {}; }

	// `lookup(address)` returns the registration currently mapped at `address` (0 if none).
	template <typename Lookup>
	[[nodiscard]] bool Matches(uint64_t address, uint64_t generation, Lookup&& lookup) {
		if (m_registration == 0 || address != m_address) {
			return false;
		}
		if (generation == m_checked_generation) {
			return true;
		}
		if (lookup(address) != m_registration) {
			return false;
		}
		m_checked_generation = generation;
		return true;
	}

private:
	uint64_t m_address            = 0;
	uint64_t m_registration       = 0;
	uint64_t m_checked_generation = 0;
};

// Every register PrepareProgram(VertexShaderInfo) and the clip-space block of
// GetGraphicsPrograms read, except the user-data SGPR values (per-draw data, never part of the
// program) and the shader's registration (checked through ShaderRegistrationStamp). Values that
// only feed an EXIT check are keyed too, so a repeat can never skip a check that would fail.
inline void BuildVertexProgramKey(const HW::VertexShaderInfo& regs, const HW::Context& context,
                                  const HW::UserConfig& user_config, VertexProgramKey& key) {
	const auto& sh     = context.GetShaderRegisters();
	const auto  stages = context.GetShaderStages();
	const auto& clip   = context.GetClipControl();
	key.Clear();
	key.Push64(regs.es_regs.data_addr);
	key.Push(regs.gs_regs.rsrc2.user_sgpr);
	key.Push(stages);
	key.Push(sh.m_paClVsOutCntl);
	key.Push(static_cast<uint32_t>(clip.clip_disable));
	if (clip.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		key.PushFloat(viewport.xscale);
		key.PushFloat(viewport.yscale);
		key.PushFloat(viewport.xoffset);
		key.PushFloat(viewport.yoffset);
	}
	if ((stages & 0x20u) != 0) {
		const auto& group = user_config.GetGeControl();
		key.Push64(regs.gs_regs.data_addr);
		key.Push(regs.gs_regs.rsrc1.gs_vgpr_component_count);
		key.Push(regs.gs_regs.rsrc2.es_vgpr_component_count);
		key.Push(regs.gs_regs.rsrc2.lds_size);
		key.Push(static_cast<uint32_t>(user_config.GetPrimType()));
		key.Push(group.primitive_group_size);
		key.Push(group.vertex_group_size);
		key.Push(sh.m_geMaxOutputPerSubgroup);
		key.Push(sh.m_vgtGsOutPrimType);
		key.Push(sh.m_vgtGsMaxVertOut);
		key.Push(static_cast<uint32_t>(context.GetModeControl().provoking_vtx_last));
	}
}

// Every register PrepareProgram(PixelShaderInfo) reads, except the user-data SGPR values and the
// shader's registration, plus the dual-source blending verdict GetGraphicsPrograms derives from
// the blend state and applies to the prepared info (it is part of the program's static key). An
// export mapping only matters for a slot that has an output format.
inline void BuildPixelProgramKey(const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
                                 std::span<const Prospero::ColorComponentMapping, 8> mapping,
                                 bool dual_source_blending, PixelProgramKey& key) {
	const auto& db = sh.db_shader_control;
	key.Clear();
	key.Push(static_cast<uint32_t>(dual_source_blending));
	key.Push64(regs.ps_regs.data_addr);
	key.Push(regs.ps_regs.rsrc2.user_sgpr);
	key.Push(sh.ps_in_control);
	key.Push(sh.ps_input_ena);
	key.Push(sh.ps_input_addr);
	key.Push(static_cast<uint32_t>(db.shader_kill_enable));
	key.Push(static_cast<uint32_t>(db.shader_z_export_enable));
	key.Push(static_cast<uint32_t>(db.shader_mask_export_enable));
	key.Push(static_cast<uint32_t>(db.shader_execute_on_noop));
	key.Push(db.shader_z_behavior);
	const uint32_t input_num = std::min<uint32_t>(sh.ps_in_control & 0x3fu,
	                                              std::size(sh.ps_interpolator_settings));
	for (uint32_t i = 0; i < input_num; i++) {
		key.Push(sh.ps_interpolator_settings[i]);
	}
	for (uint32_t i = 0; i < 8; i++) {
		const auto mode = sh.target_output_mode[i];
		key.Push(static_cast<uint32_t>(mode) |
		         (mode != 0 ? static_cast<uint32_t>(mapping[i].packed) << 8u : 0u));
	}
}

// The part of one fetched vertex attribute that the program static key and the pipeline consume:
// everything in the V# except its base address and record count, which are per draw.
[[nodiscard]] inline bool VertexResourceLayoutEqual(const ShaderBufferResource& a,
                                                    const ShaderBufferResource& b) {
	return a.Stride() == b.Stride() && a.SwizzleEnabled() == b.SwizzleEnabled() &&
	       a.DstSelXYZW() == b.DstSelXYZW() && a.RawFormat() == b.RawFormat() &&
	       a.OutOfBounds() == b.OutOfBounds() && a.AddTid() == b.AddTid();
}

// The vertex input layout decoded from the attribute and V# tables: the per-attribute fields
// BuildStageStaticKey(ShaderVertexInputInfo) keys and the binding layout GetGraphicsPipeline
// derives its vertex-input state from. Base addresses and record counts are excluded: they move
// with every object while its program and pipeline stay the same.
[[nodiscard]] inline bool VertexInputLayoutEqual(const ShaderVertexInputInfo& a,
                                                 const ShaderVertexInputInfo& b) {
	if (a.resources_num != b.resources_num || a.buffers_num != b.buffers_num) {
		return false;
	}
	for (int i = 0; i < a.resources_num; i++) {
		const auto& da = a.resources_dst[i];
		const auto& db = b.resources_dst[i];
		if (da.register_start != db.register_start || da.registers_num != db.registers_num ||
		    da.attr_id != db.attr_id || da.fetch_index != db.fetch_index ||
		    !VertexResourceLayoutEqual(a.resources[i], b.resources[i])) {
			return false;
		}
	}
	for (int i = 0; i < a.buffers_num; i++) {
		const auto& ba = a.buffers[i];
		const auto& bb = b.buffers[i];
		if (ba.stride != bb.stride || ba.fetch_index != bb.fetch_index ||
		    ba.attr_num != bb.attr_num ||
		    !std::equal(ba.attr_indices, ba.attr_indices + ba.attr_num, bb.attr_indices) ||
		    !std::equal(ba.attr_offsets, ba.attr_offsets + ba.attr_num, bb.attr_offsets)) {
			return false;
		}
	}
	return true;
}

// Copies the decoded vertex input layout (and the addresses that come with it) up to the counts;
// entries past resources_num / buffers_num are never read.
inline void CopyVertexInputLayout(const ShaderVertexInputInfo& src, ShaderVertexInputInfo& dst) {
	EXIT_IF(src.resources_num < 0 || src.resources_num > ShaderVertexInputInfo::RES_MAX ||
	        src.buffers_num < 0 || src.buffers_num > ShaderVertexInputInfo::RES_MAX);
	std::copy_n(src.resources, src.resources_num, dst.resources);
	std::copy_n(src.resources_dst, src.resources_num, dst.resources_dst);
	std::copy_n(src.buffers, src.buffers_num, dst.buffers);
	dst.resources_num = src.resources_num;
	dst.buffers_num   = src.buffers_num;
}

// Everything PrepareProgram(VertexShaderInfo) sets besides the decoded vertex input layout (the
// fetch fields and tables, which ShaderFetchVertexInputs fills per draw) and the stage runtime.
inline void CopyPreparedVertexState(const ShaderVertexInputInfo& src, ShaderVertexInputInfo& dst) {
	dst.logical_stage       = src.logical_stage;
	dst.wave_size           = src.wave_size;
	dst.scratch_size_dwords = src.scratch_size_dwords;
	dst.pa_cl_vs_out_cntl   = src.pa_cl_vs_out_cntl;
	dst.clip_space          = src.clip_space;
	dst.mesh                = src.mesh;
	dst.tess                = src.tess;
}

// PrepareProgram(VertexShaderInfo) with a memo of the previous call. `same_program` is set when
// the program identity, its user-data count and every input of its program-cache static key equal
// the previous call's, so the program cache may reuse the previous entry without a lookup.
class VertexProgramMemo {
public:
	VertexProgramMemo();
	~VertexProgramMemo();
	KYTY_CLASS_NO_COPY(VertexProgramMemo);

	ShaderParams Prepare(const HW::VertexShaderInfo& regs, const HW::Context& context,
	                     const HW::UserConfig& user_config, ShaderVertexInputInfo& info,
	                     bool& same_program);

private:
	void VerifyAgainstFullPreparation(const HW::VertexShaderInfo& regs, const HW::Context& context,
	                                  const HW::UserConfig& user_config, const ShaderParams& params,
	                                  const ShaderVertexInputInfo& info);

	VertexProgramKey          m_key;
	VertexProgramKey          m_scratch_key;
	ShaderRegistrationStamp   m_front;
	ShaderRegistrationStamp   m_back;
	std::span<const uint32_t> m_code;
	std::span<const uint32_t> m_back_code;
	uint64_t                  m_hash = 0;
	ShaderVertexMetadata      m_metadata;
	// Prepared scalars of the previous call and the vertex input layout the program cache saw last.
	std::unique_ptr<ShaderVertexInputInfo> m_info;
	// Slow-path result for the oracle, allocated on its first use.
	std::unique_ptr<ShaderVertexInputInfo> m_oracle_info;
	bool                                   m_valid = false;
};

// PrepareProgram(PixelShaderInfo) with a memo of the previous call; every input of the pixel
// static key comes from registers, so a key match means the same program.
class PixelProgramMemo {
public:
	ShaderParams Prepare(const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
	                     std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                     bool dual_source_blending, ShaderPixelInputInfo& info,
	                     bool& same_program);

private:
	PixelProgramKey           m_key;
	PixelProgramKey           m_scratch_key;
	ShaderRegistrationStamp   m_stamp;
	std::span<const uint32_t> m_code;
	uint64_t                  m_hash = 0;
	ShaderPixelInputInfo      m_info;
	bool                      m_valid = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_SHADER_SHADERPROGRAMMEMO_H_
