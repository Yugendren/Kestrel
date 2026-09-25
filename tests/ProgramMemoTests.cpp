#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/shaderProgramMemo.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace {

using Libs::Graphics::BuildPixelProgramKey;
using Libs::Graphics::BuildVertexProgramKey;
using Libs::Graphics::CopyVertexInputLayout;
using Libs::Graphics::DstSel;
using Libs::Graphics::GenerationMemo;
using Libs::Graphics::GraphicsPipelineKey;
using Libs::Graphics::PipelineTopologyClass;
using Libs::Graphics::PixelProgramKey;
using Libs::Graphics::ShaderRegistrationStamp;
using Libs::Graphics::ShaderVertexInputInfo;
using Libs::Graphics::VertexInputLayoutEqual;
using Libs::Graphics::VertexProgramKey;
namespace HW       = Libs::Graphics::HW;
namespace Prospero = Libs::Graphics::Prospero;

int g_cases = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ProgramMemoTests: failed: %s\n", text);
		std::abort();
	}
	g_cases++;
}

// ---- Vertex program key --------------------------------------------------------------------

struct VertexInputs {
	HW::VertexShaderInfo regs;
	HW::Context          context;
	HW::UserConfig       user_config;
};

VertexInputs BaseVertexInputs(bool ngg) {
	VertexInputs in;
	in.regs.es_regs.data_addr                     = 0x10000;
	in.regs.gs_regs.data_addr                     = 0x20000;
	in.regs.gs_regs.user_data_addr                = 0x30000;
	in.regs.gs_regs.rsrc2.user_sgpr               = 6;
	in.regs.gs_regs.rsrc1.gs_vgpr_component_count = 3;
	in.regs.gs_regs.rsrc2.es_vgpr_component_count = 3;
	in.regs.gs_user_sgpr.value[0]                 = 0x1234;
	in.context.SetShaderStages(ngg ? 0x20u : 0u);
	in.context.SetClVsOutCntl(0x10);
	in.context.SetMaxOutputPerSubgroup(64);
	in.context.SetGsOutPrimType(2);
	in.context.SetGsMaxVertOut(3);
	HW::ClipControl clip;
	clip.clip_disable = true;
	in.context.SetClipControl(clip);
	in.context.SetViewportScaleOffset(0, 960.0f, 960.0f, -540.0f, 540.0f, 0.5f, 0.5f);
	in.user_config.SetPrimitiveType(Prospero::PrimitiveType::kTriList);
	in.user_config.SetGeControl({.primitive_group_size = 64, .vertex_group_size = 128});
	return in;
}

VertexProgramKey KeyOf(const VertexInputs& in) {
	VertexProgramKey key;
	BuildVertexProgramKey(in.regs, in.context, in.user_config, key);
	return key;
}

struct VertexCase {
	const char*                        name;
	bool                               ngg;
	std::function<void(VertexInputs&)> mutate;
};

void TestVertexKeyEqualInputsHit() {
	for (const bool ngg: {false, true}) {
		Check(KeyOf(BaseVertexInputs(ngg)) == KeyOf(BaseVertexInputs(ngg)),
		      "equal vertex inputs build equal keys");
		auto in = BaseVertexInputs(ngg);
		// User-data SGPR values are per-draw data (tables, constants), never program identity.
		in.regs.gs_user_sgpr.value[0] = 0x5678;
		in.regs.gs_user_sgpr.value[5] = 0x9abc;
		in.regs.gs_regs.user_data_addr = 0x40000;
		Check(KeyOf(in) == KeyOf(BaseVertexInputs(ngg)), "user-data SGPR values do not break the key");
	}
	auto in = BaseVertexInputs(false);
	// Without NGG the GS registers are never read.
	in.regs.gs_regs.data_addr = 0x50000;
	in.user_config.SetPrimitiveType(Prospero::PrimitiveType::kTriStrip);
	Check(KeyOf(in) == KeyOf(BaseVertexInputs(false)), "non-NGG key ignores the GS assembly state");
}

void TestVertexKeyFields() {
	const std::vector<VertexCase> cases = {
	    {"es data_addr", false, [](auto& in) { in.regs.es_regs.data_addr = 0x11000; }},
	    {"user_sgpr count", false, [](auto& in) { in.regs.gs_regs.rsrc2.user_sgpr = 7; }},
	    {"shader stages", false, [](auto& in) { in.context.SetShaderStages(0x00400000u); }},
	    {"PA_CL_VS_OUT_CNTL", false, [](auto& in) { in.context.SetClVsOutCntl(0x11); }},
	    {"clip_disable", false,
	     [](auto& in) {
		     HW::ClipControl clip;
		     clip.clip_disable = false;
		     in.context.SetClipControl(clip);
	     }},
	    {"viewport xscale", false,
	     [](auto& in) { in.context.SetViewportScaleOffset(0, 961.0f, 960.0f, -540.0f, 540.0f, 0.5f, 0.5f); }},
	    {"viewport xoffset", false,
	     [](auto& in) { in.context.SetViewportScaleOffset(0, 960.0f, 961.0f, -540.0f, 540.0f, 0.5f, 0.5f); }},
	    {"viewport yscale", false,
	     [](auto& in) { in.context.SetViewportScaleOffset(0, 960.0f, 960.0f, -541.0f, 540.0f, 0.5f, 0.5f); }},
	    {"viewport yoffset", false,
	     [](auto& in) { in.context.SetViewportScaleOffset(0, 960.0f, 960.0f, -540.0f, 541.0f, 0.5f, 0.5f); }},
	    {"NGG es data_addr", true, [](auto& in) { in.regs.es_regs.data_addr = 0x11000; }},
	    {"NGG wave32 stage bit", true, [](auto& in) { in.context.SetShaderStages(0x00400020u); }},
	    {"NGG gs data_addr", true, [](auto& in) { in.regs.gs_regs.data_addr = 0x21000; }},
	    {"NGG gs_vgpr_component_count", true,
	     [](auto& in) { in.regs.gs_regs.rsrc1.gs_vgpr_component_count = 2; }},
	    {"NGG es_vgpr_component_count", true,
	     [](auto& in) { in.regs.gs_regs.rsrc2.es_vgpr_component_count = 2; }},
	    {"NGG lds_size", true, [](auto& in) { in.regs.gs_regs.rsrc2.lds_size = 4; }},
	    {"NGG primitive type", true,
	     [](auto& in) { in.user_config.SetPrimitiveType(Prospero::PrimitiveType::kTriStrip); }},
	    {"NGG primitive_group_size", true,
	     [](auto& in) {
		     in.user_config.SetGeControl({.primitive_group_size = 32, .vertex_group_size = 128});
	     }},
	    {"NGG vertex_group_size", true,
	     [](auto& in) {
		     in.user_config.SetGeControl({.primitive_group_size = 64, .vertex_group_size = 96});
	     }},
	    {"NGG GE_MAX_OUTPUT_PER_SUBGROUP", true,
	     [](auto& in) { in.context.SetMaxOutputPerSubgroup(128); }},
	    {"NGG VGT_GS_OUT_PRIM_TYPE", true,
	     [](auto& in) { in.context.SetGsOutPrimType(1); }},
	    {"NGG VGT_GS_MAX_VERT_OUT", true,
	     [](auto& in) { in.context.SetGsMaxVertOut(4); }},
	    {"NGG provoking_vtx_last", true,
	     [](auto& in) {
		     HW::ModeControl mode;
		     mode.provoking_vtx_last = true;
		     in.context.SetModeControl(mode);
	     }},
	    {"NGG PA_CL_VS_OUT_CNTL", true, [](auto& in) { in.context.SetClVsOutCntl(0x11); }},
	};
	for (const auto& c: cases) {
		auto in = BaseVertexInputs(c.ngg);
		c.mutate(in);
		if (KeyOf(in) == KeyOf(BaseVertexInputs(c.ngg))) {
			std::fprintf(stderr, "ProgramMemoTests: vertex key ignores %s\n", c.name);
			std::abort();
		}
		g_cases++;
	}
}

// ---- Pixel program key ---------------------------------------------------------------------

struct PixelInputs {
	HW::PixelShaderInfo                                regs;
	HW::ShaderRegisters                                sh;
	std::array<Prospero::ColorComponentMapping, 8>     mapping {};
	bool                                               dual_source_blending = false;
};

PixelInputs BasePixelInputs() {
	PixelInputs in;
	in.regs.ps_regs.data_addr       = 0x60000;
	in.regs.ps_regs.rsrc2.user_sgpr = 4;
	in.regs.ps_user_sgpr.value[0]   = 0x1111;
	in.sh.ps_in_control             = 3;
	in.sh.ps_input_ena              = 0x302;
	in.sh.ps_input_addr             = 0x302;
	in.sh.ps_interpolator_settings[0] = 0;
	in.sh.ps_interpolator_settings[1] = 1;
	in.sh.ps_interpolator_settings[2] = 2;
	in.sh.db_shader_control.shader_z_behavior = 1;
	in.sh.target_output_mode[0]               = 4;
	in.sh.target_output_mode[1]               = 9;
	return in;
}

PixelProgramKey KeyOf(const PixelInputs& in) {
	PixelProgramKey key;
	BuildPixelProgramKey(in.regs, in.sh, std::span<const Prospero::ColorComponentMapping, 8>(in.mapping),
	                     in.dual_source_blending, key);
	return key;
}

void TestPixelKeyEqualInputsHit() {
	Check(KeyOf(BasePixelInputs()) == KeyOf(BasePixelInputs()), "equal pixel inputs build equal keys");
	auto in                        = BasePixelInputs();
	in.regs.ps_user_sgpr.value[0]  = 0x2222;
	in.sh.ps_interpolator_settings[5] = 7; // beyond NUM_INTERP
	in.mapping[3].packed           = 0x1b; // slot without an output format
	Check(KeyOf(in) == KeyOf(BasePixelInputs()),
	      "user data, unused interpolators and unused export mappings do not break the key");
}

void TestPixelKeyFields() {
	struct PixelCase {
		const char*                       name;
		std::function<void(PixelInputs&)> mutate;
	};
	const std::vector<PixelCase> cases = {
	    {"ps data_addr", [](auto& in) { in.regs.ps_regs.data_addr = 0x61000; }},
	    {"user_sgpr count", [](auto& in) { in.regs.ps_regs.rsrc2.user_sgpr = 5; }},
	    {"SPI_PS_IN_CONTROL num_interp", [](auto& in) { in.sh.ps_in_control = 4; }},
	    {"SPI_PS_IN_CONTROL wave32", [](auto& in) { in.sh.ps_in_control = 3u | 0x8000u; }},
	    {"SPI_PS_INPUT_ENA", [](auto& in) { in.sh.ps_input_ena = 0x303; }},
	    {"SPI_PS_INPUT_ADDR", [](auto& in) { in.sh.ps_input_addr = 0x303; }},
	    {"shader_kill_enable", [](auto& in) { in.sh.db_shader_control.shader_kill_enable = true; }},
	    {"shader_z_export_enable",
	     [](auto& in) { in.sh.db_shader_control.shader_z_export_enable = true; }},
	    {"shader_mask_export_enable",
	     [](auto& in) { in.sh.db_shader_control.shader_mask_export_enable = true; }},
	    {"shader_execute_on_noop",
	     [](auto& in) { in.sh.db_shader_control.shader_execute_on_noop = true; }},
	    {"shader_z_behavior", [](auto& in) { in.sh.db_shader_control.shader_z_behavior = 0; }},
	    {"interpolator setting", [](auto& in) { in.sh.ps_interpolator_settings[2] = 0x22; }},
	    {"target_output_mode[0]", [](auto& in) { in.sh.target_output_mode[0] = 9; }},
	    {"target_output_mode[7]", [](auto& in) { in.sh.target_output_mode[7] = 4; }},
	    {"export mapping of an active slot", [](auto& in) { in.mapping[1].packed = 0x1b; }},
	    {"dual-source blending", [](auto& in) { in.dual_source_blending = true; }},
	};
	for (const auto& c: cases) {
		auto in = BasePixelInputs();
		c.mutate(in);
		if (KeyOf(in) == KeyOf(BasePixelInputs())) {
			std::fprintf(stderr, "ProgramMemoTests: pixel key ignores %s\n", c.name);
			std::abort();
		}
		g_cases++;
	}
}

// ---- Shader registration identity ----------------------------------------------------------

void TestRegistrationStamp() {
	uint64_t   current_registration = 7;
	int        lookups              = 0;
	const auto lookup               = [&](uint64_t address) {
        lookups++;
        return address == 0x10000 ? current_registration : 0;
	};
	ShaderRegistrationStamp stamp;
	Check(!stamp.Matches(0x10000, 7, lookup), "an empty stamp never matches");
	stamp.Record(0x10000, 7, 7);
	Check(stamp.Matches(0x10000, 7, lookup) && lookups == 0,
	      "an unchanged registry generation matches without a lookup");
	Check(!stamp.Matches(0x11000, 7, lookup), "another address misses");
	Check(stamp.Matches(0x10000, 8, lookup) && lookups == 1,
	      "a registration elsewhere costs one lookup and still matches");
	Check(stamp.Matches(0x10000, 8, lookup) && lookups == 1,
	      "the re-checked generation is remembered");
	current_registration = 9;
	Check(!stamp.Matches(0x10000, 9, lookup),
	      "a new registration of a shader at the same address misses");
	stamp.Clear();
	Check(!stamp.Matches(0x10000, 9, lookup), "a cleared stamp misses");
}

// ---- Vertex input layout -------------------------------------------------------------------

std::unique_ptr<ShaderVertexInputInfo> BaseLayout() {
	auto info           = std::make_unique<ShaderVertexInputInfo>();
	info->resources_num = 2;
	info->buffers_num   = 1;
	for (int i = 0; i < 2; i++) {
		auto& r     = info->resources[i];
		r.fields[0] = 0x00100000u + static_cast<uint32_t>(i) * 12u; // base address low
		r.fields[1] = (32u << 16u) | 0x0012u;                      // stride 32, base address high
		r.fields[2] = 1000;                                         // num_records
		r.fields[3] = DstSel(4, 5, 6, 7) | (77u << 12u) | (3u << 28u);
		auto& d          = info->resources_dst[i];
		d.register_start = 4 * i;
		d.registers_num  = 4;
		d.attr_id        = i;
		d.fetch_index    = 0;
	}
	auto& b           = info->buffers[0];
	b.addr            = 0x1200100000ull;
	b.stride          = 32;
	b.num_records     = 1000;
	b.fetch_index     = 0;
	b.attr_num        = 2;
	b.attr_indices[0] = 0;
	b.attr_indices[1] = 1;
	b.attr_offsets[0] = 0;
	b.attr_offsets[1] = 12;
	return info;
}

void TestLayoutIgnoresAddressesAndRecordCounts() {
	const auto base = BaseLayout();
	auto       moved = BaseLayout();
	for (int i = 0; i < 2; i++) {
		moved->resources[i].UpdateAddress48(0x3400200000ull + static_cast<uint64_t>(i) * 12u);
		moved->resources[i].fields[2] = 5000;
	}
	moved->buffers[0].addr        = 0x3400200000ull;
	moved->buffers[0].num_records = 5000;
	Check(VertexInputLayoutEqual(*base, *moved),
	      "a new vertex buffer base address or record count keeps the layout");
	// Entries past the counts are never read, so stale values there must not matter.
	moved->resources[5].fields[3] = 0xffffffffu;
	moved->buffers[3].stride      = 99;
	Check(VertexInputLayoutEqual(*base, *moved), "entries past the counts are ignored");
	auto copy = std::make_unique<ShaderVertexInputInfo>();
	CopyVertexInputLayout(*moved, *copy);
	Check(VertexInputLayoutEqual(*moved, *copy) && copy->buffers[0].addr == 0x3400200000ull,
	      "CopyVertexInputLayout copies the layout and its addresses");
}

void TestLayoutFields() {
	struct LayoutCase {
		const char*                                 name;
		std::function<void(ShaderVertexInputInfo&)> mutate;
	};
	const std::vector<LayoutCase> cases = {
	    {"resources_num", [](auto& i) { i.resources_num = 1; }},
	    {"buffers_num", [](auto& i) { i.buffers_num = 2; }},
	    {"register_start", [](auto& i) { i.resources_dst[1].register_start = 12; }},
	    {"registers_num", [](auto& i) { i.resources_dst[1].registers_num = 3; }},
	    {"attr_id", [](auto& i) { i.resources_dst[1].attr_id = 5; }},
	    {"attribute fetch_index", [](auto& i) { i.resources_dst[1].fetch_index = 1; }},
	    {"V# stride", [](auto& i) { i.resources[1].fields[1] = (16u << 16u) | 0x0012u; }},
	    {"V# swizzle enable", [](auto& i) { i.resources[1].fields[1] |= 1u << 31u; }},
	    {"V# dst_sel x", [](auto& i) { i.resources[1].fields[3] ^= DstSel(1); }},
	    {"V# dst_sel w", [](auto& i) { i.resources[1].fields[3] ^= DstSel(0, 0, 0, 3); }},
	    {"V# format", [](auto& i) { i.resources[1].fields[3] ^= 1u << 12u; }},
	    {"V# out-of-bounds select", [](auto& i) { i.resources[1].fields[3] ^= 1u << 28u; }},
	    {"V# add_tid", [](auto& i) { i.resources[1].fields[3] ^= 1u << 23u; }},
	    {"binding stride", [](auto& i) { i.buffers[0].stride = 16; }},
	    {"binding fetch_index", [](auto& i) { i.buffers[0].fetch_index = 1; }},
	    {"binding attr_num", [](auto& i) { i.buffers[0].attr_num = 1; }},
	    {"binding attr_indices", [](auto& i) { i.buffers[0].attr_indices[1] = 0; }},
	    {"binding attr_offsets", [](auto& i) { i.buffers[0].attr_offsets[1] = 16; }},
	};
	const auto base = BaseLayout();
	for (const auto& c: cases) {
		auto info = BaseLayout();
		c.mutate(*info);
		if (VertexInputLayoutEqual(*base, *info) || VertexInputLayoutEqual(*info, *base)) {
			std::fprintf(stderr, "ProgramMemoTests: vertex layout ignores %s\n", c.name);
			std::abort();
		}
		g_cases++;
	}
}

// ---- Graphics pipeline key -----------------------------------------------------------------

GraphicsPipelineKey BasePipelineKey() {
	GraphicsPipelineKey key {};
	key.vertex_shader_ids[0]                  = 11;
	key.ps_shader_id                          = 12;
	key.rendering.color_count                 = 1;
	key.rendering.color_formats[0]            = vk::Format::eR8G8B8A8Unorm;
	key.rendering.depth_format                = vk::Format::eD32Sfloat;
	key.vertex_input.binding_count            = 1;
	key.vertex_input.attribute_count          = 1;
	key.vertex_input.bindings[0]              = {.stride = 32, .instance = false};
	key.vertex_input.attributes[0]            = {.offset = 0, .binding = 0};
	key.static_params.samples                 = 1;
	key.static_params.topology_class          = PipelineTopologyClass::Triangle;
	key.static_params.color_mask[0]           = 0xf;
	key.static_params.polygon_mode            = vk::PolygonMode::eFill;
	return key;
}

void TestPipelineMemo() {
	using Memo = GenerationMemo<GraphicsPipelineKey, int>;
	Memo memo;
	memo.Store(BasePipelineKey(), 0, 1);
	const auto* hit = memo.Find(BasePipelineKey(), 0);
	Check(hit != nullptr && *hit == 1, "an equal pipeline key hits");

	struct PipelineCase {
		const char*                               name;
		std::function<void(GraphicsPipelineKey&)> mutate;
	};
	const std::vector<PipelineCase> cases = {
	    {"vs program id", [](auto& k) { k.vertex_shader_ids[0] = 21; }},
	    {"hs program id", [](auto& k) { k.vertex_shader_ids[1] = 31; }},
	    {"tes program id", [](auto& k) { k.vertex_shader_ids[2] = 41; }},
	    {"ps program id", [](auto& k) { k.ps_shader_id = 22; }},
	    {"color_count", [](auto& k) { k.rendering.color_count = 2; }},
	    {"color format", [](auto& k) { k.rendering.color_formats[0] = vk::Format::eB8G8R8A8Unorm; }},
	    {"depth format", [](auto& k) { k.rendering.depth_format = vk::Format::eD16Unorm; }},
	    {"stencil format",
	     [](auto& k) { k.rendering.stencil_format = vk::Format::eD32SfloatS8Uint; }},
	    {"binding_count", [](auto& k) { k.vertex_input.binding_count = 2; }},
	    {"attribute_count", [](auto& k) { k.vertex_input.attribute_count = 2; }},
	    {"binding stride", [](auto& k) { k.vertex_input.bindings[0].stride = 16; }},
	    {"binding input rate", [](auto& k) { k.vertex_input.bindings[0].instance = true; }},
	    {"attribute offset", [](auto& k) { k.vertex_input.attributes[0].offset = 4; }},
	    {"attribute binding", [](auto& k) { k.vertex_input.attributes[0].binding = 1; }},
	    {"negative_one_to_one", [](auto& k) { k.static_params.negative_one_to_one = true; }},
	    {"depth_clip_enable", [](auto& k) { k.static_params.depth_clip_enable = false; }},
	    {"topology class",
	     [](auto& k) { k.static_params.topology_class = PipelineTopologyClass::Line; }},
	    {"samples", [](auto& k) { k.static_params.samples = 4; }},
	    {"sample_shading_enable", [](auto& k) { k.static_params.sample_shading_enable = true; }},
	    {"color_mask", [](auto& k) { k.static_params.color_mask[0] = 0x7; }},
	    {"provoking_vtx_last", [](auto& k) { k.static_params.provoking_vtx_last = true; }},
	    {"polygon_mode", [](auto& k) { k.static_params.polygon_mode = vk::PolygonMode::eLine; }},
	    {"color_srcblend", [](auto& k) { k.static_params.color_srcblend[0] = 4; }},
	    {"color_comb_fcn", [](auto& k) { k.static_params.color_comb_fcn[0] = 1; }},
	    {"color_destblend", [](auto& k) { k.static_params.color_destblend[0] = 5; }},
	    {"alpha_srcblend", [](auto& k) { k.static_params.alpha_srcblend[0] = 4; }},
	    {"alpha_comb_fcn", [](auto& k) { k.static_params.alpha_comb_fcn[0] = 1; }},
	    {"alpha_destblend", [](auto& k) { k.static_params.alpha_destblend[0] = 5; }},
	    {"separate_alpha_blend", [](auto& k) { k.static_params.separate_alpha_blend[0] = true; }},
	    {"blend_enable", [](auto& k) { k.static_params.blend_enable[0] = true; }},
	    {"blend state of attachment 7", [](auto& k) { k.static_params.blend_enable[7] = true; }},
	};
	for (const auto& c: cases) {
		auto key = BasePipelineKey();
		c.mutate(key);
		if (memo.Find(key, 0) != nullptr) {
			std::fprintf(stderr, "ProgramMemoTests: pipeline memo ignores %s\n", c.name);
			std::abort();
		}
		g_cases++;
	}
}

} // namespace

int main() {
	TestVertexKeyEqualInputsHit();
	TestVertexKeyFields();
	TestPixelKeyEqualInputsHit();
	TestPixelKeyFields();
	TestRegistrationStamp();
	TestLayoutIgnoresAddressesAndRecordCounts();
	TestLayoutFields();
	TestPipelineMemo();
	std::printf("ProgramMemoTests: all %d cases passed\n", g_cases);
	return 0;
}
