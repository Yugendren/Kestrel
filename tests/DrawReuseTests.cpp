#include "graphics/guest_gpu/command_processor/drawStateTracker.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/drawReuse.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <vector>

// DrawStateTracker (the PM4 whitelist) and DrawReuseRecord (the renderer's comparison against the
// previous draw) are header-only and take no Vulkan objects, so both are exercised directly.

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool value, const char* text, uint32_t detail = 0) {
	if (!value) {
		std::fprintf(stderr, "DrawReuseTests: failed: %s (0x%x)\n", text, detail);
		g_failures++;
	}
}

// A packet with `body` after the header; KYTY_PM4 encodes the total length including the header.
std::vector<uint32_t> Packet(uint32_t opcode, std::initializer_list<uint32_t> body, uint32_t r = 0) {
	std::vector<uint32_t> packet {KYTY_PM4(body.size() + 1u, opcode, r)};
	packet.insert(packet.end(), body.begin(), body.end());
	return packet;
}

std::vector<uint32_t> SetShReg(uint32_t first, uint32_t count) {
	std::vector<uint32_t> packet {KYTY_PM4(count + 2u, Pm4::IT_SET_SH_REG, 0), first};
	packet.resize(count + 2u, 0x1234u);
	return packet;
}

std::vector<uint32_t> SetShRegIndirect(const std::vector<uint32_t>& pairs) {
	const auto address = reinterpret_cast<uint64_t>(pairs.data());
	return Packet(Pm4::IT_SET_SH_REG_INDIRECT,
	              {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), 0,
	               static_cast<uint32_t>(pairs.size() / 2u)});
}

const std::vector<uint32_t> g_draw = Packet(Pm4::IT_DRAW_INDEX_AUTO, {3, 2});

// Runs a direct draw, then `between`, then another direct draw, and returns the second draw's
// verdict.
bool VerdictAcross(const std::vector<std::vector<uint32_t>>& between) {
	DrawStateTracker tracker;
	tracker.NotePacket(g_draw.data());
	(void)tracker.TakeDrawVerdict();
	for (const auto& packet: between) {
		tracker.NotePacket(packet.data());
	}
	tracker.NotePacket(g_draw.data());
	return tracker.TakeDrawVerdict();
}

bool IsWhitelisted(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_NOP:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

void TestInitialStateAndConsumption() {
	DrawStateTracker tracker;
	tracker.NotePacket(g_draw.data());
	Check(!tracker.TakeDrawVerdict(), "the first draw is never eligible");
	tracker.NotePacket(g_draw.data());
	Check(tracker.TakeDrawVerdict(), "a draw right after a draw is eligible");
	Check(!tracker.TakeDrawVerdict(), "a verdict is consumed by the draw that takes it");

	for (const auto opcode: {Pm4::IT_DRAW_INDEX_2, Pm4::IT_DRAW_INDEX_OFFSET_2,
	                         Pm4::IT_DRAW_INDEX_AUTO}) {
		const auto draw = Packet(opcode, {0, 0, 0, 0});
		Check(VerdictAcross({draw}), "a direct draw consumes the state", opcode);
	}
}

void TestOpcodes() {
	for (uint32_t opcode = 0; opcode < 256; opcode++) {
		if (IsWhitelisted(opcode)) {
			continue;
		}
		const auto packet = Packet(opcode, {0, 0, 0, 0});
		Check(!VerdictAcross({packet}), "an opcode off the whitelist invalidates", opcode);
	}
	for (const auto opcode: {Pm4::IT_INDEX_BASE, Pm4::IT_INDEX_BUFFER_SIZE, Pm4::IT_INDEX_TYPE,
	                         Pm4::IT_NUM_INSTANCES, Pm4::IT_SET_BASE, Pm4::IT_INDIRECT_BUFFER}) {
		const auto packet = Packet(opcode, {0, 0, 0});
		Check(VerdictAcross({packet}), "a draw-argument packet keeps the state", opcode);
	}
}

void TestIndirectDraws() {
	for (const auto opcode: {Pm4::IT_DRAW_INDIRECT, Pm4::IT_DRAW_INDEX_INDIRECT,
	                         Pm4::IT_DRAW_INDIRECT_MULTI, Pm4::IT_DRAW_INDEX_INDIRECT_MULTI}) {
		DrawStateTracker tracker;
		tracker.NotePacket(g_draw.data());
		(void)tracker.TakeDrawVerdict();
		const auto indirect = Packet(opcode, {0, 0, 0, 0});
		tracker.NotePacket(indirect.data());
		Check(!tracker.TakeDrawVerdict(), "an indirect draw is never eligible", opcode);
		tracker.NotePacket(indirect.data());
		Check(!tracker.TakeDrawVerdict(), "an indirect draw after a draw is never eligible",
		      opcode);
	}
}

void TestShRegisters() {
	using Pm4::SPI_SHADER_USER_DATA_GS_0;
	using Pm4::SPI_SHADER_USER_DATA_HS_0;
	using Pm4::SPI_SHADER_USER_DATA_PS_0;
	Check(VerdictAcross({SetShReg(SPI_SHADER_USER_DATA_PS_0, 32)}), "PS user data keeps the state");
	Check(VerdictAcross({SetShReg(SPI_SHADER_USER_DATA_GS_0 + 4, 8)}),
	      "GS user data keeps the state");
	Check(VerdictAcross({SetShReg(SPI_SHADER_USER_DATA_HS_0 + 31, 1)}),
	      "HS user data keeps the state");
	Check(VerdictAcross({SetShReg(Pm4::SPI_SHADER_USER_DATA_ADDR_LO_GS, 2),
	                     SetShReg(Pm4::SPI_SHADER_USER_DATA_ADDR_LO_HS, 2)}),
	      "the GS/HS user-data address pairs keep the state");
	Check(VerdictAcross({SetShReg(Pm4::SH_NOP, 1)}), "a SET_SH_REG to SH_NOP writes nothing");
	Check(VerdictAcross({SetShReg(SPI_SHADER_USER_DATA_PS_0, 0)}),
	      "a SET_SH_REG without values writes nothing");

	Check(!VerdictAcross({SetShReg(SPI_SHADER_USER_DATA_PS_0 - 1, 3)}),
	      "a non-user-data register first in the range invalidates");
	Check(!VerdictAcross({SetShReg(Pm4::SPI_SHADER_USER_DATA_ADDR_HI_GS, 10)}),
	      "a non-user-data register in the middle of the range invalidates");
	Check(!VerdictAcross({SetShReg(Pm4::SPI_SHADER_USER_DATA_PS_31 - 1, 3)}),
	      "a non-user-data register last in the range invalidates");
	Check(!VerdictAcross({SetShReg(Pm4::COMPUTE_USER_DATA_0, 4)}),
	      "compute user data invalidates");
	Check(!VerdictAcross({SetShReg(Pm4::SPI_SHADER_PGM_LO_GS, 2)}),
	      "a program address invalidates");

	for (uint32_t offset = 0; offset < Pm4::SH_NUM; offset++) {
		const bool user_data =
		    (offset >= 0x0cu && offset <= 0x2bu) || (offset >= 0x8cu && offset <= 0xabu) ||
		    (offset >= 0x10cu && offset <= 0x12bu) || offset == 0x82u || offset == 0x83u ||
		    offset == 0x102u || offset == 0x103u;
		Check(DrawStateTracker::IsUserDataRegister(offset) == user_data,
		      "user-data classification", offset);
		if (offset != Pm4::SH_NOP) {
			Check(VerdictAcross({SetShReg(offset, 1)}) == user_data,
			      "a single-register SET_SH_REG follows the classification", offset);
		}
	}
}

void TestShRegistersIndirect() {
	const auto ps = Pm4::SPI_SHADER_USER_DATA_PS_0;
	const std::vector<uint32_t> user_data {ps, 1, ps + 1, 2, Pm4::SPI_SHADER_USER_DATA_GS_0, 3,
	                                       Pm4::SPI_SHADER_USER_DATA_HS_0 | 0x10000000u, 4};
	Check(VerdictAcross({SetShRegIndirect(user_data)}),
	      "an indirect list of user data keeps the state");
	const std::vector<uint32_t> skipped {ps, 1, Pm4::SH_NOP, 2, 0xffffffffu, 3};
	Check(VerdictAcross({SetShRegIndirect(skipped)}),
	      "entries the handler skips are not classified");
	const std::vector<uint32_t> middle {ps, 1, Pm4::SPI_SHADER_PGM_LO_GS, 2, ps + 2, 3};
	Check(!VerdictAcross({SetShRegIndirect(middle)}),
	      "a non-user-data register in an indirect list invalidates");
	const std::vector<uint32_t> last {ps, 1, ps + 1, 2, Pm4::COMPUTE_USER_DATA_0, 3};
	Check(!VerdictAcross({SetShRegIndirect(last)}),
	      "a non-user-data register last in an indirect list invalidates");
	const std::vector<uint32_t> none;
	Check(VerdictAcross({SetShRegIndirect(none)}), "an empty indirect list writes nothing");
	Check(!VerdictAcross({Packet(Pm4::IT_SET_SH_REG_INDIRECT, {0, 0, 0})}),
	      "a malformed SET_SH_REG_INDIRECT invalidates");
	Check(!VerdictAcross({Packet(Pm4::IT_SET_SH_REG_INDIRECT, {0, 0, 0, 1})}),
	      "a SET_SH_REG_INDIRECT without a list invalidates");
}

void TestNops() {
	Check(VerdictAcross({Packet(Pm4::IT_NOP, {0, 0}, Pm4::R_ZERO)}), "padding keeps the state");
	for (const uint32_t marker: {0x68750000u, 0x68750004u, 0x6875000du}) {
		Check(VerdictAcross({Packet(Pm4::IT_NOP, {marker, 0}, Pm4::R_ZERO)}),
		      "a user-data marker keeps the state", marker);
	}
	for (const uint32_t marker: {0x68750777u, 0x68750778u, 0x68750781u, 0x68750005u}) {
		Check(!VerdictAcross({Packet(Pm4::IT_NOP, {marker, 0, 0, 0, 0, 0}, Pm4::R_ZERO)}),
		      "a flip or unknown marker invalidates", marker);
	}
	for (uint32_t r = 0; r < Pm4::R_NUM; r++) {
		const bool keeps = r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER;
		Check(VerdictAcross({Packet(Pm4::IT_NOP, {0, 0}, r)}) == keeps,
		      "NOP subtypes other than padding and debug markers invalidate", r);
	}
}

void TestEvents() {
	for (uint32_t event = 0; event < static_cast<uint32_t>(DrawStateTracker::Event::Count);
	     event++) {
		DrawStateTracker tracker;
		tracker.NotePacket(g_draw.data());
		(void)tracker.TakeDrawVerdict();
		tracker.NoteEvent(static_cast<DrawStateTracker::Event>(event));
		tracker.NotePacket(g_draw.data());
		Check(!tracker.TakeDrawVerdict(), "an out-of-stream event invalidates", event);

		// Also between a draw packet and the draw that takes its verdict.
		tracker.NotePacket(g_draw.data());
		tracker.NoteEvent(static_cast<DrawStateTracker::Event>(event));
		Check(!tracker.TakeDrawVerdict(), "an event clears a pending verdict", event);
	}
}

DrawReuseInputs BaseInputs() {
	static int kept_state = 0;
	DrawReuseInputs inputs;
	inputs.position = {.tick                        = 7,
	                   .command_buffer              = 0x1000,
	                   .render_pass_epoch           = 3,
	                   .rendering                   = true,
	                   .dynamic_state_invalidations = 2,
	                   .deferred_operations         = 5};
	inputs.kept_state                 = &kept_state;
	inputs.texture_generation         = 11;
	inputs.es_address                 = 0x100000;
	inputs.gs_address                 = 0x200000;
	inputs.ls_address                 = 0x400000;
	inputs.hs_address                 = 0x500000;
	inputs.ps_address                 = 0x300000;
	inputs.render_target_slice_offset = 1;
	inputs.indexed                    = true;
	inputs.index_type_and_size        = 1;
	return inputs;
}

DrawPipelineFeed BaseFeed() {
	static int vs = 0;
	static int ps = 0;
	DrawPipelineFeed feed;
	feed.programs.vertex_programs[0]         = &vs;
	feed.programs.ps_program                 = &ps;
	feed.programs.vertex_ids[0]              = 21;
	feed.programs.ps_id                      = 22;
	feed.programs.vertex_modules[0]          = 0x2100;
	feed.programs.ps_module                  = 0x2200;
	feed.programs.ps_active                  = true;
	feed.ps_sample_shading                   = false;
	feed.topology                            = 3;
	feed.primitive_restart                   = false;
	feed.vertex_input.binding_count          = 2;
	feed.vertex_input.attribute_count        = 2;
	feed.vertex_input.bindings[0]            = {.stride = 16, .instance = false};
	feed.vertex_input.bindings[1]            = {.stride = 8, .instance = true};
	feed.vertex_input.attributes[0]          = {.offset = 0, .binding = 0};
	feed.vertex_input.attributes[1]          = {.offset = 4, .binding = 1};
	return feed;
}

void TestRecord() {
	PipelineCache::Pipeline pipeline;
	const auto              inputs = BaseInputs();
	const auto              feed   = BaseFeed();

	DrawReuseRecord empty;
	Check(!empty.KeepsRenderState(inputs), "an empty record keeps nothing");
	Check(empty.ReusablePipeline(inputs.position, feed) == nullptr,
	      "an empty record has no pipeline");

	DrawReuseRecord record;
	record.Store(inputs, feed, nullptr);
	Check(!record.KeepsRenderState(inputs), "a draw without a pipeline is not recorded");

	record.Store(inputs, feed, &pipeline);
	Check(record.KeepsRenderState(inputs), "identical inputs keep the render state");
	Check(record.ReusablePipeline(inputs.position, feed) == &pipeline,
	      "an identical position and feed keep the pipeline");

	const auto previous = record.Take();
	Check(!record.KeepsRenderState(inputs), "a taken record is invalid until stored again");
	Check(previous.KeepsRenderState(inputs), "Take() hands out the previous draw");

	using Mutation = std::function<void(DrawReuseInputs&)>;
	const std::vector<std::pair<const char*, Mutation>> input_changes {
	    {"tick", [](auto& i) { i.position.tick++; }},
	    {"command buffer", [](auto& i) { i.position.command_buffer++; }},
	    {"render pass epoch", [](auto& i) { i.position.render_pass_epoch++; }},
	    {"rendering", [](auto& i) { i.position.rendering = false; }},
	    {"dynamic state invalidations", [](auto& i) { i.position.dynamic_state_invalidations++; }},
	    {"deferred operations", [](auto& i) { i.position.deferred_operations++; }},
	    {"kept state", [](auto& i) { i.kept_state = nullptr; }},
	    {"texture generation", [](auto& i) { i.texture_generation++; }},
	    {"ES address", [](auto& i) { i.es_address += 0x100; }},
	    {"GS address", [](auto& i) { i.gs_address += 0x100; }},
	    {"LS address", [](auto& i) { i.ls_address += 0x100; }},
	    {"HS address", [](auto& i) { i.hs_address += 0x100; }},
	    {"PS address", [](auto& i) { i.ps_address += 0x100; }},
	    {"render-target slice offset", [](auto& i) { i.render_target_slice_offset++; }},
	    {"indexed", [](auto& i) { i.indexed = false; }},
	    {"index type", [](auto& i) { i.index_type_and_size = 0; }},
	};
	for (const auto& [name, change]: input_changes) {
		auto changed = inputs;
		change(changed);
		if (!previous.KeepsRenderState(changed)) {
			continue;
		}
		std::fprintf(stderr, "DrawReuseTests: failed: changing the %s keeps the render state\n",
		             name);
		g_failures++;
	}
	for (const auto& [name, change]: input_changes) {
		auto changed = inputs;
		change(changed);
		if (changed.position == inputs.position) {
			continue;
		}
		if (previous.ReusablePipeline(changed.position, feed) != nullptr) {
			std::fprintf(stderr, "DrawReuseTests: failed: changing the %s keeps the pipeline\n",
			             name);
			g_failures++;
		}
	}

	using FeedMutation = std::function<void(DrawPipelineFeed&)>;
	static int other = 0;
	const std::vector<std::pair<const char*, FeedMutation>> feed_changes {
	    {"VS program", [](auto& f) { f.programs.vertex_programs[0] = &other; }},
	    {"PS program", [](auto& f) { f.programs.ps_program = &other; }},
	    {"VS id", [](auto& f) { f.programs.vertex_ids[0]++; }},
	    {"PS id", [](auto& f) { f.programs.ps_id++; }},
	    {"VS module", [](auto& f) { f.programs.vertex_modules[0]++; }},
	    {"PS module", [](auto& f) { f.programs.ps_module++; }},
	    {"HS program", [](auto& f) { f.programs.vertex_programs[1] = &other; }},
	    {"TES id", [](auto& f) { f.programs.vertex_ids[2]++; }},
	    {"TES module", [](auto& f) { f.programs.vertex_modules[2]++; }},
	    {"pixel activity", [](auto& f) { f.programs.ps_active = false; }},
	    {"sample shading", [](auto& f) { f.ps_sample_shading = true; }},
	    {"topology", [](auto& f) { f.topology++; }},
	    {"primitive restart", [](auto& f) { f.primitive_restart = true; }},
	    {"binding count", [](auto& f) { f.vertex_input.binding_count--; }},
	    {"attribute count", [](auto& f) { f.vertex_input.attribute_count--; }},
	    {"binding stride", [](auto& f) { f.vertex_input.bindings[1].stride = 12; }},
	    {"binding rate", [](auto& f) { f.vertex_input.bindings[0].instance = true; }},
	    {"attribute offset", [](auto& f) { f.vertex_input.attributes[1].offset = 8; }},
	    {"attribute binding", [](auto& f) { f.vertex_input.attributes[1].binding = 0; }},
	};
	for (const auto& [name, change]: feed_changes) {
		auto changed = feed;
		change(changed);
		if (previous.ReusablePipeline(inputs.position, changed) != nullptr) {
			std::fprintf(stderr, "DrawReuseTests: failed: changing the %s keeps the pipeline\n",
			             name);
			g_failures++;
		}
		if (changed.programs != feed.programs && previous.KeepsPrograms(changed.programs)) {
			std::fprintf(stderr, "DrawReuseTests: failed: changing the %s keeps the programs\n",
			             name);
			g_failures++;
		}
	}
	Check(previous.KeepsPrograms(feed.programs), "the same programs keep the render state");
	Check(!empty.KeepsPrograms(feed.programs), "an empty record keeps no programs");
}

void TestDynamicState() {
	GraphicsDynamicState a {};
	a.viewport_count = 1;
	a.viewports[0]   = vk::Viewport {0.0f, 0.0f, 1920.0f, 1080.0f, 0.0f, 1.0f};
	a.scissors[0]    = vk::Rect2D {{0, 0}, {1920, 1080}};
	auto b           = a;
	Check(a == b, "identical dynamic state compares equal");
	b.scissors[0].extent.width = 960;
	Check(!(a == b), "a changed scissor differs");
	b = a;
	b.stencil[1].reference = 1;
	Check(!(a == b), "a changed stencil reference differs");
	b = a;
	b.stencil[0].write_mask = 0xff;
	Check(!(a == b), "a changed stencil write mask differs");
	b = a;
	b.color_write_enable[0] = VK_TRUE;
	Check(!(a == b), "a changed colour write enable differs");
	b = a;
	b.feedback_aspects = vk::ImageAspectFlagBits::eDepth;
	Check(!(a == b), "a changed attachment feedback loop differs");
}

} // namespace

int main() {
	TestInitialStateAndConsumption();
	TestOpcodes();
	TestIndirectDraws();
	TestShRegisters();
	TestShRegistersIndirect();
	TestNops();
	TestEvents();
	TestRecord();
	TestDynamicState();
	if (g_failures != 0) {
		std::fprintf(stderr, "DrawReuseTests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("DrawReuseTests: all tests passed\n");
	return EXIT_SUCCESS;
}
