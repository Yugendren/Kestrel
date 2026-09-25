#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAWSTATETRACKER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAWSTATETRACKER_H_

#include "graphics/guest_gpu/pm4.h"

#include <cstdint>

namespace Libs::Graphics {

// Tells the renderer whether a draw packet follows the previous draw packet with nothing executed
// in between that could change what the draw renders to or with, other than its own arguments and
// shader user data. Such a draw may keep the previous draw's render targets, pipeline, dynamic
// state and render pass (see DrawReuseRecord, and RenderExecutor::DrawIndex()/DrawAuto()).
//
// The tracker is a whitelist: the command processor reports every packet it is about to execute,
// and every packet not listed in NotePacket() invalidates -- context, uconfig and non-user-data
// SH register writes, dispatches, synchronisation, memory writes, flips, unknown NOP subtypes and
// any opcode added in the future. Work that runs on the GPU thread outside the packet stream is
// reported with NoteEvent() and invalidates as well; see graphicsRun.cpp.
//
// GPU thread only.
class DrawStateTracker {
public:
	// GPU-thread work outside the packet stream that can change register or renderer state.
	enum class Event : uint8_t {
		QueuedCommand,     // a GuestGpu command runs (SendCommand(), ProcessCommands())
		SubmissionSlice,   // a submission, or the next slice of a blocked one, starts executing
		ProcessorReset,    // CommandProcessor::Reset() clears the registers
		Flush,             // the command buffer is submitted, waited on or restarted
		GarbageCollection, // RenderContext::RunGarbageCollector()
		FlipPreparation,   // a CPU flip is prepared
		Count,
	};

	// Mirrors NormalizeRegisterOffset() in pm4Handlers.cpp: the selector bits an indirect register
	// list may carry above the register offset.
	static constexpr uint32_t RegisterSelectorMask = 0x70000000u;

	// SPI_SHADER_USER_DATA_{PS,GS,HS}_0..31 and the GS/HS user-data address pairs: registers whose
	// only effect is the user data the next draw's programs read, which the renderer re-reads for
	// every draw. VS/ES/LS user data do not exist on this hardware and compute user data only
	// feeds dispatches, so neither is on the list.
	[[nodiscard]] static constexpr bool IsUserDataRegister(uint32_t offset) noexcept {
		return (offset >= Pm4::SPI_SHADER_USER_DATA_PS_0 &&
		        offset <= Pm4::SPI_SHADER_USER_DATA_PS_31) ||
		       (offset >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		        offset <= Pm4::SPI_SHADER_USER_DATA_GS_31) ||
		       (offset >= Pm4::SPI_SHADER_USER_DATA_HS_0 &&
		        offset <= Pm4::SPI_SHADER_USER_DATA_HS_31) ||
		       offset == Pm4::SPI_SHADER_USER_DATA_ADDR_LO_GS ||
		       offset == Pm4::SPI_SHADER_USER_DATA_ADDR_HI_GS ||
		       offset == Pm4::SPI_SHADER_USER_DATA_ADDR_LO_HS ||
		       offset == Pm4::SPI_SHADER_USER_DATA_ADDR_HI_HS;
	}

	// Called with the packet header at `packet[0]` before the command processor runs the packet's
	// handler. A packet may only be read as far as the handler itself reads it.
	void NotePacket(const uint32_t* packet) noexcept {
		const uint32_t header = packet[0];
		switch ((header >> 8u) & 0xffu) {
			case Pm4::IT_SET_SH_REG: NoteShRegisters(packet); return;
			case Pm4::IT_SET_SH_REG_INDIRECT: NoteShRegistersIndirect(packet); return;
			// Draw arguments: the renderer compares the index encoding itself and binds the index
			// buffer and instance count for every draw.
			case Pm4::IT_INDEX_BASE:
			case Pm4::IT_INDEX_BUFFER_SIZE:
			case Pm4::IT_INDEX_TYPE:
			case Pm4::IT_NUM_INSTANCES:
			case Pm4::IT_SET_BASE:
			// Only calls into another command buffer, whose packets are reported one by one.
			case Pm4::IT_INDIRECT_BUFFER: return;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_AUTO: BeginDraw(true); return;
			// The arguments live in guest memory the command processor does not look at here, and
			// the renderer never reuses state for them; they still end the span the next draw is
			// compared across.
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: BeginDraw(false); return;
			case Pm4::IT_NOP: NoteNop(packet); return;
			default: Invalidate(); return;
		}
	}

	// Every event invalidates: the next draw packet is not eligible.
	void NoteEvent(Event event) noexcept {
		static_cast<void>(event);
		Invalidate();
	}

	// Whether the draw now being executed comes from a direct draw packet with only whitelisted
	// packets since the previous draw packet. The verdict belongs to one draw: taking it clears it,
	// so a packet that executes several draws, or a draw reached some other way, reads false.
	[[nodiscard]] bool TakeDrawVerdict() noexcept {
		const bool verdict = m_verdict;
		m_verdict          = false;
		return verdict;
	}

private:
	// Something outside the whitelist ran: the next draw packet is not eligible.
	void Invalidate() noexcept {
		m_clean   = false;
		m_verdict = false;
	}

	// Marker ids CpOpMarker() (pm4Handlers.cpp) handles without leaving the command stream: a plain
	// marker and the user-data type tags that precede user-data writes. Its other ids flip.
	static constexpr uint32_t AgcMarkerTag         = 0x68750000u;
	static constexpr uint32_t AgcMarkerTagMask     = 0xffff0000u;
	static constexpr uint32_t AgcMarkerIdMask      = 0xfffu;
	static constexpr uint32_t AgcMarkerPlain       = 0x0u;
	static constexpr uint32_t AgcMarkerVsharpData  = 0x4u;
	static constexpr uint32_t AgcMarkerRegionData  = 0xdu;

	void BeginDraw(bool direct) noexcept {
		m_verdict = direct && m_clean;
		// The draw itself is what the next one is compared against.
		m_clean = true;
	}

	// SET_SH_REG writes KYTY_PM4_LEN - 2 consecutive registers starting at packet[1]. Every one of
	// them has to be a user-data register.
	void NoteShRegisters(const uint32_t* packet) noexcept {
		const uint32_t first = packet[1];
		if (first == Pm4::SH_NOP) {
			return; // CpOpSetShaderReg() writes nothing for it.
		}
		const uint32_t count = KYTY_PM4_LEN(packet[0]) - 2u;
		for (uint32_t i = 0; i < count; i++) {
			if (!IsUserDataRegister(first + i)) {
				Invalidate();
				return;
			}
		}
	}

	// SET_SH_REG_INDIRECT names a list of (offset, value) pairs in guest memory, which its handler
	// reads right after this; classify every register the handler will write, skipping the same
	// entries it skips.
	void NoteShRegistersIndirect(const uint32_t* packet) noexcept {
		if (KYTY_PM4_LEN(packet[0]) != 5u) {
			Invalidate();
			return;
		}
		const uint32_t count = packet[4] & 0x3fffu;
		if (count == 0) {
			return;
		}
		const auto* list = reinterpret_cast<const uint32_t*>(
		    (static_cast<uint64_t>(packet[1]) & 0xfffffffcu) |
		    (static_cast<uint64_t>(packet[2]) << 32u));
		if (list == nullptr) {
			Invalidate();
			return;
		}
		for (uint32_t i = 0; i < count; i++) {
			const uint32_t raw    = list[i * 2u];
			const uint32_t offset = raw & ~RegisterSelectorMask;
			if (offset == Pm4::SH_NOP || raw == 0xffffffffu) {
				continue;
			}
			if (!IsUserDataRegister(offset)) {
				Invalidate();
				return;
			}
		}
	}

	void NoteNop(const uint32_t* packet) noexcept {
		switch (KYTY_PM4_R(packet[0])) {
			case Pm4::R_ZERO: {
				if ((packet[1] & AgcMarkerTagMask) != AgcMarkerTag) {
					return; // padding
				}
				const uint32_t id = packet[1] & AgcMarkerIdMask;
				if (id != AgcMarkerPlain && id != AgcMarkerVsharpData && id != AgcMarkerRegionData) {
					Invalidate();
				}
				return;
			}
			case Pm4::R_PUSH_MARKER:
			case Pm4::R_POP_MARKER: return; // debug labels
			default: Invalidate(); return;
		}
	}

	// Only whitelisted packets ran since the previous draw packet.
	bool m_clean = false;
	// The verdict for the draw packet being executed; see TakeDrawVerdict().
	bool m_verdict = false;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAWSTATETRACKER_H_ */
