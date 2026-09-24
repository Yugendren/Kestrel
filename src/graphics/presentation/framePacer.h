#ifndef EMULATOR_INCLUDE_GRAPHICS_PRESENTATION_FRAMEPACER_H_
#define EMULATOR_INCLUDE_GRAPHICS_PRESENTATION_FRAMEPACER_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// Host-side pacing for --frame-cap. Flips can only be presented on a vblank, so the cap is a
// running deadline on the ideal cap grid, checked with half a vblank of slack:
//  - Comparing the time since the last flip against exactly 1/cap does not work on a vblank
//    grid: a 30 fps cap on 60 Hz needs every 2nd vblank, and timer jitter lands that vblank a few
//    microseconds short of 1/30 s about half the time, so the flip slipped to the 3rd vblank and
//    --frame-cap 30 delivered ~23 fps.
//  - Advancing the deadline by whole cap periods (instead of restarting it at the present time)
//    makes caps that are not a divisor of the refresh average out correctly: 45 on 60 Hz
//    alternates 1- and 2-vblank gaps instead of collapsing to 30.
//  - The deadline is never allowed to trail the last present by less than 3/4 of a cap period,
//    so a stretch with nothing to present cannot bank credit and release a burst of flips at the
//    full refresh rate afterwards.
// Only presented frames consume a slot: a vblank on which the guest had nothing ready does not
// delay its next frame by a whole cap period.
class FramePacer {
public:
	// Times are in host counter ticks; vblank_period and flip_period in the same unit.
	[[nodiscard]] bool IsFlipDue(uint64_t now, uint64_t vblank_period) const {
		return !m_started || now + vblank_period / 2 >= m_next_flip;
	}

	void OnPresented(uint64_t now, uint64_t flip_period) {
		const uint64_t earliest = now + flip_period - flip_period / 4;
		m_next_flip             = m_started ? std::max(m_next_flip + flip_period, earliest) : now + flip_period;
		m_started               = true;
	}

	void Reset() { m_started = false; }

private:
	uint64_t m_next_flip = 0;
	bool     m_started   = false;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_GRAPHICS_PRESENTATION_FRAMEPACER_H_ */
