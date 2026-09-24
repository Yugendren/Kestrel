#include "graphics/presentation/framePacer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using Libs::Graphics::FramePacer;

void Check(bool value, const char* message, double got) {
	if (!value) {
		std::fprintf(stderr, "FramePacerTests: failed: %s (got %.3f)\n", message, got);
		std::abort();
	}
}

constexpr uint64_t FREQ = 10000000; // 10 MHz host counter

// Simulates a 60 Hz vblank grid with +/-jitter ticks of timer noise and a guest that always has a
// frame ready (or only every `ready_every` vblanks). Returns presented frames per second.
double Simulate(uint32_t cap, int64_t jitter, int seconds, int ready_every = 1) {
	const uint64_t period      = FREQ / 60;
	const uint64_t flip_period = FREQ / cap;
	FramePacer     pacer;
	int            presented = 0;
	const int      vblanks   = 60 * seconds;
	for (int i = 1; i <= vblanks; i++) {
		// Deterministic pseudo-jitter, alternating early and late.
		const int64_t  noise = (i % 2 == 0 ? -jitter : jitter) * ((i * 7919) % 5) / 4;
		const uint64_t now   = static_cast<uint64_t>(static_cast<int64_t>(period * i) + noise);
		if (i % ready_every != 0) {
			continue;
		}
		if (pacer.IsFlipDue(now, period)) {
			pacer.OnPresented(now, flip_period);
			presented++;
		}
	}
	return static_cast<double>(presented) / seconds;
}

void TestCapsOnVblankGrid() {
	for (const int64_t jitter : {int64_t {0}, int64_t {200}, int64_t {3000}}) {
		const double f30 = Simulate(30, jitter, 20);
		Check(f30 > 29.9 && f30 <= 30.05, "cap 30 must present every 2nd vblank", f30);
		const double f20 = Simulate(20, jitter, 20);
		Check(f20 > 19.9 && f20 <= 20.05, "cap 20 must present every 3rd vblank", f20);
		const double f45 = Simulate(45, jitter, 20);
		Check(f45 > 44.0 && f45 <= 45.1, "cap 45 must average 45 on a 60 Hz grid", f45);
		const double f15 = Simulate(15, jitter, 20);
		Check(f15 > 14.9 && f15 <= 15.05, "cap 15 must present every 4th vblank", f15);
	}
}

void TestNoBurstAfterIdle() {
	const uint64_t period      = FREQ / 60;
	const uint64_t flip_period = FREQ / 30;
	FramePacer     pacer;
	pacer.OnPresented(period, flip_period);
	// One second with nothing to present, then frames on every vblank: the cap must hold at once.
	uint64_t now       = period * 61;
	int      presented = 0;
	for (int i = 0; i < 60; i++, now += period) {
		if (pacer.IsFlipDue(now, period)) {
			pacer.OnPresented(now, flip_period);
			presented++;
		}
	}
	Check(presented == 30, "no catch-up burst after an idle stretch", presented);
}

void TestSlowGuestIsNotDelayed() {
	// A guest that only has a frame every 3rd vblank under a 30 cap must still present all of them.
	const double f = Simulate(30, 200, 20, 3);
	Check(f > 19.9 && f <= 20.05, "a slow guest keeps its own rate under a higher cap", f);
}

} // namespace

int main() {
	TestCapsOnVblankGrid();
	TestNoBurstAfterIdle();
	TestSlowGuestIsNotDelayed();
	std::printf("FramePacerTests: all passed\n");
	return 0;
}
