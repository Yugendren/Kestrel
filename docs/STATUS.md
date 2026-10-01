# Status

Last updated: 2026-10-01

## What works

- **Astro's Playroom** boots, shows menus and plays through the hub. Frame rate is 25-60 fps
  depending on the scene on the reference PC. The game ties its speed to its frame rate, so
  below 60 fps it runs in slow motion unless the game-speed patch is on. The patch loader is
  on `main`: with the "correct speed at 30 fps" patch and `--frame-cap 30` (both enabled by the
  `astro-low` and `astro-high` presets) the game runs at correct speed at 30 fps. The Cooling
  Springs crash is fixed on `main` (tri-fan geometry shader support, by nmzik). Known crash:
  entering the SSD Speedway portal.
- **Demon's Souls** reached gameplay on 2026-10-01 on a development branch. The fixes are not on
  `main` yet. It is not playable (see below).
- **Astro Bot** is untested on Kestrel.

## Reference hardware numbers

Ryzen 5 3600 (6 cores, Zen 2) and RTX 3060 12 GB. 1080p internal (render scale 0.5), ray
tracing off, vertex fetch cpu. Astro's Playroom, average fps:

| Scene | Windows | Linux (PGO build) |
| --- | --- | --- |
| Pre-plaza (opening), still camera | ~53 | ~60 |
| Pre-plaza, moving | - | ~54 |
| CPU Plaza (hub), still camera | ~25-27 | ~34 |
| CPU Plaza, moving | ~41 | ~40-41 (median 59) |

- At native 4K internal (render scale 1.0) the RTX 3060 is GPU-bound at 22-31 fps.
- In the CPU Plaza the emulator's command-processor thread is the limit: about 23.5 ms per
  frame, slightly more than the GPU at about 21.5 ms. CPU-side work is the current focus.
- Demon's Souls: about 2 fps in gameplay. The game's job threads spin.
- PGO is profile-guided optimisation. It is available on Linux builds (+18-22%). Windows PGO is
  under test.

## In progress

- Demon's Souls: dark square specks, and two intermittent crashes (one on save load, one about
  two minutes in).
- Variable-step game speed for Astro's Playroom (correct at any frame rate), in testing.
- Optional FSR 1 upscaler.
- Frame-time log and performance overlay (F10).
- Performance ports in A/B testing: SRT JIT (+4-8% expected), queue-drain (from upstream, by
  nmzik), hardware buffer bounds (from BryanKAdams/KytyPS5), relocatable draw records (the
  biggest CPU lever).

## Roadmap to 1.0

Version 1.0 means: Astro's Playroom playable at correct game speed, and Demon's Souls stable in
gameplay with no crash in a 30-minute session.

1. Demon's Souls: fix the red-zone crash and the save-load crash, and calm the spinning job
   threads. Aim for a stable 30 fps-capped session. Frame generation such as Lossless Scaling
   can take 30 to 60.
2. Astro's Playroom: the patch loader and the 30 fps correct-speed patch are on `main`; next is
   the SSD Speedway portal crash, and frame generation to 60.
3. Astro's Playroom: variable-step game speed, correct at any frame rate.
4. Astro Bot: first run and fixes, including porting freeze fixes from BryanKAdams/KytyPS5.
5. CPU performance: relocatable draw records, SRT JIT, recording worker, Windows PGO.
6. Then Ghost of Yotei and packaged releases.

Results from your own machine are welcome. See [QUICKSTART.md](QUICKSTART.md#reporting-results).
