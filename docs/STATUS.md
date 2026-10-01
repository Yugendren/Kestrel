# Status

Last updated: 2026-10-01

See the [roadmap](ROADMAP.md) for where to help.

## What works

- **Astro's Playroom** boots, shows menus and plays through the hub. Frame rate is 25-55 fps
  depending on the scene on the reference PC (table below). The game ties its speed to its frame
  rate, so below 60 fps it runs in slow motion unless a game-speed patch is on. Both patches are
  on `main`: "correct speed at any frame rate" (the `astro-low-unlocked` and
  `astro-high-unlocked` presets, with a 30 fps cap by default) and "correct speed at 30 fps"
  (the `astro-low` and `astro-high` presets). The Cooling Springs crash is fixed on `main`
  (tri-fan geometry shader support, by nmzik). Known crash: entering the SSD Speedway portal.
- The performance overlay (F10), clip recording (F9) and `--frame-time-log` are on `main`.
- **Demon's Souls** reached gameplay on 2026-10-01 on a development branch. The fixes are not on
  `main` yet. It is not playable (see below).
- **Astro Bot** is untested on Kestrel.

## Reference hardware numbers

Linux, `main` at `bbfff154`, non-PGO build. Ryzen 5 3600 (6 cores, Zen 2) and RTX 3060 12 GB.
1080p internal (render scale 0.5), ray tracing off, vertex fetch cpu. Astro's Playroom.

| Scene | Avg fps | 1% low fps | CP ms | GPU ms |
| --- | --- | --- | --- | --- |
| CPU Plaza, still | 26.9 | 19.8 | ~37 | ~25 |
| CPU Plaza, moving | 33.9 | 17.4 | ~29 | ~22 |
| Pre-plaza, still | 53.5 | 28.4 | ~18 | ~16 |
| Pre-plaza, moving | 43.2 | 27.5 | ~23 | ~18 |
| Cooling Springs, still | 24.9 | 17.9 | ~37 | ~40 |

- CP is the command-processor thread, the single thread that turns PS5 GPU commands into
  Vulkan. CP and GPU times are busy-time estimates. 60 fps needs both under 16.7 ms.
- At native 4K internal (render scale 1.0) the RTX 3060 is GPU-bound at 22-31 fps.
- A Windows PGO build measured +21% in the moving CPU Plaza (39.9 to 48.2 fps) on an older
  build. PGO is profile-guided optimisation. Linux PGO builds gain about 18-22%.
- Demon's Souls: about 2 fps in gameplay. The game's job threads spin.

The full plan, measuring method and work items are in the [roadmap](ROADMAP.md).

## In progress

- Demon's Souls: dark square specks, and two intermittent crashes (one on save load, one about
  two minutes in).
- Optional FSR 1 upscaler.
- Performance ports in A/B testing: SRT JIT, queue-drain (from upstream, by
  nmzik), hardware buffer bounds (from BryanKAdams/KytyPS5), relocatable draw records (the
  biggest CPU lever).

## Roadmap to 1.0

Version 1.0 means: Astro's Playroom playable at correct game speed, and Demon's Souls stable in
gameplay with no crash in a 30-minute session.

1. Demon's Souls: fix the red-zone crash and the save-load crash, and calm the spinning job
   threads. Aim for a stable 30 fps-capped session. Frame generation such as Lossless Scaling
   can take 30 to 60.
2. Astro's Playroom: the patch loader and both speed patches are on `main`; next is the SSD
   Speedway portal crash, and frame generation to 60.
3. Astro Bot: first run and fixes, including porting freeze fixes from BryanKAdams/KytyPS5.
4. CPU performance: relocatable draw records, SRT JIT, recording worker, PGO builds.
5. Then Ghost of Yotei and packaged releases.

Results from your own machine are welcome. See [QUICKSTART.md](QUICKSTART.md#reporting-results).
