# Where Kestrel is and how to help

Last updated: 2026-10-01

## Goal

Native 60 fps at 1080p internal on a Ryzen 5 3600 and RTX 3060 class PC. A PS5 runs Astro's
Playroom at 60 fps, and an RTX 3060 is roughly PS5-class for GPU power, so this is a realistic
ceiling.

Until we get there, the goal is 40 or more real fps everywhere. At that rate frame generation
(Lossless Scaling today, a built-in option later) looks good.

## The budget

60 fps is 16.7 ms per frame. Two things must both fit in that time:

- The command-processor (CP) thread. It translates PS5 GPU commands into Vulkan calls. It runs
  on a single thread and is the main bottleneck today.
- The GPU.

## Baseline scorecard

Linux, `main` at `bbfff154`, non-PGO build, render scale 0.5 (1080p internal), `--rt-mode off`,
`--vertex-fetch cpu`. Ryzen 5 3600 and RTX 3060.

| Scene | Avg fps | 1% low fps | CP ms | GPU ms |
| --- | --- | --- | --- | --- |
| CPU Plaza, still | 26.9 | 19.8 | ~37 | ~25 |
| CPU Plaza, moving | 33.9 | 17.4 | ~29 | ~22 |
| Pre-plaza, still | 53.5 | 28.4 | ~18 | ~16 |
| Pre-plaza, moving | 43.2 | 27.5 | ~23 | ~18 |
| Cooling Springs, still | 24.9 | 17.9 | ~37 | ~40 |

CP and GPU milliseconds are busy-time estimates, not exact.

A Windows PGO build measured +21% in the moving CPU Plaza scene (39.9 to 48.2 fps) on an older
build. PGO means profile-guided optimisation.

## How to measure

Use three scenes of Astro's Playroom, each still and moving:

1. CPU Plaza (the hub).
2. Pre-plaza (the opening area).
3. Cooling Springs. Enter it from CPU Plaza through the fan portal.

Run with `--frame-time-log <file>` and `--perf-overlay` (F10 toggles the overlay). Report:

- average fps
- 1% low and 10% low fps
- the share of frames longer than 33.3 ms
- the ray-tracing mode line the emulator prints at startup
- the git hash of your build

Every performance PR should include before and after numbers on these scenes. A change that
helps one scene and hurts another is fine, but say so.

## Work items

Estimates are rough. "Saving" means milliseconds per frame on the reference PC unless noted.

### CPU (command-processor thread)

| Item | Saving | Effort | Status | How to start |
| --- | --- | --- | --- | --- |
| PGO builds of `main` (Windows and Linux) | 15-20% fps (measured +21%) | 1-2 days per OS | Linux works, Windows under test | CMake option `KYTY_PGO` in `cmake/KytyOptimization.cmake`. Build with `GENERATE`, play the three scenes, merge the profile, rebuild with `USE`. |
| Queue-drain fix from upstream (nmzik, KytyPS5 commit `840b9f57`). The game thread waits about 13 ms per frame at suspend points in pre-plaza. | up to ~13 ms of waiting in pre-plaza | ~1 day | Ported, needs A/B | `src/graphics/guest_gpu/graphicsRun.cpp` and `src/graphics/guest_gpu/command_processor`. Run the scorecard with and without it. |
| Recording and submission worker thread. Design from the 3-stage pipeline in TheCruZ/KytyPS5-GTA. | ~2.5 ms | 1.5-2 weeks | Not started | `src/graphics/host_gpu/renderer/commandScheduler.cpp` and `renderDraw.cpp`. |
| SRT walk JIT. An x86-64 JIT of the shader resource table (SRT) walk. | ~1.5-2 ms | 1-2 weeks | Prototype exists on an older line. Needs a redesign for the current `SrtWalker`. Measure first. | `src/graphics/shader/recompiler/ir/passes/SrtWalker.cpp` and `SrtWalker.h`. |
| Spin calming for guest threads that busy-wait | Unknown. A few ms on 6-core CPUs. | Few days | Idea | Guest threads that spin steal cores from the CP thread. Start in `src/loader` and the kernel thread code to find the spin sites. |
| Relocatable draw records. Reuse last frame's translated draws. 97% of draws match the previous frame. | 5-8 ms | 3-4 weeks | Prototype hangs the GPU on replay. Record storage must be in place first. | `src/graphics/host_gpu/renderer/drawReuse.h`, `drawDelta.h` and `renderDraw.cpp`. |
| Speculative parallel SRT workers with O(1) global-epoch validation | 1-2.5 ms | 1-2 weeks | Idea | Same SRT code as above. Walk in parallel, then validate with one global epoch counter. |

### GPU

| Item | Saving | Effort | Status | How to start |
| --- | --- | --- | --- | --- |
| Hardware buffer bounds via `VK_EXT_robustness2` (from BryanKAdams/KytyPS5) | 1.5-2.7 ms | Few days | Ported, needs A/B | Device setup in `src/graphics/presentation/window/vulkanWindow.cpp` and the shader backend in `src/graphics/shader/recompiler/backend/spirv`. |
| Cooling Springs GPU census. The scene costs about 40 ms of GPU time at 1080p. Find the expensive passes. | Unknown, likely large | 1-2 days | Not started. A good first task. | Use RenderDoc (F1 captures a frame) or the Vulkan timestamp tools of your choice. Draws are issued from `src/graphics/host_gpu/renderer/renderDraw.cpp` and `renderCompute.cpp`. |
| Real ray-tracing skip. `--rt-mode off` still runs BVH traversal and only forces misses. | Unknown | 1-2 weeks | Idea | `src/graphics/shader/recompiler/backend/spirv` (ray query code). |
| Lower render scale (0.4 to 0.33) plus FSR 1 for GPU-bound scenes | Large on GPU-bound scenes | Few days | FSR 1 is on a branch | Render scale code is in `src/graphics/host_gpu/renderer/renderScale.h` and `passScale.h`. |

### Smoothness

| Item | Saving | Effort | Status | How to start |
| --- | --- | --- | --- | --- |
| Shader and pipeline pre-load before play. Removes 100+ ms hitches on first use of a shader. | Removes hitches, not average fps | 1-2 weeks | Idea. See the ideas in upstream KytyPS5 PR 939. | `src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp` and `shaders.cpp`. |
| Built-in frame generation. AMD FSR 3.1 frame generation using the game's motion vectors and depth. lsfg-vk (optical flow) as a fallback. | Doubles presented fps | Months | Idea. Do it after base fps is 40 or more. | `src/graphics/presentation`. |

### Games

| Item | Status | How to start |
| --- | --- | --- |
| Astro's Playroom: SSD Speedway portal crash. Null dereference at `eboot+0x8524f`. It also happens on builds without patches. | Open | Reproduce it, then find the caller of the faulting function in the crash log. The game's kernel and loader calls are in `src/loader`. |
| Astro Bot (PPSA21564) first boot. Upstream issue #835: crash after the Team Asobi splash. BryanKAdams/KytyPS5 has freeze fixes. | Untested on Kestrel | Try it, report the log. Port freeze fixes with the original author kept. |
| Astro Bot any-frame-rate patch | Draft needs the game version and a runtime check of 10 fixed 1/60 s sites | Patch format is in [PATCHES.md](../PATCHES.md). Loader code is in `src/loader/patches`. |
| Demon's Souls (PPSA01342 v01.005). It reaches gameplay on a development line at about 2 fps. | Open problems: a save-load crash (a lost list update in the game's resource loader), a red-zone crash about 2 minutes in (`--redzone` on Windows covers it), job threads that spin (a sleep patch is drafted), and dark square specks. | Start with the spinning job threads. A sleep patch frees cores. |

## Known behaviour

- The any-frame-rate patch keeps game speed exact at any fps. Jumps are lower below 30 fps,
  because that is how the game's physics works.
- Uncapped on a 60 Hz display the picture judders. Frames alternate between 16.7 and 33.3 ms
  under vsync. Use `--frame-cap 30` on a 60 Hz display. On 120 or 144 Hz use a divisor of the
  refresh rate (40 on 120 Hz, 48 on 144 Hz). On a VRR display you can leave it uncapped.

## Credits

Kestrel reuses work from upstream KytyPS5 and Kyty, nmzik, BryanKAdams, chenxiao07 (PR 939),
TheCruZ and Jetsku. Cherry-picks keep their original authors.
