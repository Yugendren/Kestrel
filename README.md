# Kestrel

Kestrel is a PlayStation 5 emulator for Windows and Linux. It is a fork of
[KytyPS5](https://github.com/KytyPS5/KytyPS5), which is based on
[Kyty](https://github.com/InoriRus/Kyty). Kestrel focuses on low-end playability: getting PS5
games to run acceptably on modest PCs. The reference machine is a Ryzen 5 3600 with an RTX 3060
12 GB. Kestrel reuses proven work from KytyPS5 and its forks and keeps the original authors on
their commits. Our own effort goes into render scale, a lighting remap, a shader fix for
speckles, game-speed patches, portability, careful measurement and low-end tuning.

> [!IMPORTANT]
> Kestrel is not affiliated with Sony Interactive Entertainment or PlayStation. It does not
> distribute games or system software. Use only game dumps that you obtained legally from a
> console you own. Never upload or share game files.

The emulator binary is still named `kyty_emulator` (`kyty_emulator.exe` on Windows).

## Games

| Game | Title ID | State | Notes |
| --- | --- | --- | --- |
| Astro's Playroom | PPSA01325 | Playable-ish | Boots, menus and gameplay through the hub work. 25-60 fps depending on the scene. Correct game speed at any frame rate with the any-frame-rate patch (the `astro-low-unlocked` and `astro-high-unlocked` presets enable it). Use `--frame-cap 30` on a 60 Hz display, or a divisor of your refresh rate, or VRR, to avoid judder. The 30 fps patch plus cap is in the `astro-low` and `astro-high` presets. The Cooling Springs crash is fixed on `main` (tri-fan geometry shader support, by nmzik). Known crash: entering the SSD Speedway portal. |
| Demon's Souls (v01.005) | PPSA01342 | Not playable yet | Reaches gameplay on a development branch. About 2 fps, dark square specks, two intermittent crashes. |
| Astro Bot | PPSA21564 | Untested | Not tried on Kestrel yet. Upstream KytyPS5 reports a crash after the Team Asobi splash. |
| Ghost of Yotei | - | Planned | Later target. |

Details and numbers are on the [status page](docs/STATUS.md).

## Requirements

- CPU: x86-64 with AVX2 (x86-64-v3). That means Intel Haswell (4th gen Core) or newer, or AMD
  Zen 1 or newer. The default build refuses to start on an older CPU and says why. 6 or more
  cores are recommended.
- GPU: Vulkan 1.3 with a current driver. NVIDIA, AMD and Intel are supported. Tested on an
  RTX 3060 (12 GB). 8 GB or more VRAM is recommended. A 4K internal render needs more.
- RAM: 16 GB or more recommended.
- OS: Windows 10/11 x64, or a current x86-64 Linux. macOS is not a Kestrel target.
- A controller is recommended. A keyboard also works, see [Keyboard](#keyboard).

## Features

On `main` today:

- **Render scale** (`--render-scale`, `--internal-resolution`): draws the game at a lower or
  higher internal resolution. 0.5 is half width and height, a quarter of the pixels. This is
  the biggest GPU saver. Astro's Playroom renders at 4K natively, so 0.5 is 1080p internal.
- **Lighting remap** (`--compute-rescale off|auto|verify`, default `auto`): makes the game's
  per-pixel lighting and compute passes run at the reduced resolution too, when the shader
  recompiler can prove it is safe. Render scale then saves GPU time on lighting as well as
  drawing. `verify` only cross-checks.
- **Wave-reduction recovery** (the speckle fix): the shader recompiler recognises the game's
  cross-lane reduction patterns, so effects that use them render without speckles.
- **Ray-tracing mode** (`--rt-mode full|reduced|off`): `off` is fastest and is what our
  numbers use.
- **Frame cap** (`--frame-cap <fps>`): caps presented frames per second, paced on the vblank
  grid. It does not change game speed.
- **Display options**: `--fullscreen`, `--screen-width`, `--screen-height`,
  `--present-mode Fifo|Mailbox|Immediate`, `--vertex-fetch cpu|gpu`.
- **Red-zone protection** (`--redzone`): needed by Demon's Souls on Windows.
- **fps log** (`--fps-log <seconds>`): writes `fps: N frame: N t=Ns` lines to the guest log.
  The window title also shows `frame: N, fps: N` while running.
- **Performance overlay and clips** (F10 overlay, F9 clip recording, `--perf-overlay`,
  `--record`, `--frame-time-log <path>`): shows fps, frame time, 1% low, a frame-time graph, CPU,
  GPU and memory use. F9 records an H.264 clip with a per-frame timestamp CSV and needs
  [ffmpeg](https://ffmpeg.org/). F9 and F10 cannot be used in `--keymap`. Also `--clip-overlay`,
  `--ffmpeg`, `--clips-dir`.
- **Launcher scripts** (`kestrel.bat`, `kestrel.sh`): presets for each game and GPU class, a
  dry-run mode, and a report mode that records a run for bug reports.
- **Game-speed patches** (`--patch "<name>"`, `--patches <dir>`): a patch loader that reads
  GoldHEN/shadPS4 XML patch files. Patches are opt-in by name and off by default. For Astro's
  Playroom there are two: "correct speed at 30 fps" (needs `--frame-cap 30`) and "correct speed
  at any frame rate". With the any-frame-rate patch, use `--frame-cap 30` on a 60 Hz display (or
  refresh/3 on 120/144 Hz, or VRR) to avoid judder. See [PATCHES.md](PATCHES.md).
- **Shader and pipeline caches**: kept in `_PipelineCache` so later runs stutter less.

### Coming soon

These are built and tested on branches but are not on `main` yet. The launcher skips them with
a warning if your build lacks them.

- **FSR 1 upscaler** (`--upscaler fsr1`, `--fsr-sharpness`): optional AMD FSR 1 upscale from
  render resolution to the window. It improves the final image, not performance.
- **Performance ports in A/B testing**: SRT JIT, hardware buffer bounds (from
  BryanKAdams/KytyPS5), queue-drain (from upstream, by nmzik) and relocatable draw records.
  See the [roadmap](docs/ROADMAP.md).

## Quick start

1. Clone with submodules and build `kyty_emulator` (Windows: clang-cl; Linux: clang).
2. Create `kestrel.local.ini` with a preset and your own game dump folder:

   ```ini
   [launcher]
   preset = astro-low

   [game]
   path = D:\Games\PS5\PPSA01325
   ```

3. Run `kestrel.bat` (Windows) or `./kestrel.sh` (Linux). Add `-DryRun` / `--dry-run` to see the
   command without starting the game.

The full steps are in [docs/QUICKSTART.md](docs/QUICKSTART.md).

More: [docs/STATUS.md](docs/STATUS.md) (progress), [docs/ROADMAP.md](docs/ROADMAP.md) (where
to help) and [docs/BRANCHES.md](docs/BRANCHES.md) (branch policy and contributing).

## Keyboard

A keyboard works with no setup. You can remap with `--keymap Control=Input` (see
[docs/QUICKSTART.md](docs/QUICKSTART.md)).

| PS5 control | Key |
| --- | --- |
| Left stick | W A S D |
| Right stick (up, left, down, right) | T F G H |
| D-pad | Arrow keys |
| Cross, Circle, Square, Triangle | J, L, K, I |
| L1, R1 | Q, E |
| L2, R2 | Z, C |
| L3, R3 | Left Shift, Left Ctrl |
| Options | Enter |
| Touchpad left half, right half | Backspace, Tab |

Reserved keys (they cannot be remapped): F1 RenderDoc frame capture, F7 mouse-to-joystick, F9
clip recording, F10 performance overlay, F11 fullscreen toggle, Esc.

## Contributing / where to help

The [roadmap](docs/ROADMAP.md) lists what is slow, what is broken, and where to start in the
source. Good first tasks:

- **Cooling Springs GPU census**: the scene takes about 40 ms of GPU time at 1080p. Find the
  expensive passes.
- **Queue-drain A/B**: measure the upstream queue-drain fix (by nmzik) against `main`.
- **Hardware buffer bounds A/B**: measure `VK_EXT_robustness2` bounds (from BryanKAdams) against
  `main`.
- **PGO builds** of `main` for Windows and Linux.

Rule: every performance PR includes before and after scorecard numbers (CPU Plaza, pre-plaza and
Cooling Springs, still and moving). The roadmap explains how to measure. Branch policy is in
[docs/BRANCHES.md](docs/BRANCHES.md).

## Reporting results

Run `kestrel.bat -Report -Seconds 300` (Windows) or `./kestrel.sh --report 300` (Linux) and paste
the printed block into a GitHub issue on [Yugendren/Kestrel](https://github.com/Yugendren/Kestrel).
See [docs/QUICKSTART.md](docs/QUICKSTART.md#reporting-results).

## Developer information

The PS5 graphics architecture is based on AMD RDNA 2. Use AMD's
[RDNA 2 Instruction Set Architecture Reference Guide (document 70648)](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
as the primary instruction-encoding reference when working on shader decoding and recompilation.

Important areas of the codebase:

- [`src/graphics/shader/recompiler`](src/graphics/shader/recompiler): instruction decoding,
  intermediate representation, control flow, resource tracking, and SPIR-V emission
- [`src/graphics/guest_gpu`](src/graphics/guest_gpu): PS5 (Prospero) GPU formats and command
  processing
- [`src/graphics/host_gpu`](src/graphics/host_gpu): Vulkan host backend and resource management
- [`tests`](tests): focused memory, shader, and resource-tracking regression tests

The renderer targets Vulkan 1.3. Keep shader changes aligned with both the RDNA 2 ISA semantics
and the Vulkan/SPIR-V validation rules. To contribute, see [docs/BRANCHES.md](docs/BRANCHES.md).

## Credits

- [KytyPS5](https://github.com/KytyPS5/KytyPS5) and its contributors (nmzik and many others):
  the emulator Kestrel is built on.
- [Kyty](https://github.com/InoriRus/Kyty) by InoriRus: the original project.
- KytyPS5 PR #558 authors, whose work is on `main`: Çağatay Güçlü (defektu), TarkusR and EmK530.
  Their rendering fixes cover texture streaming and residency, depth layouts and shaders, aimed
  at Astro Bot.
- [shadPS4](https://github.com/shadps4-emu/shadPS4): reference, and the patch file format.
- Forks whose work we are porting, credited on their commits when merged: BryanKAdams/KytyPS5,
  chenxiao07 (KytyPS5 PR #939 notes), TheCruZ/KytyPS5-GTA, Jetsku/KytyPS5, and Myoko's
  Demon's Souls branch.
- AMD FidelityFX Super Resolution 1 (MIT), when the FSR 1 option lands.

## Licence

Kestrel is licensed under the GNU General Public License version 2 (`GPL-2.0-only`), see
[LICENSE](LICENSE). It is based on Kyty, which was released under the MIT License. Kyty's
copyright and licence notice are preserved in [LICENSES/Kyty-MIT.txt](LICENSES/Kyty-MIT.txt).
Third-party components keep their own licences.
