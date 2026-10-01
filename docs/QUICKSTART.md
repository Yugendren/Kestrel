# Quick start

This guide takes you from a fresh clone to a running game. It assumes a desktop PC, for example
a Ryzen 7 9800X3D with an RTX 5090 on Windows. Linux is covered too.

Kestrel never distributes games. You need your own legally obtained dump.

## 1. Prerequisites

Hardware: a CPU with AVX2 (Intel Haswell or newer, AMD Zen 1 or newer), a Vulkan 1.3 GPU with a
current driver, 16 GB RAM or more, and a controller. See the
[requirements](../README.md#requirements).

Windows:

- Git
- Visual Studio 2022 or Build Tools 2022 with "Desktop development with C++" and "C++ Clang
  tools for Windows". This provides clang-cl, CMake and Ninja.
- glslang: download `glslangValidator.exe` from the
  [glslang releases](https://github.com/KhronosGroup/glslang/releases) and unpack it, for
  example to `C:\tools\glslang`.
- A current GPU driver.

You do not need Qt. Kestrel builds the emulator only, not the Qt launcher. MSVC `cl.exe` is not
supported; use clang-cl.

Linux (Ubuntu/Debian example):

```
sudo apt-get install --no-install-recommends clang lld ninja-build cmake git glslang-tools \
  pkg-config libgl1-mesa-dev libx11-dev libxcursor-dev libxext-dev libxfixes-dev libxi-dev \
  libxrandr-dev libxss-dev libxtst-dev libxkbcommon-dev libasound2-dev libpulse-dev \
  libudev-dev libdbus-1-dev libwayland-dev wayland-protocols
```

`ccache` is optional. You need a Vulkan driver: NVIDIA proprietary or Mesa (RADV/ANV).

## 2. Build

The first configure downloads FFmpeg prebuilts and some sources, so you need internet. A full
build takes about 10-20 minutes on 6 cores.

Windows. Open the "x64 Native Tools Command Prompt for VS 2022" or a Developer PowerShell:

```
git clone --recurse-submodules https://github.com/Yugendren/Kestrel.git
cd Kestrel
cmake -S . -B _Build\windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DKYTY_BUILD_LAUNCHER=OFF -DKYTY_TRACY=OFF -DKYTY_GLSLANG_VALIDATOR=C:\tools\glslang\bin\glslangValidator.exe
cmake --build _Build\windows --target kyty_emulator --parallel
cmake --install _Build\windows --prefix _Build\windows\install
```

If you already cloned without submodules, run
`git submodule update --init --recursive` first.

The result is `_Build\windows\install\kyty_emulator.exe` with `libwinpthread-1.dll` next to it.
The DLL comes from `3rdparty\winpthread\bin\` and the build copies it. If Windows says
`libwinpthread-1.dll` is missing, copy it from `3rdparty\winpthread\bin\` next to
`kyty_emulator.exe`.

Linux:

```
git clone --recurse-submodules https://github.com/Yugendren/Kestrel.git && cd Kestrel
cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKYTY_BUILD_LAUNCHER=OFF -DKYTY_TRACY=OFF
cmake --build _Build/linux --target kyty_emulator --parallel
cmake --install _Build/linux --prefix _Build/linux/install
```

The result is `_Build/linux/install/kyty_emulator`.

## 3. Your game dump

Use your own decrypted PS5 game dump from a console you own. How to make a dump is out of
scope. Put the folder anywhere, for example `D:\Games\PS5\PPSA01325` or `~/games/PPSA01325`.

The folder must contain `eboot.bin` and `sce_sys/param.json`.

Demon's Souls needs version 01.005. Never upload or share game files, and do not attach them to
issues.

## 4. Configure

Settings are layered. The launcher loads the preset, then `kestrel.ini`, then
`kestrel.local.ini`. A key overrides only if its value is not empty. Put your own settings in
`kestrel.local.ini` in the repository root, next to `kestrel.ini`. Git ignores it, so
`git pull` never conflicts.

Presets:

| Preset | Game | GPU class |
| --- | --- | --- |
| `astro-low` | Astro's Playroom | 3060-class, render scale 0.5 (1080p internal) |
| `astro-high` | Astro's Playroom | 4080/5090-class, render scale 1.0 (native 4K internal), fullscreen |
| `astro-low-unlocked`, `astro-high-unlocked` | Astro's Playroom | Same as `astro-low` / `astro-high`, but with the correct-speed-at-any-frame-rate patch and no frame cap |
| `ds-low` | Demon's Souls | 3060-class, render scale 0.5 |
| `ds-high` | Demon's Souls | 4080/5090-class, render scale 1.0, fullscreen |
| `default` | any | plain emulator defaults |

Demon's Souls presets turn on red-zone protection. All presets turn ray tracing off.

The Astro's Playroom presets also enable the game-speed patch ("Astro's Playroom - correct
speed at 30 fps") and a 30 fps frame cap (`game_speed_patch = on`, `patch_frame_cap = 30`).
The launcher passes `--patch "<name>" --frame-cap 30` when your build supports `--patch`. To
turn it off, add this to `kestrel.local.ini`:

```ini
[speed]
game_speed_patch = off
```

See [PATCHES.md](../PATCHES.md) for the patch loader itself.

Minimal `kestrel.local.ini` for Astro's Playroom on an RTX 5090 (Windows):

```ini
[launcher]
preset = astro-high

[game]
path = D:\Games\PS5\PPSA01325
```

Minimal `kestrel.local.ini` for Astro's Playroom on an RTX 3060 (Linux):

```ini
[launcher]
preset = astro-low

[game]
path = /home/me/games/PPSA01325
```

Values are not quoted. Paths may contain spaces. Other settings, such as `render_scale`,
`frame_cap` or `extra_args`, are documented in comments in `kestrel.ini`.

## 5. Dry run

A dry run prints the preset, the config files used, any warnings and the full emulator command
as one line. It does not start the game. It works even before you build.

```
kestrel.bat -DryRun          (Windows)
./kestrel.sh --dry-run       (Linux)
```

## 6. Launch

```
kestrel.bat                  (Windows)
./kestrel.sh                 (Linux)
```

Options: `-Preset NAME`, `-Config FILE`, `-Game DIR`, `-Help` on Windows, and `--preset NAME`,
`--config FILE`, `--game DIR`, `--help` on Linux. The launcher prints the command and the
`saves:` folder before it starts the emulator.

A controller is required. Kestrel reads DualSense, DualShock 4 and Xbox pads through SDL. There
is no keyboard mapping. You can remap with `--keymap Control=Input` in `extra_args`.

## 7. Reading what you see

The window title shows `frame: N, fps: N` while the game runs.

Coming soon: a performance overlay, toggled with F10. It shows:

- fps: frames per second now
- frame time: how long the last frame took
- 1% low: the average fps of the slowest 1% of frames, a measure of stutter
- a graph of frame times
- CPU and GPU load
- VRAM use

To start with the overlay shown, set `overlay = on` under `[debug]`. If your build lacks the
overlay, the launcher says so and the window title shows fps instead.

Astro's Playroom runs in slow motion below 60 fps, because the game's speed is tied to its
frame rate. The `astro-low` and `astro-high` presets enable the game-speed patch with a 30 fps
cap, which gives correct speed at 30 fps (see section 4 to turn it off). Correct speed at any
frame rate is in testing. See the [status page](STATUS.md).

## 8. First-run shader stutter

The first time you see a scene, the game stutters while shaders compile. Later runs are
smoother, because the caches are kept in `_PipelineCache` in the emulator's folder.

## 9. Saves

The game writes saves itself, so saving in game works as on console. They live in `_SaveData`
in the emulator's folder, for example `_Build\windows\install\_SaveData` or
`_Build/linux/install/_SaveData`. The launcher prints this path as `saves:`.

To back up, copy the `_SaveData` folder somewhere safe. To restore, copy it back into the
emulator's folder while the game is closed. If you delete `_Build`, copy your saves out first.

## 10. Reporting results

One command per OS. It runs the game for 300 seconds, closes it, and prints a report:

```
kestrel.bat -Report -Seconds 300     (Windows)
./kestrel.sh --report 300            (Linux)
```

While it runs, play normally. The launcher prints progress every 30 seconds and closes the game
when the time is up. Ctrl+C also stops the game.

The report block lists the commit, OS, CPU, GPU, RAM, preset, settings and fps statistics. The
statistics are average fps, 1% low, 10% low and the share of frames slower than 33.3 ms when the
frame-time log is available. Otherwise you get per-second fps samples. Only the game folder name
is shown, not the full path.

Paste everything between these two lines into a GitHub issue on
[Yugendren/Kestrel](https://github.com/Yugendren/Kestrel/issues):

```
----- Kestrel report (paste everything below) -----
----- end of report -----
```

The launcher also saves its files in `kestrel-logs/report-<time>/`. Attach them only if asked.
Do not attach game files.

## 11. Troubleshooting

- **"kyty_emulator was built for x86-64-v3 (AVX2/BMI2/FMA); this CPU lacks: ..." at start.**
  Your CPU lacks AVX2 (x86-64-v3). The default build refuses to start on it. Building with `-DKYTY_MARCH=` at a lower level is possible but
  untested.
- **`libwinpthread-1.dll` is missing.** Copy it from `3rdparty\winpthread\bin\` next to
  `kyty_emulator.exe`.
- **Vulkan errors.** Update your GPU driver. You need Vulkan 1.3. On Linux use
  the NVIDIA proprietary driver or Mesa (RADV/ANV).
- **Demon's Souls.** It needs game version 01.005. It is not playable on `main` yet: the fixes
  that reach gameplay are still on a development branch (see the [status page](STATUS.md)).
- **Out of VRAM.** Lower `render_scale` in `kestrel.local.ini`, for example from 1.0 to 0.5. 8 GB or
  more is recommended, and a 4K internal render needs more.
- **"warning: ... (coming soon, not in this build); skipped".** Harmless. A setting needs a
  feature that is not on `main` yet, so the launcher skipped it.
- **"build first, see docs/QUICKSTART.md".** The launcher found no emulator. Build it (step 2) or
  set `emulator =` under `[launcher]`.
- **Unknown preset.** The launcher lists the presets that exist in `presets/`.

## 12. Updating

```
git pull
git submodule update --init --recursive
```

Then run the build commands from step 2 again.
