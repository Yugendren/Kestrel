# Game patches

KytyPS5 can apply per-game binary patches to a module (normally `eboot.bin`) while it is loaded,
before any guest code runs. The loader (`src/loader/patches`) is generic; only the patch files are
title-specific. It reads two formats: GoldHEN/shadPS4 XML patch files (below) and etaHEN/GoldHEN
cheat JSON files (`--game-patch`, written by the launcher's Cheats dialog).

## Using patches

- Patch files are `*.xml` files in a patch directory: by default `patches/` next to the emulator
  executable (the build copies the shipped `patches/` folder there), or `--patches <dir>`.
- Every patch is **off by default** and stays off unless it is explicitly enabled, in one of three
  ways (to switch a patch off again, drop the option / list entry):
  - `--patch "<name>"` on the command line (can be repeated; names match case-insensitively),
  - `<patch dir>/patches.json`: `{"enabled": ["<name>", "..."]}`,
  - `isEnabled="true"` on the patch's `Metadata` element (shadPS4 convention).
- At startup the emulator prints one `[patch]` line per decision, on stdout and in the log file:

  ```
  [patch] applied "Astro's Playroom - correct speed at 30 fps" to eboot.bin (4 lines, 115 bytes)
  [patch] available (disabled): Astro's Playroom - correct speed at 30 fps
  [patch] patch "..." requires --frame-cap 30 (current: uncapped); not applied
  [patch] skipped "..." for eboot.bin: original bytes differ at +0x1b4b8c0 (expected c5fb..., found 90...)
  [patch] warning: no patch named "..." for PPSA01325 version 01.905.000
  ```

Example (Astro's Playroom at correct game speed with 30 presented fps; both options are needed,
without `--patch` the game runs unmodified, without `--frame-cap 30` the patch is refused):

```
kyty_emulator --game <Astro dir> --frame-cap 30 --patch "Astro's Playroom - correct speed at 30 fps"
```

## File format

The schema is the GoldHEN/etaHEN PS5 (`xml_prospero`) and shadPS4 patch format, so their files
load unchanged, plus three optional attributes (`Original`, `RequiresFrameCap`, `ElfXXH3`) that
other tools ignore.

```xml
<?xml version="1.0" encoding="utf-8"?>
<Patch>
  <TitleID>
    <ID>PPSA01325</ID>          <!-- one or more regional title ids -->
  </TitleID>
  <Metadata Title="Astro's Playroom"
            Name="Astro's Playroom - correct speed at 30 fps"   (the name used to enable it)
            Note="..." Author="..." PatchVer="1.0"
            AppVer="01.905.000"     (exact match against param.json contentVersion)
            AppElf="eboot.bin"      (module file name, case-insensitive)
            ImageBase="0x0"         (Address - ImageBase = ELF p_vaddr; default 0)
            RequiresFrameCap="30"   (optional: only applied when --frame-cap equals this)
            ElfXXH3="..."           (optional: XXH3-64 of the module file, 16 hex digits)
            isEnabled="false">      (optional, shadPS4)
    <PatchList>
      <Line Type="bytes"   Address="0x01b4b8c0" Value="c5fb59..." Original="c5fb10..."/>
      <Line Type="float64" Address="0x02721278" Value="30.0"      Original="0000000000004e40"/>
    </PatchList>
  </Metadata>
  <!-- more Metadata blocks: other versions, modules or patches -->
</Patch>
```

Line types (all multi-byte values are written little-endian):

| Type | Value |
|---|---|
| `bytes` | hex byte string (spaces allowed) |
| `byte`, `bytes16`, `bytes32`, `bytes64` | unsigned integer; `0x`, `#` or `$` prefix = hex, else decimal; must fit the width |
| `float32`, `float64` | decimal floating-point number |
| `utf8` | text, written with a terminating NUL |
| `utf16` | text, written as UTF-16LE with a terminating 2-byte NUL |

`mask` / `mask_jump32` (pattern-scanned code caves) are not supported; a patch that uses them is
skipped with a message.

## Cheat JSON files (`--game-patch <file>`)

`--game-patch <file>` (may be repeated) loads an etaHEN/GoldHEN cheat file:

```json
{"id": "PPSA01325", "version": "01.905.000", "process": "eboot.bin",
 "mods": [{"name": "...", "enabled": true,
           "memory": [{"offset": "1f4b8c0", "off": "c5fb100d", "on": "90909090"}]}]}
```

Each mod is one patch (enabled unless `"enabled": false`); each memory write is a line with
`Original` = `off` and `Value` = `on`. Cheat offsets are absolute addresses in an unknown image
base, so the base is found by searching the module for the first non-zero `off` bytes and
validating the other writes at each candidate. Writes outside the module whose `off` is all zero
are code caves: the pages are mapped zero-filled at those addresses and freed with the module.

## Safety rules

- **Verify before write.** Each line's `Original` bytes must equal the module's bytes at that
  address. A line without `Original` is only accepted when the entry is pinned to the exact file
  with `ElfXXH3`. A patch is atomic: every line is checked (inside one segment's file data, no
  overlapping lines, original bytes match) before any byte is written; on any failure nothing is
  written and the reason is logged.
- **Segment tail code caves.** Besides a segment's file data, a `bytes` line may lie entirely in
  the zero-filled tail of an executable `PT_LOAD` segment: from `p_vaddr + p_filesz` up to
  `AlignUp(p_vaddr + p_memsz, 0x4000)` (the guest page the loader maps with), cut off at the start
  of the next segment and at the end of the range the loader protects for this segment. Such a line
  must have an `Original` that is all zero (a pin with `ElfXXH3` does not replace it), and the
  zeros are verified against the live memory like any other `Original`. The tail is part of the
  module's single mapping and receives the segment's executable protection, the red-zone patcher
  only scans `p_filesz` and relocation does not target it, so code placed there is ordinary
  executable guest code; a line that starts in file data and runs into the tail is rejected. The
  atomic all-or-nothing rule applies unchanged.
- **Exact version.** `TitleID` + `AppVer` + `AppElf` (+ `ElfXXH3`) must all match; a patch for
  another version is never applied.
- **Load order.** Patches are applied to the module's pristine file bytes right after its
  segments are copied, before the loader's own code rewriting (TLS/fs-access rewrites, Windows
  red-zone patching), before memory protections are set, and before relocation and module start.
  If relocation later changes a patched byte, a warning names the patch (a patch error).
- **No patch, no change.** When no patch is enabled for a module nothing is written.
- **`RequiresFrameCap`.** Patches that change a game's per-frame timestep are only correct at one
  presentation rate. The emulator refuses them unless `--frame-cap` is exactly that value.

## Writing a patch

1. Find the ELF virtual addresses (module-relative) of the bytes to change; KytyPS5 maps the
   module so that address = base + p_vaddr.
2. Copy the current bytes at those addresses into `Original`, and the new bytes/values into
   `Value`. Keep each patch self-contained (all the lines needed for one behaviour change).
3. Give the patch a descriptive `Name`; that is what users pass to `--patch`.
4. Run with `--patch "<name>"` and check the `[patch] applied` line.

## Shipped patches

| File | Game | Version | Patch | Requires |
|---|---|---|---|---|
| `patches/PPSA01325.xml` | Astro's Playroom | 01.905.000 | `Astro's Playroom - correct speed at 30 fps`: the engine steps a fixed 1/60 s per presented frame (`SetFrameRate(double fps)` at 0x1b4b8c0). The patch makes the setter use fps x 0.5 and sets the static initial values to 30.0 / 1/30 / 33333 us, so each frame advances 1/30 s. | `--frame-cap 30` |
| `patches/PPSA01325.xml` | Astro's Playroom | 01.905.000 | `Astro's Playroom - correct speed at any frame rate` (off unless enabled): a code cave in the text segment's zero tail (+0x1f36b00), called instead of the per-frame function 0x1842eb0 at 0x1842e83, measures the real time since the previous frame, clamps it to 10..120 fps, and stores it as the engine's timestep (`period_us` and float `dt`). The 1/fps getter at 0x1b4b930 returns `dt`, which frees the fps double at 0x2721278 to hold the previous timestamp. | |

### Playing Astro's Playroom unlocked

Run with `--patch "Astro's Playroom - correct speed at any frame rate"` and no frame cap (or any
cap). Game logic follows real time, clamped to 10..120 fps, so game speed is right at any frame
rate. Jumps get lower below about 30 fps, because the game's own physics integrates at large steps
(measured jump apex -10% at 20 fps and -24% at 15 fps, relative to 30 fps). F10 / `--perf-overlay`
shows fps and frame time. Do not enable it together with the 30 fps patch: the loader refuses the
second one.
