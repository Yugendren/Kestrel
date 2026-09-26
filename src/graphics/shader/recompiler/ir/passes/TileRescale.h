#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Resolution rescaling of per-pixel tile compute.
//
// With internal resolution scaling the renderer keeps a render target at 1/k of its guest extent
// (k = 2^s) and serves compute storage bindings from a native-size twin, so a full-screen compute
// pass still runs one invocation per guest pixel and resamples its inputs and output. For a
// program that provably computes each pixel independently, the recompiler can instead run it at
// the scaled resolution:
//
//  * Invocation remap (backend prologue, active when RescaleControl::RemapBit is set): a W x H
//    workgroup is split into k^2 whole-warp slots of (W*H)/k^2 invocations. Slot j runs guest
//    workgroup k^2*WorkGroupID.x + j, and invocation i of a slot sees
//    LocalInvocationID = k*(i % (W/k), i / (W/k)): one representative guest pixel per k x k block.
//    Slots past NumWorkgroups.x return at once (whole warps, and accepted programs have no
//    barrier).
//  * Texel rescale (ApplyTileRescale): every texel address (image load/store) into an image
//    whose RescaleControl mask bit is set is shifted right by s, so it lands in the scaled image.
//    Normalised sampling needs no change.
//
// AnalyzeTileRescale proves on the final, specialised SSA IR that this is sound, refusing
// otherwise with every failed proof (design and rationale:
// research/remap-detector-2026-09-26.md, sections 2 and 3):
//  S0 structure       2D power-of-two workgroup, guest wave32 == host subgroup 32, whole-warp
//                     slots, no tg_size_en, not the dispatcher fallback.
//  S1 side effects    no LDS/GDS, barrier, DPP/WQM/permute/swizzle, append/consume, buffer or
//                     global stores and atomics (scratch is private), image atomics, size
//                     queries, GlobalInvocationID/LocalInvocationIndex/LocalInvocationID.z;
//                     texel image ops only on single-level 2D images with 32-bit x/y.
//  S2 affine domain   each U32 value is ax*Lx + ay*Ly + sum(coef*uniform atom) + c with known
//                     alignment of the uniform part, or wave-uniform, or varying.
//  S3 store ownership every image store is at exactly (Lx + Ux, Ly + Uy), with the tile origin
//                     U provably a multiple of k (address read under the store's predicate).
//  S4 loads           classified; a load of a written image must be the own pixel of the store.
//  S5 parity          LocalInvocationID reaches integer consumers only through monotone ops
//                     (no checkerboards, low-bit masks, edge-lane equality except load guards).
//  S6 cross-lane      exec-mask algebra (C0), waterfalls (C1), lane-serialised wave code (C2);
//                     anything else refuses. C1/C2 are confirmed at runtime by the renderer.
//  SCCP               code the compile options made unreachable never causes a refusal.
//
// Assumption A1: an indexed register write (SelectFlags::indexed_register_write) stays inside
// its array -- out of bounds is undefined in the source language -- so its rungs never carry a
// coordinate register away.

namespace Libs::Graphics::ShaderRecompiler::IR {

// How the program is dispatched: the parts of the compute input the proofs depend on.
struct TileRescaleShape {
	std::array<uint32_t, 3> threads {};
	uint32_t                wave_size          = 0;
	uint32_t                host_subgroup_size = 0;
	bool                    tg_size_en         = false;
	// log2 of the downscale factor k to prove the program for (render scale 2^-s).
	uint32_t                scale_log2         = 1;
};

struct TileRescalePlan {
	bool                     accepted   = false;
	// Every failed proof, in program order, each with the offending instruction when there is
	// one. Empty when accepted.
	std::vector<std::string> reasons;
	CrossLaneClass           cross_lane = CrossLaneClass::ExecMaskOnly;
	uint32_t                 scale_log2 = 0;
	// Largest s the store tile origins allow (min alignment over all stores; 32 = no store).
	uint32_t                 store_align = 32;
	// Float conversions (ConvertF32U32/ConvertF32S32) whose operand is exactly the invocation's
	// own pixel coordinate on one axis, Lx + Ux or Ly + Uy. ApplyTileRescale offsets these by
	// +0.5*(k-1) when RescaleControl::CentreBit is set. Null-free; empty unless accepted.
	std::vector<const Inst*> own_coordinate_conversions;
	// One-line human-readable account: load classes, parity and cross-lane notes.
	std::string              summary;
};

// Pure analysis: never modifies the program.
[[nodiscard]] TileRescalePlan AnalyzeTileRescale(const Program& program,
                                                 const TileRescaleShape& shape);

// Applies an accepted plan: rewrites every texel address to follow the runtime per-image mask,
// adds the optional centre correction, and records Program::tile_rescale (which makes
// AllocateBindings reserve the control word and the backend emit the prologue). Must run before
// CollectShaderInfo/AllocateBindings.
void ApplyTileRescale(Program& program, const TileRescalePlan& plan);

} // namespace Libs::Graphics::ShaderRecompiler::IR
