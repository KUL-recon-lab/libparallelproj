#pragma once
// =====================================================================================
// Sort key of a TOF listmode event (issue #54), CPU and CUDA
// =====================================================================================
//
// Why sort listmode events?
// -------------------------
// The TOF listmode projectors (joseph3d_tof_lm_fwd / _back) use one GPU thread per event.
// 32 neighbouring threads (a "warp") run in lock step, so 32 consecutive events are
// projected at the same time. Every event reads (forward) or writes (back) the voxels
// along a short piece of its LOR: the image planes of its TOF window around the
// "TOF center" (the center of the event's TOF bin on the LOR).
//
// Listmode data are usually in time order, i.e. the events of a warp lie anywhere in the
// image. Then almost every voxel access of a warp needs its own memory transaction (the
// image does not fit into the GPU caches), which makes the projectors memory bound:
// per image plane, they were 4x (forward) to 12x (back) slower than the TOF sinogram
// projectors. If consecutive events have nearby TOF centers, the threads of a warp read
// and write the same image region at the same time; the voxels then come from the caches.
// Measured on a desktop GPU (10 M events): ~9x faster forward and back projections.
//
// The sort key
// ------------
// To bring events with nearby TOF centers together, every event gets a 64-bit number (the
// sort key); sorting the events by it (once, e.g. once per subset, and reusing the sorted
// events in all iterations) gives a good order:
//
//   bits 40-41  principal axis of the LOR (0, 1, 2; 3 for degenerate events, sorted last).
//               Events of a warp then step through the image along the same axis.
//   bits  0-29  3D Morton code ("Z-order") of the grid cell that contains the TOF center,
//               on a grid of TOF_LM_SORT_KEY_CELL_VOXELS^3 voxels (10 bits per axis).
//
// Morton code: write the cell coordinates (qx, qy, qz) in binary and interleave their
// bits, ... qz1 qy1 qx1 qz0 qy0 qx0. Example in 2D: (x, y) = (5, 3) = (101b, 011b) ->
// y2 x2 y1 x1 y0 x0 = 01 10 11b = 27. Sorting cells by their Morton code orders them along
// a space-filling "Z" curve: cells with close codes are (mostly) close in space, in all
// three directions at once (sorting by one coordinate only would keep events with
// different other coordinates together). Benchmarks: grid cells of 1 to 4 voxels perform
// equally; additional direction bins helped the forward projection by ~5 % but slowed the
// back projection down, so they are not used.
//
// The TOF center is computed exactly as in the TOF listmode "point" projectors
// (tof_bin_center_voxel_coords in utils.h): at the distance
// (tof_bin - (num_tof_bins - 1) / 2) * tof_bin_width + tof_center_offset from the LOR
// center, towards lor_end. Cells outside 0 .. 1023 (TOF centers outside the image or
// images with more than 2048 voxels along an axis) are clamped; this only affects the
// order (never the projections).
#include <cstdint>
#include "cuda_compat.h"
#include "utils.h"

/// size of the grid cells of the sort key in voxels (along every axis)
#define TOF_LM_SORT_KEY_CELL_VOXELS 2

/// Insert two zero bits between the lowest 10 bits of x (bit k moves to bit 3 k): the
/// building block of the 3D Morton code. Classic "magic number" bit trick: each step
/// moves groups of bits apart (by 16, 8, 4, 2 positions) and masks out the rest.
WORKER_QUALIFIER inline std::int64_t morton_spread3(std::int64_t x)
{
  x &= 0x3FF;                         // 10 bits: ---- ---- ---- ---- ---- --98 7654 3210
  x = (x | (x << 16)) & 0x030000FF;   // ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x | (x << 8)) & 0x0300F00F;    // ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x | (x << 4)) & 0x030C30C3;    // ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x | (x << 2)) & 0x09249249;    // ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}

/// Cell coordinate (0 .. 1023) along one axis from the continuous voxel coordinate f
/// (voxel centers at integers, voxel k covers [k - 0.5, k + 0.5)). The comparisons are
/// done in float before the conversion to int (no overflow for huge / non-finite f).
WORKER_QUALIFIER inline std::int64_t tof_lm_sort_key_cell(float f)
{
  const float q = floorf((f + 0.5f) / static_cast<float>(TOF_LM_SORT_KEY_CELL_VOXELS));
  if (!(q > 0.0f)) // also NaN
    return 0;
  if (q > 1023.0f)
    return 1023;
  return static_cast<std::int64_t>(q);
}

/// Sort key of event i (see above), written to sort_key[i].
WORKER_QUALIFIER inline void tof_lm_sort_key_worker(std::size_t i,
                                                    const float *lor_start,
                                                    const float *lor_end,
                                                    const float *image_origin,
                                                    const float *voxel_size,
                                                    std::int64_t *sort_key,
                                                    float tof_bin_width,
                                                    const float *tof_center_offset,
                                                    const short *tof_bin_index,
                                                    short num_tof_bins,
                                                    unsigned char is_lor_dependent_tof_center_offset)
{
  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;

  // TOF center in continuous voxel coordinates (as in the TOF listmode point projectors)
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  float f[3];
  if (!tof_bin_center_voxel_coords(p0, p1, image_origin, voxel_size, tof_bin_width, local_tof_center_offset,
                                   tof_bin_index[i], num_tof_bins, f))
  {
    // degenerate event (lor_start == lor_end or non-finite input): "axis 3", i.e. after
    // all other events (such events are skipped by the projectors anyway)
    sort_key[i] = std::int64_t(3) << 40;
    return;
  }

  // principal axis: the same rule as ray_cube_intersection_joseph (utils.h), which the
  // projectors use (largest |component| of lor_end - lor_start; ties go to the larger axis)
  const float d0 = p1[0] - p0[0], d1 = p1[1] - p0[1], d2 = p1[2] - p0[2];
  const float s0 = d0 * d0, s1 = d1 * d1, s2 = d2 * d2;
  std::int64_t axis = 0;
  if (s1 >= s0 && s1 >= s2)
    axis = 1;
  else if (s2 >= s0 && s2 >= s1)
    axis = 2;

  // Morton code of the grid cell of the TOF center: bits of axis 0 at positions 0, 3, 6, ...,
  // axis 1 at 1, 4, 7, ..., axis 2 at 2, 5, 8, ...
  const std::int64_t morton = morton_spread3(tof_lm_sort_key_cell(f[0])) |
                              (morton_spread3(tof_lm_sort_key_cell(f[1])) << 1) |
                              (morton_spread3(tof_lm_sort_key_cell(f[2])) << 2);
  sort_key[i] = (axis << 40) | morton;
}
