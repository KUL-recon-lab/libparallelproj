#pragma once
// =====================================================================================
// TOF sinogram forward projection of one LOR (Joseph's method), CPU and CUDA
// =====================================================================================
//
// Prerequisites: joseph3d_lor_geometry.h (planes, interpolation, TOF position of a plane)
// and tof_kernel.h (TOF bin coordinates, TOF window, weights).
//
// 1. What is computed
// -------------------
// For every image plane ip intersected by the LOR:
//   val  = cf * (bilinear interpolation of the image in plane ip)  (non-TOF contribution)
//   it_f = TOF bin coordinate of the intersection point
//   for every TOF bin t in the TOF window of it_f:  projection[t] += w_t(it_f) * val
// i.e. the non-TOF contribution of each plane is distributed over the TOF bins near the
// plane with the TOF weights w_t (which sum to 1).
//
// 2. The straightforward implementation, and why it is slow on GPUs
// ------------------------------------------------------------------
// Adding w_t * val directly to projection[t] in memory for every plane means that every
// TOF bin is read, modified and written back many times: with TOF bins of ~10 planes and
// a window of ~7 bins, ~70 times per bin. Each GPU thread handles one LOR, and the TOF
// bins of neighbouring LORs (threads) are num_tof_bins floats apart in memory, so these
// accesses cannot be combined into few memory transactions. This was the main cost of the
// previous implementation (up to 20x slower than this one).
//
// 3. The sliding window ("accumulator")
// -------------------------------------
// Key observation: if the planes are visited in the order in which the TOF position it_f
// increases, the TOF window (bins it_min .. it_min + n - 1) only moves forward. A bin that
// the window has left behind (t < it_min) will not get any further contribution - it is
// final. So we
//   - keep the sums of the bins of the current window in a small array acc[0 .. W-1],
//     where acc[k] holds bin base + k,
//   - add the contributions of a plane to acc (fast, in registers),
//   - when the window moves on (it_min increases), write the finished bins acc[0], ...
//     to memory and shift the array down, and
//   - at the end, write the remaining bins.
// Every TOF bin of the output is written exactly once (also the bins that never get a
// contribution: they are written as zeros).
//
// Example with W = 8, V = 1 (window of n = 5 bins moving forward):
//
//   plane 1: it_min = 3  -> base = 3,  acc = [b3 b4 b5 b6 b7 .  .  . ]  (add 5 weights)
//   plane 2: it_min = 3  -> no shift,  acc = [b3 b4 b5 b6 b7 .  .  . ]
//   plane 3: it_min = 4  -> bin 3 is final: write b3, shift down by one, base = 4
//                                      acc = [b4 b5 b6 b7 b8 .  .  . ]
//   ...
//   end:     write all W bins of acc (bins base .. base + W - 1), then zeros for the bins
//            before the first base and after the last bin of acc.
//
// 4. Why templates and "#pragma unroll" everywhere
// -------------------------------------------------
// On the GPU, acc only lives in (fast) registers if every access acc[k] uses an index k
// that the compiler knows at compile time. Therefore
//   - the size W is a template parameter (compile-time constant), and the projector is
//     compiled for several sizes (tof_acc_size in tof_kernel.h); each LOR uses the
//     smallest size that fits its TOF window,
//   - all loops over the W slots are fully unrolled (#pragma unroll) and loop over all W
//     slots, with an `if` that skips the slots outside the window (instead of looping
//     from k_lo to k_hi, whose bounds are only known at run time).
// A version with run-time indices (a ring buffer) was 1.5x slower than the old code,
// because the array was placed in slow "local" memory.
//
// 5. Groups of 4 bins (template parameter V)
// ------------------------------------------
// With V = 4, finished bins are written in groups of 4 neighbouring bins with one 16-byte
// store (float4) instead of 4 single stores, and the array is shifted by 4. A float4 store
// must start at an address that is a multiple of 16 bytes. The first bin of the row
// generally is not aligned, so `base` is always kept at a bin number whose address is a
// multiple of 16 bytes ("phase" below). Because base can then lie up to 3 bins before
// it_min, the accumulator needs V - 1 = 3 extra slots. tof_store_width (tof_kernel.h)
// decides when this pays off. On the CPU, V = 4 uses 4 normal stores (same results).
//
// Template parameters:
//   W: accumulator size (TOF bins), at least the largest window size plus V - 1 (see
//      tof_acc_size_class). The largest window size is the number of bins overlapping the
//      kernel support, but never more than num_tof_bins (windows only contain existing
//      bins, see tof_window). Only if this is more than 64 (wide kernel AND more than 64
//      TOF bins), windows are truncated to W - V + 1 = 64 bins around the support center.
//   V: store width: 1 (one bin at a time) or 4 (address-aligned groups of 4 bins).
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"

// Write the V bins t .. t+V-1 of a projection row (bins outside [0, num_tof_bins) are
// skipped, they do not exist). Specialized for V = 1 and V = 4.
template <int V>
struct TOFSinoStore;

template <>
struct TOFSinoStore<1>
{
  WORKER_QUALIFIER static inline void store(float *row, int t, const float (&v)[1], int num_tof_bins)
  {
    if (t >= 0 && t < num_tof_bins)
      row[t] = v[0];
  }
};

template <>
struct TOFSinoStore<4>
{
  // if all 4 bins are inside the row, row + t is 16-byte aligned (see `phase` in the worker)
  WORKER_QUALIFIER static inline void store(float *row, int t, const float (&v)[4], int num_tof_bins)
  {
    if (t >= 0 && t + 3 < num_tof_bins)
    {
#ifdef __CUDA_ARCH__
      // GPU: one 16-byte store
      *reinterpret_cast<float4 *>(row + t) = make_float4(v[0], v[1], v[2], v[3]);
#else
      row[t] = v[0];
      row[t + 1] = v[1];
      row[t + 2] = v[2];
      row[t + 3] = v[3];
#endif
    }
    else
    {
      // group at the start or end of the row: only the existing bins
      for (int j = 0; j < 4; ++j)
        if (t + j >= 0 && t + j < num_tof_bins)
          row[t + j] = v[j];
    }
  }
};

/// TOF sinogram forward projection of LOR i: writes all num_tof_bins values of
/// projection_values[i * num_tof_bins ...]. See the explanation at the top of this file.
template <int W, int V, typename KP>
WORKER_QUALIFIER inline void joseph3d_tof_sino_fwd_worker(std::size_t i,
                                                          const float *lor_start,
                                                          const float *lor_end,
                                                          const float *image,
                                                          const float *image_origin,
                                                          const float *voxel_size,
                                                          float *projection_values,
                                                          const int *image_dim,
                                                          float tof_bin_width,
                                                          const float *tof_center_offset,
                                                          short num_tof_bins,
                                                          unsigned char is_lor_dependent_tof_center_offset,
                                                          const KP &kp)
{
  float *row = projection_values + i * num_tof_bins; // the TOF bins of this LOR
  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;

  // ---------------------------------------------------------------------------------
  // Geometry of the LOR (planes, interpolation); LORs that miss the image get zeros
  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
  {
    for (short it = 0; it < num_tof_bins; ++it)
      row[it] = 0.0f;
    return;
  }
  const int istart = lor.istart;
  const int iend = lor.iend;

  // ---------------------------------------------------------------------------------
  // TOF parameters: truncated kernel of this LOR and the TOF bin coordinate of plane ip,
  // it_f = ip * at + bt
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  // TOF kernel of this LOR (Gaussian or table, see tof_kernel.h): built by the per-call
  // kernel parameters kp from the LOR's sigma or kernel scale
  const auto kernel = kp.kernel(i);
  if (!kernel.valid())
  {
    // degenerate TOF parameters (e.g. sigma <= 0): no TOF contribution
    for (short it = 0; it < num_tof_bins; ++it)
      row[it] = 0.0f;
    return;
  }
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // Visit the planes in the order of increasing it_f: in increasing plane order if
  // at >= 0, in decreasing order otherwise. (The sum of the contributions does not depend
  // on the order; only the window logic below needs it.)
  const bool forward = (at >= 0.0f);
  const int num_planes = iend - istart + 1;
  // largest window that fits into the accumulator (with groups of 4, base can be up to 3
  // bins before it_min, see above)
  const int max_window = W - (V - 1);

  // Alignment for groups of V bins: the bins t with t = phase (mod V) start at an address
  // that is a multiple of 4 * V bytes. (row / sizeof(float)) % V is the position of bin 0
  // within its group of V floats in memory. For V = 1 every bin is "aligned" (phase 0).
  const int phase = (V > 1) ? static_cast<int>((V - (reinterpret_cast<std::size_t>(row) / sizeof(float)) % V) % V) : 0;

  // the accumulator: acc[k] holds the sum of TOF bin base + k
  float acc[W];
#pragma unroll
  for (int k = 0; k < W; ++k)
    acc[k] = 0.0f;

  int base = 0;          // TOF bin of acc[0]
  int first = 0;         // first TOF bin covered by the accumulator (bins below are zero)
  bool started = false;  // true once a plane with a non-empty TOF window was found

  // ---------------------------------------------------------------------------------
  // Loop over the planes, in the order of increasing TOF position
  for (int n = 0; n < num_planes; ++n)
  {
    const int ip = forward ? (istart + n) : (iend - n);
    const float it_f = tof_plane_position(ip, at, bt);

    // TOF window of this plane: bins it_min .. it_min + n_tof_weights - 1, only bins that
    // exist (0 .. num_tof_bins - 1), see tof_window
    int it_min, n_tof_weights;
    bool truncated;
    tof_window(it_f, kernel, max_window, num_tof_bins, it_min, n_tof_weights, truncated);

    // A plane whose window is empty is so far outside the TOF range that it contributes
    // to none of the TOF bins: skip it. (Such planes are at the beginning or at the end of
    // the traversal, since the window only moves forward.)
    if (n_tof_weights <= 0)
      continue;

    // --- move the accumulator such that base <= it_min < base + V
    if (!started)
    {
      // first plane: start the accumulator at it_min, rounded down to an aligned bin
      // ((x % V + V) % V is the mathematical modulo, also for negative x)
      base = it_min - (((it_min - phase) % V) + V) % V;
      first = base;
      started = true;
    }
    // defensive: the window start never decreases along the traversal (also not where
    // the 64-bin truncation applies to some planes only); never move the accumulator back
    it_min = (it_min < base) ? base : it_min;
    if (it_min - base >= W)
    {
      // Rare case: the window jumped past the whole accumulator (TOF kernel and bins much
      // narrower than the plane spacing). Write all W bins, write zeros for the skipped
      // bins between the old accumulator and the new start, and restart at it_min.
#pragma unroll
      for (int g = 0; g < W / V; ++g)
      {
        float v[V];
#pragma unroll
        for (int j = 0; j < V; ++j)
          v[j] = acc[g * V + j];
        TOFSinoStore<V>::store(row, base + g * V, v, num_tof_bins);
      }
      const int new_base = it_min - (((it_min - phase) % V) + V) % V;
      for (int t = base + W; t < new_base; ++t)
        if (t >= 0 && t < num_tof_bins)
          row[t] = 0.0f;
#pragma unroll
      for (int k = 0; k < W; ++k)
        acc[k] = 0.0f;
      base = new_base;
    }
    while (it_min >= base + V)
    {
      // Normal case: the window moved on by one group or more. The first V bins
      // (base .. base + V - 1) are final: write them and shift the accumulator down by V
      // (acc[k] = acc[k + V]); the V freed slots at the top start at zero.
      float v[V];
#pragma unroll
      for (int j = 0; j < V; ++j)
        v[j] = acc[j];
      TOFSinoStore<V>::store(row, base, v, num_tof_bins);
#pragma unroll
      for (int k = 0; k < W - V; ++k)
        acc[k] = acc[k + V];
#pragma unroll
      for (int k = W - V; k < W; ++k)
        acc[k] = 0.0f;
      base += V;
    }
    // now base <= it_min < base + V, and the whole window (at most max_window bins) fits
    // into acc[it_min - base .. it_min - base + n_tof_weights - 1]

    // --- non-TOF contribution of this plane (cf times the bilinear interpolation)
    const float val = lor.interp(image, ip);

    // --- distribute val over the window bins with the TOF weights
    // weight of bin t: (cdf(upper edge) - cdf(lower edge)) * inv_norm, see tof_kernel.h.
    // Neighbouring bins share an edge: e_lo is the lower edge of the current bin, e_hi its
    // upper edge, which becomes e_lo of the next bin. Bin base + k is acc[k]; the loop runs
    // over all W slots (compile-time indices, see above) and only adds to the slots
    // k_lo <= k < k_hi of the window.
    const float inv_norm = tof_window_inv_norm(kernel, it_f, it_min, n_tof_weights, truncated);
    float e_lo = kernel.cdf(it_min - 0.5f - it_f);
    const int k_lo = it_min - base;
    const int k_hi = k_lo + n_tof_weights;
#pragma unroll
    for (int k = 0; k < W; ++k)
    {
      if (k >= k_lo && k < k_hi)
      {
        const float e_hi = kernel.cdf(base + k + 0.5f - it_f);
        acc[k] += (e_hi - e_lo) * inv_norm * val;
        e_lo = e_hi;
      }
    }
  }

  // ---------------------------------------------------------------------------------
  // No plane had a non-empty window (the LOR crosses the image only outside the TOF
  // range): all TOF bins are zero.
  if (!started)
  {
    for (short it = 0; it < num_tof_bins; ++it)
      row[it] = 0.0f;
    return;
  }

  // After the last plane: write the remaining W bins of the accumulator (bins outside
  // [0, num_tof_bins) are skipped by the store)
#pragma unroll
  for (int g = 0; g < W / V; ++g)
  {
    float v[V];
#pragma unroll
    for (int j = 0; j < V; ++j)
      v[j] = acc[g * V + j];
    TOFSinoStore<V>::store(row, base + g * V, v, num_tof_bins);
  }

  // Bins that the accumulator never covered get no contribution: write zeros. Together
  // with the writes above, the accumulator covered the bins first .. base + W - 1 without
  // gaps, so these are the bins before `first` and after base + W - 1.
  for (int t = 0; t < first && t < num_tof_bins; ++t)
    row[t] = 0.0f;
  for (int t = (base + W > 0 ? base + W : 0); t < num_tof_bins; ++t)
    row[t] = 0.0f;
}

// Call CALL(W, V) with the compile-time accumulator size W (one of tof_acc_size()) and
// store width V (1 or 4) that are only known at run time: one branch per compiled
// combination (18 combinations). Used by the CPU loop and to launch the GPU kernels.
#define TOF_SINO_FWD_DISPATCH(W_RT, V_RT, CALL)       \
  do                                                  \
  {                                                   \
    const int w_ = (W_RT);                            \
    if ((V_RT) == 4)                                  \
    {                                                 \
      if (w_ == 8) { CALL(8, 4); }                    \
      else if (w_ == 12) { CALL(12, 4); }             \
      else if (w_ == 16) { CALL(16, 4); }             \
      else if (w_ == 20) { CALL(20, 4); }             \
      else if (w_ == 24) { CALL(24, 4); }             \
      else if (w_ == 32) { CALL(32, 4); }             \
      else if (w_ == 40) { CALL(40, 4); }             \
      else if (w_ == 48) { CALL(48, 4); }             \
      else { CALL(64, 4); }                           \
    }                                                 \
    else                                              \
    {                                                 \
      if (w_ == 8) { CALL(8, 1); }                    \
      else if (w_ == 12) { CALL(12, 1); }             \
      else if (w_ == 16) { CALL(16, 1); }             \
      else if (w_ == 20) { CALL(20, 1); }             \
      else if (w_ == 24) { CALL(24, 1); }             \
      else if (w_ == 32) { CALL(32, 1); }             \
      else if (w_ == 40) { CALL(40, 1); }             \
      else if (w_ == 48) { CALL(48, 1); }             \
      else { CALL(64, 1); }                           \
    }                                                 \
  } while (0)
