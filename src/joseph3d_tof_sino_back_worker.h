#pragma once
// =====================================================================================
// TOF sinogram back projection of one LOR (Joseph's method), CPU and CUDA
// =====================================================================================
//
// Read joseph3d_tof_sino_fwd_worker.h first: this is its mirror image (the adjoint).
//
// 1. What is computed
// -------------------
// For every image plane ip intersected by the LOR:
//   it_f = TOF bin coordinate of the intersection point
//   val  = sum over the TOF bins t of the TOF window of it_f of  w_t(it_f) * projection[t]
//   add cf * val to the image, distributed with the bilinear weights of plane ip
// i.e. each plane collects the TOF bins near it with the same TOF weights as in the
// forward projection, and the result is "smeared" back into the image. Because the
// planes, weights and interpolation are exactly those of the forward projection
// (shared code in joseph3d_lor_geometry.h and tof_kernel.h), the back projection is the
// adjoint (transpose) of the forward projection: <A x, y> = <x, A^T y>.
//
// 2. The sliding window ("buffer")
// --------------------------------
// The straightforward implementation reads all bins of the TOF window from memory for
// every plane (each bin ~70 times per LOR for typical settings). As in the forward
// projection, the planes are visited in the order of increasing TOF position, so the
// window only moves forward. We keep the TOF values of the bins base .. base + W - 1 in
// a small array buf[0 .. W-1] (in registers on the GPU) and, when the window moves on,
// shift buf down and load the next bins: every TOF bin of a LOR is read from memory only
// once.
//
// Example with W = 8, V = 1 (window of n = 5 bins moving forward):
//
//   plane 1: it_min = 3 -> base = 3, load bins 3..10:  buf = [y3 y4 y5 y6 y7 y8 y9 y10]
//   plane 2: it_min = 3 -> no change, use buf[0..4]
//   plane 3: it_min = 4 -> shift down by one, load bin 11 into the last slot:
//                                                      buf = [y4 y5 y6 y7 y8 y9 y10 y11]
//
// Bins outside [0, num_tof_bins) are loaded as 0 (they do not exist, so they contribute
// nothing). As in the forward projection, W is a template parameter and all loops over
// the W slots are unrolled, so that buf stays in registers (see section 4 of
// joseph3d_tof_sino_fwd_worker.h), and V = 4 loads groups of 4 aligned bins with one
// 16-byte load.
//
// 3. Where the time goes
// ----------------------
// The atomic additions to the image (4 per plane) usually dominate; they are the same as
// in the non-TOF back projection. Their speed depends a lot on the order of the LORs
// and the image layout (neighbouring threads should add to neighbouring voxels), see
// docs/performance.rst.
//
// Template parameters as for the forward projection:
//   W: buffer size (TOF bins), at least the window size plus V - 1 (tof_acc_size_class)
//   V: load width: 1 (one bin at a time) or 4 (address-aligned groups of 4 bins)
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"

// Read the V bins t .. t+V-1 of a TOF projection row into v (bins outside
// [0, num_tof_bins) do not exist and are read as 0). Specialized for V = 1 and V = 4.
template <int V>
struct TOFSinoLoad;

template <>
struct TOFSinoLoad<1>
{
  WORKER_QUALIFIER static inline void load(const float *row, int t, float (&v)[1], int num_tof_bins)
  {
    v[0] = (t >= 0 && t < num_tof_bins) ? row[t] : 0.0f;
  }
};

template <>
struct TOFSinoLoad<4>
{
  // if all 4 bins are inside the row, row + t is 16-byte aligned (see `phase` in the worker)
  WORKER_QUALIFIER static inline void load(const float *row, int t, float (&v)[4], int num_tof_bins)
  {
    if (t >= 0 && t + 3 < num_tof_bins)
    {
#ifdef __CUDA_ARCH__
      // GPU: one 16-byte load
      const float4 q = *reinterpret_cast<const float4 *>(row + t);
      v[0] = q.x;
      v[1] = q.y;
      v[2] = q.z;
      v[3] = q.w;
#else
      for (int j = 0; j < 4; ++j)
        v[j] = row[t + j];
#endif
    }
    else
    {
      // group at the start or end of the row: missing bins are 0
      for (int j = 0; j < 4; ++j)
        v[j] = (t + j >= 0 && t + j < num_tof_bins) ? row[t + j] : 0.0f;
    }
  }
};

/// TOF sinogram back projection of LOR i: adds the back projection of the
/// num_tof_bins values projection_values[i * num_tof_bins ...] to the image (atomic
/// additions). See the explanation at the top of this file.
template <int W, int V>
WORKER_QUALIFIER inline void joseph3d_tof_sino_back_worker(std::size_t i,
                                                           const float *lor_start,
                                                           const float *lor_end,
                                                           float *image,
                                                           const float *image_origin,
                                                           const float *voxel_size,
                                                           const float *projection_values,
                                                           const int *image_dim,
                                                           float tof_bin_width,
                                                           const float *tof_sigma,
                                                           const float *tof_center_offset,
                                                           float num_sigmas,
                                                           short num_tof_bins,
                                                           unsigned char is_lor_dependent_tof_sigma,
                                                           unsigned char is_lor_dependent_tof_center_offset)
{
  const float *row = projection_values + i * num_tof_bins; // the TOF bins of this LOR

  // Skip the back projection of this LOR if all of its TOF bins are zero (nothing to add).
  // NOTE: testing the sum over TOF bins is not sufficient, since signed input
  // (e.g. gradients) can have non-zero bins that sum to zero.
  bool all_tof_bins_zero = true;
  for (short j = 0; j < num_tof_bins; ++j)
  {
    if (row[j] != 0.0f)
    {
      all_tof_bins_zero = false;
      break;
    }
  }
  if (all_tof_bins_zero)
    return;

  // ---------------------------------------------------------------------------------
  // Geometry of the LOR (planes, interpolation); nothing to do if it misses the image
  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;
  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
    return;

  // ---------------------------------------------------------------------------------
  // TOF parameters: truncated kernel of this LOR and the TOF bin coordinate of plane ip,
  // it_f = ip * at + bt (identical to the forward projection)
  const float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  const TOFKernelGauss kernel(local_tof_sigma, tof_bin_width, num_sigmas);
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // visit the planes in the order of increasing it_f (as in the forward projection)
  const bool forward = (at >= 0.0f);
  const int num_planes = lor.iend - lor.istart + 1;
  // largest window that fits into the buffer (with groups of 4, base can be up to 3 bins
  // before it_min)
  const int max_window = W - (V - 1);

  // alignment for groups of V bins: the bins t with t = phase (mod V) start at an address
  // that is a multiple of 4 * V bytes (see the forward projection)
  const int phase = (V > 1) ? static_cast<int>((V - (reinterpret_cast<std::size_t>(row) / sizeof(float)) % V) % V) : 0;

  // the buffer: buf[k] holds the TOF value of bin base + k (filled at the first plane)
  float buf[W];
  int base = 0; // TOF bin of buf[0]

  // ---------------------------------------------------------------------------------
  // Loop over the planes, in the order of increasing TOF position
  for (int n = 0; n < num_planes; ++n)
  {
    const int ip = forward ? (lor.istart + n) : (lor.iend - n);
    const float it_f = tof_plane_position(ip, at, bt);

    // TOF window of this plane: bins it_min .. it_min + n_tof_weights - 1
    int it_min, n_tof_weights;
    bool truncated;
    tof_window(it_f, kernel.half_width_bins, max_window, it_min, n_tof_weights, truncated);

    // --- move the buffer such that base <= it_min < base + V
    if (n == 0 || it_min - base >= W)
    {
      // first plane, or (rare) the window jumped past the whole buffer: start the buffer
      // at it_min, rounded down to an aligned bin, and load all W bins
      base = it_min - (((it_min - phase) % V) + V) % V;
#pragma unroll
      for (int g = 0; g < W / V; ++g)
      {
        float v[V];
        TOFSinoLoad<V>::load(row, base + g * V, v, num_tof_bins);
#pragma unroll
        for (int j = 0; j < V; ++j)
          buf[g * V + j] = v[j];
      }
    }
    // defensive: the window start never decreases along the traversal; never move back
    it_min = (it_min < base) ? base : it_min;
    while (it_min >= base + V)
    {
      // normal case: the window moved on. The first V bins are not needed anymore: shift
      // the buffer down by V (buf[k] = buf[k + V]) and load the next V bins into the
      // freed slots at the top (bins base + W - V .. base + W - 1 after the shift)
#pragma unroll
      for (int k = 0; k < W - V; ++k)
        buf[k] = buf[k + V];
      base += V;
      float v[V];
      TOFSinoLoad<V>::load(row, base + W - V, v, num_tof_bins);
#pragma unroll
      for (int j = 0; j < V; ++j)
        buf[W - V + j] = v[j];
    }
    // now the whole window lies in buf[it_min - base .. it_min - base + n_tof_weights - 1]

    // --- TOF-weighted sum of the window bins (the transpose of distributing a value over
    // the window in the forward projection). Same weights and shared edges as in the
    // forward projection; bin base + k is buf[k]; the loop runs over all W slots
    // (compile-time indices) and only uses the slots k_lo <= k < k_hi of the window.
    const float inv_norm = tof_window_inv_norm(kernel, it_f, it_min, n_tof_weights, truncated);
    float e_lo = kernel.cdf2(it_min - 0.5f - it_f);
    const int k_lo = it_min - base;
    const int k_hi = k_lo + n_tof_weights;
    float val = 0.0f;
#pragma unroll
    for (int k = 0; k < W; ++k)
    {
      if (k >= k_lo && k < k_hi)
      {
        const float e_hi = kernel.cdf2(base + k + 0.5f - it_f);
        val += (e_hi - e_lo) * buf[k];
        e_lo = e_hi;
      }
    }

    // --- add cf * val to the 4 voxels around the intersection point of plane ip
    // (adjoint of the bilinear interpolation, atomic additions)
    lor.interp_adjoint(image, ip, val * inv_norm);
  }
}
