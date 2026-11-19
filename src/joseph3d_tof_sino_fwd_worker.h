#pragma once
// TOF sinogram forward projection of one LOR (Joseph's method), CPU and CUDA.
//
// TOF kernel: truncated kernel, see tof_kernel.h.
//
// Register accumulator: the image planes are traversed in the direction of increasing
// TOF position, so the TOF window only moves forward along the LOR. The sums of the W
// TOF bins starting at bin `base` are kept in acc[0 .. W-1]; all loops over the W slots
// are fully unrolled (compile-time indices), so that the compiler keeps acc in
// registers on the GPU. When the window has moved past a bin, the bin is final and is
// written to the output - every output bin is written exactly once (bins outside the
// range covered by the window as zeros). This avoids the read-modify-write of all
// window bins in global memory for every image plane.
//
// Template parameters:
//   W: accumulator size (TOF bins), at least the number of bins overlapping the kernel
//      support plus V - 1 (see tof_acc_size_class); wider windows are truncated to
//      W - V + 1 bins centered on the TOF position.
//   V: store width: 1 (one bin at a time) or 4 (address-aligned groups of 4 bins, one
//      16-byte store on the GPU; `base` is then kept aligned). Groups of 4 pay off if
//      the window moves fast along the LOR (many stores per image plane).
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"

// write the V bins t .. t+V-1 of a projection row (bins outside [0, num_tof_bins) are skipped)
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
  // if all 4 bins are inside the row, row + t is 16-byte aligned (see `phase` below)
  WORKER_QUALIFIER static inline void store(float *row, int t, const float (&v)[4], int num_tof_bins)
  {
    if (t >= 0 && t + 3 < num_tof_bins)
    {
#ifdef __CUDA_ARCH__
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
      for (int j = 0; j < 4; ++j)
        if (t + j >= 0 && t + j < num_tof_bins)
          row[t + j] = v[j];
    }
  }
};

template <int W, int V>
WORKER_QUALIFIER inline void joseph3d_tof_sino_fwd_worker(std::size_t i,
                                                          const float *lor_start,
                                                          const float *lor_end,
                                                          const float *image,
                                                          const float *image_origin,
                                                          const float *voxel_size,
                                                          float *projection_values,
                                                          const int *image_dim,
                                                          float tof_bin_width,
                                                          const float *tof_sigma,
                                                          const float *tof_center_offset,
                                                          float num_sigmas,
                                                          short num_tof_bins,
                                                          unsigned char is_lor_dependent_tof_sigma,
                                                          unsigned char is_lor_dependent_tof_center_offset)
{
  float *row = projection_values + i * num_tof_bins;
  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;

  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
  {
    for (short it = 0; it < num_tof_bins; ++it)
      row[it] = 0.0f;
    return;
  }
  const int istart = lor.istart;
  const int iend = lor.iend;

  // --- TOF parameters: truncated kernel and TOF bin coordinate it_f = ip * at + bt of plane ip
  const float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  const TOFKernelGauss kernel(local_tof_sigma, tof_bin_width, num_sigmas);
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // traverse the planes such that it_f increases
  const bool forward = (at >= 0.0f);
  const int num_planes = iend - istart + 1;
  const int max_window = W - (V - 1); // largest window that fits with aligned groups

  // aligned groups of V bins: bins t with t = phase (mod V) start at a multiple of 4 * V bytes
  const int phase = (V > 1) ? static_cast<int>((V - (reinterpret_cast<std::size_t>(row) / sizeof(float)) % V) % V) : 0;

  float acc[W];
#pragma unroll
  for (int k = 0; k < W; ++k)
    acc[k] = 0.0f;

  int base = 0;  // TOF bin of acc[0]
  int first = 0; // first TOF bin covered by the accumulator (bins below are zero)

  for (int n = 0; n < num_planes; ++n)
  {
    const int ip = forward ? (istart + n) : (iend - n);
    const float it_f = tof_plane_position(ip, at, bt);

    // TOF window of this plane
    int it_min, n_tof_weights;
    bool truncated;
    tof_window(it_f, kernel.half_width_bins, max_window, it_min, n_tof_weights, truncated);

    // move the accumulator forward such that base <= it_min < base + V
    if (n == 0)
    {
      base = it_min - (((it_min - phase) % V) + V) % V;
      first = base;
    }
    // defensive: the window start never decreases along the traversal (also not where
    // the 64-bin truncation applies to some planes only); never move the accumulator back
    it_min = (it_min < base) ? base : it_min;
    if (it_min - base >= W)
    {
      // the window jumped past the whole accumulator: write it, zeros up to the new start
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
      // the first V bins are final: write them, shift the accumulator down by V
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

    // --- non-TOF contribution of this plane (bilinear interpolation, zero outside the image)
    const float val = lor.interp(image, ip);

    // --- TOF weights of the window bins it_min .. it_min + n - 1 (acc slots it_min - base + k),
    // looped over all W slots (compile-time indices) with a range check; shared edges
    const float inv_norm = tof_window_inv_norm(kernel, it_f, it_min, n_tof_weights, truncated);
    float e_lo = kernel.cdf2(it_min - 0.5f - it_f);
    const int k_lo = it_min - base;
    const int k_hi = k_lo + n_tof_weights;
#pragma unroll
    for (int k = 0; k < W; ++k)
    {
      if (k >= k_lo && k < k_hi)
      {
        const float e_hi = kernel.cdf2(base + k + 0.5f - it_f);
        acc[k] += (e_hi - e_lo) * inv_norm * val;
        e_lo = e_hi;
      }
    }
  }

  // write the remaining bins
#pragma unroll
  for (int g = 0; g < W / V; ++g)
  {
    float v[V];
#pragma unroll
    for (int j = 0; j < V; ++j)
      v[j] = acc[g * V + j];
    TOFSinoStore<V>::store(row, base + g * V, v, num_tof_bins);
  }

  // bins outside the range covered by the accumulator
  for (int t = 0; t < first && t < num_tof_bins; ++t)
    row[t] = 0.0f;
  for (int t = (base + W > 0 ? base + W : 0); t < num_tof_bins; ++t)
    row[t] = 0.0f;
}

// Call CALL(W, V) with the compile-time accumulator size W (one of tof_acc_size()) and
// store width V (1 or 4) given at run time.
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
