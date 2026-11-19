#pragma once
// TOF sinogram back projection of one LOR (Joseph's method), CPU and CUDA.
//
// Adjoint of joseph3d_tof_sino_fwd_worker: same planes, same traversal order, same TOF
// windows and weights (truncated TOF kernel, see tof_kernel.h).
//
// Register window: the image planes are traversed in the direction of increasing TOF
// position, so the TOF window only moves forward along the LOR. The TOF values of the W
// bins starting at bin `base` are kept in buf[0 .. W-1]; all loops over the W slots are
// fully unrolled (compile-time indices), so that the compiler keeps buf in registers on
// the GPU. When the window moves on, buf is shifted down by V and the next V bins are
// loaded - every TOF bin of a LOR is read once from memory, instead of reading all
// window bins for every image plane. The contribution of a plane (TOF-weighted sum of
// the window bins) is added to the image with the adjoint of the bilinear interpolation
// (atomic additions).
//
// Template parameters as for the forward projection:
//   W: buffer size (TOF bins), at least the window size plus V - 1 (tof_acc_size_class)
//   V: load width: 1 (one bin at a time) or 4 (address-aligned groups of 4 bins, one
//      16-byte load on the GPU)
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"

// read the V bins t .. t+V-1 of a TOF projection row (bins outside [0, num_tof_bins) read as 0)
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
  // if all 4 bins are inside the row, row + t is 16-byte aligned (see `phase` below)
  WORKER_QUALIFIER static inline void load(const float *row, int t, float (&v)[4], int num_tof_bins)
  {
    if (t >= 0 && t + 3 < num_tof_bins)
    {
#ifdef __CUDA_ARCH__
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
      for (int j = 0; j < 4; ++j)
        v[j] = (t + j >= 0 && t + j < num_tof_bins) ? row[t + j] : 0.0f;
    }
  }
};

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
  const float *row = projection_values + i * num_tof_bins;

  // skip the back projection of this LOR if all of its TOF bins are zero.
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

  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;
  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
    return;

  // --- TOF parameters: truncated kernel and TOF bin coordinate it_f = ip * at + bt of plane ip
  const float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  const TOFKernelGauss kernel(local_tof_sigma, tof_bin_width, num_sigmas);
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // traverse the planes such that it_f increases (as in the forward projection)
  const bool forward = (at >= 0.0f);
  const int num_planes = lor.iend - lor.istart + 1;
  const int max_window = W - (V - 1); // largest window that fits with aligned groups

  // aligned groups of V bins: bins t with t = phase (mod V) start at a multiple of 4 * V bytes
  const int phase = (V > 1) ? static_cast<int>((V - (reinterpret_cast<std::size_t>(row) / sizeof(float)) % V) % V) : 0;

  float buf[W];
  int base = 0; // TOF bin of buf[0]

  for (int n = 0; n < num_planes; ++n)
  {
    const int ip = forward ? (lor.istart + n) : (lor.iend - n);
    const float it_f = tof_plane_position(ip, at, bt);

    // TOF window of this plane
    int it_min, n_tof_weights;
    bool truncated;
    tof_window(it_f, kernel.half_width_bins, max_window, it_min, n_tof_weights, truncated);

    if (n == 0 || it_min - base >= W)
    {
      // first plane, or the window jumped past the whole buffer: (re)load all W bins
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
      // the first V bins are not needed anymore: shift down by V, load the next V bins
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

    // --- TOF-weighted sum of the window bins it_min .. it_min + n - 1 (buf slots
    // it_min - base + k), looped over all W slots (compile-time indices) with a range
    // check; shared edges. Bins outside [0, num_tof_bins) were loaded as 0.
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

    // --- adjoint of the bilinear interpolation (atomic additions)
    lor.interp_adjoint(image, ip, val * inv_norm);
  }
}
