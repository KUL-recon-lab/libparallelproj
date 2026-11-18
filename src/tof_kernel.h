#pragma once
// Shared TOF kernel code of the TOF projectors (CPU and CUDA).
//
// TOF kernel model (truncated kernel): the weight of TOF bin t for a point at the
// continuous TOF bin coordinate it_f (bin t covers [t - 0.5, t + 0.5]) is
//
//   w_t = (F(t + 0.5 - it_f) - F(t - 0.5 - it_f)) / (F(+m) - F(-m)),
//
// where F is the cumulative TOF kernel in units of TOF bins, held constant outside
// the support [-m, m] with m = num_sigmas * sigma / tof_bin_width. The weights are
// continuous in it_f (no jump when a bin enters or leaves the support), neighbouring
// bins share an evaluation of F (edge), and the weights of all bins overlapping the
// support sum to exactly 1.
//
// Gaussian kernel: F(u) = 0.5 * (1 + erf(u * tof_bin_width / (sqrt(2) * sigma))).
// The functions below work with 2 F - 1 (the erf itself), the factor 0.5 is part of
// the normalization.
#include "cuda_compat.h"
#include "utils.h"

#ifdef __CUDACC__
#define TOF_HOST_DEVICE __host__ __device__
#else
#define TOF_HOST_DEVICE
#endif

// maximum number of TOF bins a TOF window can have; wider windows are truncated to
// this many bins centered on the TOF position and renormalized
#define TOF_MAX_WINDOW_BINS 64

/// Truncated Gaussian TOF kernel of one LOR (or event).
struct TOFKernelGauss
{
  float half_width_bins; ///< m = num_sigmas * sigma / tof_bin_width
  float scale;           ///< tof_bin_width / (sqrt(2) * sigma): erf argument per TOF bin
  float e_support;       ///< erf(m * scale) = erf(num_sigmas / sqrt(2)): 2 F - 1 at the support end
  float inv_norm;        ///< 0.5 / e_support: normalization of the weights over the support

  WORKER_QUALIFIER TOFKernelGauss(float sigma, float tof_bin_width, float num_sigmas)
  {
    half_width_bins = num_sigmas * sigma / tof_bin_width;
    scale = tof_bin_width / (1.414213562373095f * sigma);
    e_support = erff(half_width_bins * scale);
    inv_norm = (e_support > 0.0f) ? 0.5f / e_support : 0.0f;
  }

  /// 2 F(u) - 1 of the truncated kernel, u in TOF bins relative to the TOF position
  WORKER_QUALIFIER float cdf2(float u) const
  {
    if (u <= -half_width_bins)
      return -e_support;
    if (u >= half_width_bins)
      return e_support;
    return erff(u * scale);
  }
};

/// Continuous TOF bin coordinate of image plane ip: ip * at + bt, rounded explicitly on
/// the GPU (no fused multiply-add), so that all TOF projectors compute the same value.
WORKER_QUALIFIER inline float tof_plane_position(int ip, float at, float bt)
{
#ifdef __CUDA_ARCH__
  return __fadd_rn(__fmul_rn(static_cast<float>(ip), at), bt);
#else
  const float p = static_cast<float>(ip) * at;
  return p + bt;
#endif
}

/// TOF window of a point at TOF coordinate it_f: the TOF bins overlapping the support,
/// it_min .. it_min + n - 1 (at most floor(2 m) + 2 bins). If the window has more than
/// max_bins bins, it is truncated to max_bins bins centered on it_f (truncated = true;
/// the weights are then normalized over the window instead of the support).
WORKER_QUALIFIER inline void tof_window(float it_f, float half_width_bins, int max_bins,
                                        int &it_min, int &n, bool &truncated)
{
  it_min = static_cast<int>(floorf(it_f - half_width_bins + 0.5f));
  n = static_cast<int>(floorf(it_f + half_width_bins + 0.5f)) - it_min + 1;
  truncated = false;
  if (n > max_bins)
  {
    it_min = static_cast<int>(lroundf(it_f)) - max_bins / 2;
    n = max_bins;
    truncated = true;
  }
}

/// Normalization factor of the window weights (2 F - 1 differences -> weights).
WORKER_QUALIFIER inline float tof_window_inv_norm(const TOFKernelGauss &kernel, float it_f, int it_min, int n,
                                                  bool truncated)
{
  if (!truncated)
    return kernel.inv_norm;
  const float sum = 0.5f * (kernel.cdf2(it_min + n - 0.5f - it_f) - kernel.cdf2(it_min - 0.5f - it_f));
  return (sum > 0.0f) ? 0.5f / sum : 0.0f;
}

/// Call f(t, weight) for every TOF bin t of the window of a point at it_f that lies in
/// [0, num_tof_bins) (shared edges: one evaluation of F per bin).
template <typename Func>
WORKER_QUALIFIER inline void tof_for_each_weight(const TOFKernelGauss &kernel, float it_f, short num_tof_bins,
                                                 Func f)
{
  int it_min, n;
  bool truncated;
  tof_window(it_f, kernel.half_width_bins, TOF_MAX_WINDOW_BINS, it_min, n, truncated);
  const float inv = tof_window_inv_norm(kernel, it_f, it_min, n, truncated);
  const int k_start = (it_min < 0) ? -it_min : 0;
  const int k_end = ((it_min + n) > num_tof_bins) ? (num_tof_bins - it_min) : n;
  if (k_start >= k_end)
    return;
  float e_lo = kernel.cdf2(it_min + k_start - 0.5f - it_f);
  for (int k = k_start; k < k_end; ++k)
  {
    const float e_hi = kernel.cdf2(it_min + k + 0.5f - it_f);
    f(it_min + k, (e_hi - e_lo) * inv);
    e_lo = e_hi;
  }
}

/// Weight of the single TOF bin t for a point at TOF coordinate it_f (same value as the
/// weight of bin t in tof_for_each_weight and in the TOF sinogram forward projection).
WORKER_QUALIFIER inline float tof_bin_weight(const TOFKernelGauss &kernel, float it_f, int t)
{
  int it_min, n;
  bool truncated;
  tof_window(it_f, kernel.half_width_bins, TOF_MAX_WINDOW_BINS, it_min, n, truncated);
  if (t < it_min || t >= it_min + n)
    return 0.0f;
  const float inv = tof_window_inv_norm(kernel, it_f, it_min, n, truncated);
  return (kernel.cdf2(t + 0.5f - it_f) - kernel.cdf2(t - 0.5f - it_f)) * inv;
}

// ---------------------------------------------------------------------------
// Accumulator size classes of the TOF sinogram forward projector

#define TOF_NUM_ACC_SIZES 9

/// supported accumulator sizes (TOF bins kept in registers per LOR)
TOF_HOST_DEVICE inline int tof_acc_size(int k)
{
  return (k == 0) ? 8 : (k == 1) ? 12 : (k == 2) ? 16 : (k == 3) ? 20 : (k == 4) ? 24 : (k == 5) ? 32 : (k == 6) ? 40 : (k == 7) ? 48 : 64;
}

/// Number of accumulator slots needed for windows of a kernel with half width m (in
/// bins) and stores of V bins: floor(2 m) + 2 window bins, plus V - 1 for alignment.
TOF_HOST_DEVICE inline int tof_acc_slots_needed(float half_width_bins, int V)
{
  return static_cast<int>(floorf(2.0f * half_width_bins)) + 2 + (V - 1);
}

/// smallest supported accumulator size >= the slots needed (64 if none; the window
/// is then truncated)
TOF_HOST_DEVICE inline int tof_acc_size_class(float half_width_bins, int V)
{
  const int needed = tof_acc_slots_needed(half_width_bins, V);
  for (int k = 0; k < TOF_NUM_ACC_SIZES - 1; ++k)
    if (tof_acc_size(k) >= needed)
      return tof_acc_size(k);
  return TOF_MAX_WINDOW_BINS;
}

// LORs whose TOF window moves at least this many TOF bins per image plane write the
// TOF bins in aligned groups of 4 (one 16-byte store); otherwise one bin at a time
// with a smaller accumulator. Determined with benchmarks on an RTX A6000.
#define TOF_STORE4_MIN_BINS_PER_PLANE 0.7f

/// TOF bins the window moves per image plane (cf / tof_bin_width), from the LOR end
/// points only (principal axis as in ray_cube_intersection_joseph)
WORKER_QUALIFIER inline float tof_bins_per_plane(const float *p0, const float *p1, const float *voxel_size,
                                                 float tof_bin_width)
{
  const float d0 = p1[0] - p0[0], d1 = p1[1] - p0[1], d2 = p1[2] - p0[2];
  const float s0 = d0 * d0, s1 = d1 * d1, s2 = d2 * d2;
  const float sum = s0 + s1 + s2;
  if (sum == 0.0f)
    return 0.0f;
  int dir = 0;
  if (s1 >= s0 && s1 >= s2)
    dir = 1;
  else if (s2 >= s0 && s2 >= s1)
    dir = 2;
  const float sdir = (dir == 0) ? s0 : ((dir == 1) ? s1 : s2);
  return voxel_size[dir] * sqrtf(sum / sdir) / tof_bin_width;
}

/// store width (1 or 4 TOF bins) of a LOR
WORKER_QUALIFIER inline int tof_store_width(float half_width_bins, float bins_per_plane)
{
  return (bins_per_plane >= TOF_STORE4_MIN_BINS_PER_PLANE &&
          tof_acc_slots_needed(half_width_bins, 4) <= TOF_MAX_WINDOW_BINS)
             ? 4
             : 1;
}
