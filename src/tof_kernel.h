#pragma once
// =====================================================================================
// TOF kernel: how much of a point on a LOR ends up in each TOF bin
// =====================================================================================
//
// Shared by all TOF projectors (sinogram and listmode, CPU and CUDA). See also the
// section "Truncation of the Gaussian TOF kernel" in docs/theory_background.rst.
//
// 1. Units: TOF bin coordinates
// -----------------------------
// Along a LOR, the TOF bins are numbered 0 .. num_tof_bins - 1. We describe positions on
// the LOR by a *continuous TOF bin coordinate*: bin t covers the interval
// [t - 0.5, t + 0.5], i.e. the center of bin t is at coordinate t. A point (e.g. the
// intersection of the LOR with an image plane) at coordinate it_f = 7.3 lies in bin 7,
// 0.3 bin widths from its center. All distances below are measured in TOF bins
// (millimeters divided by tof_bin_width).
//
// 2. The TOF kernel and its cumulative function F
// -----------------------------------------------
// Because of the finite time resolution, an annihilation at coordinate it_f is detected
// with a TOF position that is spread around it_f by the (Gaussian) TOF kernel. The
// fraction of events that ends up in TOF bin t is the integral of the kernel over the
// bin:
//
//      w_t = F(t + 0.5 - it_f) - F(t - 0.5 - it_f)        (upper edge minus lower edge)
//
// where F(u) is the *cumulative* kernel (the integral of the kernel from -infinity to
// u). For a Gaussian with standard deviation sigma (in mm):
//
//      F(u) = 0.5 * (1 + erf(u * tof_bin_width / (sqrt(2) * sigma)))
//
// Example (sigma = 1 bin width): a point at it_f = 7.3 gets
//      w_7 = F(0.2) - F(-0.8) ~ 0.367,  w_8 = F(1.2) - F(0.2) ~ 0.306, ...
//
// 3. Truncation (the "support")
// ------------------------------
// The Gaussian has infinite tails, but bins far away get negligible weights and cost
// time. We therefore truncate the kernel at m = num_sigmas * sigma (in bins:
// m = num_sigmas * sigma / tof_bin_width): the kernel is set to zero outside
// [it_f - m, it_f + m] (the *support*) and renormalized to unit area. In terms of F:
// F is held constant outside the support, and the weights are divided by the mass
// inside the support, F(m) - F(-m):
//
//      w_t = (F_trunc(t + 0.5 - it_f) - F_trunc(t - 0.5 - it_f)) / (F(m) - F(-m))
//
//      F_trunc(u) = F(-m) for u <= -m,   F(u) for -m < u < m,   F(m) for u >= m
//
// Consequences (this is why we do it this way):
//  - only bins that overlap the support get a non-zero weight (the "TOF window"),
//  - the weights of all these bins sum to exactly 1 (the sum of the differences
//    "telescopes" to F(m) - F(-m)), so the sum over all TOF bins of a TOF sinogram
//    equals the non-TOF projection (if the TOF bins cover the support),
//  - the weights are continuous in it_f: a bin at the edge of the window enters or leaves
//    it with weight 0. (With the previous model, a bin entered with its full Gaussian
//    weight, so a tiny rounding difference of it_f could make a jump.)
//
// 4. Implementation details
// --------------------------
//  - The code works with cdf2(u) = 2 F(u) - 1 = erf(...) to save operations; the factor
//    0.5 is part of the normalization constant inv_norm = 0.5 / erf(m * scale).
//  - Neighbouring bins share an edge (the upper edge of bin t is the lower edge of bin
//    t + 1), so a window of n bins needs n + 1 evaluations of F, not 2 n.
//  - Windows wider than TOF_MAX_WINDOW_BINS (64) bins are truncated further (64 bins
//    centered on it_f) and normalized over these 64 bins. This only happens for very
//    narrow TOF bins (2 num_sigmas sigma > ~62 bin widths).
//  - The sinogram projectors keep the TOF bins of the current window in registers (see
//    joseph3d_tof_sino_fwd_worker.h); the second half of this file decides how many
//    ("accumulator size") and whether they are written in groups of 4.
//  - A future tabulated (non-Gaussian) kernel (issue #46) only needs to provide another
//    struct with the same members as TOFKernelGauss (half_width_bins, cdf2, inv_norm).
#include "cuda_compat.h"
#include "utils.h"

// functions that are called on the host (to choose kernel variants) and on the device
#ifdef __CUDACC__
#define TOF_HOST_DEVICE __host__ __device__
#else
#define TOF_HOST_DEVICE
#endif

// maximum number of TOF bins a TOF window can have; wider windows are truncated to
// this many bins centered on the TOF position and renormalized
#define TOF_MAX_WINDOW_BINS 64

/// Truncated Gaussian TOF kernel of one LOR (or event), see the explanation above.
/// All lengths are in units of TOF bins.
struct TOFKernelGauss
{
  float half_width_bins; ///< m = num_sigmas * sigma / tof_bin_width: half width of the support
  float scale;           ///< tof_bin_width / (sqrt(2) * sigma): erf argument per TOF bin
  float e_support;       ///< erf(m * scale) = erf(num_sigmas / sqrt(2)): cdf2 at the end of the support
  float inv_norm;        ///< 0.5 / e_support: turns cdf2 differences into weights that sum to 1

  WORKER_QUALIFIER TOFKernelGauss(float sigma, float tof_bin_width, float num_sigmas)
  {
    half_width_bins = num_sigmas * sigma / tof_bin_width;
    scale = tof_bin_width / (1.414213562373095f * sigma);
    // the mass of the Gaussian inside the support, F(m) - F(-m), is
    // 0.5 * (erf(m scale) - erf(-m scale)) = erf(m scale) = e_support; the weight of a bin
    // is 0.5 * (cdf2 difference) / e_support = (cdf2 difference) * inv_norm
    e_support = erff(half_width_bins * scale);
    inv_norm = (e_support > 0.0f) ? 0.5f / e_support : 0.0f;
  }

  /// cdf2(u) = 2 F_trunc(u) - 1, u = distance from the point in TOF bins (held constant
  /// outside the support [-m, m]). The weight of TOF bin t for a point at it_f is
  /// (cdf2(t + 0.5 - it_f) - cdf2(t - 0.5 - it_f)) * inv_norm.
  WORKER_QUALIFIER float cdf2(float u) const
  {
    if (u <= -half_width_bins)
      return -e_support;
    if (u >= half_width_bins)
      return e_support;
    return erff(u * scale);
  }
};

/// Continuous TOF bin coordinate of image plane ip: ip * at + bt (see
/// JosephLOR::tof_line in joseph3d_lor_geometry.h for at and bt).
/// On the GPU, the compiler could compute ip * at + bt with one fused multiply-add
/// (FMA, one rounding) in one kernel and with two operations (two roundings) in another
/// one. The intrinsics __fmul_rn / __fadd_rn forbid this, so that the forward and the
/// back projection compute bit-identical TOF positions (and therefore weights).
WORKER_QUALIFIER inline float tof_plane_position(int ip, float at, float bt)
{
#ifdef __CUDA_ARCH__
  return __fadd_rn(__fmul_rn(static_cast<float>(ip), at), bt);
#else
  const float p = static_cast<float>(ip) * at;
  return p + bt;
#endif
}

/// TOF window of a point at TOF coordinate it_f: the TOF bins it_min .. it_min + n - 1
/// that overlap the support [it_f - m, it_f + m].
///
/// Bin t overlaps the support if t + 0.5 > it_f - m and t - 0.5 < it_f + m, so the first
/// bin is floor(it_f - m + 0.5) and the last bin is floor(it_f + m + 0.5). A window has at
/// most floor(2 m) + 2 bins. (Bins that only touch the support boundary get weight 0,
/// which is harmless.)
///
/// Example: it_f = 7.3, m = 3 -> support [4.3, 10.3], bins 4 .. 10 (n = 7).
///
/// If the window has more than max_bins bins, it is truncated to max_bins bins centered
/// on it_f (truncated = true; the weights are then normalized over the window instead of
/// the support, see tof_window_inv_norm).
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

/// Factor that turns the cdf2 differences of the window bins into weights:
/// inv_norm of the kernel (weights sum to 1 over the support), or - if the window was
/// truncated - 0.5 / (mass inside the truncated window), so that the weights of the
/// window still sum to 1.
WORKER_QUALIFIER inline float tof_window_inv_norm(const TOFKernelGauss &kernel, float it_f, int it_min, int n,
                                                  bool truncated)
{
  if (!truncated)
    return kernel.inv_norm;
  // mass inside the window = F(upper edge of the last bin) - F(lower edge of the first bin)
  const float sum = 0.5f * (kernel.cdf2(it_min + n - 0.5f - it_f) - kernel.cdf2(it_min - 0.5f - it_f));
  return (sum > 0.0f) ? 0.5f / sum : 0.0f;
}

/// Weight of the single TOF bin t for a point at TOF coordinate it_f. Used by the TOF
/// listmode projectors (an event has one TOF bin); gives exactly the same value as the
/// weight of bin t in the TOF sinogram projections.
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

// =====================================================================================
// Accumulator sizes of the TOF sinogram projectors
// =====================================================================================
//
// The TOF sinogram projectors keep the TOF bins of the current TOF window of a LOR in a
// small array ("accumulator" in the forward projection, "buffer" in the back projection;
// see joseph3d_tof_sino_fwd_worker.h). On the GPU this array must live in registers,
// which is only possible if its size W is a compile-time constant. The workers are
// therefore C++ templates in W, compiled for the sizes listed in tof_acc_size(), and each
// LOR uses the smallest size that fits its TOF window ("size class"). A small W is
// faster (fewer registers, short unrolled loops), a W that is too small would truncate
// the window.

#define TOF_NUM_ACC_SIZES 9

/// the supported accumulator sizes W (number of TOF bins kept in registers per LOR):
/// 8, 12, 16, 20, 24, 32, 40, 48, 64 for k = 0 .. 8
TOF_HOST_DEVICE inline int tof_acc_size(int k)
{
  return (k == 0) ? 8 : (k == 1) ? 12 : (k == 2) ? 16 : (k == 3) ? 20 : (k == 4) ? 24 : (k == 5) ? 32 : (k == 6) ? 40 : (k == 7) ? 48 : 64;
}

/// Number of accumulator slots needed for the TOF windows of a kernel with half width m
/// (in TOF bins) and stores of V bins: a window has at most floor(2 m) + 2 bins (see
/// tof_window); with groups of V = 4 bins, the accumulator start is rounded down to a
/// multiple of 4 (alignment), which can cost up to V - 1 = 3 extra slots.
TOF_HOST_DEVICE inline int tof_acc_slots_needed(float half_width_bins, int V)
{
  return static_cast<int>(floorf(2.0f * half_width_bins)) + 2 + (V - 1);
}

/// Size class of a LOR: the smallest supported accumulator size >= the slots needed.
/// If none is large enough, 64 (the window is then truncated to fit, see tof_window).
/// Example: sigma = 25.5 mm, bin width 25.3 mm, num_sigmas 3: m = 3.02, needed slots
/// floor(6.04) + 2 = 8 (V = 1) -> W = 8.
TOF_HOST_DEVICE inline int tof_acc_size_class(float half_width_bins, int V)
{
  const int needed = tof_acc_slots_needed(half_width_bins, V);
  for (int k = 0; k < TOF_NUM_ACC_SIZES - 1; ++k)
    if (tof_acc_size(k) >= needed)
      return tof_acc_size(k);
  return TOF_MAX_WINDOW_BINS;
}

// =====================================================================================
// Store / load width: single TOF bins or groups of 4
// =====================================================================================
//
// The TOF bins of a LOR are written (forward projection) or read (back projection) one
// at a time (V = 1), or in groups of 4 neighbouring bins with one 16-byte memory access
// (V = 4, a "float4"). On the GPU, a memory access costs about the same for 4 bytes as for
// 16 bytes, so groups of 4 reduce the number of memory transactions by up to 4x - but
// they need an accumulator that is 3 bins larger (alignment, see above).
//
// Groups of 4 pay off if the TOF window moves fast along the LOR, i.e. if many TOF bins
// are finished per image plane. The window moves cf / tof_bin_width bins per image plane
// (cf = distance between two planes along the LOR >= the voxel size), so
// min(voxel_size) / tof_bin_width is a lower bound of this speed.
//
// The decision is made for all LORs of a call from this lower bound (not per LOR from
// the exact speed, which depends on the angle of the LOR): 32 neighbouring LORs run
// together on the GPU (a "warp"), and if some of them use V = 1 and others V = 4, the
// warp needs the time of both (measured: ~30 % slower than either width alone).
//
// Benchmarks on a desktop GPU (5.5 M LORs): single bins are better for
// min(voxel) / bin width ~0.1 (wide TOF bins), groups of 4 for ~0.75 (2x faster),
// both equal for ~0.46. Threshold: 0.4.
#define TOF_STORE4_MIN_VOXEL_PER_BIN 0.4f

/// store / load width (1 or 4 TOF bins) of the TOF bins of a LOR (also 1 if the larger
/// accumulator needed for groups of 4 would exceed 64 bins)
WORKER_QUALIFIER inline int tof_store_width(float half_width_bins, const float *voxel_size, float tof_bin_width)
{
  const float vmin = fminf(voxel_size[0], fminf(voxel_size[1], voxel_size[2]));
  return (vmin >= TOF_STORE4_MIN_VOXEL_PER_BIN * tof_bin_width &&
          tof_acc_slots_needed(half_width_bins, 4) <= TOF_MAX_WINDOW_BINS)
             ? 4
             : 1;
}
