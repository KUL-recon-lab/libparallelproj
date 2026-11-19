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
//  - For the Gaussian, the code works with cdf(u) = 2 F(u) - 1 = erf(...) to save
//    operations; the factor 0.5 is part of the normalization constant
//    inv_norm = 0.5 / erf(m * scale).
//  - Neighbouring bins share an edge (the upper edge of bin t is the lower edge of bin
//    t + 1), so a window of n bins needs n + 1 evaluations of F, not 2 n.
//  - Windows wider than TOF_MAX_WINDOW_BINS (64) bins are truncated further (64 bins
//    centered on it_f) and normalized over these 64 bins. This only happens for very
//    narrow TOF bins (2 num_sigmas sigma > ~62 bin widths).
//  - The sinogram projectors keep the TOF bins of the current window in registers (see
//    joseph3d_tof_sino_fwd_worker.h); the second half of this file decides how many
//    ("accumulator size") and whether they are written in groups of 4.
//  - Two kernel types implement this model with the same interface (see "Kernel
//    interface" below): TOFKernelGauss (Gaussian, erf) and TOFKernelTable (an arbitrary,
//    possibly asymmetric kernel given as a table of F, issue #46). The projectors are
//    templates in the kernel type, so all code below works for both.
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

// =====================================================================================
// Kernel interface
// =====================================================================================
//
// A TOF kernel of one LOR (or event) provides, in units of TOF bins:
//   u_lo, u_hi  support of the kernel relative to the TOF position: [it_f + u_lo, it_f + u_hi]
//               (Gaussian: [-m, m]; a table can be asymmetric)
//   cdf(u)      a version of the cumulative kernel, held constant outside [u_lo, u_hi]
//               (any a * F(u) + b works, since only differences are used)
//   inv_norm    factor that turns cdf differences into weights summing to 1 over the support:
//               weight of bin t = (cdf(t + 0.5 - it_f) - cdf(t - 0.5 - it_f)) * inv_norm
//   valid()     false for degenerate parameters (e.g. sigma <= 0); such LORs / events get
//               no TOF contribution
// and the per-call parameter structs (TOFGaussParams, TOFTableParams below) build the
// kernel of LOR / event i (kernel(i)) and give the width of its support in TOF bins
// (support_bins(i), used to choose the accumulator size without building the kernel).

/// Truncated Gaussian TOF kernel of one LOR (or event), see the explanation above.
/// All lengths are in units of TOF bins.
struct TOFKernelGauss
{
  float half_width_bins; ///< m = num_sigmas * sigma / tof_bin_width: half width of the support
  float u_lo;            ///< -m: start of the support relative to the TOF position
  float u_hi;            ///< +m: end of the support relative to the TOF position
  float scale;           ///< tof_bin_width / (sqrt(2) * sigma): erf argument per TOF bin
  float e_support;       ///< erf(m * scale) = erf(num_sigmas / sqrt(2)): cdf at the end of the support
  float inv_norm;        ///< 0.5 / e_support: turns cdf differences into weights that sum to 1
  bool ok;               ///< parameters valid (sigma, bin width, num_sigmas > 0 and finite)

  WORKER_QUALIFIER TOFKernelGauss(float sigma, float tof_bin_width, float num_sigmas)
  {
    ok = (sigma > 0.0f) && isfinite(sigma) && (tof_bin_width > 0.0f) && isfinite(tof_bin_width) &&
         (num_sigmas > 0.0f) && isfinite(num_sigmas);
    half_width_bins = num_sigmas * sigma / tof_bin_width;
    u_lo = -half_width_bins;
    u_hi = half_width_bins;
    scale = tof_bin_width / (1.414213562373095f * sigma);
    // the mass of the Gaussian inside the support, F(m) - F(-m), is
    // 0.5 * (erf(m scale) - erf(-m scale)) = erf(m scale) = e_support; the weight of a bin
    // is 0.5 * (cdf difference) / e_support = (cdf difference) * inv_norm
    e_support = erff(half_width_bins * scale);
    inv_norm = (e_support > 0.0f) ? 0.5f / e_support : 0.0f;
  }

  WORKER_QUALIFIER bool valid() const { return ok; }

  /// cdf(u) = 2 F_trunc(u) - 1 = erf(u * scale), u = distance from the point in TOF bins
  /// (held constant outside the support [-m, m]). The weight of TOF bin t for a point at
  /// it_f is (cdf(t + 0.5 - it_f) - cdf(t - 0.5 - it_f)) * inv_norm.
  WORKER_QUALIFIER float cdf(float u) const
  {
    if (u <= -half_width_bins)
      return -e_support;
    if (u >= half_width_bins)
      return e_support;
    return erff(u * scale);
  }
};

/// Per-call parameters of the Gaussian TOF kernel (as passed to the projectors).
struct TOFGaussParams
{
  const float *tof_sigma;      ///< TOF resolution sigma (spatial units), 1 or one per LOR / event
  unsigned char lor_dependent; ///< 1: one sigma per LOR / event
  float num_sigmas;            ///< truncation of the kernel in units of sigma
  float tof_bin_width;         ///< TOF bin width (spatial units)

  /// kernel of LOR / event i
  WORKER_QUALIFIER TOFKernelGauss kernel(std::size_t i) const
  {
    return TOFKernelGauss(lor_dependent ? tof_sigma[i] : tof_sigma[0], tof_bin_width, num_sigmas);
  }
  /// width of the support of kernel(i) in TOF bins (u_hi - u_lo, same float operations)
  WORKER_QUALIFIER float support_bins(std::size_t i) const
  {
    const float m = num_sigmas * (lor_dependent ? tof_sigma[i] : tof_sigma[0]) / tof_bin_width;
    return m - (-m);
  }
};

// =====================================================================================
// Tabulated TOF kernel (issue #46)
// =====================================================================================
//
// For TOF kernels that are not Gaussian (e.g. measured kernels, asymmetric kernels of
// detectors with fast and slow components), the cumulative kernel F is given as a table:
//
//   table[k] = F(u_min + k * h),  k = 0 .. n - 1,  h = (u_max - u_min) / (n - 1)
//
// with u in *spatial units* (the units of lor_start / lor_end, e.g. mm): the distance along
// the LOR between the TOF position and the true annihilation point, positive towards
// lor_end. [u_min, u_max] is the support of the kernel (F is held constant outside). F
// must not decrease; it does not need to start at 0 or end at 1 (the weights are
// normalized by F(u_max) - F(u_min)). Between the table points, F is interpolated
// linearly (i.e. the kernel is piecewise constant).
//
// Hardware convention: with the arrival time difference dt = t(lor_start detector) -
// t(lor_end detector), the TOF position is u = c dt / 2 (about 0.1499 mm per ps): an
// annihilation closer to the lor_end detector arrives there earlier, so dt > 0 and u > 0.
//
// Per LOR / event, the kernel can be stretched by a scale s > 0 (same shape, s times
// wider): F_s(u) = F(u / s), support [s u_min, s u_max]. The TOF center offset is a pure
// shift and is not scaled. In TOF bins (what the projectors use), u_bins = u / tof_bin_width.

/// Tabulated TOF kernel of one LOR (or event), all lengths in TOF bins (see above).
struct TOFKernelTable
{
  const float *table; ///< F at the n table points
  int n;              ///< number of table points (>= 2)
  float u_lo;         ///< s * u_min / tof_bin_width: start of the support in TOF bins
  float u_hi;         ///< s * u_max / tof_bin_width: end of the support in TOF bins
  float a, b;         ///< table index of u (TOF bins): f = u * a + b
  float t_first;      ///< table[0] = F at the start of the support
  float t_last;       ///< table[n - 1] = F at the end of the support
  float inv_norm;     ///< 1 / (t_last - t_first): weights sum to 1 over the support
  bool ok;            ///< parameters valid (scale, bin width > 0, F increases over the table)

  WORKER_QUALIFIER TOFKernelTable(const float *table_, int n_, float u_min, float u_max, float s, float tof_bin_width)
  {
    table = table_;
    n = n_;
    u_lo = s * u_min / tof_bin_width;
    u_hi = s * u_max / tof_bin_width;
    // table spacing h in spatial units; a point at u TOF bins is at u * tof_bin_width / s in
    // units of the unscaled table, i.e. at table index (u * tof_bin_width / s - u_min) / h
    const float h = (u_max - u_min) / static_cast<float>(n - 1);
    a = tof_bin_width / (s * h);
    b = -u_min / h;
    t_first = table[0];
    t_last = table[n - 1];
    inv_norm = (t_last > t_first) ? 1.0f / (t_last - t_first) : 0.0f;
    ok = (s > 0.0f) && isfinite(s) && (tof_bin_width > 0.0f) && isfinite(tof_bin_width) && (t_last > t_first);
  }

  WORKER_QUALIFIER bool valid() const { return ok; }

  /// F_s(u) (linear interpolation of the table), u in TOF bins relative to the TOF
  /// position, held constant outside the support [u_lo, u_hi]
  WORKER_QUALIFIER float cdf(float u) const
  {
    if (u <= u_lo)
      return t_first;
    if (u >= u_hi)
      return t_last;
    // continuous table index f (in (0, n - 1), clamped against rounding), table interval
    // k .. k + 1 (k <= n - 2) and position t in [0, 1] within it
    const float f = fminf(fmaxf(u * a + b, 0.0f), static_cast<float>(n - 1));
    const int k = (static_cast<int>(f) < n - 2) ? static_cast<int>(f) : n - 2;
    const float t = f - static_cast<float>(k);
    return table[k] + t * (table[k + 1] - table[k]);
  }
};

/// Per-call parameters of the tabulated TOF kernel (as passed to the *_tab projectors).
struct TOFTableParams
{
  const float *table;          ///< F at n uniformly spaced points from u_min to u_max
  int n;                       ///< number of table points
  float u_min, u_max;          ///< support of the unscaled kernel (spatial units)
  const float *scale;          ///< kernel scale s, 1 or one per LOR / event
  unsigned char lor_dependent; ///< 1: one scale per LOR / event
  float tof_bin_width;         ///< TOF bin width (spatial units)

  WORKER_QUALIFIER TOFKernelTable kernel(std::size_t i) const
  {
    return TOFKernelTable(table, n, u_min, u_max, lor_dependent ? scale[i] : scale[0], tof_bin_width);
  }
  /// width of the support of kernel(i) in TOF bins (u_hi - u_lo, same float operations)
  WORKER_QUALIFIER float support_bins(std::size_t i) const
  {
    const float s = lor_dependent ? scale[i] : scale[0];
    return s * u_max / tof_bin_width - s * u_min / tof_bin_width;
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
/// that overlap the support [it_f + u_lo, it_f + u_hi] of the kernel.
///
/// Bin t overlaps the support if t + 0.5 > it_f + u_lo and t - 0.5 < it_f + u_hi, so the
/// first bin is floor(it_f + u_lo + 0.5) and the last bin is floor(it_f + u_hi + 0.5). A
/// window has at most floor(u_hi - u_lo) + 2 bins. (Bins that only touch the support
/// boundary get weight 0, which is harmless.)
///
/// Example (Gaussian): it_f = 7.3, m = 3 -> support [4.3, 10.3], bins 4 .. 10 (n = 7).
///
/// If the window has more than max_bins bins, it is truncated to max_bins bins centered
/// on the center of the support, it_f + (u_lo + u_hi) / 2 (truncated = true; the weights
/// are then normalized over the window instead of the support, see tof_window_inv_norm).
template <typename Kernel>
WORKER_QUALIFIER inline void tof_window(float it_f, const Kernel &kernel, int max_bins,
                                        int &it_min, int &n, bool &truncated)
{
  it_min = static_cast<int>(floorf(it_f + kernel.u_lo + 0.5f));
  n = static_cast<int>(floorf(it_f + kernel.u_hi + 0.5f)) - it_min + 1;
  truncated = false;
  if (n > max_bins)
  {
    it_min = static_cast<int>(lroundf(it_f + 0.5f * (kernel.u_lo + kernel.u_hi))) - max_bins / 2;
    n = max_bins;
    truncated = true;
  }
}

/// Factor that turns the cdf differences of the window bins into weights: inv_norm of
/// the kernel (weights sum to 1 over the support), or - if the window was truncated -
/// 1 / (cdf difference over the truncated window), so that the weights of the window still
/// sum to 1.
template <typename Kernel>
WORKER_QUALIFIER inline float tof_window_inv_norm(const Kernel &kernel, float it_f, int it_min, int n,
                                                  bool truncated)
{
  if (!truncated)
    return kernel.inv_norm;
  // cdf(upper edge of the last bin) - cdf(lower edge of the first bin)
  const float d = kernel.cdf(it_min + n - 0.5f - it_f) - kernel.cdf(it_min - 0.5f - it_f);
  return (d > 0.0f) ? 1.0f / d : 0.0f;
}

/// Weight of the single TOF bin t for a point at TOF coordinate it_f. Used by the TOF
/// listmode projectors (an event has one TOF bin); gives exactly the same value as the
/// weight of bin t in the TOF sinogram projections.
template <typename Kernel>
WORKER_QUALIFIER inline float tof_bin_weight(const Kernel &kernel, float it_f, int t)
{
  int it_min, n;
  bool truncated;
  tof_window(it_f, kernel, TOF_MAX_WINDOW_BINS, it_min, n, truncated);
  if (t < it_min || t >= it_min + n)
    return 0.0f;
  const float inv = tof_window_inv_norm(kernel, it_f, it_min, n, truncated);
  return (kernel.cdf(t + 0.5f - it_f) - kernel.cdf(t - 0.5f - it_f)) * inv;
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

/// Number of accumulator slots needed for the TOF windows of a kernel whose support is
/// support_bins TOF bins wide (u_hi - u_lo; Gaussian: 2 m) and stores of V bins: a window
/// has at most floor(support_bins) + 2 bins (see tof_window); with groups of V = 4 bins,
/// the accumulator start is rounded down to a multiple of 4 (alignment), which can cost up
/// to V - 1 = 3 extra slots.
TOF_HOST_DEVICE inline int tof_acc_slots_needed(float support_bins, int V)
{
  return static_cast<int>(floorf(support_bins)) + 2 + (V - 1);
}

/// Size class of a LOR: the smallest supported accumulator size >= the slots needed.
/// If none is large enough, 64 (the window is then truncated to fit, see tof_window).
/// Example: sigma = 25.5 mm, bin width 25.3 mm, num_sigmas 3: support 2 m = 6.04 bins,
/// needed slots floor(6.04) + 2 = 8 (V = 1) -> W = 8.
TOF_HOST_DEVICE inline int tof_acc_size_class(float support_bins, int V)
{
  const int needed = tof_acc_slots_needed(support_bins, V);
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
WORKER_QUALIFIER inline int tof_store_width(float support_bins, const float *voxel_size, float tof_bin_width)
{
  const float vmin = fminf(voxel_size[0], fminf(voxel_size[1], voxel_size[2]));
  return (vmin >= TOF_STORE4_MIN_VOXEL_PER_BIN * tof_bin_width &&
          tof_acc_slots_needed(support_bins, 4) <= TOF_MAX_WINDOW_BINS)
             ? 4
             : 1;
}
