#pragma once
// =====================================================================================
// TOF listmode forward projection of one event (Joseph's method), CPU and CUDA
// =====================================================================================
//
// Prerequisites: joseph3d_lor_geometry.h (planes, interpolation, TOF position of a plane)
// and tof_kernel.h (TOF bin coordinates, TOF kernel, weights).
//
// A listmode event is a LOR together with the TOF bin `it` in which it was detected. Its
// expected value is the value of TOF bin `it` of the TOF sinogram projection of its LOR:
//
//   value = sum over the image planes ip of  w_it(it_f(ip)) * cf * interp(image, plane ip)
//
// where w_it(it_f) is the weight of TOF bin `it` for the intersection point with plane ip
// (tof_bin_weight in tof_kernel.h - exactly the weight the TOF sinogram projectors use),
// and cf * interp is the (non-TOF) contribution of the plane. Because only one TOF bin is
// needed, only the planes whose TOF kernel support overlaps bin `it` are visited (a few
// planes around the TOF bin), not all planes of the LOR.
//
// The template parameter KP is the per-call kernel description (TOFGaussParams or
// TOFTableParams, see tof_kernel.h); kp.kernel(i) is the TOF kernel of event i.
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"

/// First and last image plane (istart, iend) whose TOF kernel support can overlap TOF bin
/// `it`, clipped to the planes intersected by the LOR. Returns false if there is none.
///
/// The support of a plane at TOF coordinate it_f is [it_f + u_lo, it_f + u_hi]; it
/// overlaps bin `it` (= [it - 0.5, it + 0.5]) if it - 0.5 - u_hi < it_f < it + 0.5 - u_lo.
/// We use the slightly larger range it_f in [it - u_hi - 1, it - u_lo + 1] (planes outside
/// the overlap simply get weight 0) and convert it to plane numbers with
/// it_f = ip * at + bt, i.e. ip = (it_f - bt) / at. If at < 0 (TOF bins numbered against
/// the plane direction), the two ends swap.
template <typename Kernel>
WORKER_QUALIFIER inline bool tof_lm_plane_range(const JosephLOR &lor, const Kernel &kernel, int it, float at, float bt,
                                                int &istart, int &iend)
{
  const float ip_a = (it - kernel.u_hi - 1.0f - bt) / at;
  const float ip_b = (it - kernel.u_lo + 1.0f - bt) / at;
  // clamp to the intersected planes (+-1) before converting to int (avoids overflow)
  const float lo = fmaxf(fminf(ip_a, ip_b), static_cast<float>(lor.istart - 1));
  const float hi = fminf(fmaxf(ip_a, ip_b), static_cast<float>(lor.iend + 1));
  istart = static_cast<int>(floorf(lo));
  iend = static_cast<int>(ceilf(hi));
  if (istart < lor.istart)
    istart = lor.istart;
  if (iend > lor.iend)
    iend = lor.iend;
  return istart <= iend;
}

/// TOF listmode forward projection of event i: projection_values[i] (see above).
template <typename KP>
WORKER_QUALIFIER inline void joseph3d_tof_lm_fwd_worker(std::size_t i,
                                                        const float *lor_start,
                                                        const float *lor_end,
                                                        const float *image,
                                                        const float *image_origin,
                                                        const float *voxel_size,
                                                        float *projection_values,
                                                        const int *image_dim,
                                                        float tof_bin_width,
                                                        const float *tof_center_offset,
                                                        const short *tof_bin_index,
                                                        short num_tof_bins,
                                                        unsigned char is_lor_dependent_tof_center_offset,
                                                        const KP &kp)
{
  // events that miss the image or have degenerate TOF parameters get 0
  projection_values[i] = 0.0f;

  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;
  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
    return;

  // TOF kernel of this event (Gaussian or table); degenerate parameters (e.g. sigma <= 0):
  // no contribution (this also protects the plane range computation below)
  const auto kernel = kp.kernel(i);
  if (!kernel.valid())
    return;

  // TOF bin coordinate of plane ip: it_f = ip * at + bt (same as in the sinogram projectors)
  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // only the planes around the TOF bin of the event
  const int it = tof_bin_index[i];
  int istart, iend;
  if (!tof_lm_plane_range(lor, kernel, it, at, bt, istart, iend))
    return;

  float sum = 0.0f;
  for (int ip = istart; ip <= iend; ++ip)
  {
    const float w = tof_bin_weight(kernel, tof_plane_position(ip, at, bt), it);
    if (w != 0.0f)
      sum += w * lor.interp(image, ip);
  }
  projection_values[i] = sum;
}
