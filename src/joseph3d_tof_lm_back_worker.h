#pragma once
// =====================================================================================
// TOF listmode back projection of one event (Joseph's method), CPU and CUDA
// =====================================================================================
//
// Adjoint of joseph3d_tof_lm_fwd_worker.h (read that first): the value of the event is
// added to the image along the planes around its TOF bin `it`:
//
//   for every plane ip:  add  w_it(it_f(ip)) * value, distributed with the bilinear
//                        weights (times cf), to the voxels of plane ip
//
// with the same planes, TOF weights and interpolation as the forward projection (atomic
// additions, since other events can add to the same voxels at the same time).
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"
#include "joseph3d_lor_geometry.h"
#include "joseph3d_tof_lm_fwd_worker.h" // tof_lm_plane_range

/// TOF listmode back projection of event i (see above).
template <typename KP>
WORKER_QUALIFIER inline void joseph3d_tof_lm_back_worker(std::size_t i,
                                                         const float *lor_start,
                                                         const float *lor_end,
                                                         float *image,
                                                         const float *image_origin,
                                                         const float *voxel_size,
                                                         const float *projection_values,
                                                         const int *image_dim,
                                                         float tof_bin_width,
                                                         const float *tof_center_offset,
                                                         const short *tof_bin_index,
                                                         short num_tof_bins,
                                                         unsigned char is_lor_dependent_tof_center_offset,
                                                         const KP &kp)
{
  // nothing to add for events with value 0
  const float value = projection_values[i];
  if (value == 0.0f)
    return;

  const float *p0 = lor_start + 3 * i;
  const float *p1 = lor_end + 3 * i;
  JosephLOR lor;
  if (!lor.init(p0, p1, image_origin, voxel_size, image_dim))
    return;

  // TOF kernel of this event; degenerate parameters: no contribution
  const auto kernel = kp.kernel(i);
  if (!kernel.valid())
    return;

  const float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];
  float at, bt;
  lor.tof_line(p0, p1, tof_bin_width, num_tof_bins, local_tof_center_offset, image_origin, voxel_size, at, bt);

  // only the planes around the TOF bin of the event (same as the forward projection)
  const int it = tof_bin_index[i];
  int istart, iend;
  if (!tof_lm_plane_range(lor, kernel, it, at, bt, istart, iend))
    return;

  for (int ip = istart; ip <= iend; ++ip)
  {
    const float w = tof_bin_weight(kernel, tof_plane_position(ip, at, bt), it, num_tof_bins);
    if (w != 0.0f)
      lor.interp_adjoint(image, ip, w * value);
  }
}
