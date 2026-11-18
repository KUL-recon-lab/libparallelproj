#pragma once
#include "cuda_compat.h"
#include "utils.h"

// TOF listmode "point" forward projection of event i: trilinear interpolation
// of the image at the center of the event's TOF bin.
WORKER_QUALIFIER inline void tof_lm_point_fwd_worker(std::size_t i,
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
                                                     unsigned char is_lor_dependent_tof_center_offset)
{
  // always initialize the output, also for events that do not contribute
  projection_values[i] = 0.0f;

  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  float f[3];
  if (!tof_bin_center_voxel_coords(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size,
                                   tof_bin_width, local_tof_center_offset, tof_bin_index[i], num_tof_bins, f))
  {
    return;
  }

  projection_values[i] = trilinear_interp(image, image_dim[0], image_dim[1], image_dim[2], f);
}
