#pragma once
#include "cuda_compat.h"
#include "utils.h"

// TOF listmode "point" back projection of event i (adjoint of
// tof_lm_point_fwd_worker): trilinear splatting of the event value at the
// center of the event's TOF bin.
WORKER_QUALIFIER inline void tof_lm_point_back_worker(std::size_t i,
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
                                                      unsigned char is_lor_dependent_tof_center_offset)
{
  float val = projection_values[i];
  if (val == 0.0f)
  {
    return;
  }

  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  float f[3];
  if (!tof_bin_center_voxel_coords(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size,
                                   tof_bin_width, local_tof_center_offset, tof_bin_index[i], num_tof_bins, f))
  {
    return;
  }

  trilinear_interp_adj(image, image_dim[0], image_dim[1], image_dim[2], f, val);
}
