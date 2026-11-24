#pragma once
#include "cuda_compat.h"
#include "joseph3d_lor_geometry.h"

/// Non-TOF back projection of LOR i (the exact adjoint of joseph3d_fwd_worker): add
/// cf * projection_values[i], distributed with the bilinear weights, to the 4 voxels
/// around the intersection of the LOR with every image plane along the principal axis;
/// see joseph3d_lor_geometry.h. Atomic additions (other LORs can add to the same voxels).
///
/// One loop for all principal axes (the axis only changes the memory strides): on GPUs,
/// the threads of a warp do not diverge when their LORs have different principal axes.
WORKER_QUALIFIER inline void joseph3d_back_worker(std::size_t i,
                                                  const float *lor_start,
                                                  const float *lor_end,
                                                  float *image,
                                                  const float *image_origin,
                                                  const float *voxel_size,
                                                  const float *projection_values,
                                                  const int *image_dim)
{
  if (projection_values[i] == 0)
    return;

  JosephLOR lor;
  // LORs that miss the image do not contribute
  if (!lor.init(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim))
    return;

  const float val = lor.cf * projection_values[i];
  for (int ip = lor.istart; ip <= lor.iend; ++ip)
    lor.bilinear_adjoint(image, ip, val);
}
