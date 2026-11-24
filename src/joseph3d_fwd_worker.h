#pragma once
#include "cuda_compat.h"
#include "joseph3d_lor_geometry.h"

/// Non-TOF forward projection of LOR i (Joseph's method): the sum over the image planes
/// along the principal axis of the bilinear interpolation of the image at the intersection
/// with the LOR, times cf (the length of the LOR between two planes); see
/// joseph3d_lor_geometry.h. The sum is accumulated in a register and written once.
///
/// One loop for all principal axes (the axis only changes the memory strides): on GPUs,
/// the threads of a warp do not diverge when their LORs have different principal axes.
WORKER_QUALIFIER inline void joseph3d_fwd_worker(std::size_t i,
                                                 const float *lor_start,
                                                 const float *lor_end,
                                                 const float *image,
                                                 const float *image_origin,
                                                 const float *voxel_size,
                                                 float *projection_values,
                                                 const int *image_dim)
{
  JosephLOR lor;
  // LORs that miss the image get 0
  if (!lor.init(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim))
  {
    projection_values[i] = 0.0f;
    return;
  }

  float sum = 0.0f;
  for (int ip = lor.istart; ip <= lor.iend; ++ip)
    sum += lor.bilinear(image, ip);

  projection_values[i] = lor.cf * sum;
}
