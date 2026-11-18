#pragma once
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"

// Helper: TOF-weighted sum of the TOF bins of a LOR for one image plane at TOF bin
// coordinate it_f (truncated TOF kernel, same weights as the forward projection, see
// tof_kernel.h).
WORKER_QUALIFIER static inline float _gather_back_tof_weights(
    float it_f,
    const TOFKernelGauss &kernel,
    const float *projection_values, // TOF projection values
    std::size_t lor_offset,         // offset at which TOF values for given LOR start in projection_values
    short num_tof_bins)
{
  float toAdd = 0.0f;
  tof_for_each_weight(kernel, it_f, num_tof_bins, [&](int t, float w)
                      { toAdd += projection_values[lor_offset + t] * w; });
  return toAdd;
}

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
  // skip the backprojection of this LOR if all of its TOF bins are zero.
  // NOTE: testing the sum over TOF bins is not sufficient, since signed input
  // (e.g. gradients) can have non-zero bins that sum to zero.
  std::size_t lor_offset = (std::size_t)i * (std::size_t)num_tof_bins;
  bool all_tof_bins_zero = true;
  for (short j = 0; j < num_tof_bins; ++j)
  {
    if (projection_values[lor_offset + j] != 0.0f)
    {
      all_tof_bins_zero = false;
      break;
    }
  }

  if (all_tof_bins_zero)
  {
    return;
  }

  int n0 = image_dim[0];
  int n1 = image_dim[1];
  int n2 = image_dim[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float cf;

  float a0, a1, a2;
  float b0, b1, b2;

  int istart = -1;
  int iend = -1;

  float d0 = lor_end[3 * i + 0] - lor_start[3 * i + 0];
  float d1 = lor_end[3 * i + 1] - lor_start[3 * i + 1];
  float d2 = lor_end[3 * i + 2] - lor_start[3 * i + 2];

  float dr;
  float val;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxel_size[dir]/cos[dir]
  ray_cube_intersection_joseph(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim, direction, cf, istart, iend);

  // if the ray does not intersect the image cube, return
  // istart and iend are set to -1
  if (istart == -1)
  {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// calculate TOF-related parameters
  float costheta = voxel_size[direction] / cf; // cosine of angle between ray and principal axis

  // get the tof_sigma and tof_center_offset for this LOR depending on whether they are constant or LOR-dependent
  float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  // truncated TOF kernel of this LOR (same as in the forward projection)
  const TOFKernelGauss kernel(local_tof_sigma, tof_bin_width, num_sigmas);

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (lor_end[3 * i + direction] >= lor_start[3 * i + direction]) ? 1.0f : -1.0f;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  // (same float expression as in the forward projection)
  float tof_origin = 0.5f * (lor_start[3 * i + direction] + lor_end[3 * i + direction]) - sign * (0.5f * num_tof_bins - 0.5f) * (tof_bin_width * costheta) + sign * local_tof_center_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tof_bin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tof_bin_width;
  float bt = (image_origin[direction] - tof_origin) / tof_slope;
  float it_f;

  //////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // float val = cf * projection_values[i];

  if (direction == 0)
  {
    dr = d0;

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    for (i0 = istart; i0 <= iend; ++i0)
    {
      i1_f = i0 * a1 + b1;
      i2_f = i0 * a2 + b2;
      it_f = tof_plane_position(i0, at, bt);
      val = cf * _gather_back_tof_weights(it_f, kernel, projection_values, lor_offset, num_tof_bins);
      bilinear_interp_adj_fixed0(image, n0, n1, n2, i0, i1_f, i2_f, val);
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    for (i1 = istart; i1 <= iend; ++i1)
    {
      i0_f = i1 * a0 + b0;
      i2_f = i1 * a2 + b2;
      it_f = tof_plane_position(i1, at, bt);
      val = cf * _gather_back_tof_weights(it_f, kernel, projection_values, lor_offset, num_tof_bins);
      bilinear_interp_adj_fixed1(image, n0, n1, n2, i0_f, i1, i2_f, val);
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    for (i2 = istart; i2 <= iend; ++i2)
    {
      i0_f = i2 * a0 + b0;
      i1_f = i2 * a1 + b1;
      it_f = tof_plane_position(i2, at, bt);
      val = cf * _gather_back_tof_weights(it_f, kernel, projection_values, lor_offset, num_tof_bins);
      bilinear_interp_adj_fixed2(image, n0, n1, n2, i0_f, i1_f, i2, val);
    }
  }
}
