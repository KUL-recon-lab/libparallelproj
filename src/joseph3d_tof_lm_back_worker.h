#pragma once
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"

WORKER_QUALIFIER inline void joseph3d_tof_lm_back_worker(std::size_t i,
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
                                                         const short *tof_bin_index,
                                                         short num_tof_bins,
                                                         unsigned char is_lor_dependent_tof_sigma,
                                                         unsigned char is_lor_dependent_tof_center_offset)
{
  if (projection_values[i] == 0)
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

  // start and end plane index along the principal axis direction according to TOF limits
  int istart = -1;
  int iend = -1;
  // start / end plane where the ray intersects the image cube along the principal axis direction
  int istart_vol = -1;
  int iend_vol = -1;
  int it = tof_bin_index[i];

  float d0 = lor_end[3 * i + 0] - lor_start[3 * i + 0];
  float d1 = lor_end[3 * i + 1] - lor_start[3 * i + 1];
  float d2 = lor_end[3 * i + 2] - lor_start[3 * i + 2];

  float dr;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxel_size[dir]/cos[dir]
  ray_cube_intersection_joseph(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim, direction, cf, istart_vol, iend_vol);

  // if the ray does not intersect the image cube, return
  // istart and iend are set to -1
  if (istart_vol == -1)
  {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// calculate TOF-related parameters
  float toAdd = 0.0;                           // variable LM forward projection (to reduce access to projection_values[i])
  float costheta = voxel_size[direction] / cf; // cosine of angle between ray and principal axis

  // get the tof_sigma and tof_center_offset for this LOR depending on whether they are constant or LOR-dependent
  float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  // Guard against degenerate TOF parameters that would make the plane-range
  // computation below divide by zero or yield non-finite istart/iend (whose cast
  // to int is undefined). Python-level validation normally prevents this; this
  // protects direct C-API callers. Such an event contributes nothing.
  if (!(tof_bin_width > 0.0f) || !isfinite(tof_bin_width) ||
      !(local_tof_sigma > 0.0f) || !isfinite(local_tof_sigma) ||
      !(num_sigmas > 0.0f) || !isfinite(num_sigmas))
  {
    return;
  }

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (lor_end[3 * i + direction] >= lor_start[3 * i + direction]) ? 1.0f : -1.0f;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  // (same float expression as in the TOF sinogram projectors)
  float tof_origin = 0.5f * (lor_start[3 * i + direction] + lor_end[3 * i + direction]) - sign * (0.5f * num_tof_bins - 0.5f) * (tof_bin_width * costheta) + sign * local_tof_center_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tof_bin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tof_bin_width;
  float bt = (image_origin[direction] - tof_origin) / tof_slope;

  // Truncated TOF kernel (same weights as the TOF sinogram projectors, see tof_kernel.h):
  // the weight of TOF bin `it` for an image plane at TOF bin coordinate it_f is the
  // weight of bin `it` in the sinogram projection of that plane, so a listmode event
  // gets exactly the value of its TOF sinogram bin. Only planes with |it_f - it| <
  // m + 0.5 (m = num_sigmas * sigma / tof_bin_width) can contribute; the plane range
  // below uses m + 1 and is clipped to the planes intersected by the image.
  const TOFKernelGauss kernel(local_tof_sigma, tof_bin_width, num_sigmas);
  const float max_tof_bin_diff = num_sigmas * local_tof_sigma / tof_bin_width + 1.0f;
  istart = static_cast<int>(floorf(((it - sign * max_tof_bin_diff) - bt) / at));
  iend = static_cast<int>(ceilf(((it + sign * max_tof_bin_diff) - bt) / at));
  if (istart < istart_vol)
    istart = istart_vol;
  if (iend > iend_vol)
    iend = iend_vol;

  float it_f;

  float tof_plane_weight;

  // value to distribute: event value times the non-TOF correction factor cf; each plane
  // gets it times its TOF weight (adjoint of the listmode forward projection)
  toAdd = projection_values[i] * cf;

   
  // in case the calculated istart based on TOF is smaller than the first plane intersected by the ray through the image (istart_vol), 
  // we reset istart to istart_vol, because we need to make sure to loop through all planes that are intersected by the ray through the image
  istart = (istart < istart_vol) ? istart_vol : istart;

  // in case the calculated iend based on TOF is larger than the last plane intersected by the ray through the image (iend_vol),
  // we reset iend to iend_vol, because we need to make sure to loop through all planes that are intersected by the ray through the image
  iend = (iend > iend_vol) ? iend_vol : iend;
  
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

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
      // TOF contribution
      tof_plane_weight = tof_bin_weight(kernel, it_f, it);
      bilinear_interp_adj_fixed0(image, n0, n1, n2, i0, i1_f, i2_f, toAdd * tof_plane_weight);

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
      // TOF contribution
      tof_plane_weight = tof_bin_weight(kernel, it_f, it);
      bilinear_interp_adj_fixed1(image, n0, n1, n2, i0_f, i1, i2_f, toAdd * tof_plane_weight);

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
      // TOF contribution
      tof_plane_weight = tof_bin_weight(kernel, it_f, it);
      bilinear_interp_adj_fixed2(image, n0, n1, n2, i0_f, i1_f, i2, toAdd * tof_plane_weight);

    }
  }
}
