#pragma once
#include "cuda_compat.h"
#include "utils.h"

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
  float sign = (lor_end[3 * i + direction] >= lor_start[3 * i + direction]) ? 1.0 : -1.0;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  float tof_origin = 0.5 * (lor_start[3 * i + direction] + lor_end[3 * i + direction]) - sign * (0.5 * num_tof_bins - 0.5) * (tof_bin_width * costheta) + local_tof_center_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tof_bin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tof_bin_width;
  float bt = (image_origin[direction] - tof_origin) / tof_slope;

  // recompute istart and iend based on TOF bin and num_sigmas to consider
  // in LM processing, we don't step through all planes, but only those that are within num_sigmas*local_tof_sigma of the TOF bin of the ray
  // NOTE: that istart and iend can be outside the image boundaries
  // NOTE: we need to loop over all planes between istart and iend even if they are outside the image boundaries, to make sure the TOF weights are normalized correctly
  istart = static_cast<int>(floorf(((it - sign * num_sigmas * local_tof_sigma / tof_bin_width) - bt) / at));
  iend = static_cast<int>(ceilf(((it + sign * num_sigmas * local_tof_sigma / tof_bin_width) - bt) / at));

  float it_f;

  float tof_plane_weight;
  float tof_plane_weights_sum = 0.0;

  // first loop over all potentially contributing planes to get the sum of the TOF weights for normalization
  // this values depends on the number of sigmas, the TOF bin width relative to tof_sigma, and the sampling location of the planes
  // so far I don't see a more direct way to calculate this normalization factor, so I think we have to loop through the planes first
  // in principle, we could buffer all tof_plane_weights here, but this would require quite some memory when executed in parallel
  for (i0 = istart; i0 <= iend; ++i0)
  {
    it_f = i0 * at + bt;
    // TOF contribution
    tof_plane_weights_sum += effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
  }

  // this is the "corrected" value up to the tof_plane_weight that we have to inject into the planes using the adjoint of the bilinear interpolation
  toAdd = projection_values[i] * tof_bin_width / tof_plane_weights_sum;

   
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
      it_f = i0 * at + bt;
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
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
      it_f = i1 * at + bt;
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
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
      it_f = i2 * at + bt;
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
      bilinear_interp_adj_fixed2(image, n0, n1, n2, i0_f, i1_f, i2, toAdd * tof_plane_weight);

    }
  }
}
