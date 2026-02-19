#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_tof_lm_fwd_worker(size_t i,
                                                        const float *lor_start,
                                                        const float *lor_end,
                                                        const float *image,
                                                        const float *image_origin,
                                                        const float *voxel_size,
                                                        float *projection_values,
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
  float toAdd = 0.0;                           // variable LM forward projection (to reduce access to projection_values[i])
  float costheta = voxel_size[direction] / cf; // cosine of angle between ray and principal axis

  // get the tof_sigma and tof_center_offset for this LOR depending on whether they are constant or LOR-dependent
  float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

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

  float it_f = istart * at + bt;

  float tof_plane_weight;
  float tof_plane_weights_sum = 0.0;

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

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i0 >= 0 && i0 < n0)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed0(image, n0, n1, n2, i0, i1_f, i2_f);
      }

      i1_f += a1;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i1 >= 0 && i1 < n1)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed1(image, n0, n1, n2, i0_f, i1, i2_f);
      }

      i0_f += a0;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tof_bin_width, local_tof_sigma, tof_bin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i2 >= 0 && i2 < n2)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed2(image, n0, n1, n2, i0_f, i1_f, i2);
      }

      i0_f += a0;
      i1_f += a1;
      it_f += at;
    }
  }

  // the tof-weighted interpolated values has to be corrected for the non-TOF correction factor (cf)
  // and also for the the fact that we truncated the plane range based on num sigmas
  // the expected sum of the TOF weights is (tof_bin_width / cf)

  // projection_values[i] = toAdd * cf * (tof_bin_width / cf) / tof_plane_weights_sum;
  projection_values[i] = toAdd * tof_bin_width / tof_plane_weights_sum;
}
